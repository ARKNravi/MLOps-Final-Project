import os
import time
import copy
import torch
import torchvision
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torchsummary import summary
from prometheus_client import Gauge, CollectorRegistry, push_to_gateway
from PIL import Image

# Define Prometheus metrics
registry = CollectorRegistry()
train_loss_metric = Gauge('train_loss', 'Training Loss', registry=registry)
val_loss_metric = Gauge('val_loss', 'Validation Loss', registry=registry)
train_accuracy_metric = Gauge('train_accuracy', 'Training Accuracy', registry=registry)
val_accuracy_metric = Gauge('val_accuracy', 'Validation Accuracy', registry=registry)

# Dataset paths
base_dir = os.path.abspath("./dataset")  # Ambil path absolut dari dataset
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Hyperparameters
num_epochs = 1
batch_size = 4
num_classes = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data transformation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

dataloaders = {
    'train': data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes

# Model initialization function
def initialize_model(num_classes):
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

# Training function with Prometheus metrics
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Update Prometheus metrics
            if phase == 'train':
                train_loss_metric.set(epoch_loss)
                train_accuracy_metric.set(epoch_acc)
            elif phase == 'val':
                val_loss_metric.set(epoch_loss)
                val_accuracy_metric.set(epoch_acc)

            # Push metrics to Prometheus PushGateway
            push_to_gateway(
                os.getenv('PROMETHEUS_PUSH_GATEWAY', 'https://prometheus-prod-37-prod-ap-southeast-1.grafana.net/api/prom/push'), 
                job='ml_training', 
                registry=registry
            )

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model

# Main script
if __name__ == "__main__":
    model = initialize_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)
