import os
import copy
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.pytorch
import requests
from datetime import datetime

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')

# Hyperparameters
num_epochs = 3
batch_size = 4
learning_rate = 0.001
momentum = 0.9
scheduler_step_size = 7
scheduler_gamma = 0.1
num_classes = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Accuracy threshold for model registry
accuracy_threshold = 0.8  

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

# Dataset paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

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

# Function to log confusion matrix as an artifact
def log_confusion_matrix(model, dataloader, class_names):
    y_true, y_pred = [], []

    # Switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    # Save confusion matrix as a PNG file
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Log the confusion matrix image as an artifact in MLflow
    mlflow.log_artifact(confusion_matrix_path)

# Add Grafana Cloud logging
def log_to_grafana(metric_name, value, labels=None):
    if not labels:
        labels = {}
    
    timestamp = int(datetime.now().timestamp() * 1000)
    
    # Prometheus metric format
    metric = {
        "metrics": [{
            "name": metric_name,
            "value": value,
            "timestamp": timestamp,
            "labels": labels
        }]
    }
    
    headers = {
        "Authorization": f"Bearer {os.getenv('GRAFANA_CLOUD_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            os.getenv('PROMETHEUS_REMOTE_WRITE_URL'),
            json=metric,
            headers=headers
        )
        print(f"Metric logged: {metric_name}={value}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error logging metric: {e}")
        return False

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("scheduler_step_size", scheduler_step_size)
        mlflow.log_param("scheduler_gamma", scheduler_gamma)
        mlflow.log_param("num_epochs", num_epochs)

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

                # Log to Grafana Cloud
                log_to_grafana(f"{phase}_loss", epoch_loss, {
                    "epoch": str(epoch),
                    "phase": phase
                })
                log_to_grafana(f"{phase}_accuracy", epoch_acc, {
                    "epoch": str(epoch),
                    "phase": phase
                })

                # Save best model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # Save the best model
        model.load_state_dict(best_model_wts)

        # Log the best model to MLflow
        mlflow.pytorch.log_model(model, "best_model")

        # Perform Model Registry if accuracy threshold is met
        if best_acc >= accuracy_threshold:
            print(f"Model meets accuracy threshold ({accuracy_threshold * 100}%). Registering model...")
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/best_model",
                "ProjectAkhirModelRegistry"
            )
            print(f"Model registered with name: {result.name}, version: {result.version}")
        else:
            print(f"Model accuracy {best_acc:.4f} did not meet threshold ({accuracy_threshold * 100}%).")

        # Log confusion matrix for validation data
        log_confusion_matrix(model, dataloaders['val'], class_names)

    return model

# Main script
if __name__ == "__main__":
    model = initialize_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)