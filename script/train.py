import os
import time
import copy
import socket
import logging
import logging.handlers
import sys

import torch
import torchvision
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torchsummary import summary
from prometheus_client import Gauge, start_http_server
from PIL import Image

import mlflow
import mlflow.pytorch

# Logging Setup
def setup_logging():
    # Create logs directory if it doesn't exist
    log_dir = '/app/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(f'{log_dir}/training.log'),
            # Console handler
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

# Logger initialization
logger = setup_logging()

# Port and Service Readiness Checks
def wait_for_port(port, host='localhost', timeout=5.0):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                logger.info(f"Successfully connected to {host}:{port}")
                return True
        except OSError:
            if time.time() - start_time >= timeout:
                logger.error(f"Timeout: Unable to connect to {host}:{port}")
                return False
            time.sleep(0.1)

# Prometheus Metrics Server
def start_prometheus_server(port=8000, max_retries=3):
    for attempt in range(max_retries):
        try:
            start_http_server(port, addr='0.0.0.0')
            logger.info(f"Prometheus metrics server started successfully on port {port}")
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries}: Failed to start Prometheus metrics server: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                logger.error("Failed to start Prometheus metrics server after all retries")
                return False

# Prometheus Metrics
train_loss_metric = Gauge('train_loss', 'Training Loss')
val_loss_metric = Gauge('val_loss', 'Validation Loss')
train_accuracy_metric = Gauge('train_accuracy', 'Training Accuracy')
val_accuracy_metric = Gauge('val_accuracy', 'Validation Accuracy')
epoch_metric = Gauge('current_epoch', 'Current Training Epoch')
batch_metric = Gauge('current_batch', 'Current Training Batch')
learning_rate_metric = Gauge('learning_rate', 'Current Learning Rate')

# Dataset and Training Configuration
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Hyperparameters
num_epochs = 10
batch_size = 4
num_classes = 4
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Transformations
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

# Data Loading
def load_datasets():
    try:
        logger.info("Loading datasets...")
        datasets_dict = {
            'train': datasets.ImageFolder(train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(val_dir, data_transforms['val'])
        }

        dataloaders = {
            'train': data.DataLoader(datasets_dict['train'], batch_size=batch_size, shuffle=True, num_workers=4),
            'val': data.DataLoader(datasets_dict['val'], batch_size=batch_size, shuffle=False, num_workers=4)
        }

        dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val']}
        class_names = datasets_dict['train'].classes

        logger.info(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
        logger.info(f"Classes: {class_names}")

        return dataloaders, dataset_sizes, class_names
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

# Model Initialization
def initialize_model(num_classes):
    try:
        logger.info("Initializing the model...")
        model_ft = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        return model_ft
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise

# Training Function
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_type": "ResNet18",
            "optimizer": type(optimizer).__name__,
            "device": str(device)
        })

        for epoch in range(num_epochs):
            logger.info(f'\nEpoch {epoch + 1}/{num_epochs}')
            epoch_metric.set(epoch + 1)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                batch_count = 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    try:
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()

                        # Forward pass
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # Backward pass + optimize only in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # Statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        
                        batch_count += 1
                        batch_metric.set(batch_count)
                        
                        # Update current batch metrics
                        current_loss = loss.item()
                        current_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                        
                        if phase == 'train':
                            train_loss_metric.set(current_loss)
                            train_accuracy_metric.set(current_acc)
                        else:
                            val_loss_metric.set(current_loss)
                            val_accuracy_metric.set(current_acc)

                    except Exception as e:
                        logger.error(f"Error in {phase} phase during training batch: {e}")
                        continue

                # Phase metrics
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                logger.info(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Log to MLflow
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch + 1)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc, step=epoch + 1)

                # Update learning rate metric
                current_lr = optimizer.param_groups[0]['lr']
                learning_rate_metric.set(current_lr)

                # Deep copy the model if we got better accuracy
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Save best model with MLflow
                    mlflow.pytorch.log_model(model, "best_model")

            scheduler.step()

    logger.info(f'\nBest val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Main Execution
def main():
    try:
        # Verify Prometheus server
        if not start_prometheus_server():
            raise RuntimeError("Failed to start Prometheus metrics server")

        # Wait for MLflow service
        if not wait_for_port(5000, 'mlflow'):
            raise RuntimeError("MLflow service is not available")

        # Set up MLflow
        logger.info("Setting up MLflow...")
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("MLOps_Project-Akhir")

        # Load datasets
        dataloaders, dataset_sizes, class_names = load_datasets()

        # Initialize model and training components
        model = initialize_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Print model summary
        logger.info("\nModel Summary:")
        summary(model, (3, 224, 224))

        # Train model
        logger.info("\nStarting Training...")
        model = train_model(
            model, 
            criterion, 
            optimizer, 
            exp_lr_scheduler, 
            num_epochs, 
            dataloaders, 
            dataset_sizes
        )
        
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()