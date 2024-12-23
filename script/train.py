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
from prometheus_client import start_http_server, Gauge
import time
import json

# MLflow configuration
MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set MLflow credentials through environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('DAGSHUB_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('DAGSHUB_TOKEN', '')

# Print for debugging
print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"DagsHub Username: {os.environ.get('DAGSHUB_USERNAME')}")
print("Attempting to connect to MLflow...")

try:
    # Test MLflow connection
    mlflow.set_experiment("default")
    print("Successfully connected to MLflow")
except Exception as e:
    print(f"Error connecting to MLflow: {e}")
    raise

# Hyperparameters
num_epochs = 3
batch_size = 4
learning_rate = 0.001
momentum = 0.9
scheduler_step_size = 7
scheduler_gamma = 0.1
num_classes = 4
device = torch.device("cpu")

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
    'train': data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
    'val': data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2)
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

# Add Prometheus metrics
TRAIN_LOSS = Gauge('train_loss', 'Training loss')
TRAIN_ACC = Gauge('train_accuracy', 'Training accuracy')
VAL_LOSS = Gauge('val_loss', 'Validation loss')
VAL_ACC = Gauge('val_accuracy', 'Validation accuracy')
LEARNING_RATE = Gauge('learning_rate', 'Current learning rate')
EPOCH_TIME = Gauge('epoch_time', 'Time per epoch in seconds')
BATCH_TIME = Gauge('batch_time', 'Time per batch in seconds')
GPU_MEMORY = Gauge('gpu_memory_usage', 'GPU memory usage in MB')

# Start Prometheus metrics server
start_http_server(8000)

def log_to_grafana(metric_name, value, labels=None):
    if not labels:
        labels = {}
    
    # Update Prometheus metrics
    if metric_name == 'train_loss':
        TRAIN_LOSS.set(value)
    elif metric_name == 'train_accuracy':
        TRAIN_ACC.set(value)
    elif metric_name == 'val_loss':
        VAL_LOSS.set(value)
    elif metric_name == 'val_accuracy':
        VAL_ACC.set(value)
    
    # Log to Grafana Cloud
    timestamp = int(datetime.now().timestamp() * 1000)
    metric = {
        "metrics": [{
            "name": metric_name,
            "value": value,
            "timestamp": timestamp,
            "labels": labels
        }]
    }
    
    headers = {
        "Authorization": f"Bearer {os.environ.get('PROMETHEUS_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            os.environ.get('PROMETHEUS_REMOTE_WRITE_URL'),
            json=metric,
            headers=headers,
            auth=(os.environ.get('PROMETHEUS_USERNAME'), os.environ.get('PROMETHEUS_API_KEY'))
        )
        
        # Send to Loki
        log_message = {
            "level": "info",
            "message": f"Metric logged: {metric_name}={value}",
            "metric_name": metric_name,
            "value": value,
            "timestamp": timestamp
        }
        
        loki_payload = {
            "streams": [{
                "stream": {
                    "job": "mlops-training",
                    "level": "info"
                },
                "values": [[str(timestamp), json.dumps(log_message)]]
            }]
        }
        
        loki_response = requests.post(
            f"{os.environ.get('LOKI_URL')}/loki/api/v1/push",
            json=loki_payload,
            auth=(os.environ.get('LOKI_USERNAME'), os.environ.get('LOKI_API_KEY'))
        )
        print(f"Log sent to Loki: {log_message}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error logging metric: {e}")
        return False

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("Starting training...")
    print(f"Training on device: {device}")
    print(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("scheduler_step_size", scheduler_step_size)
        mlflow.log_param("scheduler_gamma", scheduler_gamma)
        mlflow.log_param("num_epochs", num_epochs)

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch}/{num_epochs-1}')
            print('-' * 10)
            epoch_start_time = time.time()
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                batch_times = []

                # Add progress tracking
                total_batches = len(dataloaders[phase])
                print(f"\n{phase} phase:")

                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    if batch_idx % 10 == 0:  # Print every 10 batches
                        print(f'Batch {batch_idx}/{total_batches}', end='\r')
                        
                    batch_start_time = time.time()
                    
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

                    # Log batch metrics
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    BATCH_TIME.set(batch_time)
                    
                    if batch_idx % 10 == 0:  # Log every 10 batches
                        current_loss = loss.item()
                        print(f"\nBatch {batch_idx}/{total_batches} - Loss: {current_loss:.4f}")
                
                # Calculate and log metrics
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Log metrics to MLflow
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc.item(), step=epoch)
                
                # Log metrics to Grafana
                log_to_grafana(f"{phase}_loss", epoch_loss, {
                    "epoch": str(epoch),
                    "phase": phase
                })
                log_to_grafana(f"{phase}_accuracy", epoch_acc.item(), {
                    "epoch": str(epoch),
                    "phase": phase
                })
                
                # Log learning rate
                if phase == 'train':
                    current_lr = optimizer.param_groups[0]['lr']
                    LEARNING_RATE.set(current_lr)
                    log_to_grafana("learning_rate", current_lr, {"epoch": str(epoch)})
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
                # Save best model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            EPOCH_TIME.set(epoch_time)
            log_to_grafana("epoch_time", epoch_time, {"epoch": str(epoch)})
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            if phase == 'train':
                scheduler.step()

        # Save the best model
        model.load_state_dict(best_model_wts)
        mlflow.pytorch.log_model(model, "best_model")

        # Log final metrics
        mlflow.log_metric("best_accuracy", best_acc.item())

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