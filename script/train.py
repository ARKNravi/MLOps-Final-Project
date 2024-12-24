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
from datetime import datetime, timezone
import snappy as python_snappy
from remote_write_pb2 import WriteRequest, TimeSeries, Label, Sample
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

def send_metric_to_prometheus(metric_name, value, job="mlops_training"):
    timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Ensure value is a float
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        print(f"Error converting metric value to float: {e}")
        return False
    
    write_req = WriteRequest()
    ts = write_req.timeseries.add()
    
    labels = [
        ("__name__", metric_name),
        ("job", job),
        ("environment", "github_actions")
    ]
    
    for name, label_value in labels:
        label = ts.labels.add()
        label.name = name
        label.value = str(label_value)
    
    sample = ts.samples.add()
    sample.value = value
    sample.timestamp = timestamp_ms
    
    data = write_req.SerializeToString()
    compressed_data = python_snappy.compress(data)
    
    url = os.environ['PROMETHEUS_REMOTE_WRITE_URL']
    username = os.environ['PROMETHEUS_USERNAME']
    password = os.environ['PROMETHEUS_API_KEY']
    
    headers = {
        "Content-Encoding": "snappy",
        "Content-Type": "application/x-protobuf",
        "X-Prometheus-Remote-Write-Version": "0.1.0"
    }
    
    try:
        response = requests.post(
            url,
            data=compressed_data,
            auth=(username, password),
            headers=headers
        )
        return response.status_code in [200, 204]
    except Exception as e:
        print(f"Error sending metric {metric_name}: {e}")
        return False

def send_log_to_loki(log_message, log_level="info", labels=None, numeric_values=None):
    if labels is None:
        labels = {}
    if numeric_values is None:
        numeric_values = {}
    
    # Add default labels
    labels.update({
        "job": "mlops_training",
        "environment": "github_actions",
        "level": log_level
    })
    
    timestamp = int(time.time() * 1e9)  # Convert to nanoseconds
    
    # Create log entry with numeric values
    log_entry = {
        "message": log_message,
        "level": log_level,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **numeric_values  # Include any numeric values
    }
    
    payload = {
            "streams": [{
            "stream": labels,
            "values": [
                [str(timestamp), json.dumps(log_entry)]
            ]
        }]
    }
    
    try:
        response = requests.post(
            f"{os.environ.get('LOKI_URL')}/loki/api/v1/push",
            json=payload,
            auth=(os.environ.get('LOKI_USERNAME'), os.environ.get('LOKI_API_KEY')),
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 204:
            print(f"Failed to send log to Loki: {response.text}")
            return False
        return True
    except Exception as e:
        print(f"Error sending log to Loki: {e}")
        return False

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    # Set debug mode for quick training
    debug_mode = True  # Force debug mode to be True
    print("🔧 Running in debug mode: 1 epoch, 10 batches only")
    num_epochs = 1  # Force 1 epoch
    max_batches = 10  # Force 10 batches
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        print(f"\ntrain phase:")
        for i, (images, labels) in enumerate(train_loader):
            if debug_mode and i >= max_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if i % 5 == 0:  # Log every 5 batches
                print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {loss.item():.4f}")
                
            # Send metrics every 10 batches
            if i % 10 == 0:
                send_metric_to_prometheus(
                    "train_loss",
                    loss.item(),
                    {"epoch": str(epoch + 1), "batch": str(i + 1)}
                )
        
        avg_train_loss = running_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        batch_count = 0
        
        print(f"\nvalidation phase:")
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if debug_mode and i >= max_batches:
                    break
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                batch_count += 1
                
                if i % 5 == 0:  # Log every 5 batches
                    print(f"Validation Batch {i+1}: Loss = {loss.item():.4f}")
        
        avg_val_loss = val_loss / batch_count
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.2f}%')
        
        # Send epoch metrics
        send_metric_to_prometheus("epoch_train_loss", avg_train_loss, {"epoch": str(epoch + 1)})
        send_metric_to_prometheus("epoch_val_loss", avg_val_loss, {"epoch": str(epoch + 1)})
        send_metric_to_prometheus("epoch_val_accuracy", val_accuracy, {"epoch": str(epoch + 1)})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

# Main script
if __name__ == "__main__":
    # Initialize model and move to device
    model = initialize_model(num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Create data loaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
    }
    
    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )
    
    print("Training completed!")