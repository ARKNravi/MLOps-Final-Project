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
from supabase import create_client, Client

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
num_epochs = 1
batch_size = 4
learning_rate = 0.001
momentum = 0.9
scheduler_step_size = 7
scheduler_gamma = 0.1
num_classes = 4
device = torch.device("cpu")

# Training parameters
train_batches = 100
val_batches = 50

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

def send_metric_to_prometheus(metric_name, value, labels=None):
    if labels is None:
        labels = {}
        
    timestamp_ms = int(time.time() * 1000)
    
    # Inisialisasi WriteRequest
    write_req = WriteRequest()
    ts = write_req.timeseries.add()
    
    # Tambahkan labels wajib
    base_labels = [
        ("__name__", metric_name),
        ("job", "mlops_training"),
        ("environment", "github_actions")
    ]
    
    # Tambahkan custom labels
    all_labels = base_labels + list(labels.items())
    
    for name, value in all_labels:
        label = ts.labels.add()
        label.name = name
        label.value = str(value)
    
    # Tambahkan sample
    sample = ts.samples.add()
    sample.value = float(value)
    sample.timestamp = timestamp_ms
    
    # Serialize dan kompres data
    data = write_req.SerializeToString()
    compressed_data = python_snappy.compress(data)
    
    # Kirim ke Prometheus
    try:
        response = requests.post(
            os.environ['PROMETHEUS_REMOTE_WRITE_URL'],
            data=compressed_data,
            auth=(os.environ['PROMETHEUS_USERNAME'], os.environ['PROMETHEUS_API_KEY']),
            headers={
                "Content-Encoding": "snappy",
                "Content-Type": "application/x-protobuf",
                "X-Prometheus-Remote-Write-Version": "0.1.0"
            }
        )
        if response.status_code in [200, 204]:
            print(f"✅ Metric {metric_name} sent successfully")
            return True
        else:
            print(f"❌ Failed to send metric {metric_name}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending metric {metric_name}: {e}")
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

# Supabase configuration
supabase: Client = create_client(
    os.environ.get("SUPABASE_URL", ""),
    os.environ.get("SUPABASE_KEY", "")
)

def save_metrics_to_supabase(metrics, phase="train"):
    """Save metrics to Supabase with simplified structure"""
    try:
        data = {
            "accuracy": float(metrics["accuracy"]),  # Pastikan nilai adalah float
            "loss": float(metrics["loss"]),         # Pastikan nilai adalah float
            "source": phase                         # train, validation, atau test
        }
        
        result = supabase.table("model_metrics").insert(data).execute()
        print(f"✅ Saved {phase} metrics to Supabase")
        return True
    except Exception as e:
        print(f"❌ Error saving metrics to Supabase: {e}")
        return False

def send_batch_log_to_loki(batch_num, total_batches, phase="train", level="debug"):
    """Send batch processing log to Loki"""
    timestamp = int(time.time() * 1e9)
    message = f"Processing batch {batch_num}/{total_batches}"
    
    payload = {
        "streams": [{
            "stream": {
                "level": level,
                "job": "mlops_training",
                "environment": "github_actions"
            },
            "values": [
                [str(timestamp), json.dumps({
                    "message": message,
                    "level": level,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })]
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
        return response.status_code == 204
    except Exception as e:
        print(f"Error sending batch log: {e}")
        return False

def send_batch_metrics_to_loki(batch_metrics, phase="train", level="info"):
    """Send batch metrics to Loki"""
    timestamp = int(time.time() * 1e9)
    
    payload = {
        "streams": [{
            "stream": {
                "level": level,
                "job": "mlops_training",
                "environment": "github_actions"
            },
            "values": [
                [str(timestamp), json.dumps({
                    "message": "Batch metrics",
                    "level": level,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "batch_loss": batch_metrics["loss"],
                    "batch_accuracy": batch_metrics["accuracy"],
                    "batch_time": batch_metrics["time"]
                })]
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
        return response.status_code == 204
    except Exception as e:
        print(f"Error sending batch metrics: {e}")
        return False

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    # Inisialisasi buffer untuk moving average
    train_loss_buffer = []
    train_acc_buffer = []
    val_loss_buffer = []
    val_acc_buffer = []
    window_size = 5  # Ukuran window untuk moving average
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= train_batches:
                break
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Hitung metrik per batch
            batch_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_accuracy = 100 * batch_correct / batch_total
            
            # Tambahkan ke buffer
            train_loss_buffer.append(batch_loss)
            train_acc_buffer.append(batch_accuracy)
            
            # Kirim moving average setiap 5 batch
            if len(train_loss_buffer) >= window_size:
                avg_train_loss = sum(train_loss_buffer[-window_size:]) / window_size
                avg_train_acc = sum(train_acc_buffer[-window_size:]) / window_size
                
                # Kirim ke Prometheus
                send_metric_to_prometheus(
                    "train_loss",
                    avg_train_loss,
                    {"type": "moving_average"}
                )
                send_metric_to_prometheus(
                    "train_accuracy",
                    avg_train_acc,
                    {"type": "moving_average"}
                )
                
                print(f"Batch {i+1}/{train_batches} - Moving Avg Loss: {avg_train_loss:.4f} - Acc: {avg_train_acc:.2f}%")
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                if i >= val_batches:
                    break
                
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Hitung metrik per batch
                batch_loss = loss.item()
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                batch_total = labels.size(0)
                batch_accuracy = 100 * batch_correct / batch_total
                
                # Tambahkan ke buffer
                val_loss_buffer.append(batch_loss)
                val_acc_buffer.append(batch_accuracy)
                
                # Kirim moving average setiap 5 batch
                if len(val_loss_buffer) >= window_size:
                    avg_val_loss = sum(val_loss_buffer[-window_size:]) / window_size
                    avg_val_acc = sum(val_acc_buffer[-window_size:]) / window_size
                    
                    # Kirim ke Prometheus
                    send_metric_to_prometheus(
                        "val_loss",
                        avg_val_loss,
                        {"type": "moving_average"}
                    )
                    send_metric_to_prometheus(
                        "val_accuracy",
                        avg_val_acc,
                        {"type": "moving_average"}
                    )
                    
                    print(f"Val Batch {i+1}/{val_batches} - Moving Avg Loss: {avg_val_loss:.4f} - Acc: {avg_val_acc:.2f}%")
        
        # Batasi ukuran buffer
        if len(train_loss_buffer) > train_batches:
            train_loss_buffer = train_loss_buffer[-train_batches:]
            train_acc_buffer = train_acc_buffer[-train_batches:]
        if len(val_loss_buffer) > val_batches:
            val_loss_buffer = val_loss_buffer[-val_batches:]
            val_acc_buffer = val_acc_buffer[-val_batches:]
        
        # Save best model berdasarkan rata-rata val_loss terakhir
        current_val_loss = sum(val_loss_buffer[-window_size:]) / window_size
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("✨ New best model saved!")
    
    return train_loss_buffer, val_loss_buffer

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