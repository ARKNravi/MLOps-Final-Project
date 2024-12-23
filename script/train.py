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
                print(f"\n{phase} phase - Total batches: {total_batches}")

                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    if batch_idx % 5 == 0:  # Print progress every 5 batches
                        print(f"Processing batch {batch_idx}/{total_batches}")
                    
                    try:
                        batch_start_time = time.time()
                        inputs = inputs.to(device)
                        labels = labels.to(device)

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

                        # Calculate and log batch metrics
                        batch_time = time.time() - batch_start_time
                        batch_times.append(batch_time)
                        
                        try:
                            send_metric_to_prometheus("batch_time", batch_time)
                            if batch_idx % 10 == 0:  # Log every 10 batches
                                batch_loss = loss.item()
                                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                                
                                if phase == 'train':
                                    send_metric_to_prometheus("train_loss", batch_loss)
                                    send_metric_to_prometheus("train_accuracy", float(batch_acc))
                                else:
                                    send_metric_to_prometheus("val_loss", batch_loss)
                                    send_metric_to_prometheus("val_accuracy", float(batch_acc))
                                print(f"Batch {batch_idx} - Loss: {batch_loss:.4f}, Accuracy: {float(batch_acc):.4f}")
                        except Exception as e:
                            print(f"Error sending metrics to Prometheus: {e}")
                            continue
                            
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                        continue

                print(f"Completed {phase} phase")
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Log metrics for the epoch
                try:
                    if phase == 'train':
                        current_lr = optimizer.param_groups[0]['lr']
                        send_metric_to_prometheus("learning_rate", current_lr)
                        send_metric_to_prometheus("train_loss", epoch_loss)
                        send_metric_to_prometheus("train_accuracy", float(epoch_acc))
                        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                        mlflow.log_metric("train_accuracy", float(epoch_acc), step=epoch)
                        mlflow.log_metric("learning_rate", current_lr, step=epoch)
                    else:
                        send_metric_to_prometheus("val_loss", epoch_loss)
                        send_metric_to_prometheus("val_accuracy", float(epoch_acc))
                        mlflow.log_metric("val_loss", epoch_loss, step=epoch)
                        mlflow.log_metric("val_accuracy", float(epoch_acc), step=epoch)
                except Exception as e:
                    print(f"Error logging epoch metrics: {e}")

                # Save best model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            try:
                send_metric_to_prometheus("epoch_time", epoch_time)
                mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            except Exception as e:
                print(f"Error logging epoch time: {e}")

            if phase == 'train':
                scheduler.step()

        print("Training completed!")
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