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

def send_log_to_loki(log_message, log_level="info", labels=None):
    if labels is None:
        labels = {}
    
    # Add default labels
    labels.update({
        "job": "mlops_training",
        "environment": "github_actions",
        "level": log_level
    })
    
    timestamp = int(time.time() * 1e9)  # Convert to nanoseconds
    
    payload = {
        "streams": [{
            "stream": labels,
            "values": [
                [str(timestamp), json.dumps({
                    "message": log_message,
                    "level": log_level,
                    **labels
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
        if response.status_code != 204:
            print(f"Failed to send log to Loki: {response.text}")
    except Exception as e:
        print(f"Error sending log to Loki: {e}")

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    send_log_to_loki("Starting model training", "info", {"stage": "training_start"})
    print("Starting training...")
    print(f"Training on device: {device}")
    print(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")

    with mlflow.start_run() as run:
        send_log_to_loki(f"MLflow run started with ID: {run.info.run_id}", "info", {"stage": "mlflow_start"})
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Log hyperparameters
        params = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "scheduler_step_size": scheduler_step_size,
            "scheduler_gamma": scheduler_gamma,
            "num_epochs": num_epochs
        }
        mlflow.log_params(params)
        send_log_to_loki(f"Hyperparameters set: {json.dumps(params)}", "info", {"stage": "hyperparameters"})

        for epoch in range(num_epochs):
            send_log_to_loki(f"Starting epoch {epoch}/{num_epochs-1}", "info", {
                "stage": "epoch_start",
                "epoch": str(epoch)
            })
            
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
                send_log_to_loki(f"Starting {phase} phase with {total_batches} batches", "info", {
                    "stage": "phase_start",
                    "phase": phase,
                    "epoch": str(epoch)
                })
                print(f"\n{phase} phase - Total batches: {total_batches}")

                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    if batch_idx % 5 == 0:  # Print progress every 5 batches
                        send_log_to_loki(f"Processing batch {batch_idx}/{total_batches}", "debug", {
                            "stage": "batch_progress",
                            "phase": phase,
                            "epoch": str(epoch),
                            "batch": str(batch_idx)
                        })
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
                                
                                metrics_info = {
                                    "batch_loss": batch_loss,
                                    "batch_accuracy": float(batch_acc),
                                    "batch_time": batch_time
                                }
                                send_log_to_loki(f"Batch metrics: {json.dumps(metrics_info)}", "info", {
                                    "stage": "batch_metrics",
                                    "phase": phase,
                                    "epoch": str(epoch),
                                    "batch": str(batch_idx)
                                })
                                
                                if phase == 'train':
                                    send_metric_to_prometheus("train_loss", batch_loss)
                                    send_metric_to_prometheus("train_accuracy", float(batch_acc))
                                else:
                                    send_metric_to_prometheus("val_loss", batch_loss)
                                    send_metric_to_prometheus("val_accuracy", float(batch_acc))
                                print(f"Batch {batch_idx} - Loss: {batch_loss:.4f}, Accuracy: {float(batch_acc):.4f}")
                        except Exception as e:
                            error_msg = f"Error sending metrics to Prometheus: {str(e)}"
                            send_log_to_loki(error_msg, "error", {
                                "stage": "metrics_error",
                                "phase": phase,
                                "epoch": str(epoch),
                                "batch": str(batch_idx)
                            })
                            print(error_msg)
                            continue
                            
                    except Exception as e:
                        error_msg = f"Error processing batch {batch_idx}: {str(e)}"
                        send_log_to_loki(error_msg, "error", {
                            "stage": "batch_error",
                            "phase": phase,
                            "epoch": str(epoch),
                            "batch": str(batch_idx)
                        })
                        print(error_msg)
                        continue

                send_log_to_loki(f"Completed {phase} phase", "info", {
                    "stage": "phase_complete",
                    "phase": phase,
                    "epoch": str(epoch)
                })
                print(f"Completed {phase} phase")
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Log metrics for the epoch
                try:
                    metrics_info = {
                        "epoch_loss": float(epoch_loss),
                        "epoch_accuracy": float(epoch_acc)
                    }
                    
                    if phase == 'train':
                        current_lr = optimizer.param_groups[0]['lr']
                        metrics_info["learning_rate"] = current_lr
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
                    
                    send_log_to_loki(f"Epoch metrics: {json.dumps(metrics_info)}", "info", {
                        "stage": "epoch_metrics",
                        "phase": phase,
                        "epoch": str(epoch)
                    })
                except Exception as e:
                    error_msg = f"Error logging epoch metrics: {str(e)}"
                    send_log_to_loki(error_msg, "error", {
                        "stage": "metrics_error",
                        "phase": phase,
                        "epoch": str(epoch)
                    })
                    print(error_msg)

                # Save best model weights
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    send_log_to_loki(f"New best model saved with accuracy: {float(best_acc)}", "info", {
                        "stage": "model_checkpoint",
                        "epoch": str(epoch)
                    })

            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            try:
                send_metric_to_prometheus("epoch_time", epoch_time)
                mlflow.log_metric("epoch_time", epoch_time, step=epoch)
                send_log_to_loki(f"Epoch {epoch} completed in {epoch_time:.2f} seconds", "info", {
                    "stage": "epoch_complete",
                    "epoch": str(epoch),
                    "epoch_time": str(epoch_time)
                })
            except Exception as e:
                error_msg = f"Error logging epoch time: {str(e)}"
                send_log_to_loki(error_msg, "error", {
                    "stage": "metrics_error",
                    "epoch": str(epoch)
                })
                print(error_msg)

            if phase == 'train':
                scheduler.step()

        send_log_to_loki("Training completed!", "info", {"stage": "training_complete"})
        print("Training completed!")
        
        # Save the best model
        model.load_state_dict(best_model_wts)
        mlflow.pytorch.log_model(model, "best_model")
        send_log_to_loki(f"Best model saved with accuracy: {float(best_acc)}", "info", {
            "stage": "model_saved",
            "final_accuracy": str(float(best_acc))
        })

        # Log final metrics
        mlflow.log_metric("best_accuracy", best_acc.item())

        # Perform Model Registry if accuracy threshold is met
        if best_acc >= accuracy_threshold:
            send_log_to_loki(f"Model meets accuracy threshold ({accuracy_threshold * 100}%). Registering model...", "info", {
                "stage": "model_registry",
                "accuracy": str(float(best_acc))
            })
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/best_model",
                "ProjectAkhirModelRegistry"
            )
            send_log_to_loki(f"Model registered with name: {result.name}, version: {result.version}", "info", {
                "stage": "model_registered",
                "model_name": result.name,
                "model_version": str(result.version)
            })
        else:
            send_log_to_loki(f"Model accuracy {best_acc:.4f} did not meet threshold ({accuracy_threshold * 100}%).", "warning", {
                "stage": "model_registry",
                "accuracy": str(float(best_acc))
            })

        # Log confusion matrix for validation data
        log_confusion_matrix(model, dataloaders['val'], class_names)
        send_log_to_loki("Confusion matrix generated and saved", "info", {"stage": "confusion_matrix"})

    return model

# Main script
if __name__ == "__main__":
    model = initialize_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)