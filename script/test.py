import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import initialize_model, device, save_metrics_to_supabase  # Import save_metrics_to_supabase from train.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, start_run
from prometheus_client import start_http_server, Gauge, CollectorRegistry, push_to_gateway, generate_latest
import time
import datetime
import requests
import json
import snappy
from prometheus_client.core import Sample, Metric
from google.protobuf.timestamp_pb2 import Timestamp

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')

# Dataset paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
test_dir = os.path.join(base_dir, "test")

# Validate paths
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"

# Data transformation for the test set
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the test dataset
test_dataset = datasets.ImageFolder(test_dir, data_transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
dataset_sizes = {'test': len(test_dataset)}

# Load the trained model
num_classes = 4  # Update as per your dataset
model = initialize_model(num_classes).to(device)

# Load the saved best model weights
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/best_model_weights.pth"))
assert os.path.exists(model_path), f"Model weights file not found: {model_path}"
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Function to log confusion matrix as an artifact
def log_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
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

# Test the model with MLflow logging
def computeTestSetAccuracyAndLogConfusionMatrix(model, criterion, dataloader, class_names):
    test_loss, test_corrects = 0.0, 0
    y_true, y_pred = [], []
    
    # Set debug mode for quick testing
    debug_mode = True  # Force debug mode to be True
    print("🔧 Running in debug mode: 10 batches only")
    max_batches = 10  # Force 10 batches

    with start_run():  # Start an MLflow run
        # Log testing parameters (to match training phase logs)
        mlflow.log_param("phase", "test")
        mlflow.log_param("batch_size", dataloader.batch_size)
        mlflow.log_param("dataset_size", len(dataloader.dataset))

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if debug_mode and i >= max_batches:
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                # Compute predictions and collect true/false labels
                _, predictions = torch.max(outputs, 1)
                test_corrects += torch.sum(predictions == labels.data)

                # Collect all labels and predictions
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # Calculate average loss and accuracy
        avg_loss = test_loss / len(dataloader.dataset)
        avg_accuracy = test_corrects.double() / len(dataloader.dataset)

        # Log metrics to MLflow
        mlflow.log_metric("test_loss", avg_loss)
        mlflow.log_metric("test_accuracy", avg_accuracy)

        # Save test metrics to Supabase
        save_metrics_to_supabase({
            "accuracy": float(avg_accuracy),
            "loss": float(avg_loss)
        }, phase="test")

        # Log confusion matrix as an artifact
        log_confusion_matrix(y_true, y_pred, class_names)

        # Print results
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_accuracy:.4f}")

        # Return loss and accuracy
        return avg_loss, avg_accuracy

# Add Prometheus metrics for testing
TEST_LOSS = Gauge('test_loss', 'Test loss')
TEST_ACC = Gauge('test_accuracy', 'Test accuracy')
TEST_F1 = Gauge('test_f1_score', 'Test F1 Score')
TEST_PRECISION = Gauge('test_precision', 'Test Precision')
TEST_RECALL = Gauge('test_recall', 'Test Recall')

def log_test_metrics_to_grafana(metrics_dict):
    for metric_name, value in metrics_dict.items():
        # Update Prometheus metrics
        if metric_name == 'test_loss':
            TEST_LOSS.set(value)
        elif metric_name == 'test_accuracy':
            TEST_ACC.set(value)
        elif metric_name == 'test_f1':
            TEST_F1.set(value)
        elif metric_name == 'test_precision':
            TEST_PRECISION.set(value)
        elif metric_name == 'test_recall':
            TEST_RECALL.set(value)
        
        # Log to Grafana Cloud
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        metric = {
            "metrics": [{
                "name": metric_name,
                "value": value,
                "timestamp": timestamp
            }]
        }
        
        try:
            response = requests.post(
                os.getenv('PROMETHEUS_REMOTE_WRITE_URL'),
                json=metric,
                headers={
                    "Authorization": f"Bearer {os.getenv('PROMETHEUS_API_KEY')}",
                    "Content-Type": "application/json"
                },
                auth=(os.getenv('PROMETHEUS_USERNAME'), os.getenv('PROMETHEUS_API_KEY'))
            )
            print(f"Test metric logged: {metric_name}={value}")
        except Exception as e:
            print(f"Error logging test metric: {e}")

def create_write_request(metric_name, value, labels):
    timestamp = Timestamp()
    timestamp.GetCurrentTime()
    
    # Buat sampel metrik
    sample = Sample(
        name=metric_name,
        labels=labels,
        value=float(value),
        timestamp=int(time.time() * 1000)
    )
    
    # Buat metrik
    metric = Metric(metric_name, f'Metric {metric_name}', 'gauge')
    metric.samples = [sample]
    
    return metric

def log_to_grafana(metric_name, value, labels=None):
    if not labels:
        labels = {}
    
    # Tambahkan label wajib
    labels['environment'] = 'github_actions'
    labels['job'] = 'mlops_training'
    
    try:
        timestamp_ms = int(time.time() * 1000)
        
        # Format metrik sesuai dengan Grafana Cloud Prometheus API
        metric_data = {
            "series": [{
                "labels": [
                    {"name": "__name__", "value": metric_name}
                ] + [
                    {"name": k, "value": str(v)}
                    for k, v in labels.items()
                ],
                "samples": [
                    [timestamp_ms, str(float(value))]
                ]
            }]
        }
        
        # Kirim ke Prometheus
        response = requests.post(
            os.environ.get('PROMETHEUS_REMOTE_WRITE_URL'),
            json=metric_data,
            auth=(os.environ.get('PROMETHEUS_USERNAME'), os.environ.get('PROMETHEUS_API_KEY')),
            headers={
                'Content-Type': 'application/json',
                'X-Scope-OrgID': os.environ.get('PROMETHEUS_USERNAME'),
                'User-Agent': 'mlops-monitoring/1.0.0'
            }
        )
        
        # Kirim ke Loki
        loki_timestamp = int(time.time() * 1e9)
        loki_payload = {
            "streams": [{
                "stream": {
                    "job": "mlops_training",
                    "environment": "github_actions",
                    "metric": metric_name
                },
                "values": [
                    [str(loki_timestamp), f"Metric logged: {metric_name}={value}"]
                ]
            }]
        }
        
        loki_response = requests.post(
            f"{os.environ.get('LOKI_URL')}/loki/api/v1/push",
            json=loki_payload,
            auth=(os.environ.get('LOKI_USERNAME'), os.environ.get('LOKI_API_KEY')),
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Prometheus response: {response.status_code}")
        print(f"Loki response: {loki_response.status_code}")
        
        return response.status_code in [200, 204] and loki_response.status_code == 204
    except Exception as e:
        print(f"Error logging metric: {e}")
        return False

def send_test_metrics_to_prometheus(metric_name, value, labels=None):
    try:
        timestamp_ms = int(time.time() * 1000)
        
        if not labels:
            labels = {}
            
        # Add default labels
        labels.update({
            'environment': 'github_actions',
            'job': 'mlops_testing'
        })
        
        # Format metric data
        metric_data = {
            'series': [{
                'labels': [
                    {'name': '__name__', 'value': metric_name}
                ] + [
                    {'name': k, 'value': str(v)}
                    for k, v in labels.items()
                ],
                'samples': [
                    [timestamp_ms, str(float(value))]
                ]
            }]
        }
        
        # Send to Prometheus
        response = requests.post(
            os.environ.get('PROMETHEUS_REMOTE_WRITE_URL'),
            json=metric_data,
            auth=(os.environ.get('PROMETHEUS_USERNAME'), os.environ.get('PROMETHEUS_API_KEY')),
            headers={
                'Content-Type': 'application/json',
                'X-Scope-OrgID': os.environ.get('PROMETHEUS_USERNAME')
            }
        )
        
        # Send to Loki
        loki_timestamp = int(time.time() * 1e9)
        loki_payload = {
            'streams': [{
                'stream': {
                    'job': 'mlops_testing',
                    'environment': 'github_actions',
                    'metric': metric_name
                },
                'values': [
                    [str(loki_timestamp), f"Testing metric: {metric_name}={value}"]
                ]
            }]
        }
        
        loki_response = requests.post(
            f"{os.environ.get('LOKI_URL')}/loki/api/v1/push",
            json=loki_payload,
            auth=(os.environ.get('LOKI_USERNAME'), os.environ.get('LOKI_API_KEY')),
            headers={'Content-Type': 'application/json'}
        )
        
        return True
    except Exception as e:
        print(f"Error sending metrics: {e}")
        return False

def test_model(model, test_loader, criterion, device):
    # Set debug mode for quick testing
    debug_mode = True  # Force debug mode to be True
    print("🔧 Running in debug mode: 10 batches only")
    max_batches = 10  # Force 10 batches
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    print("\nStarting test phase:")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if debug_mode and i >= max_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1
            
            if i % 5 == 0:  # Log every 5 batches
                print(f"Test Batch {i+1}: Loss = {loss.item():.4f}")
                
            # Send metrics every 5 batches
            if i % 5 == 0:
                send_metric_to_prometheus(
                    "test_loss",
                    loss.item(),
                    {"batch": str(i + 1)}
                )
    
    avg_test_loss = test_loss / batch_count
    test_accuracy = 100 * correct / total
    
    print(f'\nTest Results:')
    print(f'Average Loss: {avg_test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.2f}%')
    
    # Send metrics
    send_test_metrics_to_prometheus('test_loss', avg_test_loss)
    send_test_metrics_to_prometheus('test_accuracy', test_accuracy)
    
    return avg_test_loss, test_accuracy

# Run the testing phase
if __name__ == "__main__":
    test_loss, test_accuracy = computeTestSetAccuracyAndLogConfusionMatrix(
        model, criterion, test_loader, test_dataset.classes
    )
    log_test_metrics_to_grafana({'test_loss': test_loss, 'test_accuracy': test_accuracy})