import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import initialize_model, device  # Importing from train.py
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

    with start_run():  # Start an MLflow run
        # Log testing parameters (to match training phase logs)
        mlflow.log_param("phase", "test")
        mlflow.log_param("batch_size", dataloader.batch_size)
        mlflow.log_param("dataset_size", len(dataloader.dataset))

        with torch.no_grad():
            for inputs, labels in dataloader:
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

def log_to_grafana(metric_name, value, labels=None):
    if not labels:
        labels = {}
    
    # Tambahkan label environment
    labels['environment'] = 'github_actions'
    labels['job'] = 'mlops_training'
    
    try:
        # Setup registry dan gauge
        registry = CollectorRegistry()
        g = Gauge(metric_name, f'Metric {metric_name}', labelnames=labels.keys(), registry=registry)
        g.labels(**labels).set(float(value))
        
        # Generate dan kompres data metrik menggunakan python-snappy
        metric_data = generate_latest(registry)
        compressed_data = snappy.compress(metric_data)
        
        # Kirim ke Prometheus
        response = requests.post(
            os.environ.get('PROMETHEUS_REMOTE_WRITE_URL'),
            data=compressed_data,
            auth=(os.environ.get('PROMETHEUS_USERNAME'), os.environ.get('PROMETHEUS_API_KEY')),
            headers={
                'Content-Type': 'application/x-protobuf',
                'Content-Encoding': 'snappy',
                'X-Prometheus-Remote-Write-Version': '0.1.0',
                'X-Scope-OrgID': os.environ.get('PROMETHEUS_USERNAME')
            }
        )
        
        # Kirim ke Loki
        timestamp = int(time.time() * 1e9)
        loki_payload = {
            "streams": [{
                "stream": {
                    "job": "mlops_training",
                    "environment": "github_actions",
                    "metric": metric_name
                },
                "values": [
                    [str(timestamp), f"Metric logged: {metric_name}={value}"]
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
        
        return response.status_code == 200 and loki_response.status_code == 204
    except Exception as e:
        print(f"Error logging metric: {e}")
        return False

# Run the testing phase
if __name__ == "__main__":
    test_loss, test_accuracy = computeTestSetAccuracyAndLogConfusionMatrix(
        model, criterion, test_loader, test_dataset.classes
    )
    log_test_metrics_to_grafana({'test_loss': test_loss, 'test_accuracy': test_accuracy})