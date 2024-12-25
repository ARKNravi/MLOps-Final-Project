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

# MLflow configuration with better error handling
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    print("⚠️ MLFLOW_TRACKING_URI tidak ditemukan, menggunakan nilai default")
    MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set MLflow credentials
    dagshub_username = os.environ.get('DAGSHUB_USERNAME')
    dagshub_token = os.environ.get('DAGSHUB_TOKEN')
    
    if not dagshub_username or not dagshub_token:
        print("⚠️ DagsHub credentials tidak lengkap")
    else:
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        print(f"DagsHub Username: {dagshub_username}")
    
    print("🔄 Mencoba menghubungkan ke MLflow...")
    mlflow.set_experiment("default")
    print("✅ Berhasil terhubung ke MLflow")
except Exception as e:
    print(f"❌ Error menghubungkan ke MLflow: {str(e)}")
    print("⚠️ Melanjutkan tanpa MLflow tracking...")

# Validasi environment variables yang diperlukan
required_env_vars = [
    'PROMETHEUS_REMOTE_WRITE_URL',
    'PROMETHEUS_USERNAME',
    'PROMETHEUS_API_KEY',
    'LOKI_URL',
    'LOKI_USERNAME',
    'LOKI_API_KEY'
]

missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    print(f"⚠️ Environment variables berikut tidak ditemukan: {', '.join(missing_vars)}")
    print("⚠️ Beberapa fitur monitoring mungkin tidak akan berfungsi")

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

# Fungsi untuk mengirim metrik ke Prometheus dengan format protobuf
def send_metric_to_prometheus(metric_name, value):
    """
    Fungsi untuk mengirim metrik ke Prometheus menggunakan protobuf
    Args:
        metric_name (str): Nama metrik (train_loss, train_accuracy, val_loss, val_accuracy)
        value (float): Nilai metrik
    """
    try:
        timestamp_ms = int(time.time() * 1000)
        
        # Buat WriteRequest protobuf
        write_req = WriteRequest()
        ts = write_req.timeseries.add()
        
        # Tambahkan labels
        labels = [
            ("__name__", metric_name),
            ("job", "mlops_training"),
            ("environment", "github_actions")
        ]
        
        for name, value_label in labels:
            label = ts.labels.add()
            label.name = name
            label.value = str(value_label)
        
        # Tambahkan sample
        sample = ts.samples.add()
        sample.value = float(value)
        sample.timestamp = timestamp_ms
        
        # Serialize dan kompres data
        data = write_req.SerializeToString()
        compressed_data = python_snappy.compress(data)
        
        # Kirim ke Prometheus
        response = requests.post(
            os.environ.get('PROMETHEUS_REMOTE_WRITE_URL'),
            data=compressed_data,
            auth=(os.environ.get('PROMETHEUS_USERNAME'), os.environ.get('PROMETHEUS_API_KEY')),
            headers={
                "Content-Encoding": "snappy",
                "Content-Type": "application/x-protobuf",
                "X-Prometheus-Remote-Write-Version": "0.1.0"
            }
        )
        
        if response.status_code not in [200, 204]:
            print(f"⚠️ Error mengirim metrik {metric_name}: {response.text}")
            return False
            
        return True
    except Exception as e:
        print(f"⚠️ Error mengirim metrik {metric_name}: {str(e)}")
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

# Inisialisasi Supabase client tanpa proxy
try:
    # Dapatkan credentials
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    # Validasi credentials
    if not supabase_url or not supabase_key:
        print("⚠️ Supabase credentials tidak lengkap")
        supabase = None
    else:
        print(f"🔄 Mencoba menghubungkan ke Supabase...")
        # Inisialisasi tanpa opsi tambahan
        supabase = create_client(supabase_url, supabase_key)
        print("✅ Berhasil terhubung ke Supabase")
except Exception as e:
    print(f"❌ Error initializing Supabase client: {str(e)}")
    print("⚠️ Melanjutkan tanpa Supabase...")
    supabase = None

def save_metrics_to_supabase(metrics, phase="train"):
    """Save metrics to Supabase with error handling"""
    if supabase is None:
        print("⚠️ Supabase client tidak tersedia, melewati penyimpanan metrik")
        return False
        
    try:
        # Validasi metrics
        if not isinstance(metrics, dict) or "accuracy" not in metrics or "loss" not in metrics:
            print("❌ Format metrik tidak valid")
            return False
            
        data = {
            "accuracy": float(metrics["accuracy"]),
            "loss": float(metrics["loss"]),
            "source": phase,
            "created_at": datetime.now().isoformat()
        }
        
        result = supabase.table("model_metrics").insert(data).execute()
        print(f"✅ Berhasil menyimpan metrik {phase} ke Supabase")
        return True
    except Exception as e:
        print(f"❌ Error menyimpan metrik ke Supabase: {str(e)}")
        print("⚠️ Melanjutkan proses training...")
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

def calculate_moving_average(buffer, window_size=5):
    """Hitung moving average dari buffer"""
    if len(buffer) < window_size:
        return sum(buffer) / len(buffer)
    return sum(buffer[-window_size:]) / window_size

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    best_model_state = None
    
    # Inisialisasi buffer untuk moving average
    train_loss_buffer = []
    train_acc_buffer = []
    val_loss_buffer = []
    val_acc_buffer = []
    window_size = 5  # Ukuran window untuk moving average
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            
            try:
                for i, (images, labels) in enumerate(train_loader):
                    if i >= train_batches:
                        break
                    
                    try:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        # Hitung metrik per batch
                        batch_loss = loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        batch_correct = (predicted == labels).sum().item()
                        batch_total = labels.size(0)
                        batch_accuracy = 100 * batch_correct / batch_total
                        
                        # Update running metrics
                        running_loss += batch_loss
                        running_correct += batch_correct
                        running_total += batch_total
                        
                        # Tambahkan ke buffer
                        train_loss_buffer.append(batch_loss)
                        train_acc_buffer.append(batch_accuracy)
                        
                        # Kirim moving average setiap window_size batch
                        if len(train_loss_buffer) >= window_size:
                            # Hitung moving average
                            avg_train_loss = calculate_moving_average(train_loss_buffer, window_size)
                            avg_train_acc = calculate_moving_average(train_acc_buffer, window_size)
                            
                            # Kirim metrik ke Prometheus
                            send_metric_to_prometheus("train_loss", avg_train_loss)
                            send_metric_to_prometheus("train_accuracy", avg_train_acc)
                            
                            print(f"Batch {i+1}/{train_batches} - Moving Avg Loss: {avg_train_loss:.4f} - Acc: {avg_train_acc:.2f}%")
                    
                    except Exception as batch_error:
                        print(f"⚠️ Error pada batch {i+1}: {str(batch_error)}")
                        continue
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for i, (images, labels) in enumerate(val_loader):
                        if i >= val_batches:
                            break
                            
                        try:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            
                            # Hitung metrik
                            batch_loss = loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            batch_correct = (predicted == labels).sum().item()
                            batch_total = labels.size(0)
                            
                            # Update metrics
                            val_loss += batch_loss
                            val_correct += batch_correct
                            val_total += batch_total
                            
                            # Tambahkan ke buffer validasi
                            val_loss_buffer.append(batch_loss)
                            val_acc_buffer.append(100 * batch_correct / batch_total)
                            
                            # Kirim moving average setiap window_size batch
                            if len(val_loss_buffer) >= window_size:
                                avg_val_loss = calculate_moving_average(val_loss_buffer, window_size)
                                avg_val_acc = calculate_moving_average(val_acc_buffer, window_size)
                                
                                # Kirim metrik ke Prometheus
                                send_metric_to_prometheus("val_loss", avg_val_loss)
                                send_metric_to_prometheus("val_accuracy", avg_val_acc)
                            
                        except Exception as val_batch_error:
                            print(f"⚠️ Error pada validation batch {i+1}: {str(val_batch_error)}")
                            continue
                
                # Hitung metrik epoch
                epoch_val_loss = val_loss / val_batches
                epoch_val_acc = 100 * val_correct / val_total
                
                # Simpan model terbaik
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    print("✨ Model terbaik baru disimpan!")
                    
                    # Simpan model
                    try:
                        torch.save(best_model_state, os.path.join('model', 'best_model_weights.pth'))
                    except Exception as save_error:
                        print(f"⚠️ Error menyimpan model: {str(save_error)}")
                
            except Exception as epoch_error:
                print(f"⚠️ Error pada epoch {epoch+1}: {str(epoch_error)}")
                continue
        
        print("\n✅ Training selesai!")
        return train_loss_buffer, val_loss_buffer
        
    except Exception as training_error:
        print(f"❌ Error fatal dalam training: {str(training_error)}")
        if best_model_state is not None:
            print("⚠️ Mengembalikan model terbaik yang tersimpan...")
            model.load_state_dict(best_model_state)
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