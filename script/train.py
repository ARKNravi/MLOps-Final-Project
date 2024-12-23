import requests
from datetime import datetime
from prometheus_client import start_http_server, Gauge
import time
import json
# MLflow configuration

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
device = torch.device("cpu")
    'train': data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
    'val': data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2)
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
        labels = {"source": "github_action"}
    
    # Prometheus metrics
    try:
        prom_response = requests.post(
            os.environ.get('PROMETHEUS_REMOTE_WRITE_URL'),
            json={
                "metrics": [{
                    "name": metric_name,
                    "value": float(value),
                    "timestamp": int(time.time() * 1000),
                    "labels": labels
                }]
            },
            headers={"Content-Type": "application/json"},
            auth=(os.environ.get('PROMETHEUS_USERNAME'), os.environ.get('PROMETHEUS_API_KEY'))
        )
        print(f"Prometheus response: {prom_response.status_code} - {prom_response.text}")
    except Exception as e:
        print(f"Error sending to Prometheus: {e}")
    
    # Loki logs
    try:
        loki_response = requests.post(
            f"{os.environ.get('LOKI_URL')}/loki/api/v1/push",
            json={
                "streams": [{
                    "stream": {
                        "job": "mlops-training",
                        "level": "info"
                    },
                    "values": [[
                        str(timestamp * 1000000),  # Loki expects nanoseconds
                        json.dumps({
                            "message": f"Metric logged: {metric_name}={value}",
                            "metric_name": metric_name,
                            "value": value,
                            "labels": labels
                        })
                    ]]
                }]
            },
            auth=(os.environ.get('LOKI_USERNAME'), os.environ.get('LOKI_API_KEY'))
        )
        print(f"Loki response: {loki_response.status_code}")
    except Exception as e:
        print(f"Error sending to Loki: {e}")

    print("Starting training...")
    print(f"Training on device: {device}")
    print(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")

    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
            print(f'\nEpoch {epoch}/{num_epochs-1}')
            epoch_start_time = time.time()
            
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
                    batch_start_time = time.time()
                    
                    # Log batch metrics immediately
                    current_loss = loss.item()
                    log_to_grafana('batch_loss', current_loss, {
                        'batch': str(batch_idx),
                        'epoch': str(epoch),
                        'phase': phase
                    })
                    
                    # Log batch time
                    batch_time = time.time() - batch_start_time
                    log_to_grafana('batch_time', batch_time, {
                        'batch': str(batch_idx),
                        'epoch': str(epoch)
                    })
                    
                    if batch_idx % 10 == 0:
                        print(f"\nBatch {batch_idx}/{total_batches} - Loss: {current_loss:.4f}")
                
                # Calculate and log metrics
                
                # Log metrics to MLflow
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
            
            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            EPOCH_TIME.set(epoch_time)
            log_to_grafana("epoch_time", epoch_time, {"epoch": str(epoch)})
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            if phase == 'train':
                scheduler.step()


        # Log final metrics
        mlflow.log_metric("best_accuracy", best_acc.item())