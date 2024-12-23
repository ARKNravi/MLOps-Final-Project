import os
import copy
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
<<<<<<< Updated upstream
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
=======
from torchsummary import summary
from prometheus_client import Gauge, start_http_server
from PIL import Image
>>>>>>> Stashed changes
import mlflow
import mlflow.pytorch
import socket

def wait_for_port(port, host='localhost', timeout=5.0):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            if time.time() - start_time >= timeout:
                return False
            time.sleep(0.1)

<<<<<<< Updated upstream
# MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/salsazufar/project-akhir-mlops.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('DAGSHUB_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
=======
# Start Prometheus client server on port 8000
def start_prometheus_server(port=8000, max_retries=3):
    for attempt in range(max_retries):
        try:
            start_http_server(port, addr='0.0.0.0')  # Changed to 0.0.0.0 to allow external access
            print(f"Prometheus metrics server started successfully on port {port}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: Failed to start Prometheus metrics server: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print("Failed to start Prometheus metrics server after all retries")
                return False

if not start_prometheus_server():
    print("Exiting due to Prometheus server startup failure")
    exit(1)

# Prometheus metrics
train_loss_metric = Gauge('train_loss', 'Training Loss')
val_loss_metric = Gauge('val_loss', 'Validation Loss')
train_accuracy_metric = Gauge('train_accuracy', 'Training Accuracy')
val_accuracy_metric = Gauge('val_accuracy', 'Validation Accuracy')
epoch_metric = Gauge('current_epoch', 'Current Training Epoch')
batch_metric = Gauge('current_batch', 'Current Training Batch')
learning_rate_metric = Gauge('learning_rate', 'Current Learning Rate')

# Dataset paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")
>>>>>>> Stashed changes

# Validate dataset paths and contents
for dir_path in [train_dir, val_dir]:
    if not os.path.exists(dir_path):
        print(f"Dataset path not found: {dir_path}")
        exit(1)
    else:
        contents = os.listdir(dir_path)
        if not contents:
            print(f"Dataset directory is empty: {dir_path}")
            exit(1)
        print(f"Contents of {dir_path}: {contents[:5]}...")  # Show first 5 items for brevity

# Hyperparameters
<<<<<<< Updated upstream
num_epochs = 3
=======
num_epochs = 10  # Increased from 1 to 10
>>>>>>> Stashed changes
batch_size = 4
learning_rate = 0.001
momentum = 0.9
scheduler_step_size = 7
scheduler_gamma = 0.1
num_classes = 4
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

<<<<<<< Updated upstream
# Accuracy threshold for model registry
accuracy_threshold = 0.8  

# Data transformation
=======
# Data transformations
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}
dataloaders = {
    'train': data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
    'val': data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)
}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes
=======
try:
    print("Loading datasets...")
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
>>>>>>> Stashed changes

    print(f"Dataset sizes - Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
    print(f"Classes: {class_names}")

except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Initialize model
def initialize_model(num_classes):
    try:
        print("Initializing the model...")
        model_ft = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        return model_ft
    except Exception as e:
        print(f"Error initializing model: {e}")
        exit(1)

<<<<<<< Updated upstream
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

# Training function
=======
# Train model with enhanced monitoring
>>>>>>> Stashed changes
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    with mlflow.start_run():
<<<<<<< Updated upstream
        # Log hyperparameters
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("scheduler_step_size", scheduler_step_size)
        mlflow.log_param("scheduler_gamma", scheduler_gamma)
        mlflow.log_param("num_epochs", num_epochs)
=======
        # Log parameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_type": "ResNet18",
            "optimizer": type(optimizer).__name__,
            "device": str(device)
        })
>>>>>>> Stashed changes

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 10)
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
                        print(f"Error in {phase} phase during training batch: {e}")
                        continue

                # Phase metrics
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

<<<<<<< Updated upstream
                # Log metrics for each epoch
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc, step=epoch)

                # Save best model weights
=======
                # Log to MLflow
                mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch + 1)
                mlflow.log_metric(f"{phase}_accuracy", epoch_acc, step=epoch + 1)

                # Update learning rate metric
                current_lr = optimizer.param_groups[0]['lr']
                learning_rate_metric.set(current_lr)

                # Deep copy the model if we got better accuracy
>>>>>>> Stashed changes
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Save best model with MLflow
                    mlflow.pytorch.log_model(model, "best_model")

<<<<<<< Updated upstream
        # Save the best model
        model.load_state_dict(best_model_wts)

        # Log the best model to MLflow
        mlflow.pytorch.log_model(model, "best_model")

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

=======
            scheduler.step()

    print(f'\nBest val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
>>>>>>> Stashed changes
    return model

# Main execution
if __name__ == "__main__":
<<<<<<< Updated upstream
    model = initialize_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)
=======
    try:
        # Set up MLflow
        print("Setting up MLflow...")
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("MLOps_Project-Akhir")

        # Initialize model and training components
        model = initialize_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Print model summary
        print("\nModel Summary:")
        summary(model, (3, 224, 224))

        # Train model
        print("\nStarting Training...")
        model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs)
        
        print("Training completed successfully!")

    except Exception as e:
        print(f"Error in main execution: {e}")
        exit(1)
>>>>>>> Stashed changes
