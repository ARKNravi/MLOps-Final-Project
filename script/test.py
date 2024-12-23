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


# Run the testing phase
if __name__ == "__main__":
    test_loss, test_accuracy = computeTestSetAccuracyAndLogConfusionMatrix(
        model, criterion, test_loader, test_dataset.classes
    )
