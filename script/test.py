import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import initialize_model, device  # Importing from train.py
import mlflow
import mlflow.pytorch

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
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Testing function
def compute_test_set_accuracy(model, criterion, dataloader):
    test_acc = 0.0
    test_loss = 0.0

    with mlflow.start_run():
        for j, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Compute accuracy
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc += acc.item() * inputs.size(0)

            print(f"Test Batch number: {j:03d}, Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

        avg_test_loss = test_loss / dataset_sizes['test']
        avg_test_acc = test_acc / dataset_sizes['test']

        print(f"\nTest Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("test_loss", avg_test_loss)
        mlflow.log_metric("test_accuracy", avg_test_acc)

        return avg_test_loss, avg_test_acc

# Run the testing phase
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")  
    mlflow.set_experiment("MLOps_Project-Akhir")
    test_loss, test_accuracy = computeTestSetAccuracy(model, criterion, test_loader)
