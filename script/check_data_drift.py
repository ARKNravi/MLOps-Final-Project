import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.pipeline.column_mapping import ColumnMapping
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from train import send_log_to_loki, send_metric_to_prometheus

def extract_image_features(image_path):
    """Extract features from image using pretrained model."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).numpy().flatten()

def load_dataset_features(dataset_path):
    """Load and extract features from all images in dataset."""
    features_list = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                features = extract_image_features(image_path)
                features_list.append(features)
    return pd.DataFrame(features_list)

def check_data_drift():
    """Check for data drift between reference and current dataset."""
    try:
        # Load reference and current datasets
        reference_path = os.path.join('dataset', 'reference')
        current_path = os.path.join('dataset', 'train')
        
        print("Loading reference dataset...")
        reference_data = load_dataset_features(reference_path)
        print("Loading current dataset...")
        current_data = load_dataset_features(current_path)

        # Configure drift detection
        column_mapping = ColumnMapping()
        data_drift_profile = Profile(sections=[DataDriftProfileSection()])
        
        # Calculate drift
        data_drift_profile.calculate(reference_data, current_data, column_mapping)
        report = data_drift_profile.json()
        report_dict = json.loads(report)

        # Extract drift metrics
        drift_share = report_dict['data_drift']['data_drift_share']
        number_of_drifted_columns = report_dict['data_drift']['number_of_drifted_columns']
        total_columns = len(reference_data.columns)
        drift_percentage = (number_of_drifted_columns / total_columns) * 100

        # Log results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        drift_info = {
            "drift_share": drift_share,
            "drifted_columns": number_of_drifted_columns,
            "total_columns": total_columns,
            "drift_percentage": drift_percentage
        }

        # Send metrics to Prometheus
        send_metric_to_prometheus("data_drift_percentage", drift_percentage)
        send_metric_to_prometheus("drifted_columns", number_of_drifted_columns)
        
        # Send detailed log to Loki
        send_log_to_loki(
            "Data drift check completed",
            "info",
            {"stage": "drift_detection"},
            drift_info
        )

        # Set output for GitHub Actions
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            # Consider drift detected if more than 30% of features have drifted
            drift_detected = drift_percentage > 30
            f.write(f"drift_detected={str(drift_detected).lower()}\n")
            print(f"Drift detected: {drift_detected}")

        # Save drift report
        report_path = "drift_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Drift report saved to {report_path}")
        return drift_detected

    except Exception as e:
        error_msg = f"Error in drift detection: {str(e)}"
        send_log_to_loki(error_msg, "error", {"stage": "drift_detection"})
        print(error_msg)
        return False

if __name__ == "__main__":
    check_data_drift() 