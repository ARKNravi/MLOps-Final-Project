import os
import mlflow
import torch
import json
from datetime import datetime
from train import train_model, initialize_model, criterion, optimizer, exp_lr_scheduler, num_epochs, device
from train import send_log_to_loki, send_metric_to_prometheus

def load_drift_report():
    """Load the latest drift report."""
    try:
        with open("drift_report.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading drift report: {e}")
        return None

def get_best_model_run():
    """Get the best performing model from MLflow."""
    client = mlflow.tracking.MlflowClient()
    experiments = client.get_experiment_by_name("default")
    runs = client.search_runs(
        experiment_ids=[experiments.experiment_id],
        order_by=["metrics.val_accuracy DESC"]
    )
    return runs[0] if runs else None

def automated_retraining():
    """Perform automated retraining based on drift detection."""
    try:
        # Load drift report
        drift_report = load_drift_report()
        if not drift_report:
            raise ValueError("No drift report found")

        # Get current best model
        best_run = get_best_model_run()
        if not best_run:
            raise ValueError("No previous model runs found")

        # Log retraining start
        send_log_to_loki(
            "Starting automated retraining",
            "info",
            {"stage": "retraining_start"},
            {
                "previous_model_id": best_run.info.run_id,
                "drift_percentage": drift_report['data_drift']['data_drift_share']
            }
        )

        # Initialize new model
        model = initialize_model(num_classes=4).to(device)
        
        # Start MLflow run for retraining
        with mlflow.start_run() as run:
            # Log training parameters
            mlflow.log_params({
                "retrain_trigger": "data_drift",
                "previous_model_id": best_run.info.run_id,
                "drift_percentage": drift_report['data_drift']['data_drift_share']
            })

            # Train the model
            print("Starting model retraining...")
            model = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=exp_lr_scheduler,
                num_epochs=num_epochs
            )

            # Log the retrained model
            mlflow.pytorch.log_model(model, "retrained_model")
            
            # Get metrics from the run
            metrics = mlflow.get_run(run.info.run_id).data.metrics
            
            # Send metrics to monitoring
            for metric_name, value in metrics.items():
                send_metric_to_prometheus(f"retrained_model_{metric_name}", value)

            # Compare with previous best model
            previous_metrics = mlflow.get_run(best_run.info.run_id).data.metrics
            improvements = {}
            for metric in metrics:
                if metric in previous_metrics:
                    improvement = metrics[metric] - previous_metrics[metric]
                    improvement_percentage = (improvement / previous_metrics[metric]) * 100
                    improvements[metric] = {
                        'previous': previous_metrics[metric],
                        'new': metrics[metric],
                        'improvement_percentage': improvement_percentage
                    }

            # Log retraining results
            send_log_to_loki(
                "Model retraining completed",
                "info",
                {"stage": "retraining_complete"},
                {
                    "run_id": run.info.run_id,
                    "improvements": improvements,
                    "training_time": run.info.end_time - run.info.start_time
                }
            )

            # Save retraining report
            report = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run.info.run_id,
                "previous_model_id": best_run.info.run_id,
                "improvements": improvements,
                "drift_report": drift_report
            }
            
            with open("retraining_report.json", "w") as f:
                json.dump(report, f, indent=2)

            print(f"Retraining completed. New model run ID: {run.info.run_id}")
            return True

    except Exception as e:
        error_msg = f"Error in automated retraining: {str(e)}"
        send_log_to_loki(error_msg, "error", {"stage": "retraining"})
        print(error_msg)
        return False

if __name__ == "__main__":
    automated_retraining() 