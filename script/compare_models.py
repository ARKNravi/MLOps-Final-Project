import os
import mlflow
import pandas as pd
from datetime import datetime
from train import send_log_to_loki, send_metric_to_prometheus

def get_model_metrics(run_id):
    """Retrieve model metrics from MLflow."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics

def compare_models(previous_run_id=None, current_run_id=None):
    """Compare metrics between two model versions."""
    try:
        # Get run IDs from environment if not provided
        if not previous_run_id:
            previous_run_id = os.getenv('PREVIOUS_MODEL_RUN_ID')
        if not current_run_id:
            current_run_id = os.getenv('CURRENT_MODEL_RUN_ID')

        if not previous_run_id or not current_run_id:
            raise ValueError("Both previous and current run IDs are required")

        # Get metrics for both models
        previous_metrics = get_model_metrics(previous_run_id)
        current_metrics = get_model_metrics(current_run_id)

        # Calculate improvements
        improvements = {}
        for metric in current_metrics:
            if metric in previous_metrics:
                improvement = current_metrics[metric] - previous_metrics[metric]
                improvement_percentage = (improvement / previous_metrics[metric]) * 100
                improvements[metric] = {
                    'previous': previous_metrics[metric],
                    'current': current_metrics[metric],
                    'absolute_improvement': improvement,
                    'percentage_improvement': improvement_percentage
                }

        # Prepare comparison results
        comparison_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'previous_run_id': previous_run_id,
            'current_run_id': current_run_id,
            'improvements': improvements
        }

        # Log comparison metrics to monitoring
        for metric, data in improvements.items():
            metric_name = f"model_improvement_{metric}"
            send_metric_to_prometheus(metric_name, data['percentage_improvement'])

        # Send detailed log to Loki
        send_log_to_loki(
            "Model comparison completed",
            "info",
            {"stage": "model_comparison"},
            comparison_results
        )

        # Save comparison report
        report_path = "model_comparison_report.json"
        pd.DataFrame(improvements).to_json(report_path, orient='index', indent=2)
        
        # Determine if new model is better
        accuracy_improvement = improvements.get('val_accuracy', {}).get('percentage_improvement', 0)
        is_better = accuracy_improvement > 0

        print(f"Model comparison completed. New model is {'better' if is_better else 'worse'}.")
        print(f"Accuracy improvement: {accuracy_improvement:.2f}%")
        
        # Set output for GitHub Actions
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"model_improved={str(is_better).lower()}\n")
            f.write(f"accuracy_improvement={accuracy_improvement}\n")

        return is_better, improvements

    except Exception as e:
        error_msg = f"Error in model comparison: {str(e)}"
        send_log_to_loki(error_msg, "error", {"stage": "model_comparison"})
        print(error_msg)
        return False, {}

if __name__ == "__main__":
    compare_models() 