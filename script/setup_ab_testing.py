import os
import json
import mlflow
from datetime import datetime
from train import send_log_to_loki, send_metric_to_prometheus

class ABTestingConfig:
    def __init__(self):
        self.traffic_split = 0.5  # 50-50 split by default
        self.monitoring_period = 7  # days
        self.min_requests = 1000   # minimum requests before evaluation
        self.significance_level = 0.05

def setup_ab_testing(model_a_run_id=None, model_b_run_id=None):
    """Setup A/B testing environment for two model versions."""
    try:
        # Get run IDs from environment if not provided
        if not model_a_run_id:
            model_a_run_id = os.getenv('MODEL_A_RUN_ID')
        if not model_b_run_id:
            model_b_run_id = os.getenv('MODEL_B_RUN_ID')

        if not model_a_run_id or not model_b_run_id:
            raise ValueError("Both model run IDs are required")

        # Create AB testing configuration
        config = ABTestingConfig()
        
        # Get model information from MLflow
        client = mlflow.tracking.MlflowClient()
        model_a = client.get_run(model_a_run_id)
        model_b = client.get_run(model_b_run_id)

        # Prepare AB testing configuration
        ab_config = {
            'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_a': {
                'run_id': model_a_run_id,
                'version': model_a.data.tags.get('version', 'unknown'),
                'metrics': model_a.data.metrics
            },
            'model_b': {
                'run_id': model_b_run_id,
                'version': model_b.data.tags.get('version', 'unknown'),
                'metrics': model_b.data.metrics
            },
            'traffic_split': config.traffic_split,
            'monitoring_period_days': config.monitoring_period,
            'min_requests': config.min_requests,
            'significance_level': config.significance_level
        }

        # Save AB testing configuration
        config_path = "ab_testing_config.json"
        with open(config_path, 'w') as f:
            json.dump(ab_config, f, indent=2)

        # Initialize monitoring metrics
        send_metric_to_prometheus("ab_testing_traffic_split", config.traffic_split)
        
        # Log AB testing setup to Loki
        send_log_to_loki(
            "A/B testing environment configured",
            "info",
            {"stage": "ab_testing_setup"},
            {
                "model_a": model_a_run_id,
                "model_b": model_b_run_id,
                "traffic_split": config.traffic_split
            }
        )

        # Setup monitoring metrics for both models
        metrics_to_monitor = [
            'accuracy',
            'latency',
            'error_rate',
            'request_count'
        ]

        for metric in metrics_to_monitor:
            send_metric_to_prometheus(f"model_a_{metric}", 0)
            send_metric_to_prometheus(f"model_b_{metric}", 0)

        print(f"A/B testing configuration saved to {config_path}")
        return True

    except Exception as e:
        error_msg = f"Error setting up A/B testing: {str(e)}"
        send_log_to_loki(error_msg, "error", {"stage": "ab_testing_setup"})
        print(error_msg)
        return False

if __name__ == "__main__":
    setup_ab_testing() 