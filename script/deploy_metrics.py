import mlflow
from grafana_client import GrafanaApi
import os

# Get metrics from MLflow
client = mlflow.tracking.MlflowClient()
run_id = "<your_run_id>"  # Replace with your run id
data = client.get_run(run_id).data

# Connect to Grafana
grafana = GrafanaApi(auth=(os.getenv("GRAFANA_USER"), os.getenv("GRAFANA_PASSWORD")), host=os.getenv("GRAFANA_HOST"))

def deploy_metrics():
    metrics = data.metrics
    for key, value in metrics.items():
        # Example of how you might send metrics to Grafana
        grafana.push_metric(key, value)

if __name__ == "__main__":
    deploy_metrics()