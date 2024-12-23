from prometheus_client import start_http_server, Gauge
import time
import datetime
import requests
# Add Prometheus metrics for testing
TEST_LOSS = Gauge('test_loss', 'Test loss')
TEST_ACC = Gauge('test_accuracy', 'Test accuracy')
TEST_F1 = Gauge('test_f1_score', 'Test F1 Score')
TEST_PRECISION = Gauge('test_precision', 'Test Precision')
TEST_RECALL = Gauge('test_recall', 'Test Recall')

def log_test_metrics_to_grafana(metrics_dict):
    for metric_name, value in metrics_dict.items():
        # Update Prometheus metrics
        if metric_name == 'test_loss':
            TEST_LOSS.set(value)
        elif metric_name == 'test_accuracy':
            TEST_ACC.set(value)
        elif metric_name == 'test_f1':
            TEST_F1.set(value)
        elif metric_name == 'test_precision':
            TEST_PRECISION.set(value)
        elif metric_name == 'test_recall':
            TEST_RECALL.set(value)
        
        # Log to Grafana Cloud
        timestamp = int(datetime.datetime.now().timestamp() * 1000)
        metric = {
            "metrics": [{
                "name": metric_name,
                "value": value,
                "timestamp": timestamp
            }]
        }
        
        try:
            response = requests.post(
                os.getenv('PROMETHEUS_REMOTE_WRITE_URL'),
                json=metric,
                headers={
                    "Authorization": f"Bearer {os.getenv('PROMETHEUS_API_KEY')}",
                    "Content-Type": "application/json"
                },
                auth=(os.getenv('PROMETHEUS_USERNAME'), os.getenv('PROMETHEUS_API_KEY'))
            )
            print(f"Test metric logged: {metric_name}={value}")
        except Exception as e:
            print(f"Error logging test metric: {e}")
    log_test_metrics_to_grafana({'test_loss': test_loss, 'test_accuracy': test_accuracy})