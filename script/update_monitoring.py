import os
import json
import requests
from datetime import datetime
from train import send_log_to_loki, send_metric_to_prometheus

def load_reports():
    """Load all monitoring reports."""
    reports = {}
    try:
        if os.path.exists("drift_report.json"):
            with open("drift_report.json", "r") as f:
                reports["drift"] = json.load(f)
        
        if os.path.exists("model_comparison_report.json"):
            with open("model_comparison_report.json", "r") as f:
                reports["comparison"] = json.load(f)
        
        if os.path.exists("ab_testing_config.json"):
            with open("ab_testing_config.json", "r") as f:
                reports["ab_testing"] = json.load(f)
        
        if os.path.exists("retraining_report.json"):
            with open("retraining_report.json", "r") as f:
                reports["retraining"] = json.load(f)
        
        return reports
    except Exception as e:
        print(f"Error loading reports: {e}")
        return {}

def update_monitoring_dashboards():
    """Update monitoring dashboards with latest metrics and configurations."""
    try:
        # Load all reports
        reports = load_reports()
        if not reports:
            raise ValueError("No monitoring reports found")

        # Prepare monitoring update info
        monitoring_info = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updates": []
        }

        # Update drift monitoring
        if "drift" in reports:
            drift_metrics = {
                "drift_percentage": reports["drift"]["data_drift"]["data_drift_share"],
                "drifted_features": reports["drift"]["data_drift"]["number_of_drifted_columns"]
            }
            for metric_name, value in drift_metrics.items():
                send_metric_to_prometheus(f"latest_{metric_name}", value)
            monitoring_info["updates"].append({
                "type": "drift",
                "metrics": drift_metrics
            })

        # Update model comparison monitoring
        if "comparison" in reports:
            comparison_metrics = reports["comparison"]
            for metric, data in comparison_metrics.items():
                if isinstance(data, dict) and "percentage_improvement" in data:
                    send_metric_to_prometheus(
                        f"model_improvement_{metric}",
                        data["percentage_improvement"]
                    )
            monitoring_info["updates"].append({
                "type": "comparison",
                "metrics": comparison_metrics
            })

        # Update A/B testing monitoring
        if "ab_testing" in reports:
            ab_config = reports["ab_testing"]
            send_metric_to_prometheus("ab_testing_traffic_split", ab_config["traffic_split"])
            monitoring_info["updates"].append({
                "type": "ab_testing",
                "config": ab_config
            })

        # Update retraining monitoring
        if "retraining" in reports:
            retraining_metrics = reports["retraining"]
            if "improvements" in retraining_metrics:
                for metric, data in retraining_metrics["improvements"].items():
                    send_metric_to_prometheus(
                        f"retraining_improvement_{metric}",
                        data["improvement_percentage"]
                    )
            monitoring_info["updates"].append({
                "type": "retraining",
                "metrics": retraining_metrics
            })

        # Log monitoring update to Loki
        send_log_to_loki(
            "Monitoring dashboards updated",
            "info",
            {"stage": "monitoring_update"},
            monitoring_info
        )

        # Save monitoring update report
        with open("monitoring_update_report.json", "w") as f:
            json.dump(monitoring_info, f, indent=2)

        print("Monitoring dashboards updated successfully")
        return True

    except Exception as e:
        error_msg = f"Error updating monitoring: {str(e)}"
        send_log_to_loki(error_msg, "error", {"stage": "monitoring_update"})
        print(error_msg)
        return False

if __name__ == "__main__":
    update_monitoring_dashboards() 