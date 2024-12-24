import os
import requests
import json
import time
from datetime import datetime

def test_loki_connection():
    print("🔍 Testing Loki connection...")
    # Coba beberapa endpoint yang mungkin
    endpoints = [
        "/ready",
        "/loki/ready",
        "/loki/api/v1/ready",
        "/api/v1/ready"
    ]
    
    for endpoint in endpoints:
        url = f"{os.environ['LOKI_URL'].rstrip('/')}{endpoint}"
        try:
            print(f"Trying endpoint: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                print(f"✅ Loki is ready at {endpoint}")
                return True
            else:
                print(f"Status code for {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"Error with {endpoint}: {e}")
    
    print("❌ Could not find working Loki endpoint")
    return False

def send_test_logs():
    print("\n📝 Preparing test logs...")
    timestamp = int(time.time() * 1e9)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    test_logs = [
        {
            "level": "info",
            "message": "MLOps Training Pipeline Started",
            "value": 1.0,
            "metric_value": 100.5
        }
    ]
    
    success_count = 0
    push_endpoint = f"{os.environ['LOKI_URL'].rstrip('/')}/loki/api/v1/push"
    
    print(f"Using push endpoint: {push_endpoint}")
    
    for log in test_logs:
        log_entry = {
            "streams": [{
                "stream": {
                    "job": "mlops_training",
                    "environment": "github_actions",
                    "level": log["level"],
                    "test": "true"
                },
                "values": [
                    [str(timestamp), json.dumps({
                        "message": log["message"],
                        "level": log["level"],
                        "value": log["value"],
                        "metric_value": log["metric_value"]
                    })]
                ]
            }]
        }
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Scope-OrgID": os.environ.get('LOKI_USERNAME', 'fake')
            }
            
            if os.environ.get('LOKI_API_KEY'):
                auth = (os.environ['LOKI_USERNAME'], os.environ['LOKI_API_KEY'])
            else:
                auth = None
            
            print(f"Sending log with headers: {headers}")
            response = requests.post(
                push_endpoint,
                json=log_entry,
                headers=headers,
                auth=auth
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            
            if response.status_code in [204, 200]:
                print(f"✅ Successfully sent log: {log['message']}")
                success_count += 1
            else:
                print(f"❌ Failed to send log: {response.text}")
        except Exception as e:
            print(f"❌ Error sending log: {str(e)}")
    
    return success_count > 0

if __name__ == "__main__":
    if test_loki_connection():
        print("\n🚀 Starting log sending test...")
        if send_test_logs():
            print("\n✅ Test log sent successfully!")
        else:
            print("\n❌ Failed to send test log")
            exit(1)
    else:
        print("\n❌ Loki connection test failed")
        exit(1) 