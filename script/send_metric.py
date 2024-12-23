import os
import json
import snappy
import requests
from remote_write_pb2 import WriteRequest, TimeSeries, Label, Sample
from datetime import datetime

def send_metric():
    # Buat timestamp dalam format yang benar (nanodetik)
    timestamp_ns = int(datetime.utcnow().timestamp() * 1e9)

    # Inisialisasi WriteRequest
    write_req = WriteRequest()

    # Tambahkan TimeSeries
    ts = write_req.timeseries.add()
    ts.labels.add(name="__name__", value="test_metric")
    ts.labels.add(name="job", value="github_actions_test")
    ts.labels.add(name="instance", value="test")
    ts.labels.add(name="environment", value="github_actions")

    # Tambahkan Sample
    sample = ts.samples.add()
    sample.value = 1.0
    sample.timestamp = timestamp_ns

    # Serialize ke Protobuf
    data = write_req.SerializeToString()

    # Kompresi dengan Snappy
    compressed_data = snappy.compress(data)

    url = os.environ['PROMETHEUS_REMOTE_WRITE_URL']
    username = os.environ['PROMETHEUS_USERNAME']
    password = os.environ['PROMETHEUS_API_KEY']

    headers = {
        "Content-Encoding": "snappy",
        "Content-Type": "application/x-protobuf",
        "X-Prometheus-Remote-Write-Version": "0.1.0"
    }

    try:
        response = requests.post(
            url,
            data=compressed_data,
            auth=(username, password),
            headers=headers
        )

        print(f"Status code: {response.status_code}")
        if response.status_code in [200, 204]:
            print("✅ Metric sent successfully")
        else:
            print("❌ Failed to send metric")
            if response.text:
                print(f"Error: {response.text}")
            print(f"Request URL: {url}")
            print(f"Request headers: {headers}")
    except Exception as e:
        print(f"Error sending metric: {e}")
        exit(1)

if __name__ == "__main__":
    send_metric() 