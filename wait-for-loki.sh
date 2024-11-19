#!/bin/bash
set -e

host="loki"
port=3100

echo "Waiting for Loki to be available at $host:$port..."

# Wait until Loki is reachable
while ! nc -z $host $port; do
  sleep 1
done

echo "Loki is up - executing command"
exec "$@"
