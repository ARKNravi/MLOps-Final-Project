#!/usr/bin/env bash
set -e

# Check for netcat dependency
if ! command -v nc >/dev/null 2>&1; then
  echo "Error: 'nc' (Netcat) command not found. Please install it in your container."
  exit 1
fi

# Argument Parsing
HOST=$1
PORT=$2
TIMEOUT=${3:-30}

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
  echo "Usage: $0 <host> <port> [timeout]"
  exit 1
fi

echo "Waiting for $HOST:$PORT for up to $TIMEOUT seconds..."

# Loop to check if the host and port are available
for i in $(seq 1 $TIMEOUT); do
  if nc -z "$HOST" "$PORT"; then
    echo "$HOST:$PORT is available!"
    exit 0
  fi
  sleep 1
done

echo "Timeout reached: $HOST:$PORT is not available"
exit 1
