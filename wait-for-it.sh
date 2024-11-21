#!/bin/bash
# wait-for-it.sh

# Function to print usage
usage() {
    echo "Usage: $0 host:port [-t timeout] [-- command args]"
    exit 1
}

# Parse arguments
HOST=""
PORT=""
TIMEOUT=120
COMMAND=()

# Process arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        *:*)
            HOST=$(echo "$1" | cut -d: -f1)
            PORT=$(echo "$1" | cut -d: -f2)
            shift
            ;;
        -t)
            TIMEOUT="$2"
            shift 2
            ;;
        --)
            shift
            COMMAND=("$@")
            break
            ;;
        *)
            usage
            ;;
    esac
done

# Validate host and port
if [[ -z "$HOST" || -z "$PORT" ]]; then
    usage
fi

# Wait for the service to be available
start_time=$(date +%s)
while true; do
    # Check if the service is reachable
    nc -z -w5 "$HOST" "$PORT"
    if [ $? -eq 0 ]; then
        echo "$HOST:$PORT is available"
        break
    fi

    # Check for timeout
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -ge $TIMEOUT ]; then
        echo "Timeout: $HOST:$PORT not available after $TIMEOUT seconds"
        exit 1
    fi

    sleep 5
done

# Execute the command if provided
if [ ${#COMMAND[@]} -gt 0 ]; then
    exec "${COMMAND[@]}"
fi