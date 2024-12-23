#!/bin/bash
# wait-for-it.sh

host="$1"
port="$2"
shift 2
cmd="$@"

until nc -z "$host" "$port"; do
  >&2 echo "Service on $host:$port is unavailable - sleeping"
>&2 echo "Service is up - executing command"
exec $cmd