#!/usr/bin/env bash
# This is a generic utility script designed to wait for a network service to become available.
# It does not process or require any sensitive environment variables (e.g., API keys, connection strings).
# Such variables should be managed by the calling environment or other project-specific scripts.
#
# Usage:
# ./wait-for-it.sh host:port [-t timeout] [-- command args...]
#
# Arguments:
# host:port - The host and port to check for.
# -t timeout - Optional timeout in seconds. Default is 15.
# -- command args... - Optional command to execute after the service is up.

set -e

TIMEOUT=15
QUIET=0
HOST=
PORT=
CMD=

while [[ $# -gt 0 ]]; do
  case "$1" in
    *:* )
    HOST=$(printf "%s\n" "$1"| cut -d : -f 1)
    PORT=$(printf "%s\n" "$1"| cut -d : -f 2)
    shift 1
    ;;
    -q | --quiet)
    QUIET=1
    shift 1
    ;;
    -t)
    TIMEOUT="$2"
    if [[ $TIMEOUT -lt 1 ]]; then
      echo "Error: timeout must be a positive integer"
      exit 1
    fi
    shift 2
    ;;
    --)
    shift
    CMD=("$@")
    break
    ;;
    *)
    echo "Unknown argument: $1"
    exit 1
    ;;
  esac
done

if [[ -z "$HOST" || -z "$PORT" ]]; then
  echo "Error: host and port not specified"
  exit 1
fi

echo "Waiting for $HOST:$PORT..."

for i in $(seq $TIMEOUT); do
  if nc -z "$HOST" "$PORT"; then
    echo "$HOST:$PORT is up."
    if [[ ${#CMD[@]} -gt 0 ]]; then
        exec "${CMD[@]}"
    fi
    exit 0
  fi
  sleep 1
done

echo "Error: timed out after $TIMEOUT seconds waiting for $HOST:$PORT"
exit 1