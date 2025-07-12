#!/bin/bash
# Startup script for Sentio Worker - runs in an infinite loop

# Ensure script is executable
chmod +x /app/worker.py

# Create log directory if it doesn't exist
mkdir -p /app/data

# Run worker in an infinite loop
while true; do
  echo "$(date): Starting worker iteration..." >> /app/data/worker.log
  python -m worker >> /app/data/worker.log 2>&1
  echo "$(date): Worker iteration complete. Sleeping for 5 seconds..." >> /app/data/worker.log
  sleep 5
done 