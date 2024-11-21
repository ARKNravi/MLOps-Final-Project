# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies, including curl for healthcheck
RUN apt-get update && apt-get install -y \
    git \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --timeout=300 --retries=10 -r requirements.txt

# Copy the application code
COPY script/ /app/script
COPY wait-for-it.sh /app/

# Make wait-for-it.sh executable
RUN chmod +x /app/wait-for-it.sh

# Expose the Prometheus metrics port
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Set the default command with corrected host:port format
ENTRYPOINT ["/app/wait-for-it.sh", "mlflow:5000", "-t", "120", "--", "python", "/app/script/train.py"]