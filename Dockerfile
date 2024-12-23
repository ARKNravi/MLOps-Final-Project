# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with specific flags to avoid resolver issues
RUN pip install --no-cache-dir --no-deps -r requirements.txt && \
    pip install --no-cache-dir colorama==0.4.5 configobj==5.0.8

# Copy the rest of the application
COPY . .

# Make port 8000 available for Prometheus metrics
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100

# Default command
CMD ["python", "script/train.py"]
