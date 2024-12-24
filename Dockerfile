# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    python3-dev \
    libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with specific flags
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir \
    supabase==2.3.0 \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    numpy \
    mlflow \
    python-snappy \
    prometheus-client \
    requests \
    protobuf \
    dvclive

# Copy the rest of the application
COPY . .

# Make port 8000 available for Prometheus metrics
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100

# Default command
CMD ["python", "script/train.py"]
