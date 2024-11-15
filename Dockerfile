FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Configure Git to trust the /app directory
RUN git config --global --add safe.directory /app

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the script directory
COPY script/ /app/script

# Set default command
CMD ["python", "/app/script/train.py"]
