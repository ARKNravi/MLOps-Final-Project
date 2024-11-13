FROM python:3.11-slim

# Install git and any other dependencies
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /app

# Configure Git to trust the /app directory
RUN git config --global --add safe.directory /app

# Copy dependency files
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the script directory to the container
COPY script/ /app/script

# Set default command
CMD ["python", "/app/script/train.py"]
