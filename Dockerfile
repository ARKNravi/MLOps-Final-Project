FROM python:3.8-slim
# Install system dependencies
    gcc \
    g++ \
# Copy requirements file
# Install Python dependencies with specific flags
RUN pip install --no-cache-dir -r requirements.txt
# Copy the rest of the application
COPY . .
# Make port 8000 available for Prometheus metrics
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100

# Default command
CMD ["python", "script/train.py"]