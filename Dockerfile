FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api ./api

# Set environment variables
ENV PORT=3000
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "3000"]
