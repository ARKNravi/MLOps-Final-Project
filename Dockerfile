# Use the official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Set the environment variable to avoid buffering in Python
ENV PYTHONUNBUFFERED 1

# Expose the port your application will run on
EXPOSE 5000

# Command to run your app (Flask app or any API service)
CMD ["python", "app.py"]
