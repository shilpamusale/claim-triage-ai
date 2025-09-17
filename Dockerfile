# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and models into the container
COPY ./app ./app
COPY ./claimtriageai ./claimtriageai
COPY ./models ./models

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run your app using uvicorn
# We use 0.0.0.0 to make it accessible from outside the container
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}