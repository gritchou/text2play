# Use an NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python 3.10 and required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev && \
    apt-get install -y wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable to improve Python logging (optional)
ENV PYTHONUNBUFFERED=1

# Run the Uvicorn server
CMD ["uvicorn", "text2play.api.api:app", "--host", "0.0.0.0", "--port", "80"]
