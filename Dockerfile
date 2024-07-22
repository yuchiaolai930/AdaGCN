# Use a base image with ARM support
FROM python:3.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install the required packages
RUN pip install --upgrade pip
RUN pip install tensorflow==2.10.1 scipy==1.7.3 networkx scikit-learn

# Run the script
CMD ["python", "train_WD.py", "--config", "config.yaml"]

