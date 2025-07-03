
# Use CUDA base with Ubuntu 22.04
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install Python 3.10, pip, and basic tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Symlink python and pip to point to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set the working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .

# Install Python dependencies, ensuring CUDA-enabled torch is pulled
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy app source code
COPY . .

# Run the application
CMD ["python", "handler.py"]
