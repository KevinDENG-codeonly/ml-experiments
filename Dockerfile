FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up directory structure
WORKDIR /usr/ml-experiments

# Copy project files
COPY . /usr/ml-experiments/

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the import fix script
RUN python fix_imports.py /usr/ml-experiments

# Create necessary directories
RUN mkdir -p /usr/ml-experiments/models
RUN mkdir -p /outputs
RUN mkdir -p /data

# Make training script executable
RUN chmod +x /usr/ml-experiments/train_cloud.sh

# Set environment variables
ENV PYTHONPATH=$PYTHONPATH:/usr/ml-experiments
ENV MODEL_PATH=/usr/ml-experiments/models/best_model.pth

# Starting command
ENTRYPOINT ["/usr/ml-experiments/train_cloud.sh"] 