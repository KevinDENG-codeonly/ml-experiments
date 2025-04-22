FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /usr/ml-experiments

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first (including setup.py)
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/usr/ml-experiments
ENV MODEL_PATH=/usr/ml-experiments/models/best_model.pth

# Make training script executable
RUN chmod +x train_cloud.sh

# Create necessary directories
RUN mkdir -p /usr/ml-experiments/models
RUN mkdir -p /outputs
RUN mkdir -p /data

# Expose port for API
EXPOSE 5000

# Run the server for inference, or use train_cloud.sh for training
CMD ["uvicorn", "ml_experiments.deploy.serve:get_application", "--host", "0.0.0.0", "--port", "5000"] 