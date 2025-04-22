#!/bin/bash
# Cloud training script for ViT with PyTorch 2.1.0, CUDA 12.1.0 and Python 3.10

# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs

# Print versions
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA version:"
python -c "import torch; print(torch.version.cuda)"

# Generate unique experiment name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="vit_cloud_${TIMESTAMP}"

# Create directories for logs, checkpoints, and results
mkdir -p /outputs/${EXP_NAME}/logs
mkdir -p /outputs/${EXP_NAME}/checkpoints
mkdir -p /outputs/${EXP_NAME}/results

# Copy config to the output directory for reproducibility
cp config_cloud.yaml /outputs/${EXP_NAME}/config.yaml

# Optional: data preparation
echo "Preparing data..."
# Add data preparation commands here if needed

# Start training
echo "Starting training with config_cloud.yaml..."
python main.py train --config config_cloud.yaml 2>&1 | tee /outputs/${EXP_NAME}/logs/training.log

# Optional: run evaluation on test set after training
echo "Evaluating model..."
python main.py eval --config config_cloud.yaml --model-path /outputs/${EXP_NAME}/checkpoints/best_model.pth 2>&1 | tee /outputs/${EXP_NAME}/logs/evaluation.log

echo "Training and evaluation complete. Results saved to /outputs/${EXP_NAME}" 