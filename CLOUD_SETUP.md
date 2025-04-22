# Cloud Setup Guide for ViT Training

This guide provides instructions for setting up and running ViT model training on cloud infrastructure using PyTorch 2.1.0, CUDA 12.1.0, and Python 3.10.

## Prerequisites

- Cloud VM or instance with NVIDIA GPUs that support CUDA 12.1.0
- Sufficient storage space for datasets and model checkpoints
- Python 3.10 installed

## Setup Steps

### 1. Environment Setup

#### Option 1: Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t vit-training -f deploy/Dockerfile .

# Run the container with GPU support
docker run --gpus all -v /path/to/data:/data -v /path/to/outputs:/outputs vit-training
```

#### Option 2: Manual Setup

```bash
# Create a Python 3.10 virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install CUDA 12.1.0 compatible PyTorch 2.1.0
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
```

### 2. Verify Setup

Run the following commands to verify your setup:

```bash
# Check Python version
python --version  # Should display Python 3.10.x

# Check PyTorch and CUDA versions
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

You should see output confirming:
- Python 3.10.x
- PyTorch 2.1.0
- CUDA available: True
- CUDA version: 12.1

### 3. Data Preparation

Copy your dataset to the appropriate location:

```bash
# Create data directory
mkdir -p /data

# Copy your dataset (example for CIFAR-10)
# If using a custom dataset, modify accordingly
python -c "import torchvision; torchvision.datasets.CIFAR10('/data', download=True)"
```

### 4. Training Configuration

Review and modify `config_cloud.yaml` to suit your specific needs:

- Adjust batch size based on your GPU memory
- Modify model hyperparameters if needed
- Change dataset settings to match your data

### 5. Running Training

#### Option 1: Using the Training Script

```bash
# Make the script executable
chmod +x train_cloud.sh

# Run the training script
./train_cloud.sh
```

#### Option 2: Manual Training Command

```bash
# Set the PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Create output directories
mkdir -p /outputs/vit_experiment/logs
mkdir -p /outputs/vit_experiment/checkpoints

# Run the training
python main.py train --config config_cloud.yaml
```

### 6. Distributed Training Setup

For multi-GPU training:

```bash
# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify available GPUs

# Run with torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 main.py train --config config_cloud.yaml
```

### 7. Monitoring Training

Monitor the training progress:

```bash
# View logs in real-time
tail -f /outputs/vit_experiment/logs/training.log
```

### 8. Evaluating Trained Models

After training completes:

```bash
# Run evaluation
python main.py eval --config config_cloud.yaml --model-path /outputs/vit_experiment/checkpoints/best_model.pth
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory Error**
   - Reduce batch size in `config_cloud.yaml`
   - Enable gradient accumulation (already configured)

2. **Slow Training**
   - Verify PyTorch is using the correct CUDA version
   - Ensure `torch.compile()` is enabled in the config
   - Check if Automatic Mixed Precision (AMP) is enabled

3. **Distributed Training Issues**
   - Ensure NCCL backend is properly configured
   - Check network connectivity between nodes (for multi-node setups)

## Cloud Provider Specific Instructions

### AWS

For AWS EC2 p3/p4 instances:

```bash
# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-525

# Install NVIDIA Container Toolkit for Docker
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime
```

### Google Cloud Platform

For GCP instances with GPUs:

```bash
# Install NVIDIA drivers
curl -O https://storage.googleapis.com/nvidia-drivers-us-public/GRID/GRID13.1/NVIDIA-Linux-x86_64-525.105.17-grid.run
sudo bash NVIDIA-Linux-x86_64-525.105.17-grid.run
```

## Additional Resources

- [PyTorch 2.1.0 Documentation](https://pytorch.org/docs/2.1.0/)
- [PyTorch Distributed Training Guide](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [CUDA 12.1.0 Documentation](https://docs.nvidia.com/cuda/archive/12.1.0/) 