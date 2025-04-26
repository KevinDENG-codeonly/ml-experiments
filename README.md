# Vision Transformer Training Framework

A modular, OOP-based framework for training and fine-tuning Vision Transformers.

## Features

- Modular OOP design with clear separation of concerns
- Support for various ViT architectures and configurations
- Hyperparameter optimization capabilities
- Training progress monitoring and logging
- Model checkpointing and versioning
- Command-line interface for training control
- Support for multiple compute platforms (CPU, CUDA, MPS)

## Setup

### Local Development Environment

1. Create a Python 3.12 virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the environment setup script to check for dependencies and create necessary directories:
   ```bash
   python setup_env.py
   ```

### Cloud Training Environment

#### Setting up a Cloud VM (AWS EC2/GCP/Azure)

1. Launch a VM with GPU support:
   - AWS: Use a `p3.2xlarge` or better instance with NVIDIA V100 GPUs
   - GCP: Use an N1 instance with NVIDIA T4 or V100 GPU
   - Azure: Use an NC series VM with NVIDIA GPUs

2. Install CUDA and cuDNN:
   ```bash
   # For Ubuntu:
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda-11-8
   ```

3. Set up Python environment:
   ```bash
   sudo apt-get install -y python3.10-venv
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/yourusername/ml-experiments.git
   cd ml-experiments
   pip install -r requirements.txt
   ```

5. Verify CUDA is working with PyTorch:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0))"
   ```

## Usage

### Local Training

Run training with default parameters:
```bash
python main.py
```

For Apple Silicon (MPS) users:
```bash
python examples/train_cifar10.py
```
The script will automatically detect MPS availability and configure training appropriately.

### Cloud Training

Run training with optimized settings for cloud GPU:
```bash
python main.py --config src/configs/cloud_config.yaml
```

For batch training that continues after SSH disconnection:
```bash
nohup python main.py --config src/configs/cloud_config.yaml > training.log &
```

#### Optimizing for Cloud Training

1. Create a cloud-optimized configuration file with:
   ```bash
   cp src/configs/cifar10_config.yaml src/configs/cloud_config.yaml
   ```

2. Edit the configuration to maximize GPU utilization:
   ```yaml
   data:
     batch_size: 128  # Increase based on GPU memory
     num_workers: 8   # Increase for faster data loading
     pin_memory: true
   
   training:
     use_amp: true    # Enable mixed precision training
     epochs: 100      # Run for longer on cloud
     learning_rate: 1e-3
     gradient_clip_val: 1.0
     accumulate_grad_batches: 1
   ```

3. Monitor training with TensorBoard:
   ```bash
   # On the cloud machine
   tensorboard --logdir outputs/logs --port 6006
   
   # On your local machine, create SSH tunnel
   ssh -L 6006:localhost:6006 username@cloud-ip
   
   # Then visit http://localhost:6006 in your browser
   ```

### Configuration Options

Specify a configuration file:
```bash
python main.py --config src/configs/experiment1.yaml
```

Override specific parameters:
```bash
python main.py --config src/configs/baseline.yaml --batch_size 32 --learning_rate 1e-4
```

### Cross-Platform Training

The framework automatically detects available hardware and optimizes accordingly:

- **CUDA (NVIDIA GPUs)**: Enables full GPU acceleration with mixed precision
- **MPS (Apple Silicon)**: Enables GPU acceleration (with some limitations)
- **CPU**: Falls back to CPU training (slower but universally compatible)

To ensure consistent behavior across platforms, you can specify the device manually:
```bash
python main.py --device cuda  # Force CUDA
python main.py --device mps   # Force MPS
python main.py --device cpu   # Force CPU
```

## Project Structure

- `src/core/`: Core training, evaluation, and optimization components
- `src/models/`: Vision Transformer implementations and related components
- `src/utils/`: Utility functions for logging, visualization, etc.
- `src/data_loaders/`: Dataset and dataloader implementations
- `src/configs/`: Configuration files for experiments
- `examples/`: Example training scripts for different scenarios

## Results and Artifacts

- Training logs and model checkpoints are saved in the `outputs/` directory by default
- TensorBoard logs: `outputs/logs/[experiment_name]/[timestamp]/`
- Model checkpoints: `outputs/models/[experiment_name]/`
- Best model: `outputs/models/[experiment_name]/best.pt`
- Latest model: `outputs/models/[experiment_name]/last.pt`

## Resuming Training

To resume training from a checkpoint:
```bash
python main.py --resume outputs/models/vit_cifar10/best.pt
```

## Cloud Storage Integration

For long-running cloud training, you can sync results to cloud storage:

### AWS S3
```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure

# Sync outputs directory to S3
aws s3 sync outputs/ s3://your-bucket/ml-experiments/outputs/
```

### Google Cloud Storage
```bash
# Install Google Cloud SDK
pip install google-cloud-storage

# Authenticate with Google Cloud
gcloud auth login

# Sync outputs directory to GCS
gsutil -m rsync -r outputs/ gs://your-bucket/ml-experiments/outputs/
```

## Troubleshooting

- **CUDA out of memory**: Reduce batch size or model size in configuration
- **Slow data loading**: Increase num_workers in configuration
- **Training instability**: Reduce learning rate or enable gradient clipping
- **Cloud VM disconnection**: Use `nohup` or `tmux` to keep processes running

For more help, please open an issue on GitHub.