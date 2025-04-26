#!/usr/bin/env python
"""
Example script to train a small Vision Transformer on CIFAR-10 dataset.
"""

import os
import sys
import torch
import platform

# Add the project root to the path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging
from src.data_loaders.datasets import create_data_loaders
from src.models.model_factory import create_model, print_model_info
from src.core.trainer import ViTTrainer


def main():
    """Train a small Vision Transformer on CIFAR-10."""
    # Load configuration
    config = load_config("src/configs/cifar10_config.yaml")
    
    # Modify configuration for faster training
    config.model.model_name = "vit_tiny_patch16"  # Smaller model
    config.model.num_classes = 10  # CIFAR-10 has 10 classes
    config.data.image_size = 224  # Input image size
    config.data.batch_size = 32  # Smaller batch size for local training
    config.data.num_workers = 0  # Use single thread to avoid multiprocessing issues 
    config.training.epochs = 1  # Fewer epochs for testing
    
    # Set device - Support for CUDA, MPS (Apple Silicon), and CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        config.device = "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        config.device = "mps"
        # MPS doesn't fully support AMP yet
        config.training.use_amp = False
    else:
        device = torch.device("cpu")
        config.device = "cpu"
        config.training.use_amp = False
    
    # Setup logging
    logger = setup_logging(
        log_dir=config.logging.log_dir,
        experiment_name=config.experiment_name,
    )
    logger.info(f"Running example on device: {device}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        dataset_name=config.data.dataset,
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        image_size=config.data.image_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        train_val_split=config.data.train_val_split,
        auto_augment=config.training.auto_augment,
        random_erase=config.training.random_erase,
        mixup_alpha=config.training.mixup_alpha,
        cutmix_alpha=config.training.cutmix_alpha,
    )
    
    logger.info(f"Dataset: {config.data.dataset}")
    
    # Create model
    model = create_model(config)
    print_model_info(model)
    
    # Create trainer
    trainer = ViTTrainer(
        model=model,
        config=config,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        device=device,
    )
    
    # Train model
    results = trainer.train()
    
    # Print results
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {results['best_score']:.2f}%")
    logger.info(f"Best model saved at: {results['best_model_path']}")
    
    return results


if __name__ == "__main__":
    main() 