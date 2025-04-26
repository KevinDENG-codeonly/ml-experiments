#!/usr/bin/env python
"""
Main script for training and evaluating Vision Transformers.
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path

from src.utils.config_utils import load_config, save_config
from src.utils.logging_utils import setup_logging
from src.data_loaders.datasets import create_data_loaders
from src.models.model_factory import create_model, print_model_info
from src.core.trainer import ViTTrainer


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Vision Transformer Training")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    
    # Device
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    
    # Dataset
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--image_size", type=int, help="Image size")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_workers", type=int, help="Number of workers")
    
    # Model
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--pretrained_path", type=str, help="Path to pretrained weights")
    
    # Training
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--optimizer", type=str, help="Optimizer")
    parser.add_argument("--scheduler", type=str, help="Scheduler")
    parser.add_argument("--label_smoothing", type=float, help="Label smoothing")
    parser.add_argument("--mixup_alpha", type=float, help="Mixup alpha")
    parser.add_argument("--cutmix_alpha", type=float, help="CutMix alpha")
    parser.add_argument("--auto_augment", action="store_true", help="Use auto augmentation")
    
    # Logging and checkpoints
    parser.add_argument("--log_dir", type=str, help="Log directory")
    parser.add_argument("--save_dir", type=str, help="Model save directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Mode
    parser.add_argument("--eval_only", action="store_true", help="Evaluation only mode")
    parser.add_argument("--test", action="store_true", help="Test mode (after training)")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main function."""
    # Parse arguments
    args = get_args()
    
    # Load configuration
    config_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(args.config, **config_overrides)
    
    # Create output directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.save_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(
        log_dir=config.logging.log_dir,
        experiment_name=config.experiment_name,
    )
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Set random seed
    set_seed(config.seed)
    logger.info(f"Random seed: {config.seed}")
    
    # Save configuration
    config_save_path = os.path.join(
        config.logging.save_dir,
        f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml",
    )
    save_config(config, config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Determine device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
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
    
    # Create model
    logger.info("Creating model...")
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
    
    # Train or evaluate
    if args.eval_only:
        logger.info("Evaluation mode")
        if args.resume is None:
            logger.warning("No checkpoint specified for evaluation, using randomly initialized model")
        else:
            logger.info(f"Loading checkpoint: {args.resume}")
            trainer.checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path=args.resume,
                map_location=device,
            )
        
        val_metrics = trainer.validate(epoch=0)
        logger.info(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")
    else:
        logger.info("Training mode")
        results = trainer.train(resume_from=args.resume)
        logger.info(f"Training completed with best {config.logging.monitor}: {results['best_score']:.4f} at epoch {results['best_epoch']}")
        
        # Test if requested
        if args.test:
            logger.info("Testing best model...")
            best_model_path = results["best_model_path"]
            trainer.checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path=best_model_path,
                map_location=device,
            )
            
            test_metrics = trainer.validate(epoch=0)
            logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")


if __name__ == "__main__":
    main()
