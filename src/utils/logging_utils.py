"""
Utilities for logging and monitoring training progress.
"""

import os
import logging
import time
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logging
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to {log_file}")
    
    return logger


class TensorboardLogger:
    """TensorBoard logger for training and validation metrics."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize TensorBoard logger."""
        self.log_dir = os.path.join(log_dir, experiment_name, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.global_step = 0
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to TensorBoard."""
        if step is None:
            step = self.global_step
            self.global_step += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}{key}", value, step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f"{prefix}{key}", value.item(), step)
    
    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """Log histogram to TensorBoard."""
        if step is None:
            step = self.global_step
        
        self.writer.add_histogram(name, values, step)
    
    def log_images(self, name: str, images, step: Optional[int] = None):
        """Log images to TensorBoard."""
        if step is None:
            step = self.global_step
        
        self.writer.add_images(name, images, step)
    
    def log_model_graph(self, model, input_to_model):
        """Log model graph to TensorBoard."""
        self.writer.add_graph(model, input_to_model)
    
    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, Any]):
        """Log hyperparameters and corresponding metrics to TensorBoard."""
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class MetricsTracker:
    """Tracker for training and validation metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_metrics = {}
        self.best_epoch = 0
        
    def update_train_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics."""
        for key, value in metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.train_metrics[key].append(value)
    
    def update_val_metrics(self, metrics: Dict[str, Any], epoch: int):
        """Update validation metrics and track best performance."""
        for key, value in metrics.items():
            if key not in self.val_metrics:
                self.val_metrics[key] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.val_metrics[key].append(value)
            
            # Track best performance
            if key.startswith("val_"):
                metric_key = key
                if metric_key not in self.best_val_metrics or value > self.best_val_metrics[metric_key]:
                    self.best_val_metrics[metric_key] = value
                    self.best_epoch = epoch
    
    def get_latest_train_metrics(self) -> Dict[str, float]:
        """Get latest training metrics."""
        return {key: values[-1] for key, values in self.train_metrics.items() if values}
    
    def get_latest_val_metrics(self) -> Dict[str, float]:
        """Get latest validation metrics."""
        return {key: values[-1] for key, values in self.val_metrics.items() if values}
    
    def get_best_val_metrics(self) -> Dict[str, Any]:
        """Get best validation metrics and corresponding epoch."""
        return {
            "metrics": self.best_val_metrics,
            "epoch": self.best_epoch
        }
    
    def reset_train_metrics(self):
        """Reset training metrics for a new epoch."""
        for key in self.train_metrics:
            self.train_metrics[key] = [] 