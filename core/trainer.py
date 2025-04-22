# core/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from typing import Any, Optional, Tuple
import json
import numpy as np

from .base import BaseHandler
from ..utils.logger import get_logger
from ..models import get_model

logger = get_logger(__name__)

class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer(BaseHandler):
    """Model trainer class."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.epochs = config.get("epochs", 100)
        self.save_dir = config.get("save_dir", "checkpoints")
        self.save_freq = config.get("save_freq", 10)
        self.log_freq = config.get("log_freq", 10)
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.grad_clip = config.get("grad_clip", 1.0)
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
    def setup(self, train_loader=None, val_loader=None, model=None):
        """Setup for training."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create model if not provided
        if model is not None:
            self.model = model
        else:
            from ..models.registry import create_model
            self.model = create_model(self.config)
            
        self.model = self.model.to(self.device)
        
        # Create criterion
        criterion_name = self.config.get("criterion", "cross_entropy")
        if criterion_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == "mse":
            self.criterion = nn.MSELoss()
        elif criterion_name == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_name}")
            
        # Create optimizer
        optimizer_name = self.config.get("optimizer", "adam")
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-5)
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
        # Create scheduler
        scheduler_name = self.config.get("scheduler", None)
        if scheduler_name is None:
            self.scheduler = None
        elif scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.epochs
            )
        elif scheduler_name == "step":
            step_size = self.config.get("step_size", 30)
            gamma = self.config.get("gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
            
    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc="Training")
        for i, batch in enumerate(pbar):
            # Get data
            images, targets = batch
            batch_size = images.size(0)
            
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            accuracy = correct / batch_size
            
            # Update metrics
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy, batch_size)
            
            # Update progress bar
            pbar.set_postfix({"loss": losses.avg, "acc": accuracies.avg})
            
        return losses.avg, accuracies.avg
    
    def validate(self) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for i, batch in enumerate(pbar):
                # Get data
                images, targets = batch
                batch_size = images.size(0)
                
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                accuracy = correct / batch_size
                
                # Update metrics
                losses.update(loss.item(), batch_size)
                accuracies.update(accuracy, batch_size)
                
                # Update progress bar
                pbar.set_postfix({"loss": losses.avg, "acc": accuracies.avg})
                
        return losses.avg, accuracies.avg
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, "latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if needed
        if is_best:
            best_path = os.path.join(self.save_dir, "best.pth")
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load history and best metrics
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        if "best_epoch" in checkpoint:
            self.best_epoch = checkpoint["best_epoch"]
            
        # Return checkpoint epoch
        return checkpoint.get("epoch", 0)
            
    def log_stats(self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        """Log training statistics."""
        # Print stats
        if epoch % self.log_freq == 0:
            logger.info(f"Epoch {epoch}/{self.epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
        # Save to history
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        
        # Save history to file
        with open(os.path.join(self.save_dir, "history.json"), "w") as f:
            json.dump({k: [float(v_i) for v_i in v] if isinstance(v, list) else v 
                      for k, v in self.history.items()}, f, indent=4)
            
    def run(self, start_epoch: int = 0):
        """Run training for specified number of epochs."""
        if self.train_loader is None or self.val_loader is None:
            raise ValueError("Data loaders must be set up before training")
            
        # Training loop
        for epoch in range(start_epoch, self.epochs):
            # Train one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Log stats
            self.log_stats(epoch + 1, train_loss, train_acc, val_loss, val_acc)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                
            if (epoch + 1) % self.save_freq == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
                
        return self.history
