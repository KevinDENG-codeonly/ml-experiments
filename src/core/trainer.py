"""
Vision Transformer trainer module.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nn_F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import logging

from src.utils.logging_utils import TensorboardLogger, MetricsTracker
from src.utils.model_utils import CheckpointManager

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted logits with shape (batch_size, num_classes).
            target: Target labels with shape (batch_size,) or one-hot with shape (batch_size, num_classes).
            
        Returns:
            Smoothed cross entropy loss.
        """
        if target.dim() == 1:
            # Convert to one-hot, ensuring num_classes is explicitly set to pred.size(-1)
            target = nn_F.one_hot(target.long(), num_classes=pred.size(-1)).float()
        elif target.size(-1) != pred.size(-1):
            # If one-hot but wrong number of classes, adjust
            if target.size(-1) < pred.size(-1):
                # Pad with zeros if needed
                padding = torch.zeros(target.size(0), pred.size(-1) - target.size(-1), device=target.device)
                target = torch.cat([target, padding], dim=1)
            else:
                # Truncate if needed
                target = target[:, :pred.size(-1)]
        
        pred = F.log_softmax(pred, dim=-1)
        
        # Calculate loss with smoothing
        if self.smoothing > 0:
            n_classes = pred.size(-1)
            target = target * (1 - self.smoothing) + self.smoothing / n_classes
        
        return torch.mean(torch.sum(-target * pred, dim=-1))


class ViTTrainer:
    """Trainer for Vision Transformer models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        """Initialize trainer.
        
        Args:
            model: Vision Transformer model.
            config: Training configuration.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Set up loss function
        self.criterion = self._create_criterion()
        
        # Set up metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Set up tensorboard logger
        self.tb_logger = TensorboardLogger(
            log_dir=config.logging.log_dir,
            experiment_name=config.experiment_name,
        )
        
        # Set up checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.logging.save_dir,
            model_name=config.experiment_name,
            monitor=config.logging.monitor,
            mode=config.logging.mode,
            save_top_k=config.logging.save_top_k,
            save_last=config.logging.save_last,
        )
        
        # Set up gradient scaler for mixed precision training
        self.scaler = GradScaler() if config.training.use_amp else None
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.training.scheduler.lower()
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=self.config.training.min_lr,
            )
        elif scheduler_name == "step":
            decay_steps = self.config.training.lr_scheduler_decay_steps
            if decay_steps is None:
                # Default decay steps at 1/3 and 2/3 of training
                epochs = self.config.training.epochs
                decay_steps = [epochs // 3, epochs * 2 // 3]
            
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=decay_steps,
                gamma=self.config.training.lr_scheduler_decay_rate,
            )
        elif scheduler_name == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.training.min_lr / self.config.training.learning_rate,
                total_iters=self.config.training.epochs,
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        if self.config.training.label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=self.config.training.label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train model for one epoch.
        
        Args:
            epoch: Current epoch.
            
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        self.metrics_tracker.reset_train_metrics()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Reset learning rate when using warmup
        if self.config.training.warmup_steps > 0 and epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 1e-6
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # For one-hot encoded targets from mixup/cutmix
            if targets.dim() > 1:
                target_labels = targets.argmax(dim=1)
            else:
                target_labels = targets
            
            # Handle gradient accumulation
            accumulate = self.config.training.accumulate_grad_batches > 1
            is_accumulating = accumulate and (batch_idx % self.config.training.accumulate_grad_batches != 0)
            
            # Warmup learning rate
            if self.config.training.warmup_steps > 0 and self.global_step < self.config.training.warmup_steps:
                lr_scale = min(1.0, float(self.global_step + 1) / self.config.training.warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_scale * self.config.training.learning_rate
            
            # Forward pass with mixed precision if enabled
            if self.config.training.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with scaling
                if not is_accumulating:
                    self.optimizer.zero_grad()
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping and optimizer step
                if not is_accumulating:
                    if self.config.training.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                if not is_accumulating:
                    self.optimizer.zero_grad()
                
                loss.backward()
                
                if not is_accumulating:
                    if self.config.training.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_val)
                    
                    self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(target_labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": 100.0 * correct / total,
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            
            # Log batch metrics
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                batch_metrics = {
                    "train_loss": loss.item(),
                    "train_accuracy": 100.0 * correct / total,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
                
                self.tb_logger.log_metrics(batch_metrics, step=self.global_step)
                self.metrics_tracker.update_train_metrics(batch_metrics)
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_metrics = {
            "train_loss": epoch_loss / len(self.train_loader),
            "train_accuracy": 100.0 * correct / total,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model.
        
        Args:
            epoch: Current epoch.
            
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            if self.config.training.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": val_loss / (batch_idx + 1),
                "acc": 100.0 * correct / total,
            })
        
        # Compute validation metrics
        val_metrics = {
            "val_loss": val_loss / len(self.val_loader),
            "val_accuracy": 100.0 * correct / total,
        }
        
        return val_metrics
    
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Train model for multiple epochs.
        
        Args:
            resume_from: Path to checkpoint to resume from.
            
        Returns:
            Dictionary with training results.
        """
        # Resume from checkpoint if provided
        if resume_from:
            metadata = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                checkpoint_path=resume_from,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
            
            self.current_epoch = metadata["epoch"]
            self.global_step = metadata["global_step"]
            logger.info(f"Resumed training from epoch {self.current_epoch}")
        
        # Log hyperparameters
        hparams = {
            "model/name": self.config.model.model_name,
            "model/patch_size": self.config.model.patch_size,
            "model/hidden_dim": self.config.model.hidden_dim,
            "model/num_heads": self.config.model.num_heads,
            "model/num_layers": self.config.model.num_layers,
            "model/num_classes": self.config.model.num_classes,
            "data/dataset": self.config.data.dataset,
            "data/image_size": self.config.data.image_size,
            "data/batch_size": self.config.data.batch_size,
            "training/optimizer": self.config.training.optimizer,
            "training/learning_rate": self.config.training.learning_rate,
            "training/weight_decay": self.config.training.weight_decay,
            "training/epochs": self.config.training.epochs,
            "training/scheduler": self.config.training.scheduler,
            "training/label_smoothing": self.config.training.label_smoothing,
            "training/mixup_alpha": self.config.training.mixup_alpha,
            "training/cutmix_alpha": self.config.training.cutmix_alpha,
        }
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_one_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update metrics tracker
            self.metrics_tracker.update_val_metrics(val_metrics, epoch)
            
            # Log epoch metrics
            all_metrics = {**train_metrics, **val_metrics}
            self.tb_logger.log_metrics(all_metrics, step=epoch, prefix="epoch/")
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                score=val_metrics[self.config.logging.monitor],
                metadata={
                    "metrics": all_metrics,
                    "config": self.config,
                },
            )
            
            # Display metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )
            
            # Check if we have a new best model
            if (epoch == 0 or 
                (self.config.logging.mode == "max" and val_metrics[self.config.logging.monitor] > self.best_score) or
                (self.config.logging.mode == "min" and val_metrics[self.config.logging.monitor] < self.best_score)):
                
                self.best_score = val_metrics[self.config.logging.monitor]
                best_epoch = epoch
                logger.info(f"New best model with {self.config.logging.monitor} = {self.best_score:.4f}")
        
        # Log total training time
        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time/60:.2f} minutes")
        
        # Log hyperparameters and final metrics
        best_metrics = self.metrics_tracker.get_best_val_metrics()
        final_metrics = {
            "hparam/best_val_accuracy": best_metrics["metrics"].get("val_accuracy", 0.0),
            "hparam/best_val_loss": best_metrics["metrics"].get("val_loss", 0.0),
            "hparam/best_epoch": best_metrics["epoch"],
        }
        self.tb_logger.log_hyperparams(hparams, final_metrics)
        
        # Close tensorboard logger
        self.tb_logger.close()
        
        # Return results
        return {
            "best_score": self.best_score,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "final_train_metrics": self.metrics_tracker.get_latest_train_metrics(),
            "final_val_metrics": self.metrics_tracker.get_latest_val_metrics(),
            "best_model_path": self.checkpoint_manager.get_best_checkpoint_path(),
        } 