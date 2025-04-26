"""
Utilities for model management, saving and loading checkpoints.
"""

import os
import json
import torch
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints, saving and loading."""
    
    def __init__(
        self,
        save_dir: str,
        model_name: str,
        monitor: str = "val_accuracy",
        mode: str = "max",
        save_top_k: int = 3,
        save_last: bool = True,
    ):
        """Initialize CheckpointManager.
        
        Args:
            save_dir: Directory to save checkpoints
            model_name: Name of the model
            monitor: Metric to monitor for determining best models
            mode: One of ['min', 'max'] - whether lower or higher metric is better
            save_top_k: Number of best models to save
            save_last: Whether to save the last model
        """
        self.save_dir = os.path.join(save_dir, model_name)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        
        self.best_models = []
        self.best_score = float("-inf") if mode == "max" else float("inf")
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create or load checkpoint tracker
        self.tracker_path = os.path.join(self.save_dir, "checkpoint_tracker.json")
        self._load_or_create_tracker()
    
    def _load_or_create_tracker(self):
        """Load existing checkpoint tracker or create a new one."""
        if os.path.exists(self.tracker_path):
            with open(self.tracker_path, "r") as f:
                self.tracker = json.load(f)
                
            # Extract best score and models from tracker
            if self.tracker["checkpoints"]:
                if self.mode == "max":
                    self.best_score = max([ckpt["score"] for ckpt in self.tracker["checkpoints"] if ckpt["score"] is not None])
                else:
                    self.best_score = min([ckpt["score"] for ckpt in self.tracker["checkpoints"] if ckpt["score"] is not None])
                
                self.best_models = sorted(
                    [ckpt for ckpt in self.tracker["checkpoints"] if ckpt["score"] is not None],
                    key=lambda x: x["score"],
                    reverse=(self.mode == "max")
                )[:self.save_top_k]
        else:
            self.tracker = {
                "checkpoints": [],
                "best_model_path": None,
                "last_model_path": None,
            }
    
    def _save_tracker(self):
        """Save checkpoint tracker to file."""
        with open(self.tracker_path, "w") as f:
            json.dump(self.tracker, f, indent=2)
    
    def _is_better(self, current_score: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "max":
            return current_score > self.best_score
        return current_score < self.best_score
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,  # Type can vary depending on scheduler used
        epoch: int = 0,
        global_step: int = 0,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state to save
            epoch: Current epoch
            global_step: Current global step
            score: Current score for the monitored metric
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"epoch_{epoch}_step_{global_step}"
        if score is not None:
            checkpoint_name += f"_{self.monitor}_{score:.4f}"
        
        checkpoint_path = os.path.join(self.save_dir, f"{checkpoint_name}.pt")
        
        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "score": score,
            "metadata": metadata or {},
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update tracker with new checkpoint
        checkpoint_info = {
            "path": checkpoint_path,
            "epoch": epoch,
            "global_step": global_step,
            "score": score,
        }
        
        # Handle 'last' checkpoint logic
        if self.save_last:
            last_checkpoint_path = os.path.join(self.save_dir, "last.pt")
            torch.save(checkpoint, last_checkpoint_path)
            self.tracker["last_model_path"] = last_checkpoint_path
        
        # Handle 'best' checkpoint logic
        if score is not None and (not self.best_models or self._is_better(score)):
            best_checkpoint_path = os.path.join(self.save_dir, "best.pt")
            torch.save(checkpoint, best_checkpoint_path)
            self.tracker["best_model_path"] = best_checkpoint_path
            self.best_score = score
        
        # Update list of checkpoints
        self.tracker["checkpoints"].append(checkpoint_info)
        
        # Sort and keep top-k checkpoints based on score
        if score is not None:
            self.tracker["checkpoints"] = sorted(
                [ckpt for ckpt in self.tracker["checkpoints"] if ckpt["score"] is not None],
                key=lambda x: x["score"],
                reverse=(self.mode == "max")
            )
            
            # Update best models list
            self.best_models = self.tracker["checkpoints"][:self.save_top_k]
            
            # Remove checkpoints that are not in top-k and not last
            if self.save_top_k > 0:
                checkpoints_to_keep = [ckpt["path"] for ckpt in self.best_models]
                if self.save_last and self.tracker["last_model_path"] is not None:
                    checkpoints_to_keep.append(self.tracker["last_model_path"])
                
                for ckpt in self.tracker["checkpoints"][self.save_top_k:]:
                    if ckpt["path"] not in checkpoints_to_keep and os.path.exists(ckpt["path"]):
                        try:
                            os.remove(ckpt["path"])
                            logger.info(f"Removed checkpoint: {ckpt['path']}")
                        except Exception as e:
                            logger.warning(f"Failed to remove checkpoint {ckpt['path']}: {e}")
        
        # Save tracker
        self._save_tracker()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,  # Type can vary depending on scheduler used
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint to load, or "best" or "last"
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into
            map_location: Device to map tensors to
            
        Returns:
            Dictionary with checkpoint metadata
        """
        if checkpoint_path is None or checkpoint_path == "best":
            if self.tracker["best_model_path"] is not None:
                checkpoint_path = self.tracker["best_model_path"]
            else:
                raise ValueError("No best model checkpoint found.")
        elif checkpoint_path == "last":
            if self.tracker["last_model_path"] is not None:
                checkpoint_path = self.tracker["last_model_path"]
            else:
                raise ValueError("No last model checkpoint found.")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Return metadata
        metadata = {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "score": checkpoint.get("score", None),
            "metadata": checkpoint.get("metadata", {}),
        }
        
        return metadata
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint."""
        return self.tracker["best_model_path"]
    
    def get_last_checkpoint_path(self) -> Optional[str]:
        """Get path to last checkpoint."""
        return self.tracker["last_model_path"]
    
    def list_checkpoints(self) -> Dict[str, Any]:
        """List all checkpoints."""
        return self.tracker 