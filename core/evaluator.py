import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .base import BaseHandler
from ..utils.logger import get_logger
from ..models import get_model

logger = get_logger(__name__)

class Evaluator(BaseHandler):
    """Handles model evaluation on test set."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.test_loader = None
        self.metrics = config.get("metrics", ["accuracy", "precision", "recall", "f1"])
        
    def setup(self, 
              model: Optional[nn.Module] = None, 
              test_loader: Optional[DataLoader] = None):
        """Setup evaluation components."""
        # Set up model
        if model is not None:
            self.model = model
        else:
            model_name = self.config.get("model_name")
            model_config = self.config.get("model_config", {})
            self.model = get_model(model_name, model_config)
            
            # Load checkpoint if specified
            checkpoint_path = self.config.get("checkpoint_path")
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Set up test loader
        self.test_loader = test_loader
        
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model and compute metrics."""
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and test loader must be set up before evaluation")
            
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc="Evaluating") as progress_bar:
                for inputs, targets in progress_bar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    _, predictions = outputs.max(1)
                    
                    # Store for metric calculation
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        # Calculate metrics
        results = {}
        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(all_targets, all_predictions)
            
        if "precision" in self.metrics:
            results["precision"] = precision_score(all_targets, all_predictions, average="macro")
            
        if "recall" in self.metrics:
            results["recall"] = recall_score(all_targets, all_predictions, average="macro")
            
        if "f1" in self.metrics:
            results["f1"] = f1_score(all_targets, all_predictions, average="macro")
            
        if "confusion_matrix" in self.metrics:
            results["confusion_matrix"] = confusion_matrix(all_targets, all_predictions)
            
        # Log results
        for metric, value in results.items():
            if metric != "confusion_matrix":
                logger.info(f"{metric.capitalize()}: {value:.4f}")
            
        return results
    
    def run(self):
        """Run evaluation."""
        return self.evaluate() 