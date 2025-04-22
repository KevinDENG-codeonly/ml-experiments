import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np

from ..models import get_model
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Predictor:
    """Class for model inference."""
    
    def __init__(self, 
                 model_path: str, 
                 device: Optional[str] = None, 
                 config: Optional[dict[str, Any]] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on
            config: Additional configuration
        """
        self.model_path = model_path
        self.device = torch.device(device if device else "cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        # Load model
        self._load_model()
        
        # Set up transforms
        self._setup_transforms()
        
    def _load_model(self):
        """Load model from checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")
            
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        if "config" in checkpoint:
            model_config = checkpoint["config"].get("model_config", {})
            model_name = checkpoint["config"].get("model_name", "vit")
        else:
            model_config = self.config.get("model_config", {})
            model_name = self.config.get("model_name", "vit")
            
        # Initialize model
        self.model = get_model(model_name, model_config)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
        
    def _setup_transforms(self):
        """Set up image transforms for inference."""
        img_size = self.config.get("img_size", 224)
        mean = self.config.get("mean", [0.485, 0.456, 0.406])
        std = self.config.get("std", [0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file {image} not found")
            image = Image.open(image).convert("RGB")
            
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Add batch dimension
        return image_tensor.unsqueeze(0)
    
    def predict(self, 
                image: Union[str, Image.Image, torch.Tensor], 
                return_probs: bool = False) -> Union[int, list[float]]:
        """
        Run inference on image.
        
        Args:
            image: Image path, PIL Image, or tensor
            return_probs: Whether to return class probabilities
            
        Returns:
            Class prediction or class probabilities
        """
        # Preprocess image if it's not already a tensor
        if not isinstance(image, torch.Tensor):
            image = self.preprocess(image)
            
        # Move to device
        image = image.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image)
            
        # Get predictions
        if return_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs[0].cpu().numpy().tolist()
        else:
            _, predicted = outputs.max(1)
            return predicted.item()
    
    def batch_predict(self, 
                     images: list[Union[str, Image.Image, torch.Tensor]], 
                     return_probs: bool = False) -> list[Union[int, list[float]]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of images
            return_probs: Whether to return class probabilities
            
        Returns:
            List of predictions or probabilities
        """
        # Preprocess images
        batch = []
        for image in images:
            if isinstance(image, torch.Tensor):
                batch.append(image)
            else:
                batch.append(self.preprocess(image).squeeze(0))
                
        # Stack tensors
        batch = torch.stack(batch).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch)
            
        # Get predictions
        if return_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return probs.cpu().numpy().tolist()
        else:
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy().tolist() 