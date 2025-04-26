"""
Factory for creating model instances.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import timm
import logging

from src.models.vit import (
    vit_tiny_patch16,
    vit_small_patch16,
    vit_base_patch16,
    vit_large_patch16,
)

logger = logging.getLogger(__name__)


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        Model instance.
    """
    model_name = config.model.model_name
    
    # Get model parameters
    model_params = {
        "image_size": config.data.image_size,
        "num_classes": config.model.num_classes,
        "representation_size": config.model.representation_size,
        "dropout": config.model.dropout,
        "attention_dropout": config.model.attention_dropout,
    }
    
    # Use custom Vision Transformer implementations
    if model_name == "vit_tiny_patch16":
        model = vit_tiny_patch16(**model_params)
    elif model_name == "vit_small_patch16":
        model = vit_small_patch16(**model_params)
    elif model_name == "vit_base_patch16":
        model = vit_base_patch16(**model_params)
    elif model_name == "vit_large_patch16":
        model = vit_large_patch16(**model_params)
    # Use timm models if available
    elif model_name in timm.list_models(pretrained=True):
        logger.info(f"Using pretrained model from timm: {model_name}")
        model = timm.create_model(
            model_name,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_classes,
            drop_rate=config.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load pretrained weights if specified
    if config.model.pretrained_path is not None:
        logger.info(f"Loading pretrained weights from: {config.model.pretrained_path}")
        checkpoint = torch.load(config.model.pretrained_path)
        
        if "model_state_dict" in checkpoint:
            # Load from our checkpoint format
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
    
    return model


def get_model_size(model: nn.Module) -> int:
    """Get model size in parameters.
    
    Args:
        model: Model.
        
    Returns:
        Number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def print_model_info(model: nn.Module):
    """Print model information.
    
    Args:
        model: Model.
    """
    logger.info(f"Model architecture: {type(model).__name__}")
    logger.info(f"Model parameters: {get_model_size(model):,}")
    logger.info(f"Model structure:\n{model}") 