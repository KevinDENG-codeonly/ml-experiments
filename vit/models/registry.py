from typing import Optional, Any
from .base import BaseModel

# Global registry for models
MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str):
    """
    Decorator for registering a model class.
    
    Args:
        name: The name under which the model will be registered
        
    Returns:
        The decorator function
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name} already registered")
        if not issubclass(cls, BaseModel):
            raise TypeError(f"Model {name} must be a subclass of BaseModel")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, config: Optional[dict[str, Any]] = None) -> BaseModel:
    """
    Factory function to get a model instance by name.
    
    Args:
        name: The name of the model to retrieve
        config: Configuration parameters for model initialization
        
    Returns:
        An instance of the requested model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_cls = MODEL_REGISTRY[name]
    if config is None:
        config = {}
    
    return model_cls(**config) 