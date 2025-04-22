from typing import Optional, Any
from .base import BaseDataModule

# Global registry for datasets
DATASET_REGISTRY: dict[str, type[BaseDataModule]] = {}


def register_dataset(name: str):
    """
    Decorator for registering a dataset class.
    
    Args:
        name: The name under which the dataset will be registered
        
    Returns:
        The decorator function
    """
    def decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset {name} already registered")
        if not issubclass(cls, BaseDataModule):
            raise TypeError(f"Dataset {name} must be a subclass of BaseDataModule")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str, config: Optional[dict[str, Any]] = None) -> BaseDataModule:
    """
    Factory function to get a dataset instance by name.
    
    Args:
        name: The name of the dataset to retrieve
        config: Configuration parameters for dataset initialization
        
    Returns:
        An instance of the requested dataset
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not found in registry. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    dataset_cls = DATASET_REGISTRY[name]
    if config is None:
        config = {}
    
    return dataset_cls(config) 