"""
Utilities for loading and saving configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging

from src.configs.default_config import Config, get_config

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        try:
            config_dict = yaml.safe_load(f)
            return config_dict
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise


def save_yaml_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        try:
            yaml.dump(config, f, default_flow_style=False)
        except yaml.YAMLError as e:
            logger.error(f"Error saving configuration file: {e}")
            raise


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert Config object to dictionary."""
    result = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "device": config.device,
        "precision": config.precision,
        "distributed": config.distributed,
        "world_size": config.world_size,
        
        "data": {
            "dataset": config.data.dataset,
            "data_dir": config.data.data_dir,
            "image_size": config.data.image_size,
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
            "pin_memory": config.data.pin_memory,
            "train_val_split": config.data.train_val_split,
        },
        
        "model": {
            "model_name": config.model.model_name,
            "patch_size": config.model.patch_size,
            "hidden_dim": config.model.hidden_dim,
            "num_heads": config.model.num_heads,
            "num_layers": config.model.num_layers,
            "mlp_dim": config.model.mlp_dim,
            "dropout": config.model.dropout,
            "attention_dropout": config.model.attention_dropout,
            "num_classes": config.model.num_classes,
            "representation_size": config.model.representation_size,
            "pretrained": config.model.pretrained,
            "pretrained_path": config.model.pretrained_path,
        },
        
        "training": {
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "warmup_steps": config.training.warmup_steps,
            "optimizer": config.training.optimizer,
            "scheduler": config.training.scheduler,
            "min_lr": config.training.min_lr,
            "lr_scheduler_decay_steps": config.training.lr_scheduler_decay_steps,
            "lr_scheduler_decay_rate": config.training.lr_scheduler_decay_rate,
            "label_smoothing": config.training.label_smoothing,
            "mixup_alpha": config.training.mixup_alpha,
            "cutmix_alpha": config.training.cutmix_alpha,
            "auto_augment": config.training.auto_augment,
            "random_erase": config.training.random_erase,
            "gradient_clip_val": config.training.gradient_clip_val,
            "accumulate_grad_batches": config.training.accumulate_grad_batches,
            "use_amp": config.training.use_amp,
        },
        
        "logging": {
            "log_dir": config.logging.log_dir,
            "save_dir": config.logging.save_dir,
            "log_every_n_steps": config.logging.log_every_n_steps,
            "val_check_interval": config.logging.val_check_interval,
            "save_top_k": config.logging.save_top_k,
            "save_last": config.logging.save_last,
            "monitor": config.logging.monitor,
            "mode": config.logging.mode,
        },
    }
    
    return result


def update_config_from_dict(config: Config, config_dict: Dict[str, Any]) -> Config:
    """Update Config object from dictionary."""
    # Flatten nested dictionaries
    flat_dict = {}
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat_dict[inner_key] = inner_value
        else:
            flat_dict[key] = value
    
    # Update config with flattened dictionary
    config.update(**flat_dict)
    
    return config


def load_config(config_path: Optional[str] = None, **kwargs) -> Config:
    """Load configuration from file and/or keyword arguments."""
    config = get_config()
    
    if config_path is not None:
        config_dict = load_yaml_config(config_path)
        config = update_config_from_dict(config, config_dict)
    
    # Update with keyword arguments
    config.update(**kwargs)
    
    return config


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to file."""
    config_dict = config_to_dict(config)
    save_yaml_config(config_dict, save_path) 