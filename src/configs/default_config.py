"""
Default configuration for Vision Transformer training.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "cifar10"  # Dataset name: cifar10, cifar100, imagenet, etc.
    data_dir: str = "./data"  # Data directory
    image_size: int = 224  # Image size for ViT
    batch_size: int = 64  # Batch size
    num_workers: int = 4  # Number of workers for data loading
    pin_memory: bool = True  # Pin memory for data loading
    train_val_split: float = 0.9  # Train-validation split ratio


@dataclass
class ModelConfig:
    """Vision Transformer model configuration."""
    model_name: str = "vit_base_patch16"  # Model architecture
    patch_size: int = 16  # Patch size
    hidden_dim: int = 768  # Hidden dimension
    num_heads: int = 12  # Number of attention heads
    num_layers: int = 12  # Number of transformer layers
    mlp_dim: int = 3072  # MLP dimension
    dropout: float = 0.1  # Dropout rate
    attention_dropout: float = 0.0  # Attention dropout rate
    num_classes: int = 10  # Number of classes
    representation_size: Optional[int] = None  # Representation size (None = no representation layer)
    pretrained: bool = False  # Whether to load pretrained weights
    pretrained_path: Optional[str] = None  # Path to pretrained weights


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training settings
    epochs: int = 100  # Number of epochs
    learning_rate: float = 1e-3  # Learning rate
    weight_decay: float = 0.05  # Weight decay
    warmup_steps: int = 500  # Number of warmup steps
    
    # Optimizer settings
    optimizer: str = "adamw"  # Optimizer: adam, adamw, sgd
    scheduler: str = "cosine"  # Scheduler: cosine, linear, step, none
    
    # Learning rate scheduler settings
    min_lr: float = 1e-5  # Minimum learning rate for scheduler
    lr_scheduler_decay_steps: Optional[List[int]] = None  # Steps for StepLR
    lr_scheduler_decay_rate: float = 0.1  # Decay rate for StepLR
    
    # Regularization and augmentation
    label_smoothing: float = 0.1  # Label smoothing factor
    mixup_alpha: float = 0.0  # Mixup alpha (0.0 = disabled)
    cutmix_alpha: float = 0.0  # CutMix alpha (0.0 = disabled)
    auto_augment: bool = False  # Whether to use auto augmentation
    random_erase: float = 0.0  # Random erase probability

    # Gradient settings
    gradient_clip_val: float = 1.0  # Gradient clipping value
    accumulate_grad_batches: int = 1  # Gradient accumulation steps

    # AMP (mixed precision)
    use_amp: bool = True  # Whether to use automatic mixed precision
    

@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "./outputs/logs"  # Log directory
    save_dir: str = "./outputs/models"  # Model save directory
    log_every_n_steps: int = 50  # Log every n steps
    val_check_interval: float = 1.0  # Validation check interval (1.0 = once per epoch)
    save_top_k: int = 3  # Save top k models
    save_last: bool = True  # Whether to save the last model
    monitor: str = "val_accuracy"  # Metric to monitor
    mode: str = "max"  # Mode for monitoring (max or min)


@dataclass
class Config:
    """Main configuration class that contains all other configs."""
    # Experiment info
    experiment_name: str = "vit_default"  # Experiment name
    seed: int = 42  # Random seed
    
    # Component configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # CUDA and hardware settings
    device: str = "cuda"  # Device: cuda, cpu
    precision: str = "16-mixed"  # Precision: 32, 16-mixed
    
    # Distributed training
    distributed: bool = False  # Whether to use distributed training
    world_size: int = 1  # World size for distributed training
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Check if the key is in any of the sub-configs
                for config_name in ["data", "model", "training", "logging"]:
                    config = getattr(self, config_name)
                    if hasattr(config, key):
                        setattr(config, key, value)
                        break
                else:
                    raise ValueError(f"Unknown configuration key: {key}")


def get_config(config_path: Optional[str] = None, **kwargs) -> Config:
    """Get configuration from file and/or keyword arguments."""
    config = Config()
    
    if config_path is not None:
        # TODO: Implement loading from YAML file
        pass
    
    # Update with keyword arguments
    config.update(**kwargs)
    
    return config 