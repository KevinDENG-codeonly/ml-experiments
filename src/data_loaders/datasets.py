"""
Dataset loaders for various image classification datasets.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

from src.utils.data_utils import (
    get_training_transforms,
    get_validation_transforms,
    get_test_transforms,
    worker_init_fn,
    RandomMixup,
    RandomCutMix,
)

logger = logging.getLogger(__name__)


def get_dataset(
    dataset_name: str,
    data_dir: str,
    image_size: int = 224,
    train: bool = True,
    download: bool = True,
    auto_augment: bool = False,
    random_erase: float = 0.0,
) -> Dataset:
    """Get dataset by name.
    
    Args:
        dataset_name: Name of the dataset.
        data_dir: Directory to store the dataset.
        image_size: Size to resize images to.
        train: Whether to load training or validation/test set.
        download: Whether to download the dataset if not found.
        auto_augment: Whether to use auto augmentation.
        random_erase: Probability for random erasing.
        
    Returns:
        Dataset object.
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Get appropriate transforms
    if train:
        transform = get_training_transforms(
            image_size=image_size,
            auto_augment=auto_augment,
            random_erase=random_erase,
        )
    else:
        transform = get_validation_transforms(image_size=image_size)
    
    # Handle different datasets
    if dataset_name.lower() == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=download,
            transform=transform,
        )
    elif dataset_name.lower() == "cifar100":
        dataset = datasets.CIFAR100(
            root=data_dir,
            train=train,
            download=download,
            transform=transform,
        )
    elif dataset_name.lower() == "imagenet":
        split = "train" if train else "val"
        dataset = datasets.ImageNet(
            root=data_dir,
            split=split,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def train_collate_fn(batch, mixup_alpha=0.0, cutmix_alpha=0.0, num_classes=10):
    """Collate function for training data with mixup/cutmix.
    
    Args:
        batch: Batch of data.
        mixup_alpha: Mixup alpha parameter.
        cutmix_alpha: CutMix alpha parameter.
        num_classes: Number of classes in the dataset.
    """
    batch_dict = {
        "image": torch.stack([item[0] for item in batch]),
        "label": torch.tensor([item[1] for item in batch]),
    }
    
    # Apply mixup and cutmix if enabled
    if mixup_alpha > 0:
        batch_dict = RandomMixup(alpha=mixup_alpha, num_classes=num_classes)(batch_dict)
    
    if cutmix_alpha > 0:
        batch_dict = RandomCutMix(alpha=cutmix_alpha, num_classes=num_classes)(batch_dict)
    
    return batch_dict["image"], batch_dict["label"]


def val_collate_fn(batch):
    """Collate function for validation/test data."""
    return torch.stack([item[0] for item in batch]), torch.tensor([item[1] for item in batch])


def create_data_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_val_split: float = 0.9,
    auto_augment: bool = False,
    random_erase: float = 0.0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> Dict[str, DataLoader]:
    """Create data loaders for training and validation.
    
    Args:
        dataset_name: Name of the dataset.
        data_dir: Directory to store the dataset.
        batch_size: Batch size.
        image_size: Size to resize images to.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory for data loading.
        train_val_split: Ratio for splitting training data into train and validation.
        auto_augment: Whether to use auto augmentation.
        random_erase: Probability for random erasing.
        mixup_alpha: Mixup alpha parameter.
        cutmix_alpha: CutMix alpha parameter.
        
    Returns:
        Dictionary containing data loaders for training and validation.
    """
    # Determine number of classes based on dataset
    if dataset_name.lower() == "cifar10":
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        num_classes = 100
    elif dataset_name.lower() == "imagenet":
        num_classes = 1000
    else:
        num_classes = 10  # Default
    
    # Get training dataset
    train_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size,
        train=True,
        download=True,
        auto_augment=auto_augment,
        random_erase=random_erase,
    )
    
    # Get test dataset
    test_dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size,
        train=False,
        download=True,
    )
    
    # Split training dataset into train and validation
    if train_val_split < 1.0:
        train_size = int(len(train_dataset) * train_val_split)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        val_dataset = test_dataset
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create customized collate function for training with current alpha values
    def train_collate_with_params(batch):
        return train_collate_fn(batch, mixup_alpha, cutmix_alpha, num_classes)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        collate_fn=train_collate_with_params if (mixup_alpha > 0 or cutmix_alpha > 0) else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate_fn,
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    } 