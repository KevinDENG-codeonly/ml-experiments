# datasets/cifar10.py
import os
from typing import Any, Optional, Tuple
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

from .base import BaseDataModule
from .registry import register_dataset

@register_dataset("cifar10")
class CIFAR10DataModule(BaseDataModule):
    """CIFAR10 data module."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.data_dir = config.get("data_dir", "data/cifar10")
        self.val_split = config.get("val_split", 0.1)
        self.img_size = config.get("img_size", 224)
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2470, 0.2435, 0.2616]
        
    def prepare_data(self):
        """Download CIFAR10 data if needed."""
        # This downloads the data if it doesn't exist
        torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True
        )
        torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True
        )
        
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(int(self.img_size * 1.14)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Load datasets
        cifar_full = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            transform=train_transform
        )
        
        # Split into train and validation
        val_size = int(len(cifar_full) * self.val_split)
        train_size = len(cifar_full) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            cifar_full, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Test dataset
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            transform=test_transform
        )
        
    def get_num_classes(self) -> int:
        """Get the number of classes in CIFAR10."""
        return 10
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get the input shape (C, H, W) of CIFAR10."""
        return (3, self.img_size, self.img_size)