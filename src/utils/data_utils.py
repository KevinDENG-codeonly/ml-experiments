"""
Utilities for data processing, transformations, and augmentations.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import torch.nn.functional as nn_F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random


class RandomMixup:
    """Randomly apply Mixup to images and labels.
    
    Args:
        alpha (float): Mixup alpha parameter.
        num_classes (int): Number of classes in the dataset.
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 10):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(self, batch: Dict) -> Dict:
        """Apply mixup to a batch of images.
        
        Args:
            batch: Dict containing 'image' and 'label' keys.
            
        Returns:
            Batch with mixup applied.
        """
        if self.alpha <= 0:
            return batch
        
        images, labels = batch["image"], batch["label"]
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Apply mixup
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Use the specified number of classes
        mixed_labels = (lam * nn_F.one_hot(labels.long(), num_classes=self.num_classes) + 
                        (1 - lam) * nn_F.one_hot(labels[indices].long(), num_classes=self.num_classes))
        
        return {"image": mixed_images, "label": mixed_labels}


class RandomCutMix:
    """Randomly apply CutMix to images and labels.
    
    Args:
        alpha (float): CutMix alpha parameter.
        num_classes (int): Number of classes in the dataset.
    """
    
    def __init__(self, alpha: float = 1.0, num_classes: int = 10):
        self.alpha = alpha
        self.num_classes = num_classes
    
    def __call__(self, batch: Dict) -> Dict:
        """Apply cutmix to a batch of images.
        
        Args:
            batch: Dict containing 'image' and 'label' keys.
            
        Returns:
            Batch with cutmix applied.
        """
        if self.alpha <= 0:
            return batch
        
        images, labels = batch["image"], batch["label"]
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get cutmix box parameters
        h, w = images.size(2), images.size(3)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        images_clone = images.clone()
        images_clone[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to account for actual cut size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        # Use the specified number of classes
        mixed_labels = (lam * nn_F.one_hot(labels.long(), num_classes=self.num_classes) + 
                       (1 - lam) * nn_F.one_hot(labels[indices].long(), num_classes=self.num_classes))
        
        return {"image": images_clone, "label": mixed_labels}


def get_training_transforms(
    image_size: int = 224,
    auto_augment: bool = False,
    random_erase: float = 0.0,
) -> T.Compose:
    """Get transformations for training data.
    
    Args:
        image_size: Size to resize images to.
        auto_augment: Whether to use auto augmentation.
        random_erase: Probability for random erasing.
        
    Returns:
        Composed transformations.
    """
    transforms = []
    
    # Standard augmentations
    transforms.extend([
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
    ])
    
    # Auto augmentation (RandAugment, TrivialAugment, etc.)
    if auto_augment:
        transforms.append(T.RandAugment())
    
    # Normalization and tensor conversion
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Random erasing (applied after normalization)
    if random_erase > 0:
        transforms.append(T.RandomErasing(p=random_erase))
    
    return T.Compose(transforms)


def get_validation_transforms(image_size: int = 224) -> T.Compose:
    """Get transformations for validation data.
    
    Args:
        image_size: Size to resize images to.
        
    Returns:
        Composed transformations.
    """
    return T.Compose([
        T.Resize(int(image_size * 1.14)),  # Resize to a bit larger for center crop
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_test_transforms(image_size: int = 224) -> T.Compose:
    """Get transformations for test data (same as validation)."""
    return get_validation_transforms(image_size)


def worker_init_fn(worker_id: int):
    """Initialize worker for data loading.
    
    Sets different random seed for each worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) 