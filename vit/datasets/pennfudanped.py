# datasets/pennfudanped.py
import os
from typing import Any, Optional, Tuple
import torch
import torchvision
from torchvision.datasets import VisionDataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from PIL import Image
import numpy as np

from .base import BaseDataModule
from .registry import register_dataset

class PennFudanDataset(VisionDataset):
    """PennFudan Pedestrian dataset for object detection."""
    
    def __init__(self, root, transforms=None):
        super(PennFudanDataset, self).__init__(root, transforms)
        self.root = root
        self.transforms = transforms
        
        # Load all image and mask files, sorting them to ensure correct matching
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        mask = np.array(mask)
        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        # Remove background (value 0)
        obj_ids = obj_ids[1:]
        
        # Split the mask into binary masks for each object
        masks = mask == obj_ids[:, None, None]
        
        # Get bounding boxes
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)  # All objects are pedestrians
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)


@register_dataset("pennfudan")
class PennFudanPedDataModule(BaseDataModule):
    """PennFudan Pedestrian data module for object detection."""
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.data_dir = config.get("data_dir", "data/PennFudanPed")
        self.val_split = config.get("val_split", 0.2)
        self.img_size = config.get("img_size", 224)
        self.download_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
        
    def prepare_data(self):
        """Download PennFudanPed data if needed."""
        # Check if the dataset exists
        if not os.path.exists(self.data_dir) or not os.path.exists(os.path.join(self.data_dir, "PNGImages")):
            import urllib.request
            import zipfile
            
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Download dataset
            print(f"Downloading PennFudanPed dataset from {self.download_url}")
            zip_path = os.path.join(self.data_dir, "PennFudanPed.zip")
            urllib.request.urlretrieve(self.download_url, zip_path)
            
            # Extract dataset
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.data_dir))
                
            # Rename the extracted directory
            extracted_dir = os.path.join(os.path.dirname(self.data_dir), "PennFudanPed")
            if os.path.exists(extracted_dir) and extracted_dir != self.data_dir:
                os.rename(extracted_dir, self.data_dir)
                
            # Clean up
            if os.path.exists(zip_path):
                os.remove(zip_path)
        
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test datasets."""
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        
        # Create dataset
        dataset = PennFudanDataset(self.data_dir, transforms=transform)
        
        # Split into train and validation
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # No test set for this dataset, use validation as test
        self.test_dataset = self.val_dataset
        
    def get_num_classes(self) -> int:
        """Get the number of classes (background + pedestrian)."""
        return 2  # Background and pedestrian
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get the input shape (C, H, W)."""
        return (3, self.img_size, self.img_size)