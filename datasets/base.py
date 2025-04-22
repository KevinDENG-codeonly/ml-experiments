from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
import torch
from torch.utils.data import DataLoader, Dataset

class BaseDataModule(ABC):
    """Base class for all data modules."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    @abstractmethod
    def prepare_data(self):
        """
        Download and prepare data if needed.
        This method is called only once and on the main process only.
        """
        pass
    
    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for train/val/test.
        This method is called on every GPU in distributed training.
        """
        pass
    
    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset has not been setup. Call setup() first.")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create the validation dataloader."""
        if self.val_dataset is None:
            return None
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create the test dataloader."""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        raise NotImplementedError("Subclasses must implement get_num_classes()")
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get the input shape (C, H, W) of the dataset."""
        raise NotImplementedError("Subclasses must implement get_input_shape()") 