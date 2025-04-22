from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all models."""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass
    
    def get_num_parameters(self):
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True 