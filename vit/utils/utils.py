# util/utils.py
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch
import random
import numpy as np
from typing import Optional, Any, Union

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, plots_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(plots_dir, f'training_results_{timestamp}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training plots saved to {save_path}")

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get PyTorch device.
    
    Args:
        device_str: Device string ('cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        PyTorch device
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_experiment_dir(base_dir: str, experiment_name: Optional[str] = None) -> str:
    """
    Create a directory for experiment with timestamp.
    
    Args:
        base_dir: Base directory
        experiment_name: Experiment name
        
    Returns:
        Path to experiment directory
    """
    if experiment_name is None:
        experiment_name = "experiment"
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir