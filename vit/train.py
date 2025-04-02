import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import VisionTransformer
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import copy

class VisionTransformerTrainer:
    """
    A class for training Vision Transformer models on CIFAR10 dataset.
    Includes functionality for training, validation, model saving, and visualization.
    """
    
    def __init__(self, save_weights=True, weights_dir='results/weights', plots_dir='results/plots'):
        """
        Initialize the trainer with directories for saving weights and plots.
        
        Args:
            save_weights (bool): Whether to save model weights
            weights_dir (str): Directory for saving model weights
            plots_dir (str): Directory for saving training plots
        """
        # Set up device (MPS for Apple Silicon, CPU for others)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.save_weights = save_weights
        self.weights_dir = weights_dir
        self.plots_dir = os.path.join(plots_dir, 'train')  # Save plots under plots/train
        
        # Create necessary directories
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize training state
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping parameters
        self.patience = 3  # Number of epochs to wait before early stopping
        self.min_delta = 0.005  # Minimum change in loss to be considered as improvement
        self.early_stopping_counter = 0
        self.early_stopping = False
        
        # Initialize best model state (in memory)
        self.best_model_state = None
        self.best_optimizer_state = None
        
        print(f"Using device: {self.device}")

    def load_data(self, batch_size=64, train_size=0.2):
        """
        Load and preprocess the CIFAR10 dataset.
        
        Args:
            batch_size (int): Batch size for training
            train_size (float): Fraction of training data to use (0.0 to 1.0)
        """
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 statistics
        ])
        
        # Load training dataset
        train_data = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
        
        # Reduce training data size if specified
        if train_size < 1.0:
            train_size = int(len(train_data) * train_size)
            train_data, _ = torch.utils.data.random_split(train_data, [train_size, len(train_data) - train_size])
        
        # Load validation dataset
        val_data = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
        
        # Create data loaders
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        print(f"Training set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")

    def create_model(self):
        """Create and initialize the Vision Transformer model"""
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=10,
            embed_dim=384,
            num_heads=6,
            depth=4,
            mlp_dim=512
        ).to(self.device)
        
        # Use CrossEntropyLoss for multi-class classification
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
        
        return total_loss / len(self.train_loader), 100.*correct/total

    def validate(self):
        """Validate the model on the validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return total_loss / len(self.val_loader), 100.*correct/total

    def update_best_model(self, epoch, loss):
        """
        Update the best model state if current model performs better.
        Uses memory-efficient approach by only storing the best model.
        
        Args:
            epoch (int): Current epoch number
            loss (float): Current validation loss
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.early_stopping_counter = 0
            
            # Update best model state using deep copy
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            
            print(f"New best model found at epoch {epoch + 1} (Loss: {loss:.4f})")
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.patience:
                self.early_stopping = True
                print(f"Early stopping triggered after {epoch + 1} epochs")

    def save_final_model(self):
        """Save the best model to disk at the end of training"""
        if not self.save_weights or self.best_model_state is None:
            return
            
        # Save the best model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.weights_dir, f'best_model_{timestamp}.pth')
        
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.best_optimizer_state,
            'loss': self.best_loss,
        }, save_path)
        
        print(f"\nBest model saved to {save_path}")
        print(f"Best model was at epoch {self.best_epoch + 1} with loss {self.best_loss:.4f}")

    def plot_results(self):
        """Plot training and validation metrics"""
        # Plot losses
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracies')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.plots_dir, f'training_results_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training plots saved to {save_path}")

    def train(self, num_epochs=10):
        """
        Main training loop.
        
        Args:
            num_epochs (int): Number of epochs to train
        """
        self.load_data()
        self.create_model()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
            
            # Update best model if necessary
            self.update_best_model(epoch, val_loss)
            
            if self.early_stopping:
                print("Early stopping triggered. Training stopped.")
                break
        
        # Save the best model and plot results
        self.save_final_model()
        self.plot_results()

def main():
    """Main function to run the training pipeline"""
    # Create trainer instance
    trainer = VisionTransformerTrainer(save_weights=True)
    
    # Start training
    trainer.train(num_epochs=20)

if __name__ == '__main__':
    main()
