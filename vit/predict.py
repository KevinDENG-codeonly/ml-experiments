import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import VisionTransformer
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class VisionTransformerPredictor:
    """
    A class for loading trained Vision Transformer models and making predictions on CIFAR10 images.
    This class handles model loading, image selection, prediction, and visualization of results.
    """
    
    def __init__(self, weights_dir='results/weights', plots_dir='results/plots/predict'):
        """
        Initialize the predictor with directories for model weights and prediction plots.
        
        Args:
            weights_dir (str): Directory containing saved model weights
            plots_dir (str): Directory for saving prediction visualization plots
        """
        # Set up device (MPS for Apple Silicon, CPU for others)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.weights_dir = weights_dir
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Define CIFAR10 class labels for human-readable predictions
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        print(f"Using device: {self.device}")

    def get_latest_model(self):
        """
        Load the most recently saved best model from the weights directory.
        
        Returns:
            model: Loaded Vision Transformer model in evaluation mode
            
        Raises:
            FileNotFoundError: If no model weights are found in the directory
        """
        # Find all best model files in the weights directory
        model_files = [f for f in os.listdir(self.weights_dir) if f.startswith('best_model_')]
        if not model_files:
            raise FileNotFoundError("No best model found in weights directory")
        
        # Get the most recently created model file
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.weights_dir, x)))
        model_path = os.path.join(self.weights_dir, latest_model)
        
        # Initialize model with same architecture as training
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=10,
            embed_dim=384,
            num_heads=6,
            depth=4,
            mlp_dim=512
        ).to(self.device)
        
        # Load the saved weights and set model to evaluation mode
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded model from: {model_path}")
        return model

    def get_random_images(self, num_images=10):
        """
        Randomly select images from the CIFAR10 test set.
        
        Args:
            num_images (int): Number of images to select
            
        Returns:
            tuple: (stacked_images, labels) where stacked_images is a tensor of shape (num_images, 3, 224, 224)
                  and labels is a tensor of shape (num_images,)
        """
        # Define the same image transformations as used in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Load the test dataset
        test_data = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
        
        # Randomly select indices and get corresponding images and labels
        indices = torch.randperm(len(test_data))[:num_images]
        images = []
        labels = []
        
        for idx in indices:
            img, label = test_data[idx]
            images.append(img)
            labels.append(label)
        
        # Stack images into a single tensor for batch processing
        return torch.stack(images), torch.tensor(labels)

    def predict_and_visualize(self, model, images, labels):
        """
        Make predictions on the images and visualize the results.
        
        Args:
            model: The loaded Vision Transformer model
            images: Tensor of images to predict
            labels: Tensor of true labels
            
        The function creates a visualization with:
        - 2x5 grid of images
        - Each image shows the predicted and true class
        - Saves the plot to the plots directory
        - Prints the accuracy on the selected images
        """
        # Make predictions without computing gradients
        with torch.no_grad():
            images = images.to(self.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
        # Create a 2x5 grid for visualization
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        # Plot each image with its prediction and true label
        for idx in range(len(images)):
            # Convert image tensor to numpy array and adjust channel order for display
            img = images[idx].cpu().permute(1, 2, 0)
            img = img.numpy()
            
            # Get human-readable class names for prediction and true label
            pred = self.classes[predicted[idx].item()]
            true = self.classes[labels[idx].item()]
            
            # Display image with prediction and true label
            axes[idx].imshow(img)
            axes[idx].set_title(f'Pred: {pred}\nTrue: {true}')
            axes[idx].axis('off')
        
        # Adjust layout and save the plot
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.plots_dir, f'predictions_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Predictions plot saved to {save_path}")
        
        # Calculate and print accuracy
        correct = (predicted.cpu() == labels).sum().item()
        accuracy = correct / len(labels)
        print(f"\nAccuracy on selected images: {accuracy:.2%}")

def main():
    """
    Main function to run the prediction pipeline:
    1. Initialize the predictor
    2. Load the latest trained model
    3. Get random test images
    4. Make predictions and visualize results
    """
    # Create predictor instance
    predictor = VisionTransformerPredictor()
    
    try:
        # Load the latest model
        model = predictor.get_latest_model()
        
        # Get random images
        images, labels = predictor.get_random_images()
        
        # Make predictions and visualize
        predictor.predict_and_visualize(model, images, labels)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
