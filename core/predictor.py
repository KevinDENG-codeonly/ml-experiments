# core/predictor.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.vit import VisionTransformer
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pathlib import Path

class VisionTransformerPredictor:
    def __init__(self, weights_dir='results/weights', plots_dir='results/plots/predict'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.weights_dir = weights_dir
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        print(f"Using device: {self.device}")

    def get_latest_model(self):
        model_files = [f for f in os.listdir(self.weights_dir) if f.startswith('best_model_')]
        if not model_files:
            raise FileNotFoundError("No best model found in weights directory")
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(self.weights_dir, x)))
        model_path = os.path.join(self.weights_dir, latest_model)
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=10,
            embed_dim=384,
            num_heads=6,
            depth=4,
            mlp_dim=512
        ).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from: {model_path}")
        return model

    def get_random_images(self, num_images=10):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        test_data = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
        indices = torch.randperm(len(test_data))[:num_images]
        images = []
        labels = []
        for idx in indices:
            img, label = test_data[idx]
            images.append(img)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    def predict_and_visualize(self, model, images, labels):
        with torch.no_grad():
            images = images.to(self.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        for idx in range(len(images)):
            img = images[idx].cpu().permute(1, 2, 0)
            img = img.numpy()
            pred = self.classes[predicted[idx].item()]
            true = self.classes[labels[idx].item()]
            axes[idx].imshow(img)
            axes[idx].set_title(f'Pred: {pred}\nTrue: {true}')
            axes[idx].axis('off')
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.plots_dir, f'predictions_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Predictions plot saved to {save_path}")
        correct = (predicted.cpu() == labels).sum().item()
        accuracy = correct / len(labels)
        print(f"\nAccuracy on selected images: {accuracy:.2%}")

    def main(self):
        predictor = VisionTransformerPredictor()
        try:
            model = predictor.get_latest_model()
            images, labels = predictor.get_random_images()
            predictor.predict_and_visualize(model, images, labels)
        except Exception as e:
            print(f"Error: {e}")