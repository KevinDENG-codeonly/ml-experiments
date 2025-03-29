import torch.optim as optim
from torchvision import datasets, transforms
from model import VisionTransformer
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Use a smaller subset of CIFAR10
    train_data = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    # Take only 10% of the data
    train_size = int(0.1 * len(train_data))
    train_data, _ = torch.utils.data.random_split(train_data, [train_size, len(train_data) - train_size])

    # Remove num_workers since it's causing issues with MPS
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a smaller model
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=10,
        embed_dim=384,  # Reduced from 768
        num_heads=6,    # Reduced from 8
        depth=4,        # Reduced from 6
        mlp_dim=512     # Reduced from 1024
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        # Add progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/5')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = running_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/5], Average Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()
