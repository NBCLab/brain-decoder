"""Train a model on the BrainDec dataset."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from braindec.model import MRI3DAutoencoder
from braindec.preproc import MRIDataset


# Function to calculate PSNR
def psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# Test function
def test(model, test_loader, device):
    model.eval()
    psnr_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            output = model(batch)
            psnr_scores.append(psnr(batch, output).item())
    return np.mean(psnr_scores)


# Main training loop
def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"

    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and split into train, validation, and test sets
    dataset = MRIDataset(project_dir)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = MRI3DAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

    # Test the model
    avg_psnr = test(model, test_loader, device)
    print(f"Average PSNR on test set: {avg_psnr:.2f} dB")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "mri_3d_autoencoder.pth")


if __name__ == "__main__":
    main()
