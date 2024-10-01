"""Train and evaluate a model on the BrainDec dataset."""

import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from tqdm import tqdm

from braindec.model import MRI3dCNN
from braindec.preproc import MRIDataset


def create_balanced_loaders(dataset, batch_size, train_size=0.7, val_size=0.15):
    # Get the targets from the dataset
    targets = dataset.encoded_labels

    # First split: train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=1 - train_size - val_size,
        stratify=targets,
        random_state=42,
    )

    # Second split: train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (train_size + val_size),
        stratify=[targets[i] for i in train_val_idx],
        random_state=42,
    )

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for labels, encoded_labels, images in tqdm(train_loader, desc="Training"):
        # labels = labels.to(device)
        encoded_labels = encoded_labels.to(device)
        images = images.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        loss = criterion(output, encoded_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += encoded_labels.size(0)
        correct += predicted.eq(encoded_labels).sum().item()

    accuracy = 100.0 * correct / total
    return train_loss / len(train_loader), accuracy


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, encoded_labels, images in tqdm(val_loader, desc="Validating"):
            # labels = labels.to(device)
            encoded_labels = encoded_labels.to(device)
            images = images.to(device)

            output = model(images)

            loss = criterion(output, encoded_labels)
            val_loss += loss.item()

            _, predicted = output.max(1)
            total += encoded_labels.size(0)
            correct += predicted.eq(encoded_labels).sum().item()

    accuracy = 100.0 * correct / total
    return val_loss / len(val_loader), accuracy


# Test function
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, encoded_labels, images in tqdm(test_loader, desc="Testing"):
            # labels = labels.to(device)
            images = images.to(device)

            output = model(images)
            _, predicted = output.max(1)
            total += encoded_labels.size(0)
            correct += predicted.eq(encoded_labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


# Main training loop
def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    data_dir = op.join(project_dir, "data")

    # Hyperparameters
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("cpu")  # Use CPU as MPS doesn't support max_pool3d_with_indices
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA for Nvidia GPUs
    else:
        device = torch.device("cpu")  # Default to CPU

    print(f"Using device: {device}")

    # Create dataset
    dataset_fn = op.join(data_dir, "dataset.pkl")
    if op.exists(dataset_fn):
        with open(dataset_fn, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = MRIDataset(data_dir)
        with open(dataset_fn, "wb") as f:
            pickle.dump(dataset, f)

    print(f"Number of samples: {len(dataset)}")

    # Create data loaders for training, validation, and testing
    train_loader, val_loader, test_loader = create_balanced_loaders(dataset, batch_size=32)

    # Initialize model, loss function, and optimizer
    model = MRI3dCNN(
        num_classes=dataset.num_classes,
        input_shape=dataset.image_shape,
        batch_size=batch_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

    # Test the model
    test_accuracy = test(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "mri_3d_cnn.pth")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


if __name__ == "__main__":
    main()
