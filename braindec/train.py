"""Train and evaluate a model on the BrainDec dataset."""

import os
import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from braindec.model import MRI3dCNN
from braindec.preproc import MRIDataset

# from torchsummary import summary


def _plot_training_history(
    train_losses,
    train_accs,
    train_roc_aucs,
    val_losses,
    val_accs,
    val_roc_aucs,
    out_dir,
):
    # Plot training and validation losses
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot training and validation accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 3, 3)
    plt.plot(train_roc_aucs, label="Train ROC AUC")
    plt.plot(val_roc_aucs, label="Validation ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.legend()
    plt.title("Training and Validation ROC AUC")

    plt.tight_layout()
    plt.savefig(op.join(out_dir, "training_history.png"))
    plt.show()


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


def create_random_loaders(dataset, batch_size, train_size=0.7, val_size=0.15):
    # Get the targets from the dataset
    targets = dataset.labels

    # First split: train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=1 - train_size - val_size,
        random_state=42,
    )

    # Second split: train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (train_size + val_size),
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
    targets, predictions = [], []
    for labels, soft_labels, true_labels, images in tqdm(train_loader, desc="Training"):
        # labels = labels.to(device)
        soft_labels = soft_labels.to(device)
        true_labels = true_labels.to(device)
        images = images.to(device)

        optimizer.zero_grad()  # We need to reset all gradients

        output = model(images)  # Forward pass
        loss = criterion(F.log_softmax(output, dim=1), soft_labels)  # Calculate the loss
        loss.backward()  # Backpropagate the loss

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()  # Update the weights

        train_loss += loss.item()
        _, predicted_labels = output.max(1)  # predictions

        total += true_labels.size(0)
        correct += predicted_labels.eq(true_labels).sum().item()
        targets.extend(true_labels.cpu().numpy())
        predictions.extend(F.softmax(output, dim=1).detach().cpu().numpy())

    targets = np.array(targets)
    predictions = np.array(predictions)
    roc_auc = roc_auc_score(targets, predictions, average="macro", multi_class="ovr")
    accuracy = 100.0 * correct / total
    return train_loss / len(train_loader), accuracy, roc_auc


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    targets, predictions = [], []
    with torch.no_grad():
        for labels, soft_labels, true_labels, images in tqdm(val_loader, desc="Validating"):
            soft_labels = soft_labels.to(device)
            true_labels = true_labels.to(device)
            images = images.to(device)

            output = model(images)

            loss = criterion(F.log_softmax(output, dim=1), soft_labels)

            val_loss += loss.item()
            _, predicted_labels = output.max(1)  # predictions

            total += true_labels.size(0)
            correct += predicted_labels.eq(true_labels).sum().item()
            targets.extend(true_labels.cpu().numpy())
            predictions.extend(F.softmax(output, dim=1).detach().cpu().numpy())

    targets = np.array(targets)
    predictions = np.array(predictions)
    roc_auc = roc_auc_score(targets, predictions, average="macro", multi_class="ovr")
    accuracy = 100.0 * correct / total
    return val_loss / len(val_loader), accuracy, roc_auc


# Test function
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    targets, predictions = [], []
    with torch.no_grad():
        for labels, soft_labels, true_labels, images in tqdm(test_loader, desc="Testing"):
            soft_labels = soft_labels.to(device)
            true_labels = true_labels.to(device)
            images = images.to(device)

            output = model(images)
            _, predicted_labels = output.max(1)  # predictions

            total += true_labels.size(0)
            correct += predicted_labels.eq(true_labels).sum().item()
            targets.extend(true_labels.cpu().numpy())
            predictions.extend(F.softmax(output, dim=1).detach().cpu().numpy())

    targets = np.array(targets)
    predictions = np.array(predictions)
    roc_auc = roc_auc_score(targets, predictions, average="macro", multi_class="ovr")
    accuracy = 100.0 * correct / total
    return accuracy, roc_auc


# Main training loop
def main():
    project_dir = "/Users/julioaperaza/Documents/GitHub/brain-decoder"
    data_dir = op.join(project_dir, "data")
    out_dir = op.join(project_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    best_model_fn = op.join(out_dir, "best_model.pth")

    # Hyperparameters
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4

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
    train_loader, val_loader, test_loader = create_balanced_loaders(dataset, batch_size=batch_size)
    # train_loader, val_loader, test_loader = create_random_loaders(dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = MRI3dCNN(
        num_classes=dataset.num_classes,
        input_shape=dataset.image_shape,
        batch_size=batch_size,
        dropout=0.5,
    ).to(device)

    # Get the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")

    # criterion = nn.CrossEntropyLoss() # for hard labels
    # criterion = nn.BCELoss() # when using sigmoid in the output layer
    criterion = nn.KLDivLoss(reduction="batchmean")  # for soft labels

    # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # summary(model, (1, 28, 28))

    # Training loop
    best_val_loss = float("inf")
    patience = 10
    counter = 0
    train_losses = []
    train_accs = []
    train_roc_aucs = []
    val_losses = []
    val_accs = []
    val_roc_aucs = []
    for epoch in range(num_epochs):
        train_loss, train_acc, train_roc_auc = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, val_roc_auc = validate(model, val_loader, criterion, device)
        # scheduler.step(val_loss)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_roc_aucs.append(train_roc_auc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_roc_aucs.append(val_roc_auc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
            f"Train ROC AUC: {train_roc_auc:.4f}, Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_acc:.2f}%, Val ROC AUC: {val_roc_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_fn)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Test the model
    test_acc, test_roc_auc = test(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%, Test ROC AUC: {test_roc_auc:.4f}")

    # Plot training history
    _plot_training_history(
        train_losses,
        train_accs,
        train_roc_aucs,
        val_losses,
        val_accs,
        val_roc_aucs,
        out_dir,
    )

    # Save results
    results_df = pd.DataFrame()
    results_df["train_loss"] = train_losses
    results_df["train_acc"] = train_accs
    results_df["train_roc_auc"] = train_roc_aucs
    results_df["val_loss"] = val_losses
    results_df["val_acc"] = val_accs
    results_df["val_roc_auc"] = val_roc_aucs
    results_df.to_csv(op.join(out_dir, "results.csv"), index=False)


if __name__ == "__main__":
    main()
