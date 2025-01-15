"""Train and evaluate a model on the BrainDec dataset."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# from torchsummary import summary


# Training function
def train(model, train_loader, criterion, optimizer, device, scheduler=None, clip_grad_norm=None):
    model.train()

    train_loss = 0
    for image_embeddings, text_embeddings in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()  # Reset all gradients

        image_embeddings = image_embeddings.to(device)
        text_embeddings = text_embeddings.to(device)

        image_embed, text_embed = model(image_embeddings, text_embeddings)  # Forward pass

        # Calculate the loss
        loss = criterion(image_embed, text_embed, model.logit_scale, model.logit_bias)
        train_loss += loss.item()

        loss.backward()  # Backpropagate the loss

        if clip_grad_norm is not None:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()  # Update the weights

        # Scheduler update
        if scheduler is not None:
            scheduler.step()

    return train_loss / len(train_loader)


# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for image_embeddings, text_embeddings in tqdm(val_loader, desc="Validating"):
            image_embeddings = image_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)

            image_embed, text_embed = model(image_embeddings, text_embeddings)  # Forward pass

            # Calculate the loss
            loss = criterion(image_embed, text_embed, model.logit_scale, model.logit_bias)

            val_loss += loss.item()

    return val_loss / len(val_loader)


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
