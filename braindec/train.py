"""Train and evaluate a model on the BrainDec dataset."""

import torch
from tqdm import tqdm

from braindec.plot import plot_training


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler=None,
    clip_grad_norm=None,
    verbose=1,
):
    model.train()
    tqdm_off = verbose <= 2

    train_loss = 0
    for image_emb, text_emb in tqdm(train_loader, desc="Training", disable=tqdm_off):
        optimizer.zero_grad()  # Reset all gradients

        image_emb = image_emb.to(model.device)
        text_emb = text_emb.to(model.device)

        image_emb, text_emb = model(image_emb, text_emb)  # Forward pass

        # Calculate the loss
        # logit_scale = self.logit_scale.exp()
        loss = criterion(image_emb, text_emb, model.logit_scale)
        train_loss += loss.item()

        loss.backward()  # Backpropagate the loss

        if clip_grad_norm is not None:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()  # Update the weights

        # Scheduler update
        if scheduler is not None:
            scheduler.step()

    return model, train_loss / len(train_loader)


def validate(model, val_loader, criterion, verbose=1):
    model.eval()
    tqdm_off = verbose <= 2

    val_loss = 0
    with torch.no_grad():
        for image_emb, text_emb in tqdm(val_loader, desc="Validating", disable=tqdm_off):
            image_emb = image_emb.to(model.device)
            text_emb = text_emb.to(model.device)

            image_emb, text_emb = model(image_emb, text_emb)  # Forward pass

            # Calculate the loss
            # logit_scale = self.logit_scale.exp()
            loss = criterion(image_emb, text_emb, model.logit_scale)

            val_loss += loss.item()

    return model, val_loss / len(val_loader)


def predict(model, data_loader):
    model.eval()

    with torch.no_grad():
        all_image_embeddings = []
        all_text_embeddings = []
        for image_emb, text_emb in data_loader:
            image_emb = image_emb.to(model.device)
            text_emb = text_emb.to(model.device)

            image_emb, text_emb = model(image_emb, text_emb)  # Forward pass

            all_image_embeddings.append(image_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())

    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    return all_image_embeddings, all_text_embeddings


def train_clip_model(
    model,
    criterion,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
    best_model_fn,
    last_model_fn,
    verbose=1,
    plot_verbose=False,
):
    # Training loop
    best_val_loss = float("inf")
    patience = 10
    counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model, train_loss = train(
            model,
            train_loader,
            criterion,
            optimizer,
            verbose=verbose,
        )
        model, val_loss = validate(model, val_loader, criterion, verbose=verbose)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if verbose > 1:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_fn)
        else:
            counter += 1
            if counter >= patience:
                if verbose > 1:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    torch.save(model.state_dict(), last_model_fn)

    # Check training and validation loss
    if plot_verbose:
        plot_training(train_losses, val_losses)

    return train_losses, val_losses
