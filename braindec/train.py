"""Train and evaluate a model on the BrainDec dataset."""

import torch
from tqdm import tqdm

from braindec.plot import plot_training


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

    return model, train_loss / len(train_loader)


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

    return model, val_loss / len(val_loader)


def predict(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        all_image_embeddings = []
        all_text_embeddings = []
        for image_embeddings, text_embeddings in data_loader:
            image_embeddings = image_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)

            image_embed, text_embed = model(image_embeddings, text_embeddings)  # Forward pass

            all_image_embeddings.append(image_embed.cpu())
            all_text_embeddings.append(text_embed.cpu())

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
    device,
    plot_verbose=False,
):
    # Training loop
    best_val_loss = float("inf")
    patience = 10
    counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model, train_loss = train(model, train_loader, criterion, optimizer, device)
        model, val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
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
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    torch.save(model.state_dict(), last_model_fn)

    # Check training and validation loss
    if plot_verbose:
        plot_training(train_loss, val_loss)

    return model, train_losses, val_losses


def train_decoder(model, clip_model, train_loader, criterion, optimizer, device):
    model.train()
    clip_model.eval()

    train_loss = 0
    for image_embeddings, text_embeddings in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()  # Reset all gradients

        image_embeddings = image_embeddings.to(device)
        text_embeddings = text_embeddings.to(device)

        image_embed = clip_model.encode_image(image_embeddings)
        image_embed = model(image_embed)  # Forward pass. Decoder model

        # Calculate the loss
        loss = criterion(image_embed, text_embeddings)
        train_loss += loss.item()

        loss.backward()  # Backpropagate the loss

        optimizer.step()  # Update the weights

    return model, train_loss / len(train_loader)


def validate_decoder(model, clip_model, val_loader, criterion, device):
    model.eval()
    clip_model.eval()

    val_loss = 0
    with torch.no_grad():
        for image_embeddings, text_embeddings in tqdm(val_loader, desc="Validating"):
            image_embeddings = image_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)

            image_embed = clip_model.encode_image(image_embeddings)
            image_embed = model(image_embed)

            # Calculate the loss
            loss = criterion(image_embed, text_embeddings)
            val_loss += loss.item()

    return model, val_loss / len(val_loader)


def train_decoder_model(
    model,
    clip_model,
    criterion,
    optimizer,
    num_epochs,
    train_loader,
    val_loader,
    best_model_fn,
    last_model_fn,
    device,
    plot_verbose=False,
):
    # Training loop
    best_val_loss = float("inf")
    patience = 10
    counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model, train_loss = train_decoder(
            model,
            clip_model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        model, val_loss = validate_decoder(model, clip_model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
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
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    torch.save(model.state_dict(), last_model_fn)

    # Check training and validation loss
    if plot_verbose:
        plot_training(train_loss, val_loss)

    return model, train_losses, val_losses
