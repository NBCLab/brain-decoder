"""Train a model on the BrainDec dataset."""

import torch
import torch.nn as nn
import torch.optim as optim


class CNN3DAutoencoder(nn.Module):
    def __init__(self, input_channels=1):
        super(CNN3DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, 3, stride=2, padding=1),  # [16, 16, 16, 16]
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),  # [32, 8, 8, 8]
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),  # [64, 4, 4, 4]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # [32, 8, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, 16, 3, stride=2, padding=1, output_padding=1
            ),  # [16, 16, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose3d(
                16, input_channels, 3, stride=2, padding=1, output_padding=1
            ),  # [1, 32, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Assuming we have a dataset of 32x32x32 3D images
model = CNN3DAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for img in dataloader:
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()

# Example usage
input_3d = torch.randn(1, 1, 32, 32, 32)  # Batch size 1, 1 channel, 32x32x32 volume
output_3d = model(input_3d)
print(output_3d.shape)  # Should be torch.Size([1, 1, 32, 32, 32])
