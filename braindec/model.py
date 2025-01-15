"""Model"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Encoder3D(nn.Module):
    def __init__(
        self, channels=(16, 32, 64), input_channels=1, kernel_size=3, stride=1, padding=1
    ):
        """3D Convolutional Encoder."""
        super().__init__()

        encoder_layers = []
        in_channels = input_channels
        for out_channels in channels:
            encoder_layers.extend(
                [
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample by 2x2x2
                ]
            )
            in_channels = out_channels

        encoder_layers.append(Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class Bottleneck(nn.Module):
    def __init__(self, input_size, hidden_layers=(1024, 512, 256), dropout=0.2):
        super(Bottleneck, self).__init__()

        bottleneck_layers = []
        for hidden_layer in hidden_layers:
            bottleneck_layers.extend(
                [
                    nn.Linear(input_size, hidden_layer),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_size = hidden_layer

        self.bottleneck = nn.Sequential(*bottleneck_layers)

    def forward(self, x):
        return self.bottleneck(x)


class MRI3dCNN(nn.Module):
    def __init__(
        self,
        batch_size=8,
        input_channels=1,
        num_classes=3,
        input_shape=(91, 109, 91),
        channels=(16, 32, 64),
        hidden_layers=(1024, 512, 256),
        kernel_size=3,
        stride=1,
        dropout=0.2,
    ):
        super(MRI3dCNN, self).__init__()

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.channels = channels
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.dropout = dropout

        self.encoder = Encoder3D(
            channels=self.channels,
            input_channels=self.input_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Dynamically calculate the shape after the encoder
        self.encoded_shape = self.compute_output_shape((1, *input_shape))
        flattened_size = (
            self.encoded_shape[0]
            * self.encoded_shape[1]
            * self.encoded_shape[2]
            * self.encoded_shape[3]
        )

        # Add a bottleneck layer
        self.bottleneck = Bottleneck(
            flattened_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )

        """
        self.out = nn.Sequential(
            nn.Linear(self.hidden_layers[-1], num_classes),
            nn.Sigmoid(),
        )
        """
        self.out = nn.Linear(self.hidden_layers[-1], num_classes)

    def forward(self, x):
        # Apply CNN encoder
        x = self.encoder(x)

        # Apply bottleneck. Latent space representation of the input
        x = self.bottleneck(x)

        # Apply output layer to match the number of classes
        # return F.softmax(self.out(x), dim=1)  # Return the probability of each class
        return self.out(x)

    def compute_output_shape(self, input_shape):
        """Dynamically compute the output shape."""
        output_shape = list(input_shape)
        for layer in self.encoder.encoder:
            if isinstance(layer, nn.Conv3d):
                # Compute output size after Conv3d
                D_out = (output_shape[1] - self.kernel_size + 2 * self.padding) // self.stride + 1
                H_out = (output_shape[2] - self.kernel_size + 2 * self.padding) // self.stride + 1
                W_out = (output_shape[3] - self.kernel_size + 2 * self.padding) // self.stride + 1
                output_shape = [layer.out_channels, D_out, H_out, W_out]
            elif isinstance(layer, nn.MaxPool3d):
                # Compute output size after MaxPool3d
                D_out = output_shape[1] // layer.kernel_size
                H_out = output_shape[2] // layer.kernel_size
                W_out = output_shape[3] // layer.kernel_size
                output_shape = [output_shape[0], D_out, H_out, W_out]

        return output_shape


class ResidualHead(nn.Module):
    def __init__(
        self,
        dim,
        dropout,
    ):
        super().__init__()
        self.gelu = nn.GELU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = x + out
        out = self.layer_norm(out)
        return out


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, output_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ImageModel(nn.Module):
    def __init__(self, output_dim, dropout):
        super().__init__()

        self.model = (
            nn.Sequential(
                ResidualHead(output_dim, dropout=dropout),
                ResidualHead(output_dim, dropout=dropout),
                ResidualHead(output_dim, dropout=dropout),
            ),
        )

    def forward(self, x):
        return self.model(x)


class TextModel(nn.Module):
    def __init__(self, embedding_dim, output_dim, dropout):
        super().__init__()

        self.model = (
            nn.Sequential(
                ProjectionHead(embedding_dim, output_dim, dropout=dropout),
                ResidualHead(output_dim, dropout=dropout),
                ResidualHead(output_dim, dropout=dropout),
            ),
        )

    def forward(self, x):
        return self.model(x)


class CLIP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        output_dim=512,
        dropout=0.1,
        logit_scale=np.log(1 / 0.07),
        logit_bias=None,
    ):
        super().__init__()

        self.image_model = ImageModel(output_dim, dropout)
        self.text_model = TextModel(embedding_dim, output_dim, dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)
        self.logit_bias = nn.Parameter(torch.ones([]) * logit_bias) if logit_bias else None

    def encode_image(self, image):  # DiFuMo
        return self.image_model(image)

    def encode_text(self, text):  # Embeddings
        return self.text_model(text)

    def forward(self, image, text):
        image_embeddings = self.encode_image(image)
        print(f"image_embeddings shape: {image_embeddings.shape}")

        text_embeddings = self.encode_text(text)
        print(f"text_embeddings shape: {text_embeddings.shape}")

        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        return image_embeddings, text_embeddings
