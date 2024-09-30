"""Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input):
        return input.view(self.size)


class Encoder3D(nn.Module):
    def __init__(
        self, channels=(16, 32, 64), input_channels=1, kernel_size=3, stride=1, padding=1
    ):
        """3D Convolutional Encoder."""
        super().__init__()
        # input_channels=1 for grayscale images

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
                    nn.MaxPool3d(kernel_size=2, stride=2),  # Downsample by 2
                ]
            )
            in_channels = out_channels

        encoder_layers.append(Flatten())
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.MaxPool3d):
                features.append(x)
        return x, features


class Decoder3D(nn.Module):
    def __init__(
        self,
        channels=(64, 32, 16),
        output_channels=1,
        kernel_size=3,
        stride=1,
        padding=1,
        encoded_shape=(1, 64, 8, 8, 8),
    ):
        """3D Convolutional Decoder."""
        super().__init__()

        padding = kernel_size // 2

        decoder_layers = [UnFlatten(encoded_shape)]
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            decoder_layers.extend(
                [
                    nn.ConvTranspose3d(
                        reversed_channels[i],
                        reversed_channels[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=padding,
                    ),  # deconvolution
                    nn.BatchNorm3d(reversed_channels[i + 1]),
                    nn.ReLU(),
                ]
            )

        decoder_layers.extend(
            [
                nn.ConvTranspose3d(
                    reversed_channels[-1],
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=padding,
                ),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, encoder_features):
        i = 1  # Index to keep track of encoder features
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose3d):
                # Dynamic upsampling to match encoder feature size
                # Upsample does not recover the exact size for odd dimensions, so we interpolate
                encoder_feature = encoder_features[-i]
                x = F.interpolate(
                    x,
                    size=encoder_feature.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
                x = layer(x)

                # Concatenate with encoder features
                x = torch.cat([x, encoder_feature], dim=1)

                i += 1
            else:
                x = layer(x)
        return x


class MRI3dAutoencoder(nn.Module):
    def __init__(
        self,
        batch_size=8,
        input_channels=1,
        input_shape=(91, 109, 91),
        channels=(16, 32, 64),
        kernel_size=3,
        stride=1,
    ):
        super(MRI3dAutoencoder, self).__init__()

        self.batch_size = batch_size
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

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
        bottleneck_size = channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Linear(flattened_size, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, flattened_size),
        )

        self.decoder = Decoder3D(
            channels=self.channels,
            output_channels=self.input_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            encoded_shape=(self.batch_size, *self.encoded_shape),
        )

    def forward(self, x):
        encoded, encoder_features = self.encoder(x)

        # Apply bottleneck. Latent space representation of the input
        bottleneck = self.bottleneck(encoded)

        decoded = self.decoder(bottleneck, encoder_features)
        return decoded, bottleneck

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
