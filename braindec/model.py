"""Model"""

import torch
import torch.nn as nn


class Base3DModule(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super(Base3DModule, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

    def create_conv_block(self, in_channels, out_channels, transpose=False):
        conv = nn.ConvTranspose3d if transpose else nn.Conv3d
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.padding if transpose else 0,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )


class Encoder3D(Base3DModule):
    def __init__(self, input_channels=1, hidden_layers=(16, 32, 64), kernel_size=3, stride=1):
        super(Encoder3D, self).__init__(hidden_layers, kernel_size, stride)

        encoder_layers = []
        in_channels = input_channels
        for out_channels in self.channels:
            encoder_layers.extend(
                [
                    self.create_conv_block(in_channels, out_channels),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.MaxPool3d):
                features.append(x)
        return x, features


class Decoder3D(Base3DModule):
    def __init__(self, hidden_layers=(64, 32, 16), output_channels=1, kernel_size=3, stride=1):
        super(Decoder3D, self).__init__(hidden_layers, kernel_size, stride)

        decoder_layers = []
        reversed_channels = list(reversed(self.channels))
        for i in range(len(reversed_channels) - 1):
            decoder_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    self.create_conv_block(
                        reversed_channels[i], reversed_channels[i + 1], transpose=True
                    ),
                ]
            )

        decoder_layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ConvTranspose3d(
                    reversed_channels[-1],
                    output_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.padding,
                ),
                nn.Sigmoid(),
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, encoder_features):
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if isinstance(layer, nn.Upsample) and i < len(encoder_features):
                x = torch.cat([x, encoder_features[-(i // 2 + 1)]], dim=1)
        return x


class MRI3DAutoencoder(nn.Module):
    def __init__(
        self,
        input_channels=1,
        input_shape=(64, 64, 64),
        hidden_layers=(16, 32, 64),
        kernel_size=3,
        stride=1,
    ):
        super(MRI3DAutoencoder, self).__init__()

        self.input_shape = input_shape
        self.encoder = Encoder3D(
            input_channels=input_channels,
            hidden_layers=hidden_layers,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.decoder = Decoder3D(
            hidden_layers=hidden_layers,
            output_channels=input_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Add a bottleneck layer
        bottleneck_size = hidden_layers[-1]
        self.bottleneck = nn.Sequential(
            nn.Linear(
                bottleneck_size
                * (input_shape[0] // 8)
                * (input_shape[1] // 8)
                * (input_shape[2] // 8),
                bottleneck_size,
            ),
            nn.ReLU(),
            nn.Linear(
                bottleneck_size,
                bottleneck_size
                * (input_shape[0] // 8)
                * (input_shape[1] // 8)
                * (input_shape[2] // 8),
            ),
        )

    def forward(self, x):
        encoded, encoder_features = self.encoder(x)

        # Apply bottleneck
        batch_size = encoded.size(0)
        flattened = encoded.view(batch_size, -1)
        bottleneck = self.bottleneck(flattened)
        reshaped = bottleneck.view(encoded.size())

        decoded = self.decoder(reshaped, encoder_features)
        return decoded
