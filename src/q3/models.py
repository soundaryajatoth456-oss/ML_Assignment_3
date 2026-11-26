"""
Convolutional autoencoder. Three widths, one architecture.

Question: Q3. The bottleneck width is the only thing that changes.
How to run: built by src/q3/pipeline.py.
"""

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Encode a 32x32 image to a vector, then decode it back to 32x32.

    The convolutions are identical for every bottleneck. Only the two linear
    maps that enter and leave the code change width. There is no batch norm,
    so a lower reconstruction error cannot be blamed on a different normaliser.
    The code itself has no ReLU: the width of that vector is the capacity.
    """

    def __init__(self, bottleneck):
        super().__init__()
        self.bottleneck = int(bottleneck)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 128 * 4 * 4
        self.to_code = nn.Linear(flat, self.bottleneck)
        self.from_code = nn.Linear(self.bottleneck, flat)
        # Kernel 2 and stride 2 invert the three pools: 4 -> 8 -> 16 -> 32.
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, images):
        features = self.encoder_conv(images)
        return self.to_code(features.flatten(1))

    def decode(self, code):
        flat = self.from_code(code)
        spatial = flat.reshape(code.shape[0], 128, 4, 4)
        return self.decoder_conv(spatial)

    def forward(self, images):
        return self.decode(self.encode(images))


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
