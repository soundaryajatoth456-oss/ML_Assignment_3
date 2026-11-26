"""
MLP and CNN used for the CIFAR-10 comparisons.

Question: first introduced in Q1. Later questions import these classes
and must say so when they reuse them.
How to run: built by src/q1/pipeline.py. Not a standalone program.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """A stack of dense layers on a flattened image.

    Every weight is tied to one pixel position. A pattern that moves a few
    pixels needs a new set of weights, which is the limitation Q1 is about.
    """

    def __init__(self, hidden, dropout=0.0):
        super().__init__()
        layers = []
        in_features = 32 * 32 * 3
        for width in hidden:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.ReLU())
            # Dropout is off unless this run is the regularised control.
            # It is part of the model rather than the loss so eval() can
            # turn it off when we measure the train/validation gap.
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, images):
        flat = images.reshape(images.shape[0], -1)
        return self.net(flat)


class ConvNet(nn.Module):
    """A small convolutional net with global average pooling.

    Each 3x3 filter is reused at every position, which is the inductive
    bias the matched MLP does not have. There is no BatchNorm: that would
    be an extra advantage and would muddy the comparison.
    """

    def __init__(self, widths):
        super().__init__()
        blocks = []
        in_channels = 3
        for width in widths:
            blocks.append(nn.Conv2d(in_channels, width, kernel_size=3, padding=1))
            blocks.append(nn.ReLU())
            # 32 -> 16 -> 8 -> 4 as the three pools run. Locality is kept;
            # the later linear layer never sees a raw pixel.
            blocks.append(nn.MaxPool2d(kernel_size=2))
            in_channels = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, 10)

    def encode(self, images):
        """Pooled convolutional features, before the linear classifier.

        Q4 freezes this vector. It is the same tensor the Q1 head reads,
        so the representation is not a new network.
        """
        features = self.features(images)
        pooled = self.pool(features)
        return pooled.flatten(1)

    def forward(self, images):
        return self.head(self.encode(images))


def build_model(spec):
    """Build the architecture named in the Q1 config dict."""
    kind = spec["class"]
    if kind == "MLP":
        return MLP(tuple(spec["hidden"]), dropout=float(spec["dropout"]))
    if kind == "ConvNet":
        return ConvNet(tuple(spec["widths"]))
    raise ValueError(f"Unknown model class {kind}")


def count_parameters(model):
    """Trainable weights, including biases. This is the budget we match."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
