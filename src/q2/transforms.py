"""
Disturb CIFAR-10 without changing which pixel values exist, or drop colour.

Question: Q2b and Q2c. The permutation and the grayscale weights are fixed
before training and applied to train, validation and test with the same code.
How to run: called by src/q2/pipeline.py.
"""

import numpy as np
import torch

# Rec. 601 luma. Same three constants for the saved statistics and the tensors.
LUMA_WEIGHTS = (0.2989, 0.5870, 0.1140)


def make_spatial_permutation(seed, size=32 * 32):
    """One permutation of the 32x32 grid.

    The same index list is applied to every image and every channel, so the
    multiset of values in each channel is unchanged. What changes is which
    values sit next to each other. A 3x3 convolution then sees scattered
    pixels instead of a neighbourhood.
    """
    perm = np.random.default_rng(seed).permutation(size).astype(np.int64)
    if perm.shape != (size,) or len(np.unique(perm)) != size:
        raise RuntimeError("Spatial permutation is not a permutation.")
    return perm


def apply_spatial_permutation(images, permutation):
    """Reorder the H*W axis of an NCHW tensor. Channels are not mixed."""
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(images.shape)}.")
    number, channels, height, width = images.shape
    flat = images.reshape(number, channels, height * width)
    index = torch.as_tensor(permutation, device=images.device)
    shuffled = flat[:, :, index]
    return shuffled.reshape(number, channels, height, width)


def to_grayscale(images):
    """Replace RGB with luma, repeated on three channels.

    Repeating keeps the first convolution at 3 input channels, so the CNN
    and the MLP keep the parameter counts from Q1. Colour is gone because
    the three channels are copies of one number.
    """
    weights = images.new_tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
    luma = (images * weights).sum(dim=1, keepdim=True)
    return luma.repeat(1, 3, 1, 1)


def grayscale_mean_std(train_images_uint8, train_idx):
    """Population mean and std of training luma, copied onto three channels.

    Same reduction as the colour statistics: float32 in [0, 1], then a
    float64 mean. Validation and test pixels are not included.
    """
    subset = np.asarray(train_images_uint8)[train_idx]
    pixels = subset.astype(np.float32) / np.float32(255.0)
    weights = np.asarray(LUMA_WEIGHTS, dtype=np.float32)
    luma = (pixels * weights).sum(axis=-1).astype(np.float64)
    value = float(luma.mean())
    scale = float(luma.std())
    mean = np.array([value, value, value], dtype=np.float64)
    std = np.array([scale, scale, scale], dtype=np.float64)
    return mean, std
