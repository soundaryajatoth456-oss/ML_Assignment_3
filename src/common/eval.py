"""
Full-precision loss and accuracy for a tensor already sitting in memory.

Question: first introduced in Q1. Shared by later questions.
How to run: called from the training loop. Not a standalone program.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def loss_and_accuracy(model, images, labels, batch_size):
    """Mean cross-entropy and fraction correct.

    Dropout is off because the caller has set eval mode, or this function
    sets it. Full precision keeps the logged curves free of mixed-precision
    noise. The test set is not passed in here; callers decide which split
    they are measuring.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for start in range(0, labels.shape[0], batch_size):
        batch_images = images[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        logits = model(batch_images)
        loss = F.cross_entropy(logits, batch_labels)
        total_loss += float(loss.item()) * batch_labels.shape[0]
        correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
        total += int(batch_labels.shape[0])
    if was_training:
        model.train()
    return total_loss / total, correct / total
