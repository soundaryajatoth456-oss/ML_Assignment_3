"""
Train one classifier for a fixed number of epochs.

Question: first introduced in Q1. Q2 and later reuse this loop and must say so.
How to run: called by src/q1/pipeline.py. Not a standalone program.

Images are expected as a single NCHW tensor already on the training device.
CIFAR-10 fits in GPU memory, so there is no DataLoader and no per-batch copy
from CPU. Augmentation, if a later question wants it, is a function applied
to the batch on that same device.
"""

import math
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from src.common.eval import loss_and_accuracy


def _forward_loss(model, images, labels, use_amp):
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        return loss
    logits = model(images)
    return F.cross_entropy(logits, labels)


def train_classifier(model, train_images, train_labels, val_images, val_labels, settings, extra_metrics=None):
    """Minimise cross-entropy for `epochs` passes over the training tensor.

    `settings` holds lr, weight_decay, epochs, batch_size, device and use_amp.
    The checkpoint kept in memory is the earliest epoch with the best
    validation accuracy. Validation is the only selection signal. This
    function does not receive the test set.

    `extra_metrics`, if given, is called at the end of every epoch in eval
    mode and must return a dict of floats. Q1b uses it for the clean,
    noise-fit and true-label measurements.
    """
    device = train_images.device
    use_amp = bool(settings["use_amp"]) and device.type == "cuda"
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
        betas=(0.9, 0.999),
    )
    # Same schedule shape for every model: the step size shrinks smoothly
    # to zero across the shared epoch budget.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)
    channels_last = bool(settings.get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        train_images = train_images.contiguous(memory_format=torch.channels_last)
        val_images = val_images.contiguous(memory_format=torch.channels_last)

    history = []
    best_accuracy = -1.0
    best_epoch = None
    best_state = None
    started = time.time()
    transform = settings.get("batch_transform")

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_labels.shape[0], device=device)
        learning_rate = float(optimiser.param_groups[0]["lr"])
        for start in range(0, order.shape[0], batch_size):
            batch_index = order[start:start + batch_size]
            batch_images = train_images[batch_index]
            batch_labels = train_labels[batch_index]
            if transform is not None:
                batch_images = transform(batch_images)
            optimiser.zero_grad(set_to_none=True)
            loss = _forward_loss(model, batch_images, batch_labels, use_amp)
            if not math.isfinite(float(loss.item())):
                return {
                    "history": history,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_accuracy,
                    "failed": True,
                    "seconds": time.time() - started,
                }
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
        scheduler.step()

        # Measured with dropout off, so the train/val gap is about
        # generalisation and not about the mask dropout applied this step.
        train_loss, train_accuracy = loss_and_accuracy(model, train_images, train_labels, batch_size)
        val_loss, val_accuracy = loss_and_accuracy(model, val_images, val_labels, batch_size)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "lr": learning_rate,
        }
        if extra_metrics is not None:
            model.eval()
            row.update(extra_metrics(model))
        history.append(row)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"  epoch {epoch:02d}  train_acc {train_accuracy:.3f}  "
            f"val_acc {val_accuracy:.3f}  val_loss {val_loss:.3f}",
            flush=True,
        )

    return {
        "history": history,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "failed": False,
        "seconds": time.time() - started,
    }
