"""
Train one convolutional autoencoder for a fixed number of epochs.

Question: Q3. The classifier loop in src/common/train.py is not used here
because the target is an image and the checkpoint is the lowest validation MSE.
How to run: called by src/q3/pipeline.py.
"""

import math
import time

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler


def _mse(prediction, target):
    return F.mse_loss(prediction, target)


@torch.no_grad()
def reconstruction_mse(model, inputs, targets, batch_size):
    """Mean squared error per pixel, averaged over the whole split."""
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, targets.shape[0], batch_size):
        batch_in = inputs[start:start + batch_size]
        batch_target = targets[start:start + batch_size]
        prediction = model(batch_in)
        loss = _mse(prediction, batch_target)
        total += float(loss.item()) * batch_target.shape[0]
        count += int(batch_target.shape[0])
    if was_training:
        model.train()
    return total / count


def train_autoencoder(model, train_clean, val_input, val_target, settings):
    """Minimise pixel MSE.

    When noise_sigma is positive, each training batch is corrupted and the
    target stays the clean image. Validation always uses the tensors passed
    in, so the checkpoint is chosen on a fixed validation corruption rather
    than on a new random draw. The lowest validation MSE is kept. A tie
    keeps the earlier epoch. This function does not see the test set.
    """
    device = train_clean.device
    use_amp = bool(settings["use_amp"]) and device.type == "cuda"
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    sigma = float(settings.get("noise_sigma", 0.0))
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)
    channels_last = bool(settings.get("channels_last", False)) and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        train_clean = train_clean.contiguous(memory_format=torch.channels_last)
        val_input = val_input.contiguous(memory_format=torch.channels_last)
        val_target = val_target.contiguous(memory_format=torch.channels_last)

    history = []
    best_mse = math.inf
    best_epoch = None
    best_state = None
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_clean.shape[0], device=device)
        learning_rate = float(optimiser.param_groups[0]["lr"])
        running = 0.0
        seen = 0
        for start in range(0, order.shape[0], batch_size):
            batch_index = order[start:start + batch_size]
            target = train_clean[batch_index]
            if sigma > 0:
                noisy = target + sigma * torch.randn_like(target)
                model_input = noisy.clamp(0.0, 1.0)
            else:
                model_input = target
            optimiser.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = _mse(model(model_input), target)
            else:
                loss = _mse(model(model_input), target)
            if not math.isfinite(float(loss.item())):
                return {
                    "history": history,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_val_mse": None if best_epoch is None else best_mse,
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
            running += float(loss.item()) * target.shape[0]
            seen += int(target.shape[0])
        scheduler.step()
        train_mse = running / seen
        val_mse = reconstruction_mse(model, val_input, val_target, batch_size)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_mse,
                "lr": learning_rate,
            }
        )
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(
            f"  epoch {epoch:02d}  train_mse {train_mse:.5f}  val_mse {val_mse:.5f}",
            flush=True,
        )

    return {
        "history": history,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_mse": None if best_epoch is None else best_mse,
        "failed": False,
        "seconds": time.time() - started,
    }
