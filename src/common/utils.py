"""
Seeds and the compute device used by every later experiment.

Question: shared setup for Q1–Q4.
The CIFAR-10 split does not use these global seeds. It draws from its own
numpy Generator inside src/common/data.py, so re-running training cannot
change who is in the validation set.

How to run: imported by the experiment scripts. Not a standalone program.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    """Fix the RNGs that training and augmentation will consult.

    The data split has its own Generator and does not read this state.
    cuDNN is asked to be deterministic. A few CUDA kernels still use
    atomic adds and can differ slightly across runs; each experiment
    records the seed and, where it is affordable, repeats the run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Use a GPU when the machine has one, otherwise CPU.

    If several GPUs are visible, pick the one with the most free memory
    so a busy card is left alone. Phase 1 does not need a GPU.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        free_bytes, _total = torch.cuda.mem_get_info(index)
        if free_bytes > best_free:
            best_free = free_bytes
            best_index = index
    return torch.device(f"cuda:{best_index}")
