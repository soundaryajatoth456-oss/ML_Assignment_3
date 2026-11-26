"""
Replace a fixed fraction of training labels with a wrong class.

Question: Q1b. The mask is chosen once, saved, and reused. Validation labels
are never passed to this function.
How to run: called by src/q1/pipeline.py before the memorisation training.
"""

import numpy as np

from src.common.data import NUM_CLASSES


def corrupt_training_labels(labels, fraction, seed):
    """Return noisy labels and a boolean mask of the changed rows.

    The draw is stratified so every class loses the same fraction of labels.
    The replacement is uniform over the other nine classes, so a corrupted
    row is never left with its true label by accident. Fraction 0 changes
    nothing. Fraction 1 changes every training row.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Noise fraction must be in [0, 1], got {fraction}.")
    rng = np.random.default_rng(seed)
    noisy = labels.copy()
    mask = np.zeros(labels.shape[0], dtype=bool)
    for class_id in range(NUM_CLASSES):
        members = np.flatnonzero(labels == class_id)
        # Local shuffle: this Generator is not the training RNG.
        rng.shuffle(members)
        count = int(round(fraction * members.shape[0]))
        chosen = members[:count]
        mask[chosen] = True
        # Offset by 1..9, then wrap, so the new label is a different class.
        offsets = rng.integers(1, NUM_CLASSES, size=count)
        noisy[chosen] = (class_id + offsets) % NUM_CLASSES
    return noisy, mask
