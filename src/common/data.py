"""
Load CIFAR-10 and serve the one stratified split every question must use.

Question: shared data setup for Q1–Q4. Introduced before Q1.
How to run: python scripts/prepare_data.py
Later experiments should call load_experiment_arrays() and must not resplit.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from src.common.utils import get_device

# Digits of roll m26csa003, in order. This is the only split seed.
SEED = 26003
NUM_CLASSES = 10
# Official CIFAR-10 training pool: 5,000 images in each class.
# Keeping 500 per class for validation leaves 4,500 per class to train on.
VAL_PER_CLASS = 500
TRAIN_PER_CLASS = 4500

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def project_root():
    """Repository root. This file lives at src/common/data.py."""
    return Path(__file__).resolve().parents[2]


def split_paths(root=None):
    """Locations of the download, the saved split and the train-only stats."""
    root = project_root() if root is None else Path(root)
    splits = root / "splits"
    return {
        "root": root,
        "data_dir": root / "data",
        "split_file": splits / "split_seed26003.npz",
        "stats_file": splits / "norm_stats_seed26003.npz",
        "stats_json": splits / "norm_stats_seed26003.json",
        "verification_json": splits / "verification_seed26003.json",
        "example_file": splits / "figure_example_idx_seed26003.npy",
    }


def load_cifar10(data_dir):
    """Download the official CIFAR-10 files if they are not already on disk.

    Images stay uint8 in HWC layout, which is how the dataset stores them.
    The test array is loaded so later evaluation can use it, and so this
    phase can check that the file really has 10,000 images. No statistic
    that training will use is computed from those test pixels.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_set = CIFAR10(root=str(data_dir), train=True, download=True)
    test_set = CIFAR10(root=str(data_dir), train=False, download=True)
    if list(train_set.classes) != CLASS_NAMES:
        raise RuntimeError("Class names do not match the official CIFAR-10 order.")
    bundle = {
        "train_images": np.asarray(train_set.data),
        "train_labels": np.asarray(train_set.targets, dtype=np.int64),
        "test_images": np.asarray(test_set.data),
        "test_labels": np.asarray(test_set.targets, dtype=np.int64),
    }
    if bundle["train_images"].shape != (50000, 32, 32, 3):
        raise RuntimeError(f"Unexpected training image shape {bundle['train_images'].shape}.")
    if bundle["test_images"].shape != (10000, 32, 32, 3):
        raise RuntimeError(f"Unexpected test image shape {bundle['test_images'].shape}.")
    return bundle


def _class_counts(labels):
    labels = np.asarray(labels)
    return [int(np.sum(labels == class_id)) for class_id in range(NUM_CLASSES)]


def stratified_indices(train_labels, seed=SEED, val_per_class=VAL_PER_CLASS):
    """Hold out the same number of images from every class.

    A plain random split could put most of one class into validation and
    make model selection favour that class. Shuffling inside each class,
    then taking the first `val_per_class` indices, fixes the counts exactly.
    The Generator is local: calling set_seed() elsewhere cannot change it.
    Returned indices are sorted so the file records membership, not an
    order to train in. Training code has to shuffle each epoch itself.
    """
    train_labels = np.asarray(train_labels)
    pool_counts = _class_counts(train_labels)
    for class_id, count in enumerate(pool_counts):
        if count != 5000:
            raise AssertionError(
                f"Official training class {class_id} has {count} images, expected 5000. "
                "Refusing to split something that is not standard CIFAR-10."
            )
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for class_id in range(NUM_CLASSES):
        class_indices = np.flatnonzero(train_labels == class_id).copy()
        rng.shuffle(class_indices)
        val_parts.append(class_indices[:val_per_class])
        train_parts.append(class_indices[val_per_class:])
    train_idx = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.int64)
    return train_idx, val_idx


def verify_split(train_labels, test_labels, train_idx, val_idx, val_per_class=VAL_PER_CLASS):
    """Check the rules the rest of the assignment depends on.

    Test labels are counted here so a corrupted download is caught. Test
    pixels are not read. The official test images are a separate array, so
    they cannot appear in train_idx or val_idx: those index the 50,000
    training images only.
    """
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    problems = []
    if train_idx.shape != (TRAIN_PER_CLASS * NUM_CLASSES,):
        problems.append(f"train size is {train_idx.shape[0]}, expected 45000")
    if val_idx.shape != (val_per_class * NUM_CLASSES,):
        problems.append(f"val size is {val_idx.shape[0]}, expected 5000")
    overlap = np.intersect1d(train_idx, val_idx)
    if overlap.size != 0:
        problems.append(f"train and val share {overlap.size} indices")
    covered = np.concatenate([train_idx, val_idx])
    expected = np.arange(train_labels.shape[0], dtype=np.int64)
    if covered.shape[0] != expected.shape[0] or not np.array_equal(np.sort(covered), expected):
        problems.append("train and val are not a partition of the 50,000 official training images")
    if train_idx.min() < 0 or val_idx.min() < 0:
        problems.append("negative index")
    if train_idx.max() >= train_labels.shape[0] or val_idx.max() >= train_labels.shape[0]:
        problems.append("an index points outside the official training set")

    train_counts = _class_counts(train_labels[train_idx])
    val_counts = _class_counts(train_labels[val_idx])
    test_counts = _class_counts(test_labels)
    for class_id in range(NUM_CLASSES):
        if train_counts[class_id] != TRAIN_PER_CLASS:
            problems.append(f"train class {class_id} has {train_counts[class_id]}")
        if val_counts[class_id] != val_per_class:
            problems.append(f"val class {class_id} has {val_counts[class_id]}")
        if test_counts[class_id] != 1000:
            problems.append(f"test class {class_id} has {test_counts[class_id]}")
    if problems:
        raise AssertionError("Split checks failed:\n- " + "\n- ".join(problems))

    return {
        "seed": SEED,
        "train_size": int(train_idx.shape[0]),
        "val_size": int(val_idx.shape[0]),
        "test_size": int(test_labels.shape[0]),
        "overlap": 0,
        "covers_official_training_set": True,
        "test_indices_in_split": False,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "class_names": CLASS_NAMES,
        "test_pixels_used_for_statistics": False,
    }


def channel_mean_std(train_images_uint8, train_idx):
    """Mean and population std of the training pixels, one value per channel.

    Computed on the 45,000 training images only. Validation and test images
    must not shift these constants, or the test set would have influenced
    the preprocessing. Pixels are converted with the same float32 division
    the network will see, then reduced in float64 so the average of about
    46 million values is not itself a source of noise. Population std
    (divide by N, not N-1) is the scale of this fixed set of pixels.
    """
    subset = np.asarray(train_images_uint8)[train_idx]
    mean = np.zeros(3, dtype=np.float64)
    std = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        values = subset[:, :, :, channel].astype(np.float32) / np.float32(255.0)
        values64 = values.astype(np.float64)
        mean[channel] = values64.mean()
        std[channel] = values64.std()
    return mean, std


def choose_figure_examples(train_labels, train_idx, seed=SEED, per_class=10):
    """Pick a fixed grid of training images for the dataset figure.

    This is a display sample, not another split. It uses its own Generator
    started from the same seed, so the picture is reproducible and does not
    change the membership saved in the split file.
    """
    rng = np.random.default_rng(seed)
    rows = []
    train_labels = np.asarray(train_labels)
    for class_id in range(NUM_CLASSES):
        pool = train_idx[train_labels[train_idx] == class_id]
        picked = rng.choice(pool, size=per_class, replace=False)
        rows.append(np.asarray(picked, dtype=np.int64))
    return np.stack(rows, axis=0)


def _check_figure_examples(train_labels, train_idx, example_idx):
    """The drawn images must be training-split members of the labelled class."""
    example_idx = np.asarray(example_idx)
    if example_idx.shape != (NUM_CLASSES, 10):
        raise AssertionError(f"Figure index shape is {example_idx.shape}, expected (10, 10).")
    train_set = set(np.asarray(train_idx).tolist())
    for class_id, row in enumerate(example_idx):
        for index in row.tolist():
            if index not in train_set:
                raise AssertionError(f"Figure index {index} is not in the training split.")
            label = int(train_labels[index])
            if label != class_id:
                raise AssertionError(f"Figure index {index} has label {label}, expected {class_id}.")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def save_split(path, train_idx, val_idx, seed=SEED, val_per_class=VAL_PER_CLASS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        seed=np.int64(seed),
        val_per_class=np.int64(val_per_class),
    )


def load_split(path):
    archive = np.load(path)
    return archive["train_idx"].astype(np.int64), archive["val_idx"].astype(np.int64)


def save_norm_stats(stats_file, stats_json, mean, std, train_size):
    stats_file = Path(stats_file)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    np.savez(stats_file, mean=mean, std=std, train_size=np.int64(train_size), seed=np.int64(SEED))
    _write_json(
        stats_json,
        {
            "seed": SEED,
            "computed_from": "train_split_only",
            "pixel_range": "[0, 1]",
            "dtype_seen_by_network": "float32",
            "reduction": "float64 population mean and std (divide by N)",
            "train_size": int(train_size),
            "pixels_per_channel": int(train_size) * 32 * 32,
            "mean_rgb": [float(v) for v in mean],
            "std_rgb": [float(v) for v in std],
        },
    )


def load_norm_stats(stats_file):
    archive = np.load(stats_file)
    return archive["mean"].astype(np.float64), archive["std"].astype(np.float64)


def prepare_dataset(root=None):
    """Create the split and the train-only statistics, once.

    If the files already exist they are reloaded and compared with a fresh
    draw from seed 26003. A mismatch stops the run. The saved split is never
    silently replaced, because every later question has to share it.
    """
    paths = split_paths(root)
    cifar = load_cifar10(paths["data_dir"])
    fresh_train, fresh_val = stratified_indices(cifar["train_labels"], seed=SEED)

    if paths["split_file"].exists():
        saved_train, saved_val = load_split(paths["split_file"])
        same = np.array_equal(saved_train, fresh_train) and np.array_equal(saved_val, fresh_val)
        if not same:
            raise RuntimeError(
                f"{paths['split_file']} does not match seed {SEED}. "
                "Refusing to overwrite the split the other questions rely on."
            )
        train_idx, val_idx = saved_train, saved_val
        split_status = "reloaded"
    else:
        save_split(paths["split_file"], fresh_train, fresh_val)
        train_idx, val_idx = fresh_train, fresh_val
        split_status = "created"

    report = verify_split(cifar["train_labels"], cifar["test_labels"], train_idx, val_idx)
    report["split_status"] = split_status
    report["split_file"] = str(paths["split_file"])
    # Stable id for the report cover: if this hash changes, the split changed.
    report["train_idx_sha256"] = hashlib.sha256(np.ascontiguousarray(train_idx).tobytes()).hexdigest()
    report["val_idx_sha256"] = hashlib.sha256(np.ascontiguousarray(val_idx).tobytes()).hexdigest()

    fresh_mean, fresh_std = channel_mean_std(cifar["train_images"], train_idx)
    if paths["stats_file"].exists():
        saved_mean, saved_std = load_norm_stats(paths["stats_file"])
        if not (np.allclose(saved_mean, fresh_mean) and np.allclose(saved_std, fresh_std)):
            raise RuntimeError("Saved normalisation does not match the training split.")
        mean, std = saved_mean, saved_std
        stats_status = "reloaded"
    else:
        save_norm_stats(paths["stats_file"], paths["stats_json"], fresh_mean, fresh_std, train_idx.shape[0])
        mean, std = fresh_mean, fresh_std
        stats_status = "created"
    report["stats_status"] = stats_status
    report["mean_rgb"] = [float(v) for v in mean]
    report["std_rgb"] = [float(v) for v in std]

    if paths["example_file"].exists():
        example_idx = np.load(paths["example_file"])
    else:
        example_idx = choose_figure_examples(cifar["train_labels"], train_idx, seed=SEED)
        np.save(paths["example_file"], example_idx)
    _check_figure_examples(cifar["train_labels"], train_idx, example_idx)
    report["figure_examples"] = [CLASS_NAMES[i] + ": " + str(row.tolist()) for i, row in enumerate(example_idx)]
    _write_json(paths["verification_json"], report)

    prepared = {
        "paths": paths,
        "train_images": cifar["train_images"],
        "train_labels": cifar["train_labels"],
        "test_images": cifar["test_images"],
        "test_labels": cifar["test_labels"],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "mean": mean,
        "std": std,
        "example_idx": example_idx,
        "report": report,
    }
    return prepared


def _to_float_nchw(images_uint8):
    """uint8 HWC -> float32 NCHW in [0, 1], the layout convolutions expect."""
    array = np.asarray(images_uint8)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor.float().div(255.0)


def load_experiment_arrays(root=None, device=None):
    """The arrays every later experiment should train and evaluate on.

    Images are float32 in [0, 1], shape (N, 3, 32, 32), ordered by the saved
    sorted indices. They are not shuffled and not normalised. Call
    apply_train_normalisation() for classifiers. Autoencoders can stay in
    [0, 1]. The official test set is returned in full so it can be scored
    after choices are frozen; nothing in this function selects on it.

    When a GPU is available and `device` is left as None, the tensors are
    moved there. CIFAR-10 in float32 is a few hundred megabytes, so this
    avoids reading the images from the CPU on every batch.
    """
    paths = split_paths(root)
    if not paths["split_file"].exists() or not paths["stats_file"].exists():
        prepare_dataset(root)
    cifar = load_cifar10(paths["data_dir"])
    train_idx, val_idx = load_split(paths["split_file"])
    verify_split(cifar["train_labels"], cifar["test_labels"], train_idx, val_idx)
    mean, std = load_norm_stats(paths["stats_file"])

    if device is None:
        device = get_device()
    device = torch.device(device)

    train_images = _to_float_nchw(cifar["train_images"])
    test_images = _to_float_nchw(cifar["test_images"])
    train_labels = torch.from_numpy(cifar["train_labels"])
    test_labels = torch.from_numpy(cifar["test_labels"])

    bundle = {
        "train_images": train_images[train_idx],
        "train_labels": train_labels[train_idx],
        "val_images": train_images[val_idx],
        "val_labels": train_labels[val_idx],
        "test_images": test_images,
        "test_labels": test_labels,
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(std.astype(np.float32)),
        "device": device,
    }
    try:
        bundle = _move_images(bundle, device)
    except RuntimeError as error:
        # A full copy that does not fit is a memory problem, not a reason
        # to change the split. Callers can still batch from CPU memory.
        if "out of memory" not in str(error).lower():
            raise
        bundle["device"] = torch.device("cpu")
    return bundle


def _move_images(bundle, device):
    if device.type == "cpu":
        return bundle
    keys = [
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
        "mean",
        "std",
    ]
    for key in keys:
        bundle[key] = bundle[key].to(device, non_blocking=True)
    return bundle


def apply_train_normalisation(images, mean, std):
    """Subtract the saved training mean and divide by the saved training std.

    `images` is NCHW in [0, 1]. `mean` and `std` each have three entries.
    The same constants are applied to validation and test. They were not
    estimated from those sets.
    """
    mean = mean.to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = std.to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std
