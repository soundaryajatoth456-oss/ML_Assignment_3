"""
Figures that describe CIFAR-10 under the saved split.

Question: shared data setup, before Q1.
How to run: python scripts/prepare_data.py
The drawing reads the arrays and the saved indices. It does not resplit,
and it does not estimate pixel statistics from the test set.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.data import CLASS_NAMES, SEED

# Okabe–Ito colours. They stay distinct when the report is printed in grey.
TRAIN_COLOUR = "#0072B2"
VAL_COLOUR = "#E69F00"
TEST_COLOUR = "#009E73"
CHANNEL_COLOURS = ["#d62728", "#2ca02c", "#1f77b4"]


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, path):
    """Write both a PNG for slides and a PDF for the report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_class_grid(train_images, example_idx, path):
    """One row per class, ten training images, so each class is visible."""
    _style()
    fig, axes = plt.subplots(10, 10, figsize=(10.5, 11.2))
    for row, class_id in enumerate(range(10)):
        for col in range(10):
            axis = axes[row, col]
            index = int(example_idx[row, col])
            axis.imshow(train_images[index])
            axis.set_xticks([])
            axis.set_yticks([])
            if col == 0:
                axis.set_ylabel(CLASS_NAMES[class_id], rotation=0, ha="right", va="center", labelpad=28)
    fig.suptitle(f"CIFAR-10 training split: 10 examples from each class (seed {SEED})", y=0.995)
    fig.tight_layout()
    save_figure(fig, path)


def plot_class_distribution(train_counts, val_counts, test_counts, path):
    """Show that stratification kept every class in the same proportion."""
    _style()
    fig, axis = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    train_bars = axis.bar(x - width, train_counts, width, label="train (45,000)", color=TRAIN_COLOUR)
    val_bars = axis.bar(x, val_counts, width, label="validation (5,000)", color=VAL_COLOUR)
    test_bars = axis.bar(x + width, test_counts, width, label="official test (10,000)", color=TEST_COLOUR)
    axis.set_xticks(x)
    axis.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    axis.set_ylabel("Number of images")
    # Headroom so the "4500" labels sit above the bars, and the legend sits
    # above the axes instead of covering the short validation bars.
    axis.set_ylim(0, max(train_counts) * 1.18)
    axis.set_title("Class counts after the stratified split (seed 26003)")
    # Below the class names, so the counts on the short bars stay readable.
    axis.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        borderaxespad=0.0,
    )
    for bars in (train_bars, val_bars, test_bars):
        for rect in bars:
            height = rect.get_height()
            axis.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.tight_layout()
    save_figure(fig, path)


def per_class_train_stats(train_images, train_labels, train_idx):
    """Mean image and channel moments of each class inside the training split.

    Same float32 pixel scale as training. Used for the figure and the table,
    not as a second normalisation. The saved mean/std stay global, not per class.
    """
    subset = train_images[train_idx]
    labels = train_labels[train_idx]
    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        chosen = subset[labels == class_id].astype(np.float32) / np.float32(255.0)
        mean_image = chosen.mean(axis=0)
        channel_mean = chosen.mean(axis=(0, 1, 2))
        channel_std = chosen.std(axis=(0, 1, 2))
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "count": int(chosen.shape[0]),
                "mean_image": mean_image,
                "mean_rgb": channel_mean,
                "std_rgb": channel_std,
            }
        )
    return rows


def plot_mean_images(class_rows, path):
    """Average training image of each class, on a shared [0, 1] scale.

    Nearest-neighbour display keeps the 32 by 32 grid visible. A smooth
    interpolation would hide how little spatial detail the dataset has.
    A shared colour scale stops a bright class from looking artificially sharp.
    """
    _style()
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 5.2))
    for axis, row in zip(axes.ravel(), class_rows):
        axis.imshow(np.clip(row["mean_image"], 0.0, 1.0), interpolation="nearest")
        mean = row["mean_rgb"]
        axis.set_title(
            f"{row['class_name']}\nR {mean[0]:.2f}  G {mean[1]:.2f}  B {mean[2]:.2f}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Mean training image of each class, pixels in [0, 1]", y=1.02)
    fig.tight_layout()
    save_figure(fig, path)


def plot_pixel_statistics(train_images, train_idx, mean, std, path):
    """Histogram of every training pixel, with the saved channel moments."""
    _style()
    subset = train_images[train_idx]
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    bins = np.linspace(0.0, 1.0, 51)
    for channel, name in enumerate("RGB"):
        values = subset[:, :, :, channel].astype(np.float32) / np.float32(255.0)
        histogram, edges = np.histogram(values, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        # Millions, so Matplotlib does not print a "1e6" offset against the title.
        axis.plot(centres, histogram / 1e6, color=CHANNEL_COLOURS[channel], label=f"{name} channel")
        axis.axvline(mean[channel], color=CHANNEL_COLOURS[channel], linestyle="--", linewidth=1)
    axis.set_xlabel(
        "Pixel value in [0, 1]\n"
        f"saved mean [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]    "
        f"saved std [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]"
    )
    axis.set_ylabel("Count in the training split (millions)")
    axis.set_title("Training-split pixel values (dashed lines are the saved means)")
    # Outside the axes so it does not cover the spike of saturated pixels at 1.
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.55))
    fig.tight_layout()
    save_figure(fig, path)


def write_description(report, path):
    """Short dataset note for the report introduction. Numbers come from the run."""
    mean = report["mean_rgb"]
    std = report["std_rgb"]
    text = f"""# CIFAR-10 as used in this assignment

CIFAR-10 has 60,000 colour images. Each image is 32 by 32 pixels with three channels (red, green, blue). There are 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The official division is 50,000 training images and 10,000 test images, with 5,000 and 1,000 images in each class.

This project does not train on all 50,000. Seed {SEED} (the digits of roll m26csa003, in order) draws one stratified split of the official training set: 45,000 train images and 5,000 validation images, which is 4,500 and 500 in every class. The indices are stored in `splits/split_seed26003.npz`. Train and validation do not overlap, and together they are exactly the official training set. The official test set is a separate array of 10,000 images and is not part of that file.

The validation images are reserved for choosing models, architectures and hyperparameters. The test images are scored only after those choices are fixed.

Per-channel mean and population standard deviation are computed from the 45,000 training images only, after scaling pixels from 0–255 into [0, 1]:

- mean (R, G, B) = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]
- std (R, G, B) = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]

These constants are saved in `splits/norm_stats_seed26003.npz`. Validation and test pixels were not included. The often-quoted CIFAR-10 moments are estimated on all 50,000 official training images, so they are close to these numbers but they are not the ones used here.

Two properties of the data matter for the models in later questions. The images are small: a 32 by 32 picture loses fine structure such as text, thin legs and the difference between similar outlines (cat versus dog, automobile versus truck). Within one class the pictures still vary a lot in pose, colour, background and where the object sits. A classifier that cannot reuse a local pattern at many positions has to relearn that pattern separately in every part of the image.

The class-count figure uses test labels only to show that the official test set is balanced. No test pixel was reduced into a saved statistic, and no model was selected in this step.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_count_table(report, path):
    """CSV of the counts behind the bar chart, for the report table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["class_id,class_name,train,val,test"]
    for class_id, name in enumerate(CLASS_NAMES):
        lines.append(
            f"{class_id},{name},{report['train_counts'][class_id]},"
            f"{report['val_counts'][class_id]},{report['test_counts'][class_id]}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_per_class_stats_table(class_rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "class_id,class_name,count,mean_r,mean_g,mean_b,std_r,std_g,std_b"
    lines = [header]
    for row in class_rows:
        mean = row["mean_rgb"]
        std = row["std_rgb"]
        lines.append(
            f"{row['class_id']},{row['class_name']},{row['count']},"
            f"{mean[0]:.8f},{mean[1]:.8f},{mean[2]:.8f},"
            f"{std[0]:.8f},{std[1]:.8f},{std[2]:.8f}"
        )
    path.write_text("\n".join(lines) + "\n")


def write_dataset_figures(prepared, fig_dir, results_dir):
    """Regenerate every Phase 1 figure and table from the prepared arrays."""
    fig_dir = Path(fig_dir)
    results_dir = Path(results_dir)
    report = prepared["report"]
    plot_class_grid(
        prepared["train_images"],
        prepared["example_idx"],
        fig_dir / "class_samples.png",
    )
    plot_class_distribution(
        report["train_counts"],
        report["val_counts"],
        report["test_counts"],
        fig_dir / "class_distribution.png",
    )
    class_rows = per_class_train_stats(
        prepared["train_images"],
        prepared["train_labels"],
        prepared["train_idx"],
    )
    plot_mean_images(class_rows, fig_dir / "mean_images.png")
    plot_pixel_statistics(
        prepared["train_images"],
        prepared["train_idx"],
        np.asarray(prepared["mean"]),
        np.asarray(prepared["std"]),
        fig_dir / "pixel_statistics.png",
    )
    write_description(report, fig_dir / "dataset_description.md")
    write_count_table(report, results_dir / "class_counts.csv")
    write_per_class_stats_table(class_rows, results_dir / "train_per_class_pixel_stats.csv")
