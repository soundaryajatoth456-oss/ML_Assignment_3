"""
Q3 plots from the logs.

Question: Q3.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q3_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.data import project_root
from src.common.records import write_csv
from src.common.viz import save_figure


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "savefig.bbox": "tight",
        }
    )


def write_q3_figures(root=None):
    root = project_root() if root is None else Path(root)
    selection_path = root / "results" / "q3" / "q3a_selection.json"
    if not selection_path.exists():
        return
    selection = json.loads(selection_path.read_text())
    write_csv(
        root / "results" / "q3" / "q3a_val_mse.csv",
        selection["rows"],
        ["bottleneck", "parameters", "best_val_mse", "best_epoch"],
    )
    _style()
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    widths = [row["bottleneck"] for row in selection["rows"]]
    mse = [row["best_val_mse"] for row in selection["rows"]]
    axis.plot(widths, mse, marker="o", color="#0072B2")
    axis.set_xscale("log")
    axis.set_xticks(widths)
    axis.set_xticklabels([str(width) for width in widths])
    axis.set_xlabel("Bottleneck size")
    axis.set_ylabel("Best validation MSE")
    axis.set_title("Q3a clean autoencoders")
    save_figure(fig, root / "figures" / "q3" / "q3a_val_mse.png")

    class_path = root / "results" / "q3" / "q3b_classification.json"
    if class_path.exists():
        payload = json.loads(class_path.read_text())
        write_csv(
            root / "results" / "q3" / "q3b_classification.csv",
            payload["rows"],
            ["condition", "test_accuracy", "test_loss", "pixel_mse_vs_clean", "psnr_vs_clean"],
        )
        _plot_classification(root / "figures" / "q3" / "q3b_classification.png", payload["rows"])

    rep_path = root / "results" / "q3" / "q3c_representation.json"
    if rep_path.exists():
        payload = json.loads(rep_path.read_text())
        clean = [row for row in payload["rows"] if row["role"] == "clean"]
        write_csv(
            root / "results" / "q3" / "q3c_representation.csv",
            [
                {
                    "role": row["role"],
                    "bottleneck": row["bottleneck"],
                    "val_mse": row["val_mse"],
                    "test_mse": row["test_mse"] if row["test_mse"] is not None else "",
                    "probe_C": row["probe_C"],
                    "probe_val_accuracy": row["probe_val_accuracy"],
                    "probe_test_accuracy": row["probe_test_accuracy"],
                }
                for row in payload["rows"]
            ],
            [
                "role",
                "bottleneck",
                "val_mse",
                "test_mse",
                "probe_C",
                "probe_val_accuracy",
                "probe_test_accuracy",
            ],
        )
        _plot_probe(root / "figures" / "q3" / "q3c_mse_vs_probe.png", clean)


def _plot_classification(path, rows):
    _style()
    labels = {
        "clean": "Clean",
        "corrupted": "Noisy",
        "clean_ae_of_corrupted": "Clean AE",
        "denoising_ae_of_corrupted": "Denoising AE",
    }
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    names = [labels[row["condition"]] for row in rows]
    accuracy = [100 * row["test_accuracy"] for row in rows]
    axis.bar(names, accuracy, color=["#009E73", "#D55E00", "#0072B2", "#CC79A7"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Test accuracy of the frozen CNN (%)")
    axis.set_title("Q3b classification after corruption and reconstruction")
    save_figure(fig, path)


def _plot_probe(path, rows):
    _style()
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    mse = [row["test_mse"] for row in rows]
    probe = [100 * row["probe_test_accuracy"] for row in rows]
    widths = [row["bottleneck"] for row in rows]
    axis.scatter(mse, probe, s=60, color="#0072B2")
    for width, x, y in zip(widths, mse, probe):
        axis.annotate(str(width), (x, y), textcoords="offset points", xytext=(6, 4))
    axis.set_xlabel("Test reconstruction MSE")
    axis.set_ylabel("Linear probe test accuracy (%)")
    axis.set_title("Q3c reconstruction against class information")
    save_figure(fig, path)
