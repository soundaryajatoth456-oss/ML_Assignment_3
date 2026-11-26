"""
Q2 tables and plots, drawn from the logs.

Question: Q2.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q2_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.data import CLASS_NAMES, project_root
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


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def write_q2_figures(root=None):
    root = project_root() if root is None else Path(root)
    test_path = root / "results" / "q2" / "test_results.json"
    if not test_path.exists():
        return
    rows = json.loads(test_path.read_text())["rows"]
    plan = json.loads((root / "configs" / "q2_plan.json").read_text())
    conditions = ["clean", "permuted", "grayscale"]
    models = list(plan["models"])
    summary_rows = []
    for model in models:
        for condition in conditions:
            subset = [row for row in rows if row["model"] == model and row["condition"] == condition]
            if not subset:
                continue
            acc_mean, acc_std = _mean_std([row["test_accuracy"] for row in subset])
            conf_c, conf_c_std = _mean_std([row["confidence_correct"] for row in subset])
            conf_w, conf_w_std = _mean_std([row["confidence_incorrect"] for row in subset])
            high, high_std = _mean_std([row["high_confidence_error_fraction"] for row in subset])
            summary_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "parameters": plan["parameter_counts"][model],
                    "seeds": len(subset),
                    "test_accuracy_mean": acc_mean,
                    "test_accuracy_std": acc_std,
                    "confidence_correct_mean": conf_c,
                    "confidence_correct_std": conf_c_std,
                    "confidence_incorrect_mean": conf_w,
                    "confidence_incorrect_std": conf_w_std,
                    "high_confidence_error_fraction_mean": high,
                    "high_confidence_error_fraction_std": high_std,
                    "architecture_introduced_in": "Q1a",
                }
            )
    write_csv(
        root / "results" / "q2" / "q2_summary.csv",
        summary_rows,
        list(summary_rows[0].keys()),
    )
    _write_latex(root / "results" / "q2" / "q2_summary.tex", summary_rows)
    _plot_accuracy(root / "figures" / "q2" / "q2_test_accuracy.png", summary_rows, models, conditions)
    _plot_confidence(root / "figures" / "q2" / "q2_confidence.png", summary_rows)
    _plot_confusion(root, rows)


def _write_latex(path, rows):
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Model & Condition & Params & Test acc. & Conf.\ correct & Conf.\ wrong \\",
        r"\hline",
    ]
    names = {"mlp_matched": "Matched MLP", "cnn_matched": "Matched CNN"}
    labels = {"clean": "Clean", "permuted": "Permuted", "grayscale": "Grayscale"}
    for row in rows:
        lines.append(
            f"{names[row['model']]} & {labels[row['condition']]} & {row['parameters']} & "
            f"{100 * row['test_accuracy_mean']:.1f} $\\pm$ {100 * row['test_accuracy_std']:.1f} & "
            f"{row['confidence_correct_mean']:.2f} & {row['confidence_incorrect_mean']:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _plot_accuracy(path, rows, models, conditions):
    _style()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    width = 0.35
    positions = np.arange(len(conditions))
    for offset, model, colour in (
        (-width / 2, "mlp_matched", "#0072B2"),
        (width / 2, "cnn_matched", "#D55E00"),
    ):
        subset = [row for row in rows if row["model"] == model]
        by_condition = {row["condition"]: row for row in subset}
        means = [100 * by_condition[name]["test_accuracy_mean"] for name in conditions]
        stds = [100 * by_condition[name]["test_accuracy_std"] for name in conditions]
        axis.bar(positions + offset, means, width=width, yerr=stds, color=colour, label=model, capsize=3)
    axis.set_xticks(positions)
    axis.set_xticklabels(["Clean", "Pixels permuted", "Grayscale"])
    axis.set_ylabel("Test accuracy (%)")
    axis.set_ylim(0, 100)
    axis.legend()
    axis.set_title("Q2 test accuracy, mean and sample std over 3 seeds")
    save_figure(fig, path)


def _plot_confidence(path, rows):
    _style()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    cnn = [row for row in rows if row["model"] == "cnn_matched"]
    labels = ["Clean", "Pixels permuted", "Grayscale"]
    conditions = ["clean", "permuted", "grayscale"]
    by_condition = {row["condition"]: row for row in cnn}
    x = np.arange(len(conditions))
    correct = [by_condition[name]["confidence_correct_mean"] for name in conditions]
    wrong = [by_condition[name]["confidence_incorrect_mean"] for name in conditions]
    axis.bar(x - 0.18, correct, width=0.36, label="Correct predictions", color="#009E73")
    axis.bar(x + 0.18, wrong, width=0.36, label="Incorrect predictions", color="#CC79A7")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean max-softmax")
    axis.set_title("Matched CNN confidence on the test set")
    axis.legend()
    save_figure(fig, path)


def _plot_confusion(root, rows):
    _style()
    wanted = [
        row
        for row in rows
        if row["model"] == "cnn_matched" and int(row["seed"]) == 26003
    ]
    if len(wanted) < 3:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    titles = {"clean": "Clean", "permuted": "Pixels permuted", "grayscale": "Grayscale"}
    for axis, condition in zip(axes, ("clean", "permuted", "grayscale")):
        row = next(item for item in wanted if item["condition"] == condition)
        counts = np.asarray(row["confusion"], dtype=np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        share = counts / totals
        image = axis.imshow(share, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(titles[condition])
        axis.set_xlabel("Predicted")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        axis.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
        axis.set_yticklabels(CLASS_NAMES, fontsize=7)
    axes[0].set_ylabel("True class")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle("Matched CNN, seed 26003, row-normalised test confusion")
    save_figure(fig, root / "figures" / "q2" / "q2_confusion_cnn_seed26003.png")
