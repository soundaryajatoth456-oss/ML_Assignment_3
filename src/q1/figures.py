"""
Draw the Q1 figures from the saved logs.

Question: Q1. Regenerates plots and the result table. It does not train.
How to run: python scripts/q1_figures.py
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {
    "mlp_matched": "Matched MLP",
    "cnn_matched": "Matched CNN",
    "mlp_wide": "Wide MLP",
    "mlp_regularised": "Regularised MLP",
}
COLOURS = {
    "mlp_matched": "#0072B2",
    "cnn_matched": "#D55E00",
    "mlp_wide": "#009E73",
    "mlp_regularised": "#CC79A7",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 12,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _read_csv(path):
    rows = []
    with Path(path).open() as handle:
        for row in csv.DictReader(handle):
            parsed = {"epoch": int(row["epoch"])}
            for key, value in row.items():
                if key == "epoch":
                    continue
                parsed[key] = float(value) if value != "" else None
            rows.append(parsed)
    return rows


def _mean_std(values):
    count = len(values)
    mean = sum(values) / count
    if count == 1:
        return mean, 0.0
    # Sample standard deviation: with three seeds this is the spread to quote.
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance)


def _band(axis, histories, key, colour, label):
    length = min(len(one_run) for one_run in histories)
    epochs = [histories[0][index]["epoch"] for index in range(length)]
    means = []
    spreads = []
    for index in range(length):
        values = [history[index][key] for history in histories if history[index][key] is not None]
        if not values:
            return
        mean, spread = _mean_std(values)
        means.append(mean)
        spreads.append(spread)
    axis.plot(epochs, means, color=colour, label=label, linewidth=1.8)
    lower = [mean - spread for mean, spread in zip(means, spreads)]
    upper = [mean + spread for mean, spread in zip(means, spreads)]
    axis.fill_between(epochs, lower, upper, color=colour, alpha=0.18)


def _final_groups(root):
    folder = root / "results" / "q1" / "final"
    if not folder.exists():
        return {}
    groups = {}
    for path in sorted(folder.glob("*.json")):
        summary = json.loads(path.read_text())
        groups.setdefault(summary["model"], []).append(_read_csv(summary["history_csv"]))
    return groups


def plot_final_curves(root, fig_dir):
    groups = _final_groups(root)
    if not groups:
        return
    _style()
    for key, ylabel, filename in (
        ("train_accuracy", "Training accuracy", "q1a_train_accuracy.png"),
        ("val_accuracy", "Validation accuracy", "q1a_val_accuracy.png"),
        ("train_loss", "Training loss", "q1a_train_loss.png"),
        ("val_loss", "Validation loss", "q1a_val_loss.png"),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        for model_name in ("mlp_matched", "cnn_matched", "mlp_wide", "mlp_regularised"):
            if model_name not in groups:
                continue
            _band(axis, groups[model_name], key, COLOURS[model_name], DISPLAY[model_name])
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Q1a {ylabel.lower()} (mean and sample std across seeds)")
        if "accuracy" in key:
            axis.set_ylim(0.0, 1.0)
        axis.legend(frameon=False)
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
        fig.tight_layout()
        _save(fig, fig_dir / filename)


def plot_probe(root, fig_dir):
    folder = root / "results" / "q1" / "probe"
    if not folder.exists():
        return
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        return
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    panels = {"mlp_matched": axes[0], "cnn_matched": axes[1]}
    for path in paths:
        # Replace only a trailing ".csv". A decimal point earlier in the name
        # is part of the learning-rate tag, not a file extension.
        summary_path = Path(str(path)[:-4] + ".json")
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        axis = panels.get(summary["model"])
        if axis is None:
            continue
        history = _read_csv(path)
        axis.plot(
            [row["epoch"] for row in history],
            [row["val_accuracy"] for row in history],
            label=f"lr {summary['lr']:g}",
        )
    for model_name, axis in panels.items():
        axis.set_title(DISPLAY[model_name])
        axis.set_xlabel("Probe epoch")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Validation accuracy")
    fig.suptitle("Q1 learning-rate probe (validation only, seed 26003)", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "q1a_lr_probe.png")


def plot_test_bars(root, fig_dir):
    path = root / "results" / "q1" / "test_results.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    _style()
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    names = []
    means = []
    spreads = []
    colours = []
    for model_name in ("mlp_matched", "cnn_matched", "mlp_wide", "mlp_regularised"):
        values = [row["test_accuracy"] for row in rows if row["model"] == model_name]
        if not values:
            continue
        mean, spread = _mean_std(values)
        names.append(DISPLAY[model_name])
        means.append(mean)
        spreads.append(spread)
        colours.append(COLOURS[model_name])
    positions = list(range(len(names)))
    axis.bar(positions, means, yerr=spreads, color=colours, capsize=4)
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Official test accuracy")
    axis.set_title("Q1a test accuracy after validation selection")
    axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    _save(fig, fig_dir / "q1a_test_accuracy.png")


def plot_memorisation(root, fig_dir):
    folder = root / "results" / "q1" / "q1b"
    if not folder.exists():
        return
    summaries = [json.loads(path.read_text()) for path in folder.glob("*.json")]
    # Sort by the noise fraction. Filenames do not sort 0 before 0.2.
    summaries.sort(key=lambda item: float(item["noise_fraction"]))
    if not summaries:
        return
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True, sharey=True)
    series = (
        ("clean_accuracy", "#0072B2", "Clean training labels"),
        ("noise_fit", "#D55E00", "Fit to corrupted labels"),
        ("true_on_corrupted", "#009E73", "True label on corrupted"),
        ("val_accuracy", "#000000", "Validation accuracy"),
    )
    for axis, summary in zip(axes.ravel(), summaries):
        history = _read_csv(summary["history_csv"])
        for key, colour, label in series:
            values = [row.get(key) for row in history]
            if all(value is None for value in values):
                continue
            axis.plot(
                [row["epoch"] for row in history],
                [row.get(key) if row.get(key) is not None else math.nan for row in history],
                color=colour,
                label=label,
                linewidth=1.6,
            )
        fraction = summary.get("noise_fraction", "")
        axis.set_title(f"{float(fraction):.0%} of training labels replaced")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for axis in axes[1, :]:
        axis.set_xlabel("Epoch")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy")
    fig.suptitle("Q1b wide MLP: clean fit, noise fit and validation", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "q1b_memorisation.png")


def write_tables(root):
    final_folder = root / "results" / "q1" / "final"
    test_path = root / "results" / "q1" / "test_results.json"
    fairness = json.loads((root / "configs" / "q1_fairness.json").read_text())
    if not final_folder.exists() or not test_path.exists():
        return
    test_rows = json.loads(test_path.read_text())["rows"]
    out_dir = root / "results" / "q1"
    csv_lines = [
        "model,display_name,parameters,weight_decay,dropout,epochs,learning_rate,"
        "seeds,val_accuracy_mean,val_accuracy_std,test_accuracy_mean,test_accuracy_std"
    ]
    tex_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Model & Params & Weight decay & Val.\ acc. & Test acc.\ \\",
        r"\hline",
    ]
    for model_name in ("mlp_matched", "cnn_matched", "mlp_wide", "mlp_regularised"):
        summaries = [
            json.loads(path.read_text())
            for path in sorted(final_folder.glob(f"{model_name}_seed*.json"))
        ]
        if not summaries:
            continue
        val_values = [float(item["best_val_accuracy"]) for item in summaries]
        test_values = [float(item["test_accuracy"]) for item in test_rows if item["model"] == model_name]
        val_mean, val_std = _mean_std(val_values)
        test_mean, test_std = _mean_std(test_values)
        spec = fairness["models"][model_name]
        csv_lines.append(
            f"{model_name},{DISPLAY[model_name]},{spec['parameters']},{spec['weight_decay']},"
            f"{spec['dropout']},{fairness['final']['epochs']},{summaries[0]['lr']},"
            f"{len(summaries)},{val_mean:.6f},{val_std:.6f},{test_mean:.6f},{test_std:.6f}"
        )
        tex_lines.append(
            f"{DISPLAY[model_name]} & {spec['parameters']} & {spec['weight_decay']:g} & "
            f"{100 * val_mean:.1f} $\\pm$ {100 * val_std:.1f} & "
            f"{100 * test_mean:.1f} $\\pm$ {100 * test_std:.1f} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (out_dir / "q1a_model_table.csv").write_text("\n".join(csv_lines) + "\n")
    (out_dir / "q1a_model_table.tex").write_text("\n".join(tex_lines) + "\n")

    memo_lines = ["fraction,best_epoch,best_val_accuracy,final_clean_accuracy,final_noise_fit,final_true_on_corrupted"]
    memo_dir = root / "results" / "q1" / "q1b"
    memo_paths = sorted(memo_dir.glob("*.json")) if memo_dir.exists() else []
    for path in memo_paths:
        summary = json.loads(path.read_text())
        history = _read_csv(summary["history_csv"])
        best = max(history, key=lambda row: row["val_accuracy"])
        last = history[-1]
        def cell(row, key):
            value = row.get(key)
            return "" if value is None else f"{value:.6f}"

        memo_lines.append(
            f"{summary['noise_fraction']},{best['epoch']},{best['val_accuracy']:.6f},"
            f"{cell(last, 'clean_accuracy')},{cell(last, 'noise_fit')},{cell(last, 'true_on_corrupted')}"
        )
    if len(memo_lines) > 1:
        (out_dir / "q1b_summary.csv").write_text("\n".join(memo_lines) + "\n")


def write_q1_figures(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    fig_dir = root / "figures" / "q1"
    _style()
    plot_probe(root, fig_dir)
    plot_final_curves(root, fig_dir)
    plot_test_bars(root, fig_dir)
    plot_memorisation(root, fig_dir)
    write_tables(root)
    print(f"Q1 figures written under {fig_dir}", flush=True)
