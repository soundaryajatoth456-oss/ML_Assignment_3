"""
Q4 plots from the SVM sweep, the hard pairs, and the clustering table.

Question: Q4.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q4_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.data import CLASS_NAMES, load_experiment_arrays, project_root
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


def write_q4_figures(root=None):
    root = project_root() if root is None else Path(root)
    sweep_path = root / "results" / "q4" / "svm_sweep.json"
    if not sweep_path.exists():
        return
    rows = json.loads(sweep_path.read_text())["rows"]
    write_csv(
        root / "results" / "q4" / "svm_sweep.csv",
        rows,
        ["kernel", "C", "train_accuracy", "val_accuracy", "support_fraction", "n_support"],
    )
    _write_sweep_latex(root / "results" / "q4" / "svm_sweep.tex", rows)
    for kernel in ("linear", "poly", "rbf"):
        subset = [row for row in rows if row["kernel"] == kernel]
        if subset:
            _plot_kernel(root / "figures" / "q4" / f"q4_{kernel}.png", kernel, subset)
    pairs_path = root / "results" / "q4" / "hard_pairs.json"
    if pairs_path.exists():
        _plot_pairs(root, json.loads(pairs_path.read_text())["pairs"])
    cluster_path = root / "results" / "q4" / "clustering.json"
    if cluster_path.exists():
        _plot_clusters(root, json.loads(cluster_path.read_text()))


def _write_sweep_latex(path, rows):
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Kernel & C & Train acc. & Val acc. & SV fraction \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['kernel']} & {row['C']:g} & {100 * row['train_accuracy']:.1f} & "
            f"{100 * row['val_accuracy']:.1f} & {100 * row['support_fraction']:.1f}\\% \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _plot_kernel(path, kernel, rows):
    _style()
    rows = sorted(rows, key=lambda row: row["C"])
    c_values = [row["C"] for row in rows]
    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    axis.plot(c_values, [row["val_accuracy"] for row in rows], marker="o", color="#0072B2", label="Validation accuracy")
    axis.set_xscale("log")
    axis.set_xlabel("C")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation accuracy")
    other = axis.twinx()
    other.plot(
        c_values,
        [row["support_fraction"] for row in rows],
        marker="s",
        color="#D55E00",
        label="Support vector fraction",
    )
    other.set_ylim(0, 1)
    other.set_ylabel("Fraction of training points that are support vectors")
    axis.set_title(f"Q4 {kernel} kernel")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, path)


def _plot_pairs(root, pairs):
    bundle = load_experiment_arrays(root=root, device="cpu")
    train = bundle["train_images"].numpy()
    test = bundle["test_images"].numpy()
    columns = 8
    fig, axes = plt.subplots(len(pairs) * 2, columns, figsize=(columns * 1.3, len(pairs) * 2.6))
    for pair_index, pair in enumerate(pairs):
        title = f"{pair['class_i_name']} vs {pair['class_j_name']}"
        top = axes[pair_index * 2]
        bottom = axes[pair_index * 2 + 1]
        vectors = pair["support_vectors"][:columns]
        nearest = pair["nearest_test"][:columns]
        for column in range(columns):
            _draw(top[column], train, vectors, column, "sv")
            _draw(bottom[column], test, nearest, column, "test")
        top[0].set_ylabel(f"{title}\ntrain SV", fontsize=8)
        bottom[0].set_ylabel("near test", fontsize=8)
    fig.suptitle("Q4b support vectors and test points closest to the pair boundary")
    save_figure(fig, root / "figures" / "q4" / "q4_hard_pairs.png")


def _draw(axis, images, rows, column, kind):
    axis.axis("off")
    if column >= len(rows):
        return
    row = rows[column]
    if kind == "sv":
        image = images[row["train_row"]]
        caption = f"{CLASS_NAMES[row['label']]}"
    else:
        image = images[row["test_index"]]
        mark = "ok" if row["correct"] else "err"
        caption = f"{row['decision_score']:+.2f} {mark}"
    axis.imshow(np.transpose(np.clip(image, 0, 1), (1, 2, 0)))
    axis.set_title(caption, fontsize=7)


def _plot_clusters(root, payload):
    _style()
    table = np.asarray(payload["contingency"], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    image = axis.imshow(table, aspect="auto", cmap="Blues")
    axis.set_yticks(range(10))
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Cluster")
    axis.set_ylabel("True class")
    axis.set_title(
        f"Q4c test contingency, k={payload['chosen_k']}, "
        f"ARI {payload['test_ari']:.2f}, NMI {payload['test_nmi']:.2f}"
    )
    fig.colorbar(image, ax=axis, fraction=0.03)
    save_figure(fig, root / "figures" / "q4" / "q4_contingency.png")

    recalls = payload["per_class"]
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(10)
    axis.bar(x - 0.18, [row["svm_recall"] for row in recalls], width=0.36, label="SVM recall", color="#0072B2")
    axis.bar(x + 0.18, [row["cluster_recall"] for row in recalls], width=0.36, label="Cluster majority recall", color="#E69F00")
    axis.set_xticks(x)
    axis.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test recall")
    axis.legend()
    axis.set_title("Q4 classification versus clustering, by class")
    save_figure(fig, root / "figures" / "q4" / "q4_recall_comparison.png")
