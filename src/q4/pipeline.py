"""
Q4: freeze the Q1 CNN features, sweep an SVM, then cluster the same vectors.

Question: Q4. configs/q4_plan.json is read and never written here.
The stratified subset is saved before any C or kernel is fit.
How to run: PYTHONNOUSERSITE=1 .venv/bin/python scripts/q4_train.py
Feature extraction uses a GPU when the feature file is missing. The SVM
and the clustering then run on CPU.
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.common.data import CLASS_NAMES, load_experiment_arrays, project_root
from src.common.models import build_model
from src.common.records import write_csv, write_json
from src.common.utils import get_device
from src.q3.pipeline import choose_q1_cnn
from src.q4.analysis import (
    choose_k,
    choose_svm,
    cluster_grid,
    evaluate_clustering,
    fit_svc,
    ovo_decision,
    pair_column,
    pairwise_support_mask,
    support_fraction,
)


def plan_path(root):
    return root / "configs" / "q4_plan.json"


def load_plan(root):
    path = plan_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Write the Q4 plan before fitting an SVM. This script does not create that file."
        )
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, digest, path


def _heartbeat(root, message):
    path = root / "results" / "q4" / "heartbeat.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def prepare_subset(root, plan, train_labels):
    """Stratified rows of the 45,000 training split. Saved before any SVM fit."""
    path = root / "splits" / "q4_svm_subset_seed26003.npy"
    per_class = int(plan["subset_per_class"])
    labels = np.asarray(train_labels)
    rng = np.random.default_rng(int(plan["subset_seed"]))
    chosen = []
    for class_id in range(10):
        members = np.flatnonzero(labels == class_id)
        if members.shape[0] < per_class:
            raise RuntimeError(f"Class {class_id} has only {members.shape[0]} training rows.")
        rng.shuffle(members)
        chosen.append(np.sort(members[:per_class]))
    index = np.sort(np.concatenate(chosen)).astype(np.int64)
    if path.exists():
        saved = np.load(path)
        if not np.array_equal(saved, index):
            raise RuntimeError(f"{path} does not match the planned subset.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, index)
    counts = np.bincount(labels[index], minlength=10)
    if not np.all(counts == per_class):
        raise RuntimeError(f"Subset counts are {counts.tolist()}, expected {per_class} per class.")
    return index


@torch.no_grad()
def _encode(model, images, batch_size):
    pieces = []
    for start in range(0, images.shape[0], batch_size):
        pieces.append(model.encode(images[start:start + batch_size]).float().cpu())
    return torch.cat(pieces, dim=0).numpy().astype(np.float32)


def extract_features(root, plan, device):
    """Frozen penultimate layer of the Q1 matched CNN. The head is not used."""
    path = root / "results" / "q4" / "features_cnn_matched.npz"
    summary = choose_q1_cnn(root)
    if path.exists():
        archive = np.load(path)
        if int(archive["seed"]) != int(summary["seed"]):
            raise RuntimeError("Feature file was built from a different CNN seed.")
        print(f"reloading features from seed {int(archive['seed'])}", flush=True)
        return archive
    if device.type != "cuda":
        raise SystemExit("Feature extraction needs a GPU, or an existing results/q4/features_cnn_matched.npz.")
    fairness = json.loads((root / "configs" / "q1_fairness.json").read_text())
    model = build_model(fairness["models"]["cnn_matched"]).to(device)
    payload = torch.load(summary["checkpoint"], map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    bundle = load_experiment_arrays(root=root, device=device)
    from src.common.data import apply_train_normalisation

    batch_size = int(plan["batch_size"])
    train = _encode(model, apply_train_normalisation(bundle["train_images"], bundle["mean"], bundle["std"]), batch_size)
    val = _encode(model, apply_train_normalisation(bundle["val_images"], bundle["mean"], bundle["std"]), batch_size)
    test = _encode(model, apply_train_normalisation(bundle["test_images"], bundle["mean"], bundle["std"]), batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        train=train,
        val=val,
        test=test,
        train_labels=bundle["train_labels"].detach().cpu().numpy(),
        val_labels=bundle["val_labels"].detach().cpu().numpy(),
        test_labels=bundle["test_labels"].detach().cpu().numpy(),
        seed=np.int64(summary["seed"]),
        val_accuracy=np.float64(summary["best_val_accuracy"]),
        checkpoint=np.array(summary["checkpoint"]),
    )
    print(f"features written, cnn seed {summary['seed']}, dim {train.shape[1]}", flush=True)
    return np.load(path)


def _standardise(train, arrays):
    mean = train.mean(axis=0).astype(np.float64)
    scale = train.std(axis=0).astype(np.float64)
    scale = np.where(scale < 1e-6, 1.0, scale)

    def apply(values):
        return (values.astype(np.float64) - mean) / scale

    return apply(train), [apply(values) for values in arrays]


def run_sweep(plan, train_x, train_y, val_x, val_y, root):
    out = root / "results" / "q4" / "svm_sweep.json"
    if out.exists():
        print("svm sweep exists, reloading", flush=True)
        return json.loads(out.read_text())["rows"]
    partial_path = root / "results" / "q4" / "svm_sweep_partial.json"
    done = {}
    if partial_path.exists():
        for row in json.loads(partial_path.read_text())["rows"]:
            done[(row["kernel"], float(row["C"]))] = row
    rows = []
    for kernel in plan["kernels"]:
        for C in plan["C_values"]:
            key = (kernel, float(C))
            if key in done:
                rows.append(done[key])
                print(f"reload kernel {kernel} C {C}", flush=True)
                continue
            print(f"fit kernel {kernel} C {C}", flush=True)
            clf = fit_svc(train_x, train_y, kernel, C, plan["seed"])
            row = {
                "kernel": kernel,
                "C": float(C),
                "train_accuracy": float(clf.score(train_x, train_y)),
                "val_accuracy": float(clf.score(val_x, val_y)),
                "support_fraction": support_fraction(clf, train_y.shape[0]),
                "n_support": int(clf.n_support_.sum()),
            }
            rows.append(row)
            print(
                f"  train {row['train_accuracy']:.3f}  val {row['val_accuracy']:.3f}  "
                f"sv {row['support_fraction']:.3f}",
                flush=True,
            )
            write_json(root / "results" / "q4" / "svm_sweep_partial.json", {"rows": rows})
    write_json(root / "results" / "q4" / "svm_sweep.json", {"rows": rows, "question": "Q4a"})
    return rows


def _selected_path(root):
    return root / "results" / "q4" / "selected_svm.pkl"


def fit_selected(plan, rows, train_x, train_y, root):
    chosen = choose_svm(rows)
    write_json(
        root / "results" / "q4" / "svm_selection.json",
        {
            "question": "Q4a",
            "rule": plan["svm_selection_rule"],
            "kernel": chosen["kernel"],
            "C": chosen["C"],
            "val_accuracy": chosen["val_accuracy"],
            "train_accuracy": chosen["train_accuracy"],
            "support_fraction": chosen["support_fraction"],
            "note": "The test set was not scored while choosing this row.",
        },
    )
    path = _selected_path(root)
    if path.exists():
        with path.open("rb") as handle:
            clf = pickle.load(handle)
        return chosen, clf
    print(f"refit selected {chosen['kernel']} C {chosen['C']}", flush=True)
    clf = fit_svc(train_x, train_y, chosen["kernel"], chosen["C"], plan["seed"])
    with path.open("wb") as handle:
        pickle.dump(clf, handle)
    return chosen, clf


def evaluate_selected(clf, test_x, test_y, root):
    out = root / "results" / "q4" / "svm_test.json"
    if out.exists():
        return json.loads(out.read_text())
    prediction = clf.predict(test_x)
    accuracy = float((prediction == test_y).mean())
    confusion = np.zeros((10, 10), dtype=np.int64)
    for truth, guess in zip(test_y, prediction):
        confusion[int(truth), int(guess)] += 1
    payload = {
        "question": "Q4a",
        "test_accuracy": accuracy,
        "confusion": confusion.tolist(),
        "note": "Single test evaluation of the validation-selected SVM.",
    }
    write_json(out, payload)
    print(f"svm test accuracy {accuracy:.4f}", flush=True)
    return payload


def _pair_difficulty(confusion):
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            pairs.append((int(confusion[i, j] + confusion[j, i]), i, j))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return pairs


def hard_pairs(plan, clf, subset_x, subset_y, subset_index, test_x, test_y, test_payload, root):
    out = root / "results" / "q4" / "hard_pairs.json"
    if out.exists():
        return json.loads(out.read_text())
    confusion = np.asarray(test_payload["confusion"])
    # The plan says "keep the three largest". The count is fixed there;
    # it is not a key, so it cannot be edited after the SVM test.
    ranked = _pair_difficulty(confusion)[:3]
    scores = ovo_decision(clf, test_x)
    n_classes = 10
    reports = []
    for mistakes, i, j in ranked:
        column = scores[:, pair_column(i, j, n_classes)]
        train_mask = (subset_y == i) | (subset_y == j)
        mean_i = float(ovo_decision(clf, subset_x[subset_y == i])[:, pair_column(i, j, n_classes)].mean())
        mean_j = float(ovo_decision(clf, subset_x[subset_y == j])[:, pair_column(i, j, n_classes)].mean())
        positive = int(i if mean_i >= mean_j else j)
        sv_mask = pairwise_support_mask(clf, i, j)
        strengths = np.zeros(sv_mask.shape[0], dtype=np.float64)
        starts = np.cumsum(np.concatenate([[0], clf.n_support_]))
        for owner, other in ((i, j), (j, i)):
            row = other if other < owner else other - 1
            sl = slice(int(starts[owner]), int(starts[owner + 1]))
            strengths[sl] = np.abs(np.asarray(clf.dual_coef_[row, sl], dtype=np.float64)).reshape(-1)
        sv_positions = np.flatnonzero(sv_mask)
        order = sv_positions[np.argsort(-strengths[sv_positions])][:8]
        support_rows = []
        for position in order:
            subset_row = int(clf.support_[position])
            support_rows.append(
                {
                    "subset_row": subset_row,
                    "train_row": int(subset_index[subset_row]),
                    "label": int(subset_y[subset_row]),
                    "dual_abs": float(strengths[position]),
                }
            )
        pair_test = np.flatnonzero((test_y == i) | (test_y == j))
        nearest = pair_test[np.argsort(np.abs(column[pair_test]))][:8]
        prediction = clf.predict(test_x)
        near_rows = []
        for index in nearest:
            near_rows.append(
                {
                    "test_index": int(index),
                    "label": int(test_y[index]),
                    "predicted": int(prediction[index]),
                    "decision_score": float(column[index]),
                    "correct": bool(prediction[index] == test_y[index]),
                }
            )
        pair_errors = int(mistakes)
        pair_count = int(pair_test.shape[0])
        reports.append(
            {
                "class_i": i,
                "class_j": j,
                "class_i_name": CLASS_NAMES[i],
                "class_j_name": CLASS_NAMES[j],
                "test_confusions": pair_errors,
                "test_pair_count": pair_count,
                "error_rate": pair_errors / pair_count,
                "positive_class": positive,
                "positive_class_name": CLASS_NAMES[positive],
                "score_note": "The OvO column. positive_class is the class with the higher mean score on the training subset.",
                "support_vectors": support_rows,
                "nearest_test": near_rows,
                "n_support_vectors": int(sv_mask.sum()),
            }
        )
    payload = {"question": "Q4b", "pairs": reports}
    write_json(out, payload)
    return payload


def run_clustering(plan, train_x, train_y, val_x, val_y, test_x, test_y, svm_test, root):
    out = root / "results" / "q4" / "clustering.json"
    if out.exists():
        return json.loads(out.read_text())
    rows, models = cluster_grid(train_x, train_y, val_x, val_y, plan["cluster_k"], plan["seed"])
    chosen = choose_k(rows, preferred=int(plan["preferred_k"]))
    model, train_assign = models[int(chosen["k"])]
    # Drop the numpy assignment from the json. Test labels are used only here.
    evaluated = evaluate_clustering(model, train_assign, train_y, test_x, test_y)
    confusion = np.asarray(svm_test["confusion"], dtype=np.float64)
    recalls = []
    for class_id in range(10):
        total = confusion[class_id].sum()
        svm_recall = float(confusion[class_id, class_id] / total) if total else None
        cluster_recall = evaluated["per_class"][class_id]["cluster_recall"]
        recalls.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "svm_recall": svm_recall,
                "cluster_recall": cluster_recall,
                "svm_minus_cluster": None if svm_recall is None else float(svm_recall - cluster_recall),
            }
        )
    disagreement = max(recalls, key=lambda row: row["svm_minus_cluster"])
    result = {
        "question": "Q4c",
        "method": "KMeans",
        "selection_rule": plan["cluster_selection_rule"],
        "k_rows": rows,
        "chosen_k": int(chosen["k"]),
        "chosen_val_nmi": float(chosen["val_nmi"]),
        "test_ari": evaluated["test_ari"],
        "test_nmi": evaluated["test_nmi"],
        "test_majority_accuracy": evaluated["test_majority_accuracy"],
        "svm_test_accuracy": float(svm_test["test_accuracy"]),
        "contingency": evaluated["contingency"],
        "majority_mapping_from_train": evaluated["mapping"],
        "per_class": recalls,
        "disagreement_class": disagreement,
        "note": "KMeans never receives labels. The majority map is fit on the training subset. Test labels are used only for ARI, NMI, the contingency table and the recalls.",
    }
    # test_assign is large and not required in the json.
    write_json(out, result)
    print(
        f"clustering k {chosen['k']}  test ARI {evaluated['test_ari']:.3f}  "
        f"NMI {evaluated['test_nmi']:.3f}",
        flush=True,
    )
    return result


def main():
    root = project_root()
    plan, plan_hash, path = load_plan(root)
    print(f"plan {path.name} sha256 {plan_hash}", flush=True)
    _heartbeat(root, "started")
    bundle = load_experiment_arrays(root=root, device="cpu")
    subset = prepare_subset(root, plan, bundle["train_labels"].numpy())
    print(f"subset {subset.shape[0]} training rows", flush=True)
    device = get_device()
    features = extract_features(root, plan, device)
    train = features["train"]
    subset_x_raw = train[subset]
    subset_y = features["train_labels"][subset]
    # Scale is fit on the SVM training subset only, then applied to val and test.
    subset_x, scaled_rest = _standardise(subset_x_raw, [features["val"], features["test"]])
    val_x, test_x = scaled_rest
    val_y = features["val_labels"]
    test_y = features["test_labels"]
    try:
        _heartbeat(root, "svm sweep")
        rows = run_sweep(plan, subset_x, subset_y, val_x, val_y, root)
        chosen, clf = fit_selected(plan, rows, subset_x, subset_y, root)
        print(f"selected {chosen['kernel']} C {chosen['C']} val {chosen['val_accuracy']:.4f}", flush=True)
        _heartbeat(root, "svm test")
        test_payload = evaluate_selected(clf, test_x, test_y, root)
        _heartbeat(root, "hard pairs")
        hard_pairs(plan, clf, subset_x, subset_y, subset, test_x, test_y, test_payload, root)
        _heartbeat(root, "clustering")
        run_clustering(plan, subset_x, subset_y, val_x, val_y, test_x, test_y, test_payload, root)
    finally:
        try:
            from src.q4.figures import write_q4_figures

            write_q4_figures(root)
            print(f"Q4 figures written under {root / 'figures' / 'q4'}", flush=True)
        except Exception as exc:
            print(f"figure step failed: {exc}", flush=True)
    _heartbeat(root, "finished")
    print("Q4 finished.", flush=True)
