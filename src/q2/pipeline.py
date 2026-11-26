"""
Q2: reuse the Q1 matched models, then retrain them without spatial structure
and without colour.

Question: Q2. configs/q2_plan.json is read and never written. It has to exist
before this program starts, so the expectations cannot be edited after the run.
How to run, on a GPU node, from the repository root:

    PYTHONNOUSERSITE=1 .venv/bin/python scripts/q2_train.py

Q2a does not train. It reloads the Q1 checkpoints. The test set is scored
only after the permuted and grayscale checkpoints have been chosen on validation.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.common.data import (
    apply_train_normalisation,
    load_cifar10,
    load_experiment_arrays,
    load_split,
    project_root,
    split_paths,
)
from src.common.eval import loss_and_accuracy
from src.common.models import build_model, count_parameters
from src.common.records import finished_summary, log_paths, write_csv, write_json
from src.common.train import train_classifier
from src.common.utils import get_device, set_seed
from src.q1.pipeline import load_fairness as load_q1_fairness
from src.q2.transforms import (
    apply_spatial_permutation,
    grayscale_mean_std,
    make_spatial_permutation,
    to_grayscale,
)


def plan_path(root):
    return root / "configs" / "q2_plan.json"


def load_plan(root):
    path = plan_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Write the Q2 expectations before training. This script does not create that file."
        )
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, digest, path


def _heartbeat(root, message):
    path = root / "results" / "q2" / "heartbeat.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def prepare_transforms(root, plan):
    """Save the permutation and the grayscale statistics, or reload them.

    Both are fixed from seed 26003 and the training images. A second run that
    would draw a different permutation is refused, so the test images stay
    paired with the training images.
    """
    splits = root / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    perm_path = splits / "q2_spatial_permutation_seed26003.npy"
    gray_path = splits / "q2_gray_norm_seed26003.npz"
    perm = make_spatial_permutation(int(plan["permutation_seed"]))
    if perm_path.exists():
        saved = np.load(perm_path)
        if not np.array_equal(saved, perm):
            raise RuntimeError(f"{perm_path} does not match seed {plan['permutation_seed']}.")
    else:
        np.save(perm_path, perm)

    paths = split_paths(root)
    cifar = load_cifar10(paths["data_dir"])
    train_idx, _val_idx = load_split(paths["split_file"])
    mean, std = grayscale_mean_std(cifar["train_images"], train_idx)
    if gray_path.exists():
        archive = np.load(gray_path)
        if not np.allclose(archive["mean"], mean) or not np.allclose(archive["std"], std):
            raise RuntimeError(f"{gray_path} does not match the training-split luma statistics.")
    else:
        np.savez(gray_path, mean=mean, std=std)
    write_json(
        splits / "q2_gray_norm_seed26003.json",
        {"mean": mean.tolist(), "std": std.tolist(), "weights": [0.2989, 0.5870, 0.1140]},
    )
    return perm, mean.astype(np.float32), std.astype(np.float32)


def _condition_arrays(bundle, condition, permutation, gray_mean, gray_std):
    """Train and validation tensors for one condition. Test is not included."""
    if condition == "permuted":
        train = apply_spatial_permutation(bundle["train_images"], permutation)
        val = apply_spatial_permutation(bundle["val_images"], permutation)
        mean, std = bundle["mean"], bundle["std"]
    elif condition == "grayscale":
        train = to_grayscale(bundle["train_images"])
        val = to_grayscale(bundle["val_images"])
        mean = torch.from_numpy(np.asarray(gray_mean, dtype=np.float32))
        std = torch.from_numpy(np.asarray(gray_std, dtype=np.float32))
    else:
        raise ValueError(condition)
    device = bundle["train_labels"].device
    return {
        "train_images": apply_train_normalisation(train, mean, std).to(device),
        "train_labels": bundle["train_labels"],
        "val_images": apply_train_normalisation(val, mean, std).to(device),
        "val_labels": bundle["val_labels"],
    }


def _test_arrays(bundle, condition, permutation, gray_mean, gray_std, device):
    images = bundle["test_images"]
    if condition == "clean":
        mean, std = bundle["mean"], bundle["std"]
    elif condition == "permuted":
        images = apply_spatial_permutation(images, permutation)
        mean, std = bundle["mean"], bundle["std"]
    elif condition == "grayscale":
        images = to_grayscale(images)
        mean = torch.from_numpy(np.asarray(gray_mean, dtype=np.float32))
        std = torch.from_numpy(np.asarray(gray_std, dtype=np.float32))
    else:
        raise ValueError(condition)
    return apply_train_normalisation(images, mean, std).to(device), bundle["test_labels"].to(device)


def _history_fields(rows):
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"]
    extra = [key for row in rows for key in row if key not in fields]
    ordered = []
    for key in extra:
        if key not in ordered:
            ordered.append(key)
    return fields + ordered


def _save_run(stem, result, meta):
    json_path, csv_path = log_paths(stem)
    if result["history"]:
        write_csv(csv_path, result["history"], _history_fields(result["history"]))
    best_row = None
    if result["best_epoch"] is not None:
        best_row = result["history"][result["best_epoch"] - 1]
    summary = {
        "model": meta["model"],
        "question": "Q2",
        "architecture_introduced_in": "Q1a",
        "condition": meta["condition"],
        "seed": meta["seed"],
        "lr": meta["lr"],
        "weight_decay": meta["weight_decay"],
        "epochs_requested": meta["epochs"],
        "epochs_trained": len(result["history"]),
        "parameters": result["parameter_count"],
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"] if result["best_epoch"] is not None else None,
        "train_accuracy_at_best_val": None if best_row is None else best_row["train_accuracy"],
        "failed": bool(result["failed"] or result["best_state"] is None),
        "seconds": result["seconds"],
        "plan_sha256": meta["plan_sha256"],
        "history_csv": str(csv_path),
    }
    if result["best_state"] is not None and not summary["failed"]:
        checkpoint = Path(meta["checkpoint"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": result["best_state"],
                "epoch": result["best_epoch"],
                "val_accuracy": result["best_val_accuracy"],
                "model": meta["model"],
                "seed": meta["seed"],
                "condition": meta["condition"],
                "question": "Q2",
                "architecture_introduced_in": "Q1a",
            },
            checkpoint,
        )
        summary["checkpoint"] = str(checkpoint)
    write_json(json_path, summary)
    return summary


def train_condition(plan, q1, tensors, condition, device, plan_hash, root):
    folder = root / "results" / "q2" / condition
    checkpoint_dir = root / "results" / "q2" / "checkpoints"
    summaries = []
    for model_name in plan["models"]:
        spec = q1["models"][model_name]
        for seed in plan["seeds"]:
            stem = folder / f"{model_name}_seed{seed}"
            json_path, _csv_path = log_paths(stem)
            done = finished_summary(json_path, plan["epochs"])
            if done is not None:
                print(f"{condition} {model_name} seed {seed} exists, reloading", flush=True)
                summaries.append(done)
                continue
            print(f"train {condition} {model_name} seed {seed}", flush=True)
            set_seed(int(seed))
            model = build_model(spec).to(device)
            settings = {
                "lr": float(plan["learning_rate"]),
                "weight_decay": float(spec["weight_decay"]),
                "epochs": int(plan["epochs"]),
                "batch_size": int(plan["batch_size"]),
                "use_amp": device.type == "cuda",
                "channels_last": spec["class"] == "ConvNet",
                "batch_transform": None,
            }
            result = train_classifier(
                model,
                tensors["train_images"],
                tensors["train_labels"],
                tensors["val_images"],
                tensors["val_labels"],
                settings,
            )
            result["parameter_count"] = count_parameters(model)
            if result["failed"]:
                raise RuntimeError(f"{condition} {model_name} seed {seed} hit a non-finite loss.")
            summaries.append(
                _save_run(
                    stem,
                    result,
                    {
                        "model": model_name,
                        "condition": condition,
                        "seed": int(seed),
                        "lr": float(plan["learning_rate"]),
                        "weight_decay": float(spec["weight_decay"]),
                        "epochs": int(plan["epochs"]),
                        "plan_sha256": plan_hash,
                        "checkpoint": checkpoint_dir / f"{condition}_{model_name}_seed{seed}.pt",
                    },
                )
            )
    return summaries


def _load_model(spec, checkpoint, device):
    model = build_model(spec).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def confidence_on_test(model, images, labels, batch_size):
    """Accuracy, confidence of correct and incorrect predictions, and the confusion counts.

    Confidence is the maximum softmax probability. High-confidence errors are
    incorrect predictions whose confidence is at least 0.9.
    """
    predictions = []
    confidences = []
    for start in range(0, labels.shape[0], batch_size):
        logits = model(images[start:start + batch_size])
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        predictions.append(prediction.cpu())
        confidences.append(confidence.cpu())
    prediction = torch.cat(predictions).numpy()
    confidence = torch.cat(confidences).numpy()
    truth = labels.detach().cpu().numpy()
    correct = prediction == truth
    confusion = np.zeros((10, 10), dtype=np.int64)
    for true_label, guessed in zip(truth, prediction):
        confusion[int(true_label), int(guessed)] += 1
    wrong = ~correct
    return {
        "test_accuracy": float(correct.mean()),
        "confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "confidence_incorrect": float(confidence[wrong].mean()) if wrong.any() else None,
        "high_confidence_error_fraction": float((confidence[wrong] >= 0.9).mean()) if wrong.any() else None,
        "confusion": confusion.tolist(),
    }


def evaluate_tests(plan, q1, permutation, gray_mean, gray_std, device, root):
    """Score clean Q1 checkpoints and the Q2 checkpoints. Nothing is trained."""
    bundle = load_experiment_arrays(root=root, device="cpu")
    rows = []
    batch_size = int(plan["batch_size"])
    q1_tests = json.loads((root / "results" / "q1" / "test_results.json").read_text())["rows"]
    q1_by_key = {(row["model"], int(row["seed"])): row for row in q1_tests}

    for model_name in plan["models"]:
        spec = q1["models"][model_name]
        for seed in plan["seeds"]:
            q1_row = q1_by_key[(model_name, int(seed))]
            checkpoint = root / "results" / "q1" / "checkpoints" / f"{model_name}_seed{seed}.pt"
            model, payload = _load_model(spec, checkpoint, device)
            images, labels = _test_arrays(bundle, "clean", permutation, gray_mean, gray_std, device)
            report = confidence_on_test(model, images, labels, batch_size)
            # The clean accuracy has to match the Q1 log. A mismatch means the
            # checkpoint or the normalisation changed under us.
            if abs(report["test_accuracy"] - float(q1_row["test_accuracy"])) > 1e-4:
                raise RuntimeError(
                    f"Clean re-evaluation of {model_name} seed {seed} is {report['test_accuracy']}, "
                    f"Q1 logged {q1_row['test_accuracy']}."
                )
            loss, _accuracy = loss_and_accuracy(model, images, labels, batch_size)
            rows.append(
                {
                    "model": model_name,
                    "condition": "clean",
                    "seed": int(seed),
                    "checkpoint": str(checkpoint),
                    "checkpoint_epoch": int(payload["epoch"]),
                    "introduced_in": "Q1a",
                    "test_loss": loss,
                    **{key: value for key, value in report.items() if key != "confusion"},
                    "confusion": report["confusion"],
                }
            )
            print(f"test clean {model_name} seed {seed}  acc {report['test_accuracy']:.4f}", flush=True)

    for condition in ("permuted", "grayscale"):
        for model_name in plan["models"]:
            spec = q1["models"][model_name]
            for seed in plan["seeds"]:
                summary_path = root / "results" / "q2" / condition / f"{model_name}_seed{seed}.json"
                summary = json.loads(summary_path.read_text())
                model, payload = _load_model(spec, summary["checkpoint"], device)
                images, labels = _test_arrays(bundle, condition, permutation, gray_mean, gray_std, device)
                report = confidence_on_test(model, images, labels, batch_size)
                loss, _accuracy = loss_and_accuracy(model, images, labels, batch_size)
                rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "seed": int(seed),
                        "checkpoint": summary["checkpoint"],
                        "checkpoint_epoch": int(payload["epoch"]),
                        "val_accuracy": summary["best_val_accuracy"],
                        "introduced_in": "Q2",
                        "architecture_introduced_in": "Q1a",
                        "test_loss": loss,
                        **{key: value for key, value in report.items() if key != "confusion"},
                        "confusion": report["confusion"],
                    }
                )
                print(
                    f"test {condition} {model_name} seed {seed}  acc {report['test_accuracy']:.4f}",
                    flush=True,
                )
    write_json(root / "results" / "q2" / "test_results.json", {"rows": rows})
    return rows


def assert_plan(plan, q1, q1_hash):
    if q1_hash != plan["q1_fairness_sha256"]:
        raise RuntimeError("Q1 fairness file changed after Q2's plan was written.")
    if list(q1["final"]["models"]) != plan["q1_final_models"]:
        raise RuntimeError("Q1 model list does not match the Q2 plan.")
    for name in plan["models"]:
        got = count_parameters(build_model(q1["models"][name]))
        expected = int(plan["parameter_counts"][name])
        if got != expected:
            raise AssertionError(f"{name} has {got} parameters, Q2 plan says {expected}.")


def main():
    root = project_root()
    plan, plan_hash, path = load_plan(root)
    q1, q1_hash, _q1_path = load_q1_fairness(root)
    assert_plan(plan, q1, q1_hash)
    chosen = json.loads((root / "results" / "q1" / "chosen_lr.json").read_text())
    if float(chosen["chosen_lr"]) != float(plan["learning_rate"]):
        raise RuntimeError(
            f"Q1 chose learning rate {chosen['chosen_lr']}, Q2 plan says {plan['learning_rate']}."
        )
    if not torch.cuda.is_available():
        raise SystemExit("Q2 training needs a GPU. Submit scripts/q2_srun.sh with sbatch.")
    device = get_device()
    print(f"Q2 on {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"plan {path.name} sha256 {plan_hash}", flush=True)
    _heartbeat(root, "started")
    permutation, gray_mean, gray_std = prepare_transforms(root, plan)
    try:
        for condition in plan["retrain_conditions"]:
            _heartbeat(root, f"train {condition}")
            bundle = load_experiment_arrays(root=root, device=device)
            # The loader also returns the test set. Drop it before training.
            bundle.pop("test_images")
            bundle.pop("test_labels")
            tensors = _condition_arrays(bundle, condition, permutation, gray_mean, gray_std)
            train_condition(plan, q1, tensors, condition, device, plan_hash, root)
            del bundle, tensors
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _heartbeat(root, "test")
        test_out = root / "results" / "q2" / "test_results.json"
        if test_out.exists():
            print("test results exist, reloading", flush=True)
        else:
            evaluate_tests(plan, q1, permutation, gray_mean, gray_std, device, root)
    finally:
        try:
            from src.q2.figures import write_q2_figures

            write_q2_figures(root)
            print(f"Q2 figures written under {root / 'figures' / 'q2'}", flush=True)
        except Exception as exc:
            print(f"figure step failed: {exc}", flush=True)
    _heartbeat(root, "finished")
    print("Q2 finished.", flush=True)
