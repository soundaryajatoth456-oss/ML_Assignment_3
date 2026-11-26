"""
Q1 training: a matched MLP/CNN comparison, then the label-noise trajectories.

Question: Q1. The fairness file configs/q1_fairness.json is read and never
rewritten here. It has to exist before this program is started.
How to run, on a GPU node, from the repository root:

    PYTHONNOUSERSITE=1 .venv/bin/python scripts/q1_train.py

The test set is loaded only in evaluate_test_once, after the learning rate
and the checkpoints have already been chosen from validation.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.common.data import apply_train_normalisation, load_experiment_arrays, project_root
from src.common.models import build_model, count_parameters
from src.common.train import train_classifier
from src.common.utils import get_device, set_seed
from src.q1.noise import corrupt_training_labels

FINAL_MODELS = ("mlp_matched", "cnn_matched", "mlp_wide", "mlp_regularised")
PROBE_MODELS = ("mlp_matched", "cnn_matched")


def fairness_path(root):
    return root / "configs" / "q1_fairness.json"


def load_fairness(root=None):
    root = project_root() if root is None else Path(root)
    path = fairness_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Write that plan before training. "
            "This script does not create it, so the timestamp cannot move after a run."
        )
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, digest, path


def lr_tag(learning_rate):
    # "3e-04", not "0.0003". A decimal point in the stem makes Path.with_suffix
    # treat the rest of the name as an extension, so every learning rate
    # would be written to the same file.
    return f"{float(learning_rate):.0e}".replace("+", "")


def log_paths(stem):
    """JSON and CSV paths. The extension is appended, not parsed out of the name."""
    text = str(stem)
    return Path(text + ".json"), Path(text + ".csv")


def fraction_tag(fraction):
    return f"{fraction:g}"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            stored = {}
            for key in fieldnames:
                value = row.get(key, "")
                if value is None:
                    stored[key] = ""
                elif isinstance(value, float):
                    stored[key] = f"{value:.8f}"
                else:
                    stored[key] = value
            writer.writerow(stored)


def assert_parameter_counts(fairness):
    """Stop if the code and the pre-written plan disagree about the budget."""
    for name, spec in fairness["models"].items():
        got = count_parameters(build_model(spec))
        expected = int(spec["parameters"])
        if got != expected:
            raise AssertionError(f"{name} has {got} parameters, plan says {expected}.")


def load_train_val(device):
    """Train and validation tensors only.

    The loader also opens the official test set. Those arrays are dropped
    here so the training loop cannot score them. Test images are opened
    again later, in evaluate_test_once.
    """
    bundle = load_experiment_arrays(root=project_root(), device="cpu")
    mean = bundle["mean"]
    std = bundle["std"]
    train_images = apply_train_normalisation(bundle["train_images"], mean, std).to(device)
    val_images = apply_train_normalisation(bundle["val_images"], mean, std).to(device)
    return {
        "train_images": train_images,
        "train_labels": bundle["train_labels"].to(device),
        "val_images": val_images,
        "val_labels": bundle["val_labels"].to(device),
        "train_labels_numpy": bundle["train_labels"].numpy().copy(),
    }


def prepare_noise_files(train_labels_numpy, fairness, root):
    """Save every noise mask before any memorisation epoch is run."""
    memo = fairness["memorisation"]
    seed = int(memo["corruption_seed"])
    folder = root / "splits" / "q1b_noise"
    folder.mkdir(parents=True, exist_ok=True)
    saved = {}
    for fraction in memo["fractions"]:
        tag = fraction_tag(float(fraction))
        path = folder / f"fraction_{tag}_seed{seed}.npz"
        noisy, mask = corrupt_training_labels(train_labels_numpy, float(fraction), seed)
        if path.exists():
            archive = np.load(path)
            same = np.array_equal(archive["noisy_labels"], noisy) and np.array_equal(archive["mask"], mask)
            if not same:
                raise RuntimeError(f"{path} does not match corruption seed {seed}.")
        else:
            np.savez(
                path,
                noisy_labels=noisy,
                mask=mask,
                fraction=np.float64(fraction),
                seed=np.int64(seed),
            )
        per_class = []
        for class_id in range(10):
            members = train_labels_numpy == class_id
            per_class.append(int(np.sum(mask & members)))
        saved[tag] = {"path": path, "noisy_labels": noisy, "mask": mask, "per_class": per_class}
    return saved


def _settings(fairness, spec, learning_rate, epochs, device, channels_last, weight_decay=None):
    return {
        "lr": float(learning_rate),
        "weight_decay": float(spec["weight_decay"] if weight_decay is None else weight_decay),
        "epochs": int(epochs),
        "batch_size": int(fairness["final"]["batch_size"]),
        "use_amp": device.type == "cuda",
        "channels_last": channels_last and spec["class"] == "ConvNet",
        "batch_transform": None,
    }


def _train_one(spec, tensors, settings, seed, extra_metrics=None):
    set_seed(seed)
    model = build_model(spec)
    model = model.to(tensors["train_images"].device)
    result = train_classifier(
        model,
        tensors["train_images"],
        tensors["train_labels"],
        tensors["val_images"],
        tensors["val_labels"],
        settings,
        extra_metrics=extra_metrics,
    )
    result["parameter_count"] = count_parameters(model)
    return result


def _history_fields(rows):
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"]
    extra = []
    for row in rows:
        for key in row:
            if key not in fields and key not in extra:
                extra.append(key)
    return fields + extra


def _run_record(path_stem, spec, result, meta):
    json_path, csv_path = log_paths(path_stem)
    if result["history"]:
        _write_csv(csv_path, result["history"], _history_fields(result["history"]))
    best_row = None
    if result["best_epoch"] is not None:
        best_row = result["history"][result["best_epoch"] - 1]
    summary = {
        "model": meta["model"],
        "question": "Q1",
        "first_introduced": "Q1",
        "role": meta["role"],
        "seed": meta["seed"],
        "lr": meta["lr"],
        "weight_decay": meta["weight_decay"],
        "epochs_requested": meta["epochs"],
        "epochs_trained": len(result["history"]),
        "parameters": result["parameter_count"],
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": None if result["best_epoch"] is None else result["best_val_accuracy"],
        "train_accuracy_at_best_val": None if best_row is None else best_row["train_accuracy"],
        "val_loss_at_best_val": None if best_row is None else best_row["val_loss"],
        "failed": bool(result["failed"] or result["best_state"] is None),
        "seconds": result["seconds"],
        "device": meta["device"],
        "fairness_sha256": meta["fairness_sha256"],
        "history_csv": str(csv_path),
    }
    summary.update(meta.get("extra", {}))
    checkpoint = None
    if result["best_state"] is not None and not result["failed"]:
        checkpoint = path_stem.parent.parent / "checkpoints" / f"{path_stem.name}.pt"
        # Probe and final checkpoints live in different folders; keep finals only
        # when the caller asks. The path is passed in meta when it should be saved.
        if meta.get("checkpoint") is not None:
            checkpoint = Path(meta["checkpoint"])
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": result["best_state"],
                    "epoch": result["best_epoch"],
                    "val_accuracy": result["best_val_accuracy"],
                    "model": meta["model"],
                    "seed": meta["seed"],
                    "question": "Q1",
                },
                checkpoint,
            )
            summary["checkpoint"] = str(checkpoint)
    _write_json(json_path, summary)
    return summary


def _load_summary(path):
    return json.loads(Path(path).read_text())


def _finished_summary(json_path, epochs):
    """Reload a run only when it finished the planned epochs without a NaN."""
    if not json_path.exists():
        return None
    summary = _load_summary(json_path)
    if summary.get("failed"):
        return None
    if int(summary.get("epochs_trained", 0)) != int(epochs):
        return None
    return summary


def run_probe(fairness, tensors, device_name, fairness_hash, root):
    probe = fairness["probe"]
    folder = root / "results" / "q1" / "probe"
    summaries = []
    for model_name in probe["models"]:
        spec = fairness["models"][model_name]
        for learning_rate in probe["learning_rates"]:
            stem = folder / f"{model_name}_lr{lr_tag(learning_rate)}_seed{probe['seed']}"
            json_path, _csv_path = log_paths(stem)
            finished = _finished_summary(json_path, probe["epochs"])
            if finished is not None:
                print(f"probe exists, reloading {json_path.name}", flush=True)
                summaries.append(finished)
                continue
            print(f"probe {model_name} lr {learning_rate}", flush=True)
            settings = _settings(fairness, spec, learning_rate, probe["epochs"], tensors["train_images"].device, True)
            result = _train_one(spec, tensors, settings, int(probe["seed"]))
            summaries.append(
                _run_record(
                    stem,
                    spec,
                    result,
                    {
                        "model": model_name,
                        "role": "lr_probe",
                        "seed": int(probe["seed"]),
                        "lr": float(learning_rate),
                        "weight_decay": float(spec["weight_decay"]),
                        "epochs": int(probe["epochs"]),
                        "device": device_name,
                        "fairness_sha256": fairness_hash,
                        "checkpoint": None,
                    },
                )
            )
    return summaries


def select_learning_rate(fairness, summaries):
    """Apply the rule written in the plan. Do not inspect the test set."""
    rule = fairness["probe"]["rule"]
    if rule != "mean_best_val_of_matched_mlp_and_cnn_smaller_lr_wins_ties":
        raise RuntimeError(f"Q1 code does not implement probe rule {rule}.")
    scores = {}
    details = {}
    for learning_rate in fairness["probe"]["learning_rates"]:
        values = []
        for model_name in fairness["probe"]["models"]:
            match = [
                item
                for item in summaries
                if item["model"] == model_name and math.isclose(float(item["lr"]), float(learning_rate))
            ]
            if len(match) != 1:
                raise RuntimeError(f"Expected one probe for {model_name} at {learning_rate}.")
            item = match[0]
            if item["failed"] or item["best_val_accuracy"] is None:
                values.append(0.0)
            else:
                values.append(float(item["best_val_accuracy"]))
        scores[str(learning_rate)] = sum(values) / len(values)
        details[str(learning_rate)] = values
    # Higher mean validation accuracy wins. On a tie, the smaller learning rate wins.
    chosen = max(fairness["probe"]["learning_rates"], key=lambda lr: (scores[str(lr)], -float(lr)))
    return float(chosen), scores, details


def run_final_models(fairness, tensors, chosen_lr, device_name, fairness_hash, root):
    folder = root / "results" / "q1" / "final"
    checkpoint_dir = root / "results" / "q1" / "checkpoints"
    summaries = []
    epochs = int(fairness["final"]["epochs"])
    for model_name in FINAL_MODELS:
        spec = fairness["models"][model_name]
        for seed in fairness["final"]["seeds"]:
            stem = folder / f"{model_name}_seed{seed}"
            json_path, _csv_path = log_paths(stem)
            finished = _finished_summary(json_path, epochs)
            if finished is not None:
                print(f"final exists, reloading {json_path.name}", flush=True)
                summaries.append(finished)
                continue
            print(f"final {model_name} seed {seed} lr {chosen_lr}", flush=True)
            settings = _settings(fairness, spec, chosen_lr, epochs, tensors["train_images"].device, True)
            result = _train_one(spec, tensors, settings, int(seed))
            if result["failed"] or result["best_state"] is None:
                _run_record(
                    stem,
                    spec,
                    result,
                    {
                        "model": model_name,
                        "role": "final",
                        "seed": int(seed),
                        "lr": float(chosen_lr),
                        "weight_decay": float(spec["weight_decay"]),
                        "epochs": epochs,
                        "device": device_name,
                        "fairness_sha256": fairness_hash,
                        "checkpoint": None,
                    },
                )
                raise RuntimeError(f"{model_name} seed {seed} hit a non-finite loss.")
            summaries.append(
                _run_record(
                    stem,
                    spec,
                    result,
                    {
                        "model": model_name,
                        "role": "final",
                        "seed": int(seed),
                        "lr": float(chosen_lr),
                        "weight_decay": float(spec["weight_decay"]),
                        "epochs": epochs,
                        "device": device_name,
                        "fairness_sha256": fairness_hash,
                        "checkpoint": checkpoint_dir / f"{model_name}_seed{seed}.pt",
                    },
                )
            )
    return summaries


def evaluate_test_once(fairness, summaries, device, root):
    """Score the already chosen checkpoints. Nothing is trained here."""
    from src.common.models import build_model

    bundle = load_experiment_arrays(root=root, device="cpu")
    mean, std = bundle["mean"], bundle["std"]
    test_images = apply_train_normalisation(bundle["test_images"], mean, std).to(device)
    test_labels = bundle["test_labels"].to(device)
    from src.common.eval import loss_and_accuracy

    rows = []
    batch_size = int(fairness["final"]["batch_size"])
    for summary in summaries:
        if summary["role"] != "final":
            continue
        if summary["failed"]:
            raise RuntimeError(f"Refusing to test a failed run: {summary['model']} seed {summary['seed']}.")
        spec = fairness["models"][summary["model"]]
        model = build_model(spec).to(device)
        payload = torch.load(summary["checkpoint"], map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        loss, accuracy = loss_and_accuracy(model, test_images, test_labels, batch_size)
        rows.append(
            {
                "model": summary["model"],
                "seed": summary["seed"],
                "checkpoint_epoch": payload["epoch"],
                "val_accuracy": summary["best_val_accuracy"],
                "test_loss": loss,
                "test_accuracy": accuracy,
                "question": "Q1",
                "split": "official_test",
            }
        )
        print(
            f"test {summary['model']} seed {summary['seed']}  acc {accuracy:.4f}",
            flush=True,
        )
    _write_json(root / "results" / "q1" / "test_results.json", {"rows": rows})
    return rows


def _memorisation_metrics(model, tensors, clean_index, corrupt_index, true_labels, noisy_labels, batch_size):
    from src.common.eval import loss_and_accuracy

    def score(index, labels):
        if index.numel() == 0:
            return None
        _, accuracy = loss_and_accuracy(model, tensors["train_images"][index], labels[index], batch_size)
        return accuracy

    return {
        "clean_accuracy": score(clean_index, true_labels),
        "noise_fit": score(corrupt_index, noisy_labels),
        "true_on_corrupted": score(corrupt_index, true_labels),
    }


def run_memorisation(fairness, tensors, noise_files, chosen_lr, device_name, fairness_hash, root):
    memo = fairness["memorisation"]
    spec = fairness["models"][memo["model"]]
    # Same widths as the wide MLP from Q1a. Weight decay is forced off:
    # weight decay fights the fit to random labels, and this part asks
    # whether the network can fit labels that should not generalise.
    folder = root / "results" / "q1" / "q1b"
    device = tensors["train_images"].device
    batch_size = int(fairness["final"]["batch_size"])
    true_labels = tensors["train_labels"]
    summaries = []
    for fraction in memo["fractions"]:
        tag = fraction_tag(float(fraction))
        record = noise_files[tag]
        stem = folder / f"{memo['model']}_noise{tag}_seed{memo['seed']}"
        json_path, _csv_path = log_paths(stem)
        finished = _finished_summary(json_path, memo["epochs"])
        if finished is not None:
            print(f"memorisation exists, reloading {json_path.name}", flush=True)
            summaries.append(finished)
            continue
        noisy = torch.from_numpy(record["noisy_labels"]).to(device)
        mask = torch.from_numpy(record["mask"]).to(device)
        # torch.flatten(nonzero) rather than flatnonzero: this torch is 2.1.
        clean_index = (~mask).nonzero(as_tuple=False).flatten()
        corrupt_index = mask.nonzero(as_tuple=False).flatten()
        labelled = dict(tensors)
        labelled["train_labels"] = noisy
        print(f"memorisation fraction {tag}  corrupted {int(mask.sum().item())}", flush=True)

        def extra_metrics(model, clean_index=clean_index, corrupt_index=corrupt_index, noisy=noisy):
            return _memorisation_metrics(
                model, tensors, clean_index, corrupt_index, true_labels, noisy, batch_size
            )

        settings = _settings(
            fairness,
            spec,
            chosen_lr,
            memo["epochs"],
            device,
            False,
            weight_decay=memo["weight_decay"],
        )
        result = _train_one(spec, labelled, settings, int(memo["seed"]), extra_metrics=extra_metrics)
        if result["failed"]:
            raise RuntimeError(f"Memorisation run at fraction {tag} hit a non-finite loss.")
        summaries.append(
            _run_record(
                stem,
                spec,
                result,
                {
                    "model": memo["model"],
                    "role": "memorisation",
                    "seed": int(memo["seed"]),
                    "lr": float(chosen_lr),
                    "weight_decay": float(memo["weight_decay"]),
                    "epochs": int(memo["epochs"]),
                    "device": device_name,
                    "fairness_sha256": fairness_hash,
                    "checkpoint": None,
                    "extra": {
                        "noise_fraction": float(fraction),
                        "corrupted_per_class": record["per_class"],
                        "introduced_in": "Q1b",
                        "architecture_introduced_in": "Q1a",
                    },
                },
            )
        )
    return summaries


def _heartbeat(root, message):
    path = root / "results" / "q1" / "heartbeat.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def main():
    import os

    root = project_root()
    fairness, fairness_hash, path = load_fairness(root)
    if not torch.cuda.is_available():
        raise SystemExit(
            "Full Q1 training is meant for a GPU. Submit scripts/q1_srun.sh with sbatch."
        )
    device = get_device()
    device_name = torch.cuda.get_device_name(device)
    print(f"Q1 on {device} ({device_name})", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}", flush=True)
    print(f"fairness {path.name} sha256 {fairness_hash}", flush=True)
    _heartbeat(root, f"started {device_name}")
    assert_parameter_counts(fairness)
    if list(fairness["final"]["models"]) != list(FINAL_MODELS):
        raise AssertionError("Final model list in the plan does not match the Q1 runner.")
    tensors = load_train_val(device)
    noise_files = prepare_noise_files(tensors["train_labels_numpy"], fairness, root)
    try:
        probe_summaries = run_probe(fairness, tensors, device_name, fairness_hash, root)
        chosen_lr, scores, details = select_learning_rate(fairness, probe_summaries)
        chosen_payload = {
            "question": "Q1",
            "rule": fairness["probe"]["rule"],
            "scores_mean_best_val": scores,
            "per_model_best_val": details,
            "chosen_lr": chosen_lr,
            "fairness_sha256": fairness_hash,
            "note": "Chosen from the validation probe only. The test set was not loaded for this choice.",
        }
        _write_json(root / "results" / "q1" / "chosen_lr.json", chosen_payload)
        print(f"chosen learning rate {chosen_lr} scores {scores}", flush=True)
        _heartbeat(root, f"chosen_lr {chosen_lr}")
        finals = run_final_models(fairness, tensors, chosen_lr, device_name, fairness_hash, root)
        _heartbeat(root, "finals done, evaluating test")
        evaluate_test_once(fairness, finals, device, root)
        _heartbeat(root, "test done, memorisation")
        run_memorisation(fairness, tensors, noise_files, chosen_lr, device_name, fairness_hash, root)
        print("Q1 finished.", flush=True)
        _heartbeat(root, "finished")
    finally:
        from src.q1.figures import write_q1_figures

        try:
            write_q1_figures(root)
        except Exception as error:
            print(f"figure step failed: {error}", flush=True)
