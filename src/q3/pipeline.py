"""
Q3: three bottleneck sizes, one denoiser, then linear probes.

Question: Q3. configs/q3_plan.json is read and never written here.
How to run, on a GPU node, from the repository root:

    PYTHONNOUSERSITE=1 .venv/bin/python scripts/q3_train.py

The clean autoencoder is chosen by validation MSE before any test
reconstruction number is written. The test set is not used to pick C for
the linear probes.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.common.data import apply_train_normalisation, load_experiment_arrays, project_root
from src.common.eval import loss_and_accuracy
from src.common.models import build_model
from src.common.records import finished_summary, log_paths, write_csv, write_json
from src.common.utils import get_device, set_seed
from src.q1.pipeline import lr_tag
from src.q3.models import ConvAutoencoder, count_parameters
from src.q3.probe import linear_probe
from src.q3.train_ae import reconstruction_mse, train_autoencoder


def plan_path(root):
    return root / "configs" / "q3_plan.json"


def load_plan(root):
    path = plan_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Write the Q3 plan before training. This script does not create that file."
        )
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, digest, path


def _heartbeat(root, message):
    path = root / "results" / "q3" / "heartbeat.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n")


def assert_architecture(plan):
    models = [ConvAutoencoder(int(width)) for width in plan["bottlenecks"]]
    counts = [count_parameters(model) for model in models]
    if counts != [int(value) for value in plan["parameter_counts"]]:
        raise AssertionError(f"Parameter counts are {counts}, plan says {plan['parameter_counts']}.")
    reference = dict(models[0].named_parameters())
    bottleneck_names = {"to_code.weight", "to_code.bias", "from_code.weight"}
    same_bias = "from_code.bias"
    for model in models[1:]:
        for name, parameter in model.named_parameters():
            if name in bottleneck_names:
                if parameter.shape == reference[name].shape:
                    raise AssertionError(f"{name} did not change with the bottleneck.")
            elif name == same_bias:
                if parameter.shape != reference[name].shape:
                    raise AssertionError("from_code.bias should stay one bias per decoded unit.")
            elif parameter.shape != reference[name].shape:
                raise AssertionError(f"{name} changed; only the bottleneck may differ.")


def prepare_noise(root, plan, val_shape, test_shape):
    """One frozen validation corruption and one frozen test corruption.

    The generator draws validation first and test second. Training batches
    do not read this file. They draw new noise, so the denoiser cannot
    memorise the evaluation corruption.
    """
    path = root / "splits" / "q3_gaussian_noise_seed26003.npz"
    sigma = float(plan["noise_sigma"])
    rng = np.random.default_rng(int(plan["noise_seed"]))
    val_noise = rng.normal(0.0, sigma, size=val_shape).astype(np.float32)
    test_noise = rng.normal(0.0, sigma, size=test_shape).astype(np.float32)
    if path.exists():
        archive = np.load(path)
        same = float(archive["sigma"]) == sigma and np.array_equal(archive["val_noise"], val_noise)
        same = same and np.array_equal(archive["test_noise"], test_noise)
        if not same:
            raise RuntimeError(f"{path} does not match sigma {sigma} and seed {plan['noise_seed']}.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, val_noise=val_noise, test_noise=test_noise, sigma=np.float32(sigma))
    return val_noise, test_noise


def _settings(plan, learning_rate, epochs, device, sigma):
    return {
        "lr": float(learning_rate),
        "weight_decay": float(plan["weight_decay"]),
        "epochs": int(epochs),
        "batch_size": int(plan["batch_size"]),
        "use_amp": device.type == "cuda",
        "channels_last": True,
        "noise_sigma": float(sigma),
    }


def _save_ae(stem, result, meta):
    json_path, csv_path = log_paths(stem)
    if result["history"]:
        write_csv(csv_path, result["history"], ["epoch", "train_mse", "val_mse", "lr"])
    summary = {
        "question": "Q3",
        "role": meta["role"],
        "bottleneck": int(meta["bottleneck"]),
        "seed": int(meta["seed"]),
        "lr": float(meta["lr"]),
        "noise_sigma": float(meta["noise_sigma"]),
        "epochs_requested": int(meta["epochs"]),
        "epochs_trained": len(result["history"]),
        "parameters": int(meta["parameters"]),
        "best_epoch": result["best_epoch"],
        "best_val_mse": result["best_val_mse"],
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
                "val_mse": result["best_val_mse"],
                "bottleneck": int(meta["bottleneck"]),
                "role": meta["role"],
                "question": "Q3",
            },
            checkpoint,
        )
        summary["checkpoint"] = str(checkpoint)
    write_json(json_path, summary)
    return summary


def _train_one(plan, bottleneck, role, train_clean, val_input, val_target, sigma, epochs, lr, device, stem, checkpoint, plan_hash):
    done = finished_summary(log_paths(stem)[0], epochs)
    if done is not None:
        print(f"{role} bottleneck {bottleneck} exists, reloading", flush=True)
        return done
    print(f"train {role} bottleneck {bottleneck} lr {lr}", flush=True)
    set_seed(int(plan["seed"]))
    model = ConvAutoencoder(int(bottleneck)).to(device)
    result = train_autoencoder(
        model,
        train_clean,
        val_input,
        val_target,
        _settings(plan, lr, epochs, device, sigma),
    )
    if result["failed"]:
        raise RuntimeError(f"{role} bottleneck {bottleneck} hit a non-finite loss.")
    return _save_ae(
        stem,
        result,
        {
            "role": role,
            "bottleneck": int(bottleneck),
            "seed": int(plan["seed"]),
            "lr": float(lr),
            "noise_sigma": float(sigma),
            "epochs": int(epochs),
            "parameters": count_parameters(model),
            "plan_sha256": plan_hash,
            "checkpoint": checkpoint,
        },
    )


def select_learning_rate(plan, train_clean, val_clean, device, plan_hash, root):
    """Short probe on the middle bottleneck. Lower validation MSE wins."""
    folder = root / "results" / "q3" / "probe"
    rows = []
    middle = int(plan["probe_bottleneck"])
    for learning_rate in plan["probe_learning_rates"]:
        stem = folder / f"ae_b{middle}_lr{lr_tag(learning_rate)}_seed{plan['seed']}"
        summary = _train_one(
            plan,
            middle,
            "probe",
            train_clean,
            val_clean,
            val_clean,
            0.0,
            plan["probe_epochs"],
            learning_rate,
            device,
            stem,
            root / "results" / "q3" / "checkpoints" / f"probe_b{middle}_lr{lr_tag(learning_rate)}.pt",
            plan_hash,
        )
        score = summary["best_val_mse"]
        if summary["failed"] or score is None or not np.isfinite(score):
            score = float("inf")
        rows.append({"lr": float(learning_rate), "best_val_mse": score})
    finite = [row for row in rows if np.isfinite(row["best_val_mse"])]
    if not finite:
        raise RuntimeError("Every autoencoder probe failed.")
    chosen = min(finite, key=lambda row: (row["best_val_mse"], row["lr"]))
    payload = {
        "question": "Q3",
        "rule": plan["probe_rule"],
        "rows": rows,
        "chosen_lr": chosen["lr"],
        "plan_sha256": plan_hash,
        "note": "Chosen from clean validation MSE of the middle bottleneck only.",
    }
    write_json(root / "results" / "q3" / "chosen_lr.json", payload)
    print(f"chosen autoencoder lr {chosen['lr']}", flush=True)
    return chosen["lr"]


def select_bottleneck(summaries, plan, root):
    """Lowest validation MSE. A tie keeps the smaller bottleneck."""
    chosen = min(summaries, key=lambda row: (float(row["best_val_mse"]), int(row["bottleneck"])))
    payload = {
        "question": "Q3a",
        "rule": plan["selection_rule"],
        "chosen_bottleneck": int(chosen["bottleneck"]),
        "rows": [
            {
                "bottleneck": int(row["bottleneck"]),
                "parameters": int(row["parameters"]),
                "best_val_mse": float(row["best_val_mse"]),
                "best_epoch": int(row["best_epoch"]),
                "checkpoint": row["checkpoint"],
            }
            for row in summaries
        ],
        "note": "Test reconstruction error was not available when this choice was written.",
    }
    write_json(root / "results" / "q3" / "q3a_selection.json", payload)
    return payload


def _load_ae(bottleneck, checkpoint, device):
    model = ConvAutoencoder(int(bottleneck)).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _psnr(mse):
    if mse is None or mse <= 0:
        return None
    return float(10.0 * np.log10(1.0 / mse))


@torch.no_grad()
def _reconstruct(model, images, batch_size):
    pieces = []
    for start in range(0, images.shape[0], batch_size):
        pieces.append(model(images[start:start + batch_size]).float())
    return torch.cat(pieces, dim=0).clamp(0.0, 1.0)


@torch.no_grad()
def _encode(model, images, batch_size):
    pieces = []
    for start in range(0, images.shape[0], batch_size):
        pieces.append(model.encode(images[start:start + batch_size]).float().cpu())
    return torch.cat(pieces, dim=0).numpy()


def choose_q1_cnn(root):
    """The matched CNN with the highest validation accuracy. The lower seed wins a tie."""
    best = None
    folder = root / "results" / "q1" / "final"
    for path in sorted(folder.glob("cnn_matched_seed*.json")):
        payload = json.loads(path.read_text())
        if payload.get("failed"):
            continue
        rank = (-float(payload["best_val_accuracy"]), int(payload["seed"]))
        if best is None or rank < best[0]:
            best = (rank, payload)
    if best is None:
        raise RuntimeError("No Q1 matched CNN checkpoint is available.")
    return best[1]


def _classify(model, images_01, mean, std, labels, batch_size):
    normalised = apply_train_normalisation(images_01, mean, std)
    loss, accuracy = loss_and_accuracy(model, normalised, labels, batch_size)
    return loss, accuracy


def run_classification(plan, selection, denoise_summary, device, root):
    """Frozen Q1 CNN on clean, corrupted, and reconstructed test images."""
    out = root / "results" / "q3" / "q3b_classification.json"
    if out.exists():
        print("classification exists, reloading", flush=True)
        return json.loads(out.read_text())
    bundle = load_experiment_arrays(root=root, device=device)
    noise = np.load(root / "splits" / "q3_gaussian_noise_seed26003.npz")
    test_clean = bundle["test_images"]
    test_noisy = (test_clean + torch.from_numpy(noise["test_noise"]).to(test_clean.device)).clamp(0.0, 1.0)
    batch_size = int(plan["batch_size"])
    clean_model = _load_ae(selection["chosen_bottleneck"], _checkpoint_of(selection), device)
    denoise_model = _load_ae(selection["chosen_bottleneck"], denoise_summary["checkpoint"], device)
    clean_rec = _reconstruct(clean_model, test_noisy, batch_size)
    denoise_rec = _reconstruct(denoise_model, test_noisy, batch_size)

    cnn_summary = choose_q1_cnn(root)
    spec_path_models = json.loads((root / "configs" / "q1_fairness.json").read_text())
    cnn = build_model(spec_path_models["models"]["cnn_matched"]).to(device)
    payload = torch.load(cnn_summary["checkpoint"], map_location="cpu")
    cnn.load_state_dict(payload["state_dict"])
    cnn.eval()
    labels = bundle["test_labels"]
    mean, std = bundle["mean"], bundle["std"]

    conditions = {
        "clean": test_clean,
        "corrupted": test_noisy,
        "clean_ae_of_corrupted": clean_rec,
        "denoising_ae_of_corrupted": denoise_rec,
    }
    rows = []
    for name, images in conditions.items():
        loss, accuracy = _classify(cnn, images, mean, std, labels, batch_size)
        pixel_mse = float(torch.mean((images - test_clean) ** 2).item())
        rows.append(
            {
                "condition": name,
                "test_accuracy": accuracy,
                "test_loss": loss,
                "pixel_mse_vs_clean": pixel_mse,
                "psnr_vs_clean": _psnr(pixel_mse),
            }
        )
        print(f"classify {name}  acc {accuracy:.4f}  mse {pixel_mse:.5f}", flush=True)
    result = {
        "question": "Q3b",
        "classifier": "cnn_matched",
        "classifier_introduced_in": "Q1a",
        "classifier_seed": int(cnn_summary["seed"]),
        "classifier_val_accuracy": float(cnn_summary["best_val_accuracy"]),
        "classifier_checkpoint": cnn_summary["checkpoint"],
        "clean_autoencoder_bottleneck": int(selection["chosen_bottleneck"]),
        "noise_sigma": float(plan["noise_sigma"]),
        "rows": rows,
    }
    write_json(out, result)
    return result


def _checkpoint_of(selection):
    for row in selection["rows"]:
        if int(row["bottleneck"]) == int(selection["chosen_bottleneck"]):
            return row["checkpoint"]
    raise RuntimeError("Selected bottleneck has no checkpoint.")


def run_representation_comparison(plan, clean_summaries, denoise_summary, device, root):
    """Test MSE and a linear probe for every clean bottleneck, after selection."""
    out = root / "results" / "q3" / "q3c_representation.json"
    if out.exists():
        print("representation comparison exists, reloading", flush=True)
        return json.loads(out.read_text())
    bundle = load_experiment_arrays(root=root, device=device)
    batch_size = int(plan["batch_size"])
    rows = []
    for summary in clean_summaries:
        model = _load_ae(summary["bottleneck"], summary["checkpoint"], device)
        test_mse = reconstruction_mse(model, bundle["test_images"], bundle["test_images"], batch_size)
        codes = {
            "train": _encode(model, bundle["train_images"], batch_size),
            "val": _encode(model, bundle["val_images"], batch_size),
            "test": _encode(model, bundle["test_images"], batch_size),
        }
        labels = {
            "train": bundle["train_labels"].detach().cpu().numpy(),
            "val": bundle["val_labels"].detach().cpu().numpy(),
            "test": bundle["test_labels"].detach().cpu().numpy(),
        }
        sweep, best, test_accuracy = linear_probe(
            codes["train"],
            labels["train"],
            codes["val"],
            labels["val"],
            codes["test"],
            labels["test"],
            plan["probe_C"],
        )
        rows.append(
            {
                "role": "clean",
                "bottleneck": int(summary["bottleneck"]),
                "parameters": int(summary["parameters"]),
                "val_mse": float(summary["best_val_mse"]),
                "test_mse": float(test_mse),
                "test_psnr": _psnr(float(test_mse)),
                "probe_C": best["C"],
                "probe_val_accuracy": best["val_accuracy"],
                "probe_test_accuracy": test_accuracy,
                "probe_sweep": sweep,
                "introduced_in": "Q3a",
            }
        )
        print(
            f"probe clean b{summary['bottleneck']}  test_mse {test_mse:.5f}  "
            f"probe_test {test_accuracy:.4f}",
            flush=True,
        )
    denoise = _load_ae(denoise_summary["bottleneck"], denoise_summary["checkpoint"], device)
    codes = {
        "train": _encode(denoise, bundle["train_images"], batch_size),
        "val": _encode(denoise, bundle["val_images"], batch_size),
        "test": _encode(denoise, bundle["test_images"], batch_size),
    }
    labels = {
        "train": bundle["train_labels"].detach().cpu().numpy(),
        "val": bundle["val_labels"].detach().cpu().numpy(),
        "test": bundle["test_labels"].detach().cpu().numpy(),
    }
    sweep, best, test_accuracy = linear_probe(
        codes["train"], labels["train"], codes["val"], labels["val"], codes["test"], labels["test"], plan["probe_C"]
    )
    # The denoiser's reconstruction error on clean images is not its training
    # objective. The probe is on the clean-image code, which is what part (c)
    # asks: class information in the learned representation.
    rows.append(
        {
            "role": "denoising",
            "bottleneck": int(denoise_summary["bottleneck"]),
            "parameters": int(denoise_summary["parameters"]),
            "val_mse": float(denoise_summary["best_val_mse"]),
            "test_mse": None,
            "test_psnr": None,
            "probe_C": best["C"],
            "probe_val_accuracy": best["val_accuracy"],
            "probe_test_accuracy": test_accuracy,
            "probe_sweep": sweep,
            "introduced_in": "Q3b",
            "note": "val_mse is the denoising objective on the frozen noisy validation set.",
        }
    )
    payload = {"question": "Q3c", "rows": rows}
    write_json(out, payload)
    return payload


def main():
    root = project_root()
    plan, plan_hash, path = load_plan(root)
    assert_architecture(plan)
    if not torch.cuda.is_available():
        raise SystemExit("Q3 training needs a GPU. Submit scripts/q3_srun.sh with sbatch.")
    device = get_device()
    print(f"Q3 on {device} ({torch.cuda.get_device_name(device)})", flush=True)
    print(f"plan {path.name} sha256 {plan_hash}", flush=True)
    _heartbeat(root, "started")
    bundle = load_experiment_arrays(root=root, device="cpu")
    # Noise arrays are drawn before training so the validation corruption
    # cannot be chosen after seeing a reconstruction.
    prepare_noise(root, plan, tuple(bundle["val_images"].shape), tuple(bundle["test_images"].shape))
    noise = np.load(root / "splits" / "q3_gaussian_noise_seed26003.npz")
    train_clean = bundle["train_images"].to(device)
    val_clean = bundle["val_images"].to(device)
    val_noisy = (val_clean + torch.from_numpy(noise["val_noise"]).to(device)).clamp(0.0, 1.0)
    # Test images are not kept for the training stages.
    del bundle
    try:
        _heartbeat(root, "probe")
        chosen_path = root / "results" / "q3" / "chosen_lr.json"
        if chosen_path.exists():
            learning_rate = float(json.loads(chosen_path.read_text())["chosen_lr"])
            print(f"reloading chosen lr {learning_rate}", flush=True)
        else:
            learning_rate = select_learning_rate(plan, train_clean, val_clean, device, plan_hash, root)

        _heartbeat(root, "clean autoencoders")
        summaries = []
        for bottleneck in plan["bottlenecks"]:
            stem = root / "results" / "q3" / "clean" / f"ae_b{int(bottleneck)}_seed{plan['seed']}"
            summaries.append(
                _train_one(
                    plan,
                    bottleneck,
                    "clean",
                    train_clean,
                    val_clean,
                    val_clean,
                    0.0,
                    plan["epochs"],
                    learning_rate,
                    device,
                    stem,
                    root / "results" / "q3" / "checkpoints" / f"clean_b{int(bottleneck)}_seed{plan['seed']}.pt",
                    plan_hash,
                )
            )
        selection_path = root / "results" / "q3" / "q3a_selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
        else:
            selection = select_bottleneck(summaries, plan, root)
        print(f"selected bottleneck {selection['chosen_bottleneck']}", flush=True)

        # Test reconstruction of the selected model only, for part (a).
        test_path = root / "results" / "q3" / "q3a_test_mse.json"
        if not test_path.exists():
            fresh = load_experiment_arrays(root=root, device=device)
            model = _load_ae(selection["chosen_bottleneck"], _checkpoint_of(selection), device)
            test_mse = reconstruction_mse(
                model, fresh["test_images"], fresh["test_images"], int(plan["batch_size"])
            )
            write_json(
                test_path,
                {
                    "question": "Q3a",
                    "bottleneck": int(selection["chosen_bottleneck"]),
                    "test_mse": float(test_mse),
                    "test_psnr": _psnr(float(test_mse)),
                    "note": "Computed after q3a_selection.json. Not used to choose the bottleneck.",
                },
            )
            print(f"selected test mse {test_mse:.5f}", flush=True)
            del fresh

        _heartbeat(root, "denoiser")
        width = int(selection["chosen_bottleneck"])
        denoise_summary = _train_one(
            plan,
            width,
            "denoising",
            train_clean,
            val_noisy,
            val_clean,
            float(plan["noise_sigma"]),
            plan["epochs"],
            learning_rate,
            device,
            root / "results" / "q3" / "denoise" / f"ae_b{width}_seed{plan['seed']}",
            root / "results" / "q3" / "checkpoints" / f"denoise_b{width}_seed{plan['seed']}.pt",
            plan_hash,
        )
        _heartbeat(root, "classification")
        run_classification(plan, selection, denoise_summary, device, root)
        _heartbeat(root, "probes")
        run_representation_comparison(plan, summaries, denoise_summary, device, root)
    finally:
        try:
            from src.q3.figures import write_q3_figures

            write_q3_figures(root)
            print(f"Q3 figures written under {root / 'figures' / 'q3'}", flush=True)
        except Exception as exc:
            print(f"figure step failed: {exc}", flush=True)
    _heartbeat(root, "finished")
    print("Q3 finished.", flush=True)
