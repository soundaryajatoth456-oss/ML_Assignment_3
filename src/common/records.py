"""
Small helpers for writing a run's json and csv.

Question: shared by Q2, Q3 and Q4. Q1 keeps its own copies.
How to run: imported by those pipelines. Not a standalone program.
"""

import csv
import json
from pathlib import Path


def log_paths(stem):
    """Append the extension as text.

    Path.with_suffix would treat a dot inside a learning rate or a noise
    fraction as the start of the extension and overwrite the wrong file.
    """
    text = str(stem)
    return Path(text + ".json"), Path(text + ".csv")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path, rows, fieldnames):
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


def finished_summary(path, epochs):
    """Reload a run only when it completed the planned number of epochs."""
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("failed"):
        return None
    if int(payload.get("epochs_trained", -1)) != int(epochs):
        return None
    return payload
