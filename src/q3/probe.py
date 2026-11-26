"""
Linear probe on a frozen autoencoder code.

Question: Q3c. The encoder weights are not updated. C is chosen on the
validation codes. The test codes are scored once, after that choice.
How to run: called by src/q3/pipeline.py.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _standardise(train, other):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (train - mean) / scale, (other - mean) / scale


def linear_probe(train_x, train_y, val_x, val_y, test_x, test_y, C_values):
    """Logistic regression. The smallest C wins a validation tie."""
    train_n, val_n = _standardise(train_x, val_x)
    _train_again, test_n = _standardise(train_x, test_x)
    rows = []
    best = None
    for C in C_values:
        # lbfgs fits a multinomial logistic model on this sklearn.
        clf = LogisticRegression(C=float(C), solver="lbfgs", max_iter=400)
        clf.fit(train_n, train_y)
        row = {
            "C": float(C),
            "train_accuracy": float(clf.score(train_n, train_y)),
            "val_accuracy": float(clf.score(val_n, val_y)),
        }
        rows.append(row)
        better = best is None or row["val_accuracy"] > best["val_accuracy"]
        tie = best is not None and abs(row["val_accuracy"] - best["val_accuracy"]) <= 1e-12
        if better or (tie and row["C"] < best["C"]):
            best = dict(row)
            best_model = clf
    test_accuracy = float(best_model.score(test_n, test_y))
    return rows, best, test_accuracy
