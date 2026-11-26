"""
SVM and clustering rules that do not touch the test labels while fitting.

Question: Q4. The selection rules live here so the smoke test can check them
on a toy problem before the real features are used.
How to run: called by src/q4/pipeline.py.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.svm import SVC


def pair_column(class_i, class_j, n_classes):
    """Column of an OvO decision matrix for class_i < class_j.

    libsvm lists pairs as (0,1), (0,2), ... then (1,2), and so on.
    """
    if class_j < class_i:
        class_i, class_j = class_j, class_i
    return int(class_i * (2 * n_classes - class_i - 1) // 2 + (class_j - class_i - 1))


def ovo_decision(svc, features):
    """Pairwise decision values, shape (n_samples, n_pairs).

    sklearn's default shape is one-vs-rest. The underlying values are
    one-vs-one, which is the boundary part (b) asks about. The attribute
    is put back afterwards.
    """
    saved = svc.decision_function_shape
    svc.decision_function_shape = "ovo"
    try:
        scores = np.asarray(svc.decision_function(features), dtype=np.float64)
    finally:
        svc.decision_function_shape = saved
    return scores


def pairwise_support_mask(svc, class_i, class_j):
    """Rows of support_vectors_ that the class_i vs class_j problem uses.

    dual_coef_ has one row per 'other class' and one column per support
    vector, with support vectors grouped by their own class. A coefficient
    of zero means that vector is not a support vector of this pair.
    """
    if class_j < class_i:
        class_i, class_j = class_j, class_i
    starts = np.cumsum(np.concatenate([[0], svc.n_support_]))
    mask = np.zeros(svc.support_vectors_.shape[0], dtype=bool)
    for owner, other in ((class_i, class_j), (class_j, class_i)):
        row = other if other < owner else other - 1
        sl = slice(int(starts[owner]), int(starts[owner + 1]))
        active = np.abs(svc.dual_coef_[row, sl]) > 0
        mask[sl] = np.asarray(active).reshape(-1)
    return mask


def fit_svc(features, labels, kernel, C, seed):
    """One kernel SVM. gamma and the polynomial degree are not searched."""
    clf = SVC(
        C=float(C),
        kernel=kernel,
        degree=3,
        gamma="scale",
        coef0=1.0,
        cache_size=2000,
        random_state=int(seed) if kernel == "poly" else None,
    )
    # random_state is ignored by libsvm except for the probability path,
    # which we do not use. It is set for poly so the call stays explicit.
    clf.fit(features, labels)
    return clf


def support_fraction(svc, n_train):
    return float(svc.n_support_.sum()) / float(n_train)


def choose_svm(rows):
    """Highest validation accuracy. Then fewer support vectors, then smaller C, then kernel name."""
    def rank(row):
        return (
            -float(row["val_accuracy"]),
            float(row["support_fraction"]),
            float(row["C"]),
            str(row["kernel"]),
        )

    return min(rows, key=rank)


def choose_k(rows, preferred=10):
    """Highest validation NMI. Then the k closer to 10, then the smaller k."""
    def rank(row):
        return (
            -float(row["val_nmi"]),
            abs(int(row["k"]) - int(preferred)),
            int(row["k"]),
        )

    return min(rows, key=rank)


def majority_map(train_clusters, train_labels, n_clusters):
    """Cluster id to class, using training labels only."""
    mapping = np.full(n_clusters, -1, dtype=np.int64)
    for cluster in range(n_clusters):
        labels = train_labels[train_clusters == cluster]
        if labels.size == 0:
            continue
        mapping[cluster] = int(np.bincount(labels, minlength=10).argmax())
    return mapping


def cluster_grid(features_train, labels_train, features_val, labels_val, ks, seed):
    """Fit KMeans without labels. Validation labels are used only to score k."""
    rows = []
    models = {}
    for k in ks:
        model = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed))
        train_assign = model.fit_predict(features_train)
        val_assign = model.predict(features_val)
        rows.append(
            {
                "k": int(k),
                "val_nmi": float(normalized_mutual_info_score(labels_val, val_assign)),
                "val_ari": float(adjusted_rand_score(labels_val, val_assign)),
                "inertia": float(model.inertia_),
            }
        )
        models[int(k)] = (model, train_assign)
    return rows, models


def evaluate_clustering(model, train_assign, train_labels, test_features, test_labels):
    """Assign test points first. Test labels are used only after that."""
    test_assign = model.predict(test_features)
    mapping = majority_map(train_assign, train_labels, model.n_clusters)
    predicted = mapping[test_assign]
    table = contingency_matrix(test_labels, test_assign)
    per_class = []
    for class_id in range(10):
        mask = test_labels == class_id
        recall = float((predicted[mask] == class_id).mean()) if mask.any() else None
        per_class.append({"class_id": class_id, "cluster_recall": recall})
    return {
        "test_ari": float(adjusted_rand_score(test_labels, test_assign)),
        "test_nmi": float(normalized_mutual_info_score(test_labels, test_assign)),
        "test_majority_accuracy": float((predicted == test_labels).mean()),
        "contingency": table.astype(int).tolist(),
        "mapping": mapping.tolist(),
        "per_class": per_class,
        "test_assign": test_assign,
        "predicted": predicted,
    }
