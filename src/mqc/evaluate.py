"""Metrics for QC classification, including error analysis by severity."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Overall and per-class metrics, as plain Python types (JSON-ready)."""
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> list[float]:
    """Bootstrap confidence interval for accuracy.

    With 300 test images, one image is 0.33 percentage points, so two models a
    point or two apart may not really differ. The interval makes that visible.
    """
    rng = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(float)
    idx = rng.integers(0, len(correct), size=(n_resamples, len(correct)))
    accs = correct[idx].mean(axis=1)
    alpha = (1 - level) / 2
    return [float(np.quantile(accs, alpha)), float(np.quantile(accs, 1 - alpha))]


def recall_by_severity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    severity: np.ndarray,
    class_names: list[str],
    n_bins: int = 3,
) -> dict[str, list[dict[str, float]]]:
    """For each degraded class, recall in equal-sized bins from mild to severe.

    Answers the question a lab would ask: "which failures does the model miss?"
    We expect mild degradations to be missed more often than severe ones.
    """
    result = {}
    for label, name in enumerate(class_names):
        mask = (y_true == label) & ~np.isnan(severity)
        if not mask.any():
            continue
        sev, correct = severity[mask], (y_pred[mask] == label)
        order = np.argsort(sev, kind="stable")
        bins = []
        for chunk in np.array_split(order, n_bins):
            bins.append(
                {
                    "severity_min": float(sev[chunk].min()),
                    "severity_max": float(sev[chunk].max()),
                    "n": int(len(chunk)),
                    "recall": float(correct[chunk].mean()),
                }
            )
        result[name] = bins
    return result
