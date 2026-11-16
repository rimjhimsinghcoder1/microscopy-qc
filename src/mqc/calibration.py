"""Is the model's confidence trustworthy, and when should a human take over?

A QC system that says "95% sure" should be right about 95% of the time. If it
is, we can let it decide confident cases alone and send the rest to a person.
"""

from typing import Any

import numpy as np

_EPS = 1e-12


def expected_calibration_error(
    proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> float:
    """Average gap between confidence and accuracy, weighted by bin size."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_bins(
    proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin confidence vs accuracy, for a reliability diagram."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            bins.append(
                {
                    "confidence": float(conf[in_bin].mean()),
                    "accuracy": float(correct[in_bin].mean()),
                    "n": int(in_bin.sum()),
                }
            )
    return bins


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Soften (T > 1) or sharpen (T < 1) probabilities without changing the ranking.

    log(proba) equals the model's logits up to a per-row constant, which
    cancels in the softmax, so this works for any model that outputs
    probabilities, not only neural networks.
    """
    logits = np.log(proba + _EPS) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(proba: np.ndarray, y_true: np.ndarray) -> float:
    return float(-np.log(proba[np.arange(len(y_true)), y_true] + _EPS).mean())


def fit_temperature(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    """Pick the temperature that minimises validation log-loss.

    A plain grid search: one parameter, deterministic, easy to check.
    """
    grid = np.geomspace(0.2, 10.0, 200)
    losses = [_nll(apply_temperature(proba_val, t), y_val) for t in grid]
    return float(grid[int(np.argmin(losses))])


def choose_review_threshold(
    proba_val: np.ndarray, y_val: np.ndarray, target_accuracy: float
) -> float:
    """Lowest confidence threshold whose auto-decided validation cases meet the target.

    Images below the threshold go to a human. A lower threshold means fewer
    images for the human, so we take the lowest one that still meets the target.
    """
    conf = proba_val.max(axis=1)
    correct = proba_val.argmax(axis=1) == y_val
    for t in np.unique(conf):  # ascending
        accepted = conf >= t
        if correct[accepted].mean() >= target_accuracy:
            return float(t)
    return 1.0 + _EPS  # target unreachable: send everything to review


def review_outcome(
    proba: np.ndarray, y_true: np.ndarray, threshold: float
) -> dict[str, Any]:
    """What happens on new data when we apply the review threshold."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    accepted = conf >= threshold
    errors = pred != y_true
    return {
        "threshold": float(threshold),
        "auto_decided_frac": float(accepted.mean()),
        "sent_to_review": int((~accepted).sum()),
        "auto_accuracy": float((~errors[accepted]).mean()) if accepted.any() else None,
        "total_errors": int(errors.sum()),
        "errors_caught_by_review": int((errors & ~accepted).sum()),
    }


def coverage_curve(proba: np.ndarray, y_true: np.ndarray) -> dict[str, list[float]]:
    """Accuracy of auto-decided images as more low-confidence ones go to review."""
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true)[np.argsort(-conf, kind="stable")]
    n = np.arange(1, len(correct) + 1)
    return {
        "coverage": (n / len(correct)).tolist(),
        "accuracy": (np.cumsum(correct) / n).tolist(),
    }
