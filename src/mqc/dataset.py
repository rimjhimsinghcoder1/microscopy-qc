"""Build the labelled QC dataset, split it, and save it with provenance metadata."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mqc import __version__
from mqc.degradations import apply_qc_class


def build_qc_dataset(
    images: np.ndarray,
    qc_classes: list[str],
    n_per_class: int,
    ranges: dict[str, list[float]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Create a balanced dataset of good and degraded images.

    Each source image is used at most once, so the same cell never appears in
    two classes (that would let a model cheat by memorising cells).
    """
    total = n_per_class * len(qc_classes)
    if total > len(images):
        raise ValueError(f"need {total} source images, only {len(images)} available")

    rng = np.random.default_rng(seed)
    source_index = rng.choice(len(images), size=total, replace=False)
    labels = np.repeat(np.arange(len(qc_classes)), n_per_class)

    out = np.empty((total, *images.shape[1:]), dtype=np.float32)
    severity = np.empty(total, dtype=np.float32)
    for i, (src, label) in enumerate(zip(source_index, labels, strict=True)):
        out[i], severity[i] = apply_qc_class(
            images[src], qc_classes[label], ranges, rng
        )

    return {
        "images": out,
        "labels": labels,
        "severity": severity,
        "source_index": source_index,
    }


def stratified_split(
    labels: np.ndarray, fractions: list[float], seed: int
) -> dict[str, np.ndarray]:
    """Split indices into train/val/test, keeping class proportions in each part."""
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must be three numbers summing to 1")

    rng = np.random.default_rng(seed)
    parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for cls in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == cls))
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train : n_train + n_val])
        parts["test"].append(idx[n_train + n_val :])
    return {name: np.sort(np.concatenate(chunks)) for name, chunks in parts.items()}


def save_dataset(
    path: str | Path,
    dataset: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    """Save arrays to .npz and a human-readable metadata .json next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset, **{f"split_{k}": v for k, v in splits.items()})

    metadata = {
        **metadata,
        "mqc_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_samples": int(len(dataset["labels"])),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
    }
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
