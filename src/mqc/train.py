"""Train and evaluate all models in the config; save everything to a run folder."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml

from mqc import provenance
from mqc.evaluate import bootstrap_ci, classification_metrics, recall_by_severity
from mqc.features import FEATURE_NAMES, extract_features
from mqc.models import build_feature_model
from mqc.torch_train import TORCH_MODELS, train_torch_model


def load_processed(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def feature_importance(model, class_names: list[str]) -> dict[str, Any]:
    """Model-specific importance, so a reader can see what each model relies on."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):  # logistic regression: one weight per class/feature
        return {
            cls: dict(zip(FEATURE_NAMES, map(float, row), strict=True))
            for cls, row in zip(class_names, clf.coef_, strict=True)
        }
    if hasattr(clf, "feature_importances_"):  # random forest: one value per feature
        return dict(
            zip(FEATURE_NAMES, map(float, clf.feature_importances_), strict=True)
        )
    return {}


def _fit(name, params, data, X, n_classes, seed, out: Path):
    """Train one model, save it, and return a function that gives probabilities."""
    if name in TORCH_MODELS:
        model = train_torch_model(name, params, data, X, n_classes, seed)
        torch.save(
            {
                "state_dict": model.net.state_dict(),
                # Stored as tensors so the file loads with weights_only=True,
                # which refuses to run arbitrary pickled code.
                "feature_mean": torch.from_numpy(model.feature_mean),
                "feature_std": torch.from_numpy(model.feature_std),
                "params": params,
            },
            out / "model.pt",
        )
        (out / "history.json").write_text(json.dumps(model.history, indent=2))
        return lambda idx: model.predict_proba(data["images"][idx], X[idx]), {}

    model = build_feature_model(name, params, seed)
    model.fit(X[data["split_train"]], data["labels"][data["split_train"]])
    joblib.dump(model, out / "model.joblib")
    return lambda idx: model.predict_proba(X[idx]), model


def train_models(config: dict[str, Any], run_root: str | Path | None = None) -> Path:
    """Train every model listed in the config and write one folder per model.

    All models are trained on `train`. The PyTorch models also use `val` for
    early stopping. `test` is only used for the final report. Hyperparameters
    come from the config and are not tuned on the test set.
    """
    data_path = config["data"]["processed_path"]
    data = load_processed(data_path)
    class_names = config["qc_classes"]
    X = extract_features(data["images"])
    y = data["labels"]

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_root or config["output_dir"]) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance.collect(data_path), indent=2)
    )

    summary = {}
    for name, params in config["models"].items():
        out = run_dir / name
        out.mkdir()
        predict_proba, fitted = _fit(
            name, params or {}, data, X, len(class_names), config["seed"], out
        )

        results: dict[str, Any] = {}
        probs = {}
        for split in ("val", "test"):
            idx = data[f"split_{split}"]
            proba = predict_proba(idx)
            pred = proba.argmax(axis=1)
            probs[f"{split}_proba"], probs[f"{split}_index"] = proba, idx
            results[split] = classification_metrics(y[idx], pred, class_names)
            results[split]["accuracy_95ci"] = bootstrap_ci(
                y[idx], pred, seed=config["seed"]
            )
            results[split]["recall_by_severity"] = recall_by_severity(
                y[idx], pred, data["severity"][idx], class_names
            )
        if fitted:
            results["feature_importance"] = feature_importance(fitted, class_names)

        # Saved probabilities let later steps (calibration, explanations,
        # error analysis) reuse predictions without retraining.
        np.savez_compressed(out / "predictions.npz", **probs)
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        summary[name] = {
            split: {
                "accuracy": round(results[split]["accuracy"], 4),
                "accuracy_95ci": [round(v, 4) for v in results[split]["accuracy_95ci"]],
                "macro_f1": round(results[split]["macro_f1"], 4),
            }
            for split in ("val", "test")
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir
