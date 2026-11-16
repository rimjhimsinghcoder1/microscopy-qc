"""Train and evaluate the feature-based models; save everything to a run folder."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from mqc import provenance
from mqc.evaluate import classification_metrics, recall_by_severity
from mqc.features import FEATURE_NAMES, extract_features
from mqc.models import build_feature_model


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


def train_feature_models(
    config: dict[str, Any], run_root: str | Path | None = None
) -> Path:
    """Train every model listed in the config and write one folder per model.

    Models are trained on `train`; `val` and `test` are only used for reporting.
    Hyperparameters come from the config and are not tuned on the test set.
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
        model = build_feature_model(name, params or {}, config["seed"])
        train = data["split_train"]
        model.fit(X[train], y[train])

        results: dict[str, Any] = {}
        for split in ("val", "test"):
            idx = data[f"split_{split}"]
            pred = model.predict(X[idx])
            results[split] = classification_metrics(y[idx], pred, class_names)
            results[split]["recall_by_severity"] = recall_by_severity(
                y[idx], pred, data["severity"][idx], class_names
            )
        results["feature_importance"] = feature_importance(model, class_names)

        out = run_dir / name
        out.mkdir()
        joblib.dump(model, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        summary[name] = {
            split: {k: round(results[split][k], 4) for k in ("accuracy", "macro_f1")}
            for split in ("val", "test")
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir
