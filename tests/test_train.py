import json

import numpy as np
import pytest

from mqc.dataset import build_qc_dataset, save_dataset, stratified_split
from mqc.evaluate import classification_metrics, recall_by_severity
from mqc.models import build_feature_model
from mqc.train import train_feature_models

CLASSES = ["good", "blurry", "overexposed", "noisy"]


@pytest.fixture
def config(tmp_path, ranges):
    """A tiny processed dataset on disk plus a matching config."""
    rng = np.random.default_rng(0)
    images = rng.uniform(0.2, 0.7, size=(120, 28, 28)).astype(np.float32)
    ds = build_qc_dataset(images, CLASSES, 30, ranges, seed=0)
    splits = stratified_split(ds["labels"], [0.6, 0.2, 0.2], seed=0)
    path = save_dataset(tmp_path / "qc.npz", ds, splits, {})
    return {
        "seed": 0,
        "data": {"processed_path": str(path)},
        "qc_classes": CLASSES,
        "models": {
            "logistic_regression": {"C": 1.0},
            "random_forest": {"n_estimators": 20},
        },
        "output_dir": str(tmp_path / "runs"),
    }


def test_training_writes_expected_files(config, tmp_path):
    run_dir = train_feature_models(config)
    for name in ("config.yaml", "provenance.json", "summary.json"):
        assert (run_dir / name).exists()
    for model in config["models"]:
        assert (run_dir / model / "model.joblib").exists()
        metrics = json.loads((run_dir / model / "metrics.json").read_text())
        assert 0.0 <= metrics["test"]["accuracy"] <= 1.0
        assert "recall_by_severity" in metrics["test"]


def test_models_learn_something(config):
    """Synthetic degradations are easy to spot, so accuracy should beat chance."""
    run_dir = train_feature_models(config)
    summary = json.loads((run_dir / "summary.json").read_text())
    for scores in summary.values():
        assert scores["test"]["accuracy"] > 0.5  # chance is 0.25


def test_training_is_reproducible(config, tmp_path):
    a = train_feature_models(config, run_root=tmp_path / "a")
    b = train_feature_models(config, run_root=tmp_path / "b")
    for model in config["models"]:
        ma = json.loads((a / model / "metrics.json").read_text())
        mb = json.loads((b / model / "metrics.json").read_text())
        assert ma == mb


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        build_feature_model("svm_magic", {}, seed=0)


def test_classification_metrics_perfect():
    y = np.array([0, 1, 2, 3])
    m = classification_metrics(y, y, CLASSES)
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == np.eye(4, dtype=int).tolist()


def test_recall_by_severity_bins():
    y_true = np.array([1] * 6)
    y_pred = np.array([0, 0, 1, 1, 1, 1])  # the two mildest are missed
    severity = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    bins = recall_by_severity(y_true, y_pred, severity, CLASSES)["blurry"]
    assert [b["recall"] for b in bins] == [0.0, 1.0, 1.0]
    assert bins[0]["severity_max"] < bins[-1]["severity_min"]
