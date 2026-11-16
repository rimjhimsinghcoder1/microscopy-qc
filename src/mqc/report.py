"""Turn a training run into a decision-support report: calibration, review policy,
explanations and figures. Reads only what the run saved, plus the dataset."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import yaml

from mqc import figures
from mqc.calibration import (
    apply_temperature,
    choose_review_threshold,
    coverage_curve,
    expected_calibration_error,
    fit_temperature,
    reliability_bins,
    review_outcome,
)
from mqc.data import md5sum
from mqc.explain import grad_cam
from mqc.features import FEATURE_NAMES, extract_features
from mqc.torch_train import TORCH_MODELS, load_torch_classifier
from mqc.train import load_processed

LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "cnn": "CNN",
    "fusion": "Fusion",
}


def latest_run(output_dir: str | Path) -> Path:
    runs = sorted(p for p in Path(output_dir).iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"no runs in {output_dir}; run `mqc train` first")
    return runs[-1]


def _check_dataset(run_dir: Path, data_path: str) -> None:
    """Refuse to report on a dataset that differs from the one used in training."""
    expected = json.loads((run_dir / "provenance.json").read_text())["dataset_md5"]
    actual = md5sum(data_path)
    if actual != expected:
        raise RuntimeError(
            f"dataset changed since training (md5 {actual} != {expected}); "
            "retrain before reporting"
        )


def _gradcam_examples(run_dir, data, X, classes, out_dir) -> Path | None:
    """Two correctly classified test images per class, plus up to two errors."""
    if not (run_dir / "cnn" / "model.pt").exists():
        return None
    model = load_torch_classifier(
        run_dir / "cnn" / "model.pt", "cnn", len(classes), X.shape[1]
    )
    test = data["split_test"]
    pred = model.predict(data["images"][test], X[test])
    y = data["labels"][test]

    picks, titles = [], []
    for c, name in enumerate(classes):
        ok = test[(y == c) & (pred == c)][:2]
        picks += ok.tolist()
        titles += [name] * len(ok)
    for i in np.flatnonzero(pred != y)[:2]:
        picks.append(int(test[i]))
        titles.append(f"{classes[y[i]]}\n-> {classes[pred[i]]} (error)")

    idx = np.array(picks)
    images = torch.from_numpy(data["images"][idx])
    target = torch.from_numpy(model.predict(data["images"][idx], X[idx]))
    cams = grad_cam(model.net, images, target)
    return figures.gradcam_grid(
        data["images"][idx], cams, titles, out_dir / "gradcam.png"
    )


def build_report(run_dir: str | Path, target_accuracy: float = 0.99) -> Path:
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    classes = config["qc_classes"]
    data_path = config["data"]["processed_path"]
    _check_dataset(run_dir, data_path)
    data = load_processed(data_path)
    y = data["labels"]

    out_dir = run_dir / "report"
    out_dir.mkdir(exist_ok=True)

    results: dict[str, Any] = {"target_auto_accuracy": target_accuracy, "models": {}}
    coverage = {}
    probs_test = {}
    for name in config["models"]:
        with np.load(run_dir / name / "predictions.npz") as p:
            pv, pt = p["val_proba"], p["test_proba"]
            yv, yt = y[p["val_index"]], y[p["test_index"]]

        # Temperature is fitted on validation and applied unchanged to test.
        temp = fit_temperature(pv, yv)
        pv_cal, pt_cal = apply_temperature(pv, temp), apply_temperature(pt, temp)
        threshold = choose_review_threshold(pv_cal, yv, target_accuracy)
        results["models"][name] = {
            "temperature": temp,
            "test_ece_before": expected_calibration_error(pt, yt),
            "test_ece_after": expected_calibration_error(pt_cal, yt),
            "review_policy_val": review_outcome(pv_cal, yv, threshold),
            "review_policy_test": review_outcome(pt_cal, yt, threshold),
        }
        probs_test[name] = (pt, pt_cal, yt)
        coverage[LABELS.get(name, name)] = coverage_curve(pt_cal, yt)

    # Recommend the model that automates the most on VALIDATION while meeting
    # the target there. Choosing on test would make the test numbers optimistic.
    best = max(
        results["models"],
        key=lambda m: results["models"][m]["review_policy_val"]["auto_decided_frac"],
    )
    rp = results["models"][best]["review_policy_test"]
    results["recommended_model"] = best

    pt, pt_cal, yt = probs_test[best]
    r = results["models"][best]
    figures.reliability_diagram(
        {
            f"before (ECE {r['test_ece_before']:.3f})": reliability_bins(pt, yt),
            f"after, T={r['temperature']:.2f} (ECE {r['test_ece_after']:.3f})": (
                reliability_bins(pt_cal, yt)
            ),
        },
        out_dir / "reliability.png",
        title=f"{LABELS.get(best, best)}: calibration on test",
    )
    figures.coverage_plot(
        coverage,
        out_dir / "coverage.png",
        marker={
            "xy": (rp["auto_decided_frac"], rp["auto_accuracy"]),
            "text": f"{LABELS.get(best, best)} review policy:\n"
            f"{rp['auto_decided_frac']:.0%} automatic, "
            f"{rp['auto_accuracy']:.1%} correct",
        },
    )
    metrics = json.loads((run_dir / best / "metrics.json").read_text())
    figures.confusion_plot(
        np.array(metrics["test"]["confusion_matrix"]),
        classes,
        f"{LABELS.get(best, best)}: test confusion matrix",
        out_dir / "confusion.png",
    )
    if (run_dir / "logistic_regression" / "model.joblib").exists():
        lr = joblib.load(run_dir / "logistic_regression" / "model.joblib")
        coefs = lr.named_steps["clf"].coef_
        figures.coefficient_plot(
            {
                c: dict(zip(FEATURE_NAMES, row, strict=True))
                for c, row in zip(classes, coefs, strict=True)
            },
            out_dir / "lr_weights.png",
        )
    if any(m in TORCH_MODELS for m in config["models"]):
        _gradcam_examples(
            run_dir, data, extract_features(data["images"]), classes, out_dir
        )

    (out_dir / "report.json").write_text(json.dumps(results, indent=2))
    return out_dir
