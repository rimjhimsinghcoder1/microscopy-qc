import json

import numpy as np
import pytest
import torch

from mqc.cnn import FusionNet, SmallCNN
from mqc.explain import grad_cam
from mqc.report import build_report, latest_run
from mqc.train import train_models


@pytest.mark.parametrize("net", [SmallCNN(4, width=8), FusionNet(4, 7, width=8)])
def test_grad_cam_shape_and_range(net):
    images = torch.rand(3, 28, 28)
    feats = torch.rand(3, 7) if isinstance(net, FusionNet) else None
    cams = grad_cam(net, images, torch.tensor([0, 1, 2]), feats)
    assert cams.shape == (3, 28, 28)
    assert cams.min() >= 0 and cams.max() <= 1


def test_grad_cam_rejects_other_models():
    with pytest.raises(TypeError):
        grad_cam(torch.nn.Linear(2, 2), torch.rand(1, 28, 28), torch.tensor([0]))


@pytest.fixture
def run_dir(tmp_path, ranges):
    from mqc.dataset import build_qc_dataset, save_dataset, stratified_split

    classes = ["good", "blurry", "overexposed", "noisy"]
    rng = np.random.default_rng(0)
    images = rng.uniform(0.2, 0.7, size=(120, 28, 28)).astype(np.float32)
    ds = build_qc_dataset(images, classes, 30, ranges, seed=0)
    splits = stratified_split(ds["labels"], [0.6, 0.2, 0.2], seed=0)
    path = save_dataset(tmp_path / "qc.npz", ds, splits, {})
    config = {
        "seed": 0,
        "data": {"processed_path": str(path)},
        "qc_classes": classes,
        "models": {
            "logistic_regression": {},
            "cnn": {"width": 8, "epochs": 2, "patience": 2},
        },
        "output_dir": str(tmp_path / "runs"),
    }
    return train_models(config)


def test_report_writes_figures_and_json(run_dir):
    out = build_report(run_dir, target_accuracy=0.9)
    for name in ("reliability", "coverage", "confusion", "lr_weights", "gradcam"):
        assert (out / f"{name}.png").stat().st_size > 0
    res = json.loads((out / "report.json").read_text())
    assert res["recommended_model"] in ("logistic_regression", "cnn")
    assert set(res["models"]) == {"logistic_regression", "cnn"}


def test_report_refuses_changed_dataset(run_dir):
    import yaml

    data_path = yaml.safe_load((run_dir / "config.yaml").read_text())["data"][
        "processed_path"
    ]
    with open(data_path, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(RuntimeError, match="dataset changed"):
        build_report(run_dir)


def test_latest_run_errors_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_run(tmp_path)


def test_results_markdown_is_generated(run_dir, tmp_path):
    from mqc import results_md

    build_report(run_dir, target_accuracy=0.9)
    path = results_md.write(run_dir, tmp_path / "results.md")
    text = path.read_text()
    assert "| Logistic regression" in text
    assert "(recommended)" in text
    assert "Do not edit by hand" in text
