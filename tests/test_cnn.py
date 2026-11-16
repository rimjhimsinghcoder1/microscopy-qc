import numpy as np
import pytest
import torch

from mqc.cnn import FusionNet, SmallCNN
from mqc.dataset import build_qc_dataset, stratified_split
from mqc.features import extract_features
from mqc.torch_train import train_torch_model

CLASSES = ["good", "blurry", "overexposed", "noisy"]
FAST = {"width": 8, "epochs": 3, "patience": 3, "batch_size": 32}


def test_cnn_output_shape():
    out = SmallCNN(n_classes=4, width=8)(torch.rand(5, 28, 28))
    assert out.shape == (5, 4)


def test_fusion_output_shape():
    out = FusionNet(n_classes=4, n_features=7, width=8)(
        torch.rand(5, 28, 28), torch.rand(5, 7)
    )
    assert out.shape == (5, 4)


def test_fusion_requires_features():
    with pytest.raises(ValueError, match="needs QC features"):
        FusionNet(4, 7, width=8)(torch.rand(2, 28, 28))


@pytest.fixture
def tiny(ranges):
    rng = np.random.default_rng(0)
    images = rng.uniform(0.2, 0.7, size=(80, 28, 28)).astype(np.float32)
    data = build_qc_dataset(images, CLASSES, 20, ranges, seed=0)
    data.update(
        {
            f"split_{k}": v
            for k, v in stratified_split(data["labels"], [0.6, 0.2, 0.2], 0).items()
        }
    )
    return data, extract_features(data["images"])


@pytest.mark.parametrize("name", ["cnn", "fusion"])
def test_probabilities_are_valid(tiny, name):
    data, X = tiny
    model = train_torch_model(name, FAST, data, X, n_classes=4, seed=0)
    proba = model.predict_proba(data["images"][:10], X[:10])
    assert proba.shape == (10, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert 1 <= len(model.history) <= FAST["epochs"]


def test_training_is_reproducible(tiny):
    data, X = tiny
    a = train_torch_model("fusion", FAST, data, X, n_classes=4, seed=1)
    b = train_torch_model("fusion", FAST, data, X, n_classes=4, seed=1)
    assert np.array_equal(
        a.predict_proba(data["images"], X), b.predict_proba(data["images"], X)
    )
    assert a.history == b.history


def test_feature_scaling_uses_train_split_only(tiny):
    """Scaling statistics must not see validation or test data (no leakage)."""
    data, X = tiny
    model = train_torch_model("fusion", FAST, data, X, n_classes=4, seed=0)
    assert np.allclose(model.feature_mean, X[data["split_train"]].mean(axis=0))


def test_unknown_torch_model_raises(tiny):
    data, X = tiny
    with pytest.raises(ValueError, match="unknown torch model"):
        train_torch_model("transformer", FAST, data, X, n_classes=4, seed=0)
