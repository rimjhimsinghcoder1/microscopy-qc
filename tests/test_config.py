import numpy as np
import pytest

from mqc.config import load_config, set_seed


def test_baseline_config_loads():
    config = load_config("configs/baseline.yaml")
    assert config["seed"] == 42
    assert "good" in config["qc_classes"]


def test_missing_keys_raise(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: 1\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_config(bad)


def test_same_seed_gives_same_numbers():
    set_seed(123)
    first = np.random.rand(5)
    set_seed(123)
    second = np.random.rand(5)
    assert np.array_equal(first, second)
