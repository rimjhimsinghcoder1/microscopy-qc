"""Load experiment settings and fix random seeds."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REQUIRED_KEYS = {"seed", "data", "qc_classes", "degradations", "output_dir"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and check that the required keys are present."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"config is missing keys: {sorted(missing)}")
    return config


def set_seed(seed: int) -> None:
    """Make Python, NumPy and PyTorch randomness repeatable.

    Deterministic algorithms make repeated runs on the same machine identical.
    Different hardware or BLAS libraries can still change the last decimals of
    neural-network results; the scikit-learn models match exactly across machines.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
