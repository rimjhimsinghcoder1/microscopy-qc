"""Load experiment settings and fix random seeds."""

import random
from pathlib import Path
from typing import Any

import numpy as np
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
    """Make Python and NumPy randomness repeatable.

    PyTorch seeding is added in Phase 4, when PyTorch becomes a dependency.
    """
    random.seed(seed)
    np.random.seed(seed)
