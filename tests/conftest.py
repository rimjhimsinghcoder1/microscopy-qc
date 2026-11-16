"""Shared test helpers. Tests use small synthetic images, so no download is needed."""

import numpy as np
import pytest


def sharpness(image: np.ndarray) -> float:
    """Variance of a simple Laplacian: high for sharp images, low for blurry ones."""
    lap = (
        -4 * image[1:-1, 1:-1]
        + image[:-2, 1:-1]
        + image[2:, 1:-1]
        + image[1:-1, :-2]
        + image[1:-1, 2:]
    )
    return float(lap.var())


@pytest.fixture
def image() -> np.ndarray:
    """A 28x28 textured image with values spread across (0, 1)."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.6, size=(28, 28)).astype(np.float32)


@pytest.fixture
def image_stack() -> np.ndarray:
    """40 small textured images, enough to build a tiny dataset."""
    rng = np.random.default_rng(1)
    return rng.uniform(0.2, 0.6, size=(40, 28, 28)).astype(np.float32)


@pytest.fixture
def ranges() -> dict[str, list[float]]:
    return {
        "blur_sigma": [0.8, 2.0],
        "exposure_gain": [1.5, 2.5],
        "noise_std": [0.05, 0.15],
    }
