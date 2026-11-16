"""Controlled image degradations that simulate common microscopy QC failures.

Every function takes a float image in [0, 1] and returns a new image in [0, 1]
with the same shape. Randomness only comes from the `rng` passed in, so the same
seed always produces the same degraded image.
"""

from collections.abc import Callable

import numpy as np


def gaussian_kernel(sigma: float) -> np.ndarray:
    """1-D Gaussian kernel covering +/- 3 sigma, normalised to sum to 1."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a 2-D image, simulating an out-of-focus acquisition.

    A 2-D Gaussian is separable, so we blur rows then columns with a 1-D kernel.
    Edges are padded by reflection to avoid dark borders.
    """
    kernel = gaussian_kernel(sigma)
    r = len(kernel) // 2
    padded = np.pad(image, r, mode="reflect")
    rows = np.apply_along_axis(np.convolve, 1, padded, kernel, mode="valid")
    both = np.apply_along_axis(np.convolve, 0, rows, kernel, mode="valid")
    return both.astype(np.float32)


def overexpose(image: np.ndarray, gain: float) -> np.ndarray:
    """Multiply brightness and clip, simulating too-long exposure or high gain."""
    if gain <= 1:
        raise ValueError("gain must be > 1 to overexpose")
    return np.clip(image * gain, 0.0, 1.0).astype(np.float32)


def add_noise(image: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise, simulating low light or sensor noise."""
    if std <= 0:
        raise ValueError("std must be positive")
    noisy = image + rng.normal(0.0, std, size=image.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


# Each QC class -> (config key for its severity range, function applying it).
# "good" has no entry: good images are left unchanged.
Degrader = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]
DEGRADATIONS: dict[str, tuple[str, Degrader]] = {
    "blurry": ("blur_sigma", lambda img, s, rng: gaussian_blur(img, s)),
    "overexposed": ("exposure_gain", lambda img, s, rng: overexpose(img, s)),
    "noisy": ("noise_std", add_noise),
}


def apply_qc_class(
    image: np.ndarray,
    qc_class: str,
    ranges: dict[str, list[float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Degrade `image` according to `qc_class`.

    Returns the image and the severity used (NaN for "good"), so every sample
    records exactly how it was made.
    """
    if qc_class == "good":
        return image.astype(np.float32), float("nan")
    if qc_class not in DEGRADATIONS:
        raise ValueError(f"unknown QC class {qc_class!r}")
    key, fn = DEGRADATIONS[qc_class]
    low, high = ranges[key]
    severity = float(rng.uniform(low, high))
    return fn(image, severity, rng), severity
