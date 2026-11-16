"""Interpretable image-quality features.

Each feature measures one thing a lab scientist would check by eye, so a model
built on them can explain its decisions in those same terms.
"""

import numpy as np

FEATURE_NAMES = [
    "laplacian_var",  # sharpness: high for in-focus images
    "gradient_energy",  # sharpness: mean squared edge strength (Tenengrad-style)
    "mean_intensity",  # overall brightness
    "std_intensity",  # contrast
    "p99_intensity",  # brightness of the brightest 1% of pixels
    "saturated_frac",  # fraction of pixels clipped at white
    "noise_sigma",  # estimated noise standard deviation (Immerkaer)
]


def _conv3x3(images: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'Valid' 3x3 convolution over a batch (N, H, W) -> (N, H-2, W-2)."""
    h, w = images.shape[1:]
    out = np.zeros((len(images), h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * images[:, dy : dy + h - 2, dx : dx + w - 2]
    return out


_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
# Immerkaer (1996) "Fast Noise Variance Estimation": this kernel is the
# difference of two Laplacians, so it cancels image structure and keeps noise.
_NOISE = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def laplacian_var(images: np.ndarray) -> np.ndarray:
    return _conv3x3(images, _LAPLACIAN).reshape(len(images), -1).var(axis=1)


def gradient_energy(images: np.ndarray) -> np.ndarray:
    gx = _conv3x3(images, _SOBEL_X)
    gy = _conv3x3(images, _SOBEL_X.T)
    return (gx**2 + gy**2).reshape(len(images), -1).mean(axis=1)


def noise_sigma(images: np.ndarray) -> np.ndarray:
    """Estimate the standard deviation of additive Gaussian noise per image."""
    h, w = images.shape[1:]
    total = np.abs(_conv3x3(images, _NOISE)).reshape(len(images), -1).sum(axis=1)
    return np.sqrt(np.pi / 2) * total / (6 * (w - 2) * (h - 2))


def extract_features(images: np.ndarray) -> np.ndarray:
    """Compute all features for a batch of images (N, H, W) in [0, 1].

    Returns an array of shape (N, len(FEATURE_NAMES)), columns in that order.
    """
    if images.ndim != 3:
        raise ValueError("images must have shape (N, H, W)")
    flat = images.reshape(len(images), -1)
    columns = [
        laplacian_var(images),
        gradient_energy(images),
        flat.mean(axis=1),
        flat.std(axis=1),
        np.percentile(flat, 99, axis=1),
        (flat >= 0.999).mean(axis=1),
        noise_sigma(images),
    ]
    return np.stack(columns, axis=1).astype(np.float32)
