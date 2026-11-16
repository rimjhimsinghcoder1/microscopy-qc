"""Download source images and verify them with a checksum."""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

# Source: MedMNIST v3 (https://medmnist.com), CC BY 4.0. URL and MD5 copied from
# the official medmnist package (medmnist/info.py, version 3.0.2).
DATASETS = {
    "bloodmnist": {
        "url": "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
        "md5": "7053d0359d879ad8a5505303e11de1dc",
    },
}

# ITU-R BT.601 luma weights: standard RGB -> grayscale conversion.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def md5sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the MD5 hex digest of a file, read in chunks to save memory."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, raw_dir: str | Path) -> Path:
    """Download a dataset once and check its MD5.

    If a file with the correct checksum already exists it is reused, so running
    this twice does not download twice. A wrong checksum is an error: it means
    the data is not the data this project was built and tested on.
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    info = DATASETS[name]
    path = Path(raw_dir) / f"{name}.npz"

    if path.exists() and md5sum(path) == info["md5"]:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading {name} -> {path}")
    urllib.request.urlretrieve(info["url"], tmp)

    actual = md5sum(tmp)
    if actual != info["md5"]:
        tmp.unlink()
        raise RuntimeError(
            f"checksum mismatch for {name}: expected {info['md5']}, got {actual}"
        )
    tmp.replace(path)
    return path


def to_grayscale(images: np.ndarray) -> np.ndarray:
    """Convert uint8 images (N, H, W[, 3]) to float32 grayscale in [0, 1]."""
    images = images.astype(np.float32) / 255.0
    if images.ndim == 4:
        images = images @ _LUMA
    return images


def load_images(path: str | Path) -> np.ndarray:
    """Load the MedMNIST training images as grayscale float32 (N, H, W).

    Only the official training split is used as the source pool; this project
    makes its own train/val/test split after adding degradations.
    """
    with np.load(path) as npz:
        return to_grayscale(npz["train_images"])
