import numpy as np
import pytest
from conftest import sharpness

from mqc.degradations import (
    add_noise,
    apply_qc_class,
    gaussian_blur,
    gaussian_kernel,
    overexpose,
)


def test_kernel_sums_to_one():
    assert np.isclose(gaussian_kernel(1.5).sum(), 1.0)


def test_blur_keeps_shape_and_reduces_sharpness(image):
    blurred = gaussian_blur(image, sigma=1.5)
    assert blurred.shape == image.shape
    assert sharpness(blurred) < 0.5 * sharpness(image)


def test_stronger_blur_is_blurrier(image):
    assert sharpness(gaussian_blur(image, 2.0)) < sharpness(gaussian_blur(image, 1.0))


def test_blur_of_constant_image_is_unchanged():
    flat = np.full((28, 28), 0.4, dtype=np.float32)
    assert np.allclose(gaussian_blur(flat, 2.0), flat)


def test_overexpose_brightens_and_saturates(image):
    bright = overexpose(image, gain=2.5)
    assert bright.mean() > image.mean()
    assert bright.max() <= 1.0
    assert (bright == 1.0).mean() > (image == 1.0).mean()


def test_noise_is_reproducible_with_same_seed(image):
    a = add_noise(image, 0.1, np.random.default_rng(7))
    b = add_noise(image, 0.1, np.random.default_rng(7))
    c = add_noise(image, 0.1, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_increases_sharpness(image):
    assert sharpness(add_noise(image, 0.1, np.random.default_rng(0))) > sharpness(image)


@pytest.mark.parametrize("qc_class", ["good", "blurry", "overexposed", "noisy"])
def test_outputs_stay_in_valid_range(image, ranges, qc_class):
    out, _ = apply_qc_class(image, qc_class, ranges, np.random.default_rng(0))
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_good_is_unchanged_and_has_no_severity(image, ranges):
    out, severity = apply_qc_class(image, "good", ranges, np.random.default_rng(0))
    assert np.array_equal(out, image)
    assert np.isnan(severity)


def test_severity_is_inside_configured_range(image, ranges):
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, s = apply_qc_class(image, "blurry", ranges, rng)
        assert ranges["blur_sigma"][0] <= s <= ranges["blur_sigma"][1]


@pytest.mark.parametrize(
    "call",
    [
        lambda img: gaussian_blur(img, 0),
        lambda img: overexpose(img, 0.9),
        lambda img: add_noise(img, 0, np.random.default_rng(0)),
    ],
)
def test_invalid_parameters_raise(image, call):
    with pytest.raises(ValueError):
        call(image)


def test_unknown_class_raises(image, ranges):
    with pytest.raises(ValueError, match="unknown QC class"):
        apply_qc_class(image, "scratched", ranges, np.random.default_rng(0))
