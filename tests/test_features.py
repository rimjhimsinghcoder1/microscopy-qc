import numpy as np
import pytest

from mqc.degradations import add_noise, gaussian_blur, overexpose
from mqc.features import FEATURE_NAMES, extract_features, noise_sigma


def col(name):
    return FEATURE_NAMES.index(name)


def test_shape_and_names(image_stack):
    X = extract_features(image_stack)
    assert X.shape == (len(image_stack), len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_rejects_wrong_shape(image):
    with pytest.raises(ValueError):
        extract_features(image)


def test_blur_lowers_both_sharpness_features(image):
    sharp, blurred = extract_features(np.stack([image, gaussian_blur(image, 1.0)]))
    assert blurred[col("laplacian_var")] < sharp[col("laplacian_var")]
    assert blurred[col("gradient_energy")] < sharp[col("gradient_energy")]


def test_overexposure_raises_brightness_and_saturation(image):
    normal, bright = extract_features(np.stack([image, overexpose(image, 2.0)]))
    assert bright[col("mean_intensity")] > normal[col("mean_intensity")]
    assert bright[col("saturated_frac")] > normal[col("saturated_frac")]


def test_noise_estimate_is_close_to_true_noise():
    """On a flat image the only signal is noise, so the estimate should match it."""
    flat = np.full((1, 64, 64), 0.5, dtype=np.float32)
    for std in (0.02, 0.05):
        noisy = add_noise(flat[0], std, np.random.default_rng(0))[None]
        assert noise_sigma(noisy)[0] == pytest.approx(std, rel=0.1)


def test_noise_estimate_ignores_smooth_gradient():
    ramp = np.tile(np.linspace(0.1, 0.9, 32, dtype=np.float32), (32, 1))[None]
    assert noise_sigma(ramp)[0] < 1e-5
