import numpy as np
import pytest

from mqc.calibration import (
    apply_temperature,
    choose_review_threshold,
    coverage_curve,
    expected_calibration_error,
    fit_temperature,
    review_outcome,
)


def test_ece_zero_when_confidence_matches_accuracy():
    # 4 predictions at 75% confidence, 3 of them right -> perfectly calibrated
    proba = np.array([[0.75, 0.25]] * 4)
    y = np.array([0, 0, 0, 1])
    assert expected_calibration_error(proba, y) == pytest.approx(0.0)


def test_ece_large_when_overconfident():
    proba = np.array([[0.99, 0.01]] * 4)
    y = np.array([0, 1, 1, 1])  # 25% right while claiming 99%
    assert expected_calibration_error(proba, y) == pytest.approx(0.74)


def test_temperature_keeps_ranking_and_sums_to_one():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(4), size=20)
    for t in (0.5, 1.0, 3.0):
        out = apply_temperature(proba, t)
        assert np.allclose(out.sum(axis=1), 1.0)
        assert np.array_equal(out.argmax(axis=1), proba.argmax(axis=1))
    assert np.allclose(apply_temperature(proba, 1.0), proba, atol=1e-9)


def test_fit_temperature_softens_overconfident_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    correct = rng.random(500) < 0.7  # 70% accurate
    pred = np.where(correct, y, 1 - y)
    proba = np.full((500, 2), 0.01)
    proba[np.arange(500), pred] = 0.99  # but always 99% confident
    assert fit_temperature(proba, y) > 1.0


def test_review_threshold_meets_target_on_the_data_it_was_chosen_on():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 400)
    proba = rng.dirichlet(np.ones(4) * 0.5, size=400)
    t = choose_review_threshold(proba, y, target_accuracy=0.6)
    outcome = review_outcome(proba, y, t)
    if outcome["auto_accuracy"] is not None:
        assert outcome["auto_accuracy"] >= 0.6
    assert outcome["errors_caught_by_review"] <= outcome["total_errors"]


def test_unreachable_target_sends_everything_to_review():
    proba = np.array([[0.9, 0.1], [0.8, 0.2]])
    y = np.array([1, 1])  # always wrong
    t = choose_review_threshold(proba, y, target_accuracy=0.99)
    assert review_outcome(proba, y, t)["sent_to_review"] == 2


def test_coverage_curve_ends_at_overall_accuracy():
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
    y = np.array([0, 1, 0, 1])
    c = coverage_curve(proba, y)
    assert c["coverage"][-1] == 1.0
    assert c["accuracy"][-1] == pytest.approx(0.75)
    assert c["accuracy"][0] == 1.0  # most confident prediction is right
