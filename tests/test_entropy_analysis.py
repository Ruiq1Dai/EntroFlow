import numpy as np

from experiments.musique.analyze_entropy import assignment_entropy, fit_reference


def test_assignment_entropy_is_high_for_ambiguous_equal_distances():
    assert assignment_entropy(np.array([1.0, 1.0, 1.0]), temperature=1.0) > 0.99


def test_assignment_entropy_is_low_for_one_clear_cluster():
    assert assignment_entropy(np.array([0.0, 10.0, 10.0]), temperature=1.0) < 0.01


def test_reference_fit_returns_normalized_entropy_and_distance_scale():
    reference = fit_reference(
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]),
        clusters=2,
        seed=42,
    )
    assert 0.0 <= reference["baseline_entropy"] <= 1.0
    assert reference["temperature"] > 0.0
