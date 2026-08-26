import numpy as np
import pytest

from experiments.musique.analyze_topology_entropy import (
    distribution_metrics,
    fixed_entropy,
    permutation_p_value,
)


def test_fixed_entropy_uses_shared_category_space():
    assert fixed_entropy([0, 0, 0], category_count=3) == 0.0
    assert fixed_entropy([0, 1, 2], category_count=3) == pytest.approx(1.0)


def test_js_divergence_detects_shift_without_entropy_increase():
    metrics = distribution_metrics(
        np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), category_count=2
    )
    assert metrics["entropy_delta"] == 0.0
    assert metrics["js_divergence"] > 0.99


def test_permutation_test_detects_complete_distribution_separation():
    value = permutation_p_value(
        np.zeros(20, dtype=int), np.ones(20, dtype=int), 2, seed=42, samples=200
    )
    assert value < 0.05
