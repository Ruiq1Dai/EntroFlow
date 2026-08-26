import numpy as np

from experiments.musique.analyze_conditional_entropy import (
    alignment_entropy,
    atomic_units,
    coverage_loss,
    kernel_surprisal,
)


def test_atomic_units_split_numbered_reasoning_steps():
    assert len(atomic_units("1. Find the city. 2. Find its country.")) == 2


def test_alignment_entropy_detects_ambiguous_mapping():
    source = np.array([[1.0, 0.0], [1.0, 0.0]])
    target = np.array([[1.0, 0.0]])
    assert alignment_entropy(source, target) > 0.99


def test_coverage_loss_is_low_for_identical_supported_units():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert coverage_loss(vectors, vectors) == 0.0


def test_kernel_surprisal_is_lower_for_expected_conditional_target():
    sources = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    targets = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    expected, _ = kernel_surprisal(
        sources[0], targets[0], sources, targets, bandwidth=0.2, neighbors=2
    )
    unexpected, _ = kernel_surprisal(
        sources[0], targets[2], sources, targets, bandwidth=0.2, neighbors=2
    )
    assert expected < unexpected
