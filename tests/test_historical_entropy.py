import numpy as np

from experiments.musique.analyze_historical_entropy import (
    calibrated_z,
    neighbor_failure_statistics,
    prefix_vector,
)


def test_neighbor_failure_information_increases_with_failed_neighbors():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    low = neighbor_failure_statistics(
        np.array([1.0, 0.0]), vectors, np.array([0.0, 0.0, 1.0]), neighbors=2
    )
    high = neighbor_failure_statistics(
        np.array([1.0, 0.0]), vectors, np.array([1.0, 1.0, 0.0]), neighbors=2
    )
    assert low["failure_information"] < high["failure_information"]


def test_prefix_vector_appends_messages_in_order():
    result = prefix_vector(np.array([1.0]), [np.array([2.0]), np.array([3.0])])
    assert result.tolist() == [1.0, 2.0, 3.0]


def test_calibrated_z_is_relative_to_role_reference_distribution():
    assert calibrated_z(3.0, [1.0, 2.0, 3.0]) > 1.0
