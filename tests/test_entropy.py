import numpy as np
import pytest

from entroflow.entropy import RollingEdgeEntropy, normalized_entropy, semantic_entropy


def test_entropy_is_zero_for_one_cluster():
    assert normalized_entropy([0, 0, 0]) == 0.0


def test_semantic_entropy_uses_adaptive_similarity_clusters():
    vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    entropy, clusters, labels = semantic_entropy(vectors, similarity_threshold=0.9)
    assert clusters == 2
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert 0.0 < entropy <= 1.0


def test_monitor_waits_for_a_full_minimum_window():
    monitor = RollingEdgeEntropy(window_size=4, min_samples=3)
    assert not monitor.observe("a_to_b", np.array([0.0, 0.0])).ready
    assert not monitor.observe("a_to_b", np.array([1.0, 0.0])).ready
    event = monitor.observe("a_to_b", np.array([0.0, 1.0]))
    assert event.ready
    assert event.sample_count == 3
    assert event.delta is None


def test_monitor_reports_a_delta_after_the_first_ready_event():
    monitor = RollingEdgeEntropy(window_size=4, min_samples=3)
    for vector in (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])):
        monitor.observe("retriever_to_reasoner", vector)
    event = monitor.observe("retriever_to_reasoner", np.array([1.0, 1.0]))
    assert event.ready
    assert event.delta is not None


def test_monitor_rejects_embedding_shape_changes():
    monitor = RollingEdgeEntropy(window_size=3, min_samples=2)
    monitor.observe("a_to_b", np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="same shape"):
        monitor.observe("a_to_b", np.array([0.0, 0.0, 0.0]))
