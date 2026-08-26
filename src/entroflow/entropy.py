"""Framework-independent rolling semantic entropy for communication edges."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans


def normalized_entropy(labels: Iterable[int]) -> float:
    """Return Shannon entropy normalized to [0, 1] for cluster labels."""
    labels = list(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(unique)))


@dataclass(frozen=True)
class EdgeEntropyEvent:
    edge: str
    sample_count: int
    cluster_count: int
    entropy: float | None
    delta: float | None
    ready: bool


class RollingEdgeEntropy:
    """Estimate entropy from a bounded history of embeddings per edge.

    The host framework owns embedding generation and repair decisions. This
    class intentionally has no assumptions about agent names or graph layout.
    """

    def __init__(self, window_size: int = 32, min_samples: int = 8,
                 max_clusters: int = 4, random_state: int = 42):
        if window_size < min_samples or min_samples < 2:
            raise ValueError("window_size must be >= min_samples >= 2")
        if max_clusters < 2:
            raise ValueError("max_clusters must be at least 2")
        self.window_size = window_size
        self.min_samples = min_samples
        self.max_clusters = max_clusters
        self.random_state = random_state
        self._history: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._previous: dict[str, float] = {}

    def observe(self, edge: str, embedding: np.ndarray) -> EdgeEntropyEvent:
        vector = np.asarray(embedding, dtype=float)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        history = self._history[edge]
        if history and history[0].shape != vector.shape:
            raise ValueError("all embeddings for an edge must have the same shape")
        history.append(vector)
        count = len(history)
        if count < self.min_samples:
            return EdgeEntropyEvent(edge, count, 0, None, None, False)

        vectors = np.vstack(history)
        unique_count = len(np.unique(vectors, axis=0))
        clusters = min(self.max_clusters, count - 1, unique_count)
        if clusters < 2:
            entropy = 0.0
            cluster_count = 1
        else:
            labels = KMeans(
                n_clusters=clusters, random_state=self.random_state, n_init=10
            ).fit_predict(vectors)
            entropy = normalized_entropy(labels)
            cluster_count = clusters
        previous = self._previous.get(edge)
        self._previous[edge] = entropy
        return EdgeEntropyEvent(
            edge=edge,
            sample_count=count,
            cluster_count=cluster_count,
            entropy=entropy,
            delta=None if previous is None else entropy - previous,
            ready=True,
        )


def semantic_entropy(embeddings: np.ndarray, similarity_threshold: float = 0.82) -> tuple[float, int, list[int]]:
    """Compute within-decision semantic entropy over sampled outputs."""
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or len(vectors) < 2:
        return 0.0, 1, [0] * len(vectors)
    if not 0 < similarity_threshold < 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
    ).fit_predict(vectors)
    clusters = len(np.unique(labels))
    if clusters < 2:
        return 0.0, 1, labels.tolist()
    counts = np.bincount(labels, minlength=clusters)
    probs = counts[counts > 0] / counts.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(clusters))
    return entropy, clusters, labels.tolist()
