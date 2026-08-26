"""Compare conditional communication-entropy definitions on failed traces."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.reporting import read_jsonl, write_jsonl

ROLES = ("planner", "evidence", "reasoner", "aggregator")


def hop_count(example_id: str) -> int:
    return int(example_id[0])


def response_messages(events):
    return {
        event["sender"]: event["content"]
        for event in events
        if event.get("sender") in ROLES
    }


def role_sources(question: str, messages):
    return {
        "planner": question,
        "evidence": f"{question}\n{messages['planner']}",
        "reasoner": f"{messages['planner']}\n{messages['evidence']}",
        "aggregator": messages["reasoner"],
    }


def atomic_units(text: str) -> list[str]:
    pieces = re.split(r"(?:\n+|(?<=[.!?])\s+|(?=\d+\.\s))", text)
    return [piece.strip() for piece in pieces if len(piece.strip().split()) >= 3]


def alignment_entropy(source_vectors: np.ndarray, target_vectors: np.ndarray) -> float:
    """Mean normalized entropy of target-unit alignments to source units."""
    if len(source_vectors) <= 1 or not len(target_vectors):
        return 0.0
    similarities = target_vectors @ source_vectors.T
    entropies = []
    for row in similarities:
        probabilities = np.exp((row - row.max()) / 0.1)
        probabilities /= probabilities.sum()
        value = -(probabilities * np.log(probabilities + 1e-12)).sum()
        entropies.append(value / math.log(len(source_vectors)))
    return float(np.mean(entropies))


def coverage_loss(source_vectors: np.ndarray, target_vectors: np.ndarray) -> float:
    if not len(source_vectors) or not len(target_vectors):
        return 1.0
    similarities = target_vectors @ source_vectors.T
    source_coverage = similarities.max(axis=0).mean()
    target_support = similarities.max(axis=1).mean()
    return float(1.0 - (source_coverage + target_support) / 2.0)


def z_score(value: float, reference: list[float]) -> float:
    spread = float(np.std(reference)) or 1e-12
    return (value - float(np.mean(reference))) / spread


def kernel_surprisal(
    source: np.ndarray,
    target: np.ndarray,
    reference_sources: np.ndarray,
    reference_targets: np.ndarray,
    bandwidth: float,
    neighbors: int,
    exclude: int | None = None,
) -> tuple[float, float]:
    """Return local conditional and global marginal kernel surprisals."""
    source_distances = np.linalg.norm(reference_sources - source, axis=1)
    available = np.arange(len(reference_sources))
    if exclude is not None:
        available = available[available != exclude]
    nearest = available[np.argsort(source_distances[available])[:neighbors]]
    scale = max(float(bandwidth), 1e-12)

    def surprisal(candidates):
        squared = np.sum((candidates - target) ** 2, axis=1)
        log_kernels = -squared / (2.0 * scale**2)
        maximum = float(log_kernels.max())
        log_density = maximum + math.log(float(np.exp(log_kernels - maximum).mean()))
        return -log_density

    return surprisal(reference_targets[nearest]), surprisal(reference_targets[available])


def fit_transition(source_vectors, target_vectors, clusters: int, seed: int):
    cluster_count = min(clusters, len(source_vectors))
    source_model = KMeans(cluster_count, random_state=seed, n_init=10).fit(source_vectors)
    target_model = KMeans(cluster_count, random_state=seed, n_init=10).fit(target_vectors)
    source_labels = source_model.labels_
    target_labels = target_model.labels_
    counts = np.ones((cluster_count, cluster_count), dtype=float)
    for source_label, target_label in zip(source_labels, target_labels):
        counts[source_label, target_label] += 1.0
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    residuals = target_vectors - source_vectors
    covariance = LedoitWolf().fit(residuals)
    residual_information = covariance.mahalanobis(residuals)
    pairwise = np.linalg.norm(
        target_vectors[:, None, :] - target_vectors[None, :, :], axis=2
    )
    pairwise[pairwise == 0.0] = np.nan
    bandwidth = float(np.nanmedian(np.nanmin(pairwise, axis=1)))
    conditional_surprisals = []
    information_transfers = []
    for index, (source, target) in enumerate(zip(source_vectors, target_vectors)):
        conditional, marginal = kernel_surprisal(
            source,
            target,
            source_vectors,
            target_vectors,
            bandwidth,
            neighbors=min(8, len(source_vectors) - 1),
            exclude=index,
        )
        conditional_surprisals.append(conditional)
        information_transfers.append(marginal - conditional)
    return {
        "source_model": source_model,
        "target_model": target_model,
        "transition_probabilities": probabilities,
        "covariance": covariance,
        "residual_information": residual_information.tolist(),
        "source_vectors": source_vectors,
        "target_vectors": target_vectors,
        "bandwidth": bandwidth,
        "conditional_surprisals": conditional_surprisals,
        "information_transfers": information_transfers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = read_jsonl(args.results)
    traces = read_jsonl(args.traces)
    failures = read_jsonl(args.failures)[: args.limit]
    events_by_run = defaultdict(list)
    for event in traces:
        events_by_run[event.get("run_id", "")].append(event)

    embedder = LocalTransformerEmbedder(args.embedding_model)
    cache = {}

    def embed(text: str):
        if text not in cache:
            cache[text] = embedder(text)
        return cache[text]

    def embed_units(text: str):
        units = atomic_units(text)
        return np.vstack([embed(unit) for unit in units]) if units else np.empty((0, 384))

    references = defaultdict(list)
    for row in results:
        if not bool(row.get("em", False)):
            continue
        messages = response_messages(events_by_run[row["run_id"]])
        sources = role_sources(row["question"], messages)
        hop = hop_count(row["example_id"])
        for role in ROLES:
            source_units = embed_units(sources[role])
            target_units = embed_units(messages[role])
            references[(hop, role)].append(
                {
                    "source": embed(sources[role]),
                    "target": embed(messages[role]),
                    "alignment_entropy": alignment_entropy(source_units, target_units),
                    "coverage_loss": coverage_loss(source_units, target_units),
                }
            )

    fitted = {}
    for key, rows in references.items():
        fitted[key] = fit_transition(
            np.vstack([row["source"] for row in rows]),
            np.vstack([row["target"] for row in rows]),
            args.clusters,
            args.seed,
        )

    diagnostics = []
    for failure in failures:
        messages = response_messages(failure["events"])
        sources = role_sources(failure["question"], messages)
        hop = hop_count(failure["example_id"])
        edge_scores = []
        for role in ROLES:
            rows = references[(hop, role)]
            model = fitted[(hop, role)]
            source = embed(sources[role])
            target = embed(messages[role])
            source_label = int(model["source_model"].predict(source.reshape(1, -1))[0])
            target_label = int(model["target_model"].predict(target.reshape(1, -1))[0])
            transition_surprisal = -math.log(
                model["transition_probabilities"][source_label, target_label]
            )
            residual_information = float(
                model["covariance"].mahalanobis((target - source).reshape(1, -1))[0]
            )
            conditional_surprisal, marginal_surprisal = kernel_surprisal(
                source,
                target,
                model["source_vectors"],
                model["target_vectors"],
                model["bandwidth"],
                neighbors=min(8, len(rows)),
            )
            information_transfer = marginal_surprisal - conditional_surprisal
            source_units = embed_units(sources[role])
            target_units = embed_units(messages[role])
            align = alignment_entropy(source_units, target_units)
            loss = coverage_loss(source_units, target_units)
            edge_scores.append(
                {
                    "role": role,
                    "conditional_transition_surprisal": transition_surprisal,
                    "residual_information_z": z_score(
                        residual_information, model["residual_information"]
                    ),
                    "alignment_entropy_z": z_score(
                        align, [row["alignment_entropy"] for row in rows]
                    ),
                    "coverage_loss_z": z_score(
                        loss, [row["coverage_loss"] for row in rows]
                    ),
                    "local_conditional_surprisal_z": z_score(
                        conditional_surprisal, model["conditional_surprisals"]
                    ),
                    "information_transfer_drop_z": -z_score(
                        information_transfer, model["information_transfers"]
                    ),
                }
            )
        diagnostics.append(
            {
                "example_id": failure["example_id"],
                "question": failure["question"],
                "prediction": failure["prediction"],
                "gold_answer": failure["gold_answer"],
                "edge_scores": edge_scores,
            }
        )
    write_jsonl(args.output, diagnostics)


if __name__ == "__main__":
    main()
