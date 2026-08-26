"""Passive failure-conditional entropy from historical AutoGen traces."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np

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


def neighbor_failure_statistics(
    query: np.ndarray,
    vectors: np.ndarray,
    labels: np.ndarray,
    neighbors: int,
    alpha: float = 1.0,
):
    similarities = vectors @ query
    indices = np.argsort(similarities)[-min(neighbors, len(vectors)) :]
    failures = float(labels[indices].sum())
    probability = (failures + alpha) / (len(indices) + 2.0 * alpha)
    entropy = -probability * math.log(probability) - (1.0 - probability) * math.log(
        1.0 - probability
    )
    return {
        "failure_probability": probability,
        "outcome_entropy": entropy / math.log(2.0),
        "failure_information": -math.log(1.0 - probability),
    }


def prefix_vector(question_vector: np.ndarray, message_vectors: list[np.ndarray]):
    return np.concatenate([question_vector, *message_vectors])


def leave_one_out_failure_information(vectors, labels, neighbors):
    values = []
    for index, vector in enumerate(vectors):
        available = np.arange(len(vectors)) != index
        statistics = neighbor_failure_statistics(
            vector, vectors[available], labels[available], neighbors
        )
        values.append(statistics["failure_information"])
    return values


def calibrated_z(value: float, reference: list[float]) -> float:
    spread = float(np.std(reference)) or 1e-12
    return (value - float(np.mean(reference))) / spread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=15)
    args = parser.parse_args()

    results = read_jsonl(args.results)
    traces = read_jsonl(args.traces)
    queries = read_jsonl(args.failures)[: args.limit]
    query_ids = {row["example_id"] for row in queries}
    events_by_run = defaultdict(list)
    for event in traces:
        events_by_run[event.get("run_id", "")].append(event)

    embedder = LocalTransformerEmbedder(args.embedding_model)
    cache = {}

    def embed(text: str):
        if text not in cache:
            cache[text] = embedder(text)
        return cache[text]

    edge_history = defaultdict(list)
    prefix_history = defaultdict(list)
    for row in results:
        if row["example_id"] in query_ids:
            continue
        messages = response_messages(events_by_run[row["run_id"]])
        if set(messages) != set(ROLES):
            continue
        hop = hop_count(row["example_id"])
        failed = float(not bool(row.get("em", False)))
        question_vector = embed(row["question"])
        message_vectors = []
        prefix_history[(hop, -1)].append((question_vector, failed))
        for stage, role in enumerate(ROLES):
            vector = embed(messages[role])
            edge_history[(hop, role)].append((vector, failed))
            message_vectors.append(vector)
            prefix_history[(hop, stage)].append(
                (prefix_vector(question_vector, message_vectors), failed)
            )

    edge_arrays = {
        key: (np.vstack([row[0] for row in values]), np.array([row[1] for row in values]))
        for key, values in edge_history.items()
    }
    edge_calibration = {
        key: leave_one_out_failure_information(vectors, labels, args.neighbors)
        for key, (vectors, labels) in edge_arrays.items()
    }
    prefix_arrays = {
        key: (np.vstack([row[0] for row in values]), np.array([row[1] for row in values]))
        for key, values in prefix_history.items()
    }

    diagnostics = []
    for failure in queries:
        hop = hop_count(failure["example_id"])
        messages = response_messages(failure["events"])
        question_vector = embed(failure["question"])
        baseline_vectors, baseline_labels = prefix_arrays[(hop, -1)]
        previous_probability = neighbor_failure_statistics(
            question_vector, baseline_vectors, baseline_labels, args.neighbors
        )["failure_probability"]
        message_vectors = []
        edge_scores = []
        for stage, role in enumerate(ROLES):
            vector = embed(messages[role])
            edge_vectors, edge_labels = edge_arrays[(hop, role)]
            edge_statistics = neighbor_failure_statistics(
                vector, edge_vectors, edge_labels, args.neighbors
            )
            calibrated_information = calibrated_z(
                edge_statistics["failure_information"],
                edge_calibration[(hop, role)],
            )
            message_vectors.append(vector)
            prefix = prefix_vector(question_vector, message_vectors)
            history_vectors, history_labels = prefix_arrays[(hop, stage)]
            prefix_statistics = neighbor_failure_statistics(
                prefix, history_vectors, history_labels, args.neighbors
            )
            probability = prefix_statistics["failure_probability"]
            edge_scores.append(
                {
                    "role": role,
                    **edge_statistics,
                    "calibrated_failure_information_z": calibrated_information,
                    "trajectory_failure_probability": probability,
                    "trajectory_failure_jump": probability - previous_probability,
                    "trajectory_outcome_entropy": prefix_statistics["outcome_entropy"],
                }
            )
            previous_probability = probability
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
