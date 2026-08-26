"""Offline entropy localization pilot for held-out MuSiQue failures."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.entropy import normalized_entropy
from entroflow.reporting import read_jsonl, write_json, write_jsonl

ROLES = ("planner", "evidence", "reasoner", "aggregator")


def hop_count(example_id: str) -> int:
    return int(example_id[0])


def response_messages(events):
    return {
        event["sender"]: event["content"]
        for event in events
        if event.get("sender") in ROLES
    }


def assignment_entropy(distances: np.ndarray, temperature: float) -> float:
    """Entropy of a message's soft assignment over normal semantic modes."""
    values = np.asarray(distances, dtype=float)
    if values.size <= 1:
        return 0.0
    scale = max(float(temperature), 1e-12)
    logits = -(values**2) / scale
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    return float(entropy / math.log(values.size))


def fit_reference(vectors: np.ndarray, clusters: int, seed: int):
    model = KMeans(n_clusters=min(clusters, len(vectors)), random_state=seed, n_init=10)
    labels = model.fit_predict(vectors)
    distances = model.transform(vectors)
    nearest = distances.min(axis=1)
    positive = nearest[nearest > 0]
    temperature = float(np.median(positive) ** 2) if positive.size else 1.0
    spread = float(nearest.std()) or 1e-12
    return {
        "model": model,
        "labels": labels,
        "baseline_entropy": normalized_entropy(labels),
        "temperature": temperature,
        "distance_mean": float(nearest.mean()),
        "distance_std": spread,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
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

    success_rows = [row for row in results if bool(row.get("em", False))]
    references = defaultdict(list)
    for row in success_rows:
        messages = response_messages(events_by_run[row["run_id"]])
        hop = hop_count(row["example_id"])
        for role, content in messages.items():
            references[(hop, role)].append(content)

    embedder = LocalTransformerEmbedder(args.embedding_model)
    fitted = {}
    for key, texts in references.items():
        vectors = np.vstack([embedder(text) for text in texts])
        fitted[key] = fit_reference(vectors, args.clusters, args.seed)

    analyzed = []
    localization_counts = defaultdict(int)
    for failure in failures:
        hop = hop_count(failure["example_id"])
        messages = response_messages(failure["events"])
        edge_scores = []
        for role in ROLES:
            vector = embedder(messages[role]).reshape(1, -1)
            reference = fitted[(hop, role)]
            distances = reference["model"].transform(vector)[0]
            cluster = int(np.argmin(distances))
            augmented_labels = np.append(reference["labels"], cluster)
            nearest = float(distances[cluster])
            edge_scores.append(
                {
                    "role": role,
                    "assignment_entropy": assignment_entropy(
                        distances, reference["temperature"]
                    ),
                    "entropy_delta": normalized_entropy(augmented_labels)
                    - reference["baseline_entropy"],
                    "semantic_surprise_z": (
                        nearest - reference["distance_mean"]
                    )
                    / reference["distance_std"],
                    "nearest_centroid_distance": nearest,
                    "reference_count": len(reference["labels"]),
                    "content": messages[role],
                }
            )
        located = max(edge_scores, key=lambda row: row["assignment_entropy"])["role"]
        localization_counts[located] += 1
        analyzed.append(
            {
                "example_id": failure["example_id"],
                "question": failure["question"],
                "prediction": failure["prediction"],
                "gold_answer": failure["gold_answer"],
                "located_role": located,
                "edge_scores": edge_scores,
            }
        )

    write_jsonl(args.output, analyzed)
    write_json(
        args.summary,
        {
            "query_count": len(analyzed),
            "success_reference_count": len(success_rows),
            "primary_signal": "assignment_entropy",
            "localization_counts": dict(localization_counts),
            "warning": "Exploratory localization only; causes require independent trace audit.",
        },
    )


if __name__ == "__main__":
    main()
