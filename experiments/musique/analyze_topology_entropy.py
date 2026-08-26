"""Locate weak edges and motifs in the fixed MuSiQue AutoGen topology."""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.reporting import read_jsonl, write_json

ROLES = ("planner", "evidence", "reasoner", "aggregator")
UNITS = (
    "question_to_planner",
    "planner_to_evidence",
    "evidence_to_reasoner",
    "reasoner_to_aggregator",
    "planner_evidence_reasoner",
    "evidence_reasoner_aggregator",
)


def hop_count(example_id: str) -> int:
    return int(example_id[0])


def response_messages(events):
    return {
        event["sender"]: event["content"]
        for event in events
        if event.get("sender") in ROLES
    }


def fixed_entropy(labels, category_count: int) -> float:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=category_count)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(category_count))


def label_distribution(labels, category_count: int):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=category_count).astype(float)
    return counts / counts.sum()


def distribution_metrics(success_labels, failure_labels, category_count: int):
    success_distribution = label_distribution(success_labels, category_count)
    failure_distribution = label_distribution(failure_labels, category_count)
    success_entropy = fixed_entropy(success_labels, category_count)
    failure_entropy = fixed_entropy(failure_labels, category_count)
    return {
        "success_entropy": success_entropy,
        "failure_entropy": failure_entropy,
        "entropy_delta": failure_entropy - success_entropy,
        "js_divergence": float(
            jensenshannon(success_distribution, failure_distribution, base=2.0) ** 2
        ),
    }


def assign_with_ood(model, vectors, threshold: float):
    distances = model.transform(vectors)
    labels = distances.argmin(axis=1)
    labels[distances.min(axis=1) > threshold] = model.n_clusters
    return labels


def bootstrap_intervals(success_labels, failure_labels, category_count, seed, samples):
    rng = np.random.default_rng(seed)
    deltas = []
    divergences = []
    for _ in range(samples):
        success_sample = rng.choice(success_labels, len(success_labels), replace=True)
        failure_sample = rng.choice(failure_labels, len(failure_labels), replace=True)
        metrics = distribution_metrics(success_sample, failure_sample, category_count)
        deltas.append(metrics["entropy_delta"])
        divergences.append(metrics["js_divergence"])
    return {
        "entropy_delta_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "js_divergence_ci95": np.quantile(divergences, [0.025, 0.975]).tolist(),
    }


def permutation_p_value(success_labels, failure_labels, category_count, seed, samples):
    observed = distribution_metrics(success_labels, failure_labels, category_count)[
        "js_divergence"
    ]
    pooled = np.concatenate([success_labels, failure_labels])
    success_size = len(success_labels)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(samples):
        shuffled = rng.permutation(pooled)
        divergence = distribution_metrics(
            shuffled[:success_size], shuffled[success_size:], category_count
        )["js_divergence"]
        exceedances += divergence >= observed
    return (exceedances + 1) / (samples + 1)


def matched_successes(results, failure_rows, size: int, seed: int):
    target_counts = Counter(hop_count(row["example_id"]) for row in failure_rows[:size])
    successes = defaultdict(list)
    for row in results:
        if bool(row.get("em", False)):
            successes[hop_count(row["example_id"])].append(row)
    rng = random.Random(seed)
    selected = []
    for hop, count in target_counts.items():
        selected.extend(rng.sample(successes[hop], count))
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--reference-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-failures", action="store_true")
    args = parser.parse_args()

    results = read_jsonl(args.results)
    traces = read_jsonl(args.traces)
    all_failure_rows = read_jsonl(args.failures)
    failure_rows = (
        random.Random(args.seed).sample(all_failure_rows, args.sample_size)
        if args.random_failures
        else all_failure_rows[: args.sample_size]
    )
    success_rows = matched_successes(results, failure_rows, args.sample_size, args.seed)
    selected_success_ids = {row["example_id"] for row in success_rows}
    reference_candidates = [
        row
        for row in results
        if bool(row.get("em", False)) and row["example_id"] not in selected_success_ids
    ]
    reference_rows = random.Random(args.seed + 1).sample(
        reference_candidates, min(args.reference_size, len(reference_candidates))
    )
    events_by_run = defaultdict(list)
    for event in traces:
        events_by_run[event.get("run_id", "")].append(event)

    embedder = LocalTransformerEmbedder(args.embedding_model)
    cache = {}

    def embed(text):
        if text not in cache:
            cache[text] = embedder(text)
        return cache[text]

    def unit_vectors(rows, failed):
        values = defaultdict(list)
        for row in rows:
            events = row["events"] if failed else events_by_run[row["run_id"]]
            messages = response_messages(events)
            vectors = {"question": embed(row["question"])}
            vectors.update({role: embed(messages[role]) for role in ROLES})
            residuals = {
                "question_to_planner": vectors["planner"] - vectors["question"],
                "planner_to_evidence": vectors["evidence"] - vectors["planner"],
                "evidence_to_reasoner": vectors["reasoner"] - vectors["evidence"],
                "reasoner_to_aggregator": vectors["aggregator"] - vectors["reasoner"],
            }
            for name, vector in residuals.items():
                values[name].append(vector)
            values["planner_evidence_reasoner"].append(
                np.concatenate(
                    [residuals["planner_to_evidence"], residuals["evidence_to_reasoner"]]
                )
            )
            values["evidence_reasoner_aggregator"].append(
                np.concatenate(
                    [residuals["evidence_to_reasoner"], residuals["reasoner_to_aggregator"]]
                )
            )
        return {name: np.vstack(rows) for name, rows in values.items()}

    success_vectors = unit_vectors(success_rows, failed=False)
    reference_vectors = unit_vectors(reference_rows, failed=False)
    failure_vectors = unit_vectors(failure_rows, failed=True)
    report = []
    for index, unit in enumerate(UNITS):
        model = KMeans(
            n_clusters=min(args.clusters, len(success_rows)),
            random_state=args.seed,
            n_init=10,
        ).fit(reference_vectors[unit])
        reference_distances = model.transform(reference_vectors[unit]).min(axis=1)
        threshold = float(np.quantile(reference_distances, 0.95))

        success_labels = assign_with_ood(model, success_vectors[unit], threshold)
        failure_labels = assign_with_ood(model, failure_vectors[unit], threshold)
        category_count = model.n_clusters + 1
        metrics = distribution_metrics(success_labels, failure_labels, category_count)
        metrics.update(
            bootstrap_intervals(
                success_labels,
                failure_labels,
                category_count,
                args.seed + index,
                args.bootstrap_samples,
            )
        )
        metrics["js_permutation_p_value"] = permutation_p_value(
            success_labels,
            failure_labels,
            category_count,
            args.seed + index,
            args.bootstrap_samples,
        )
        metrics.update(
            {
                "unit": unit,
                "success_ood_rate": float(np.mean(success_labels == model.n_clusters)),
                "failure_ood_rate": float(np.mean(failure_labels == model.n_clusters)),
            }
        )
        report.append(metrics)

    report.sort(key=lambda row: (row["js_divergence"], row["entropy_delta"]), reverse=True)
    write_json(
        args.output,
        {
            "dataset_type": "MuSiQue",
            "success_sample_size": len(success_rows),
            "reference_sample_size": len(reference_rows),
            "failure_sample_size": len(failure_rows),
            "failure_hop_counts": dict(Counter(hop_count(row["example_id"]) for row in failure_rows)),
            "success_example_ids": [row["example_id"] for row in success_rows],
            "failure_example_ids": [row["example_id"] for row in failure_rows],
            "ranking": report,
        },
    )


if __name__ == "__main__":
    main()
