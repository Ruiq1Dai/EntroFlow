"""Evaluate semantic entropy on six graded synthetic anomaly levels."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.entropy import semantic_entropy
from entroflow.reporting import write_json

LEVELS = (
    "off_topic", "wrong_entity", "wrong_relation", "missing_hop",
    "reasoning_error", "semantic_equivalent",
)


def group(level: str, i: int) -> list[str]:
    subject = f"Person {i}"
    bridge = f"City {i}"
    target = f"Institution {i}"
    base = [
        f"{subject} was born in {bridge}, and {bridge} is home to {target}.",
        f"The evidence links {subject} to birthplace {bridge}; {target} is located there.",
        f"According to the passages, {subject}'s birthplace is {bridge}, which contains {target}.",
        f"The chain is: {subject} -> born in -> {bridge} -> contains -> {target}.",
    ]
    anomalies = {
        "off_topic": f"The Pacific Ocean is the largest ocean on Earth in area.",
        "wrong_entity": f"Person {i+1000} was born in {bridge}, and {bridge} is home to {target}.",
        "wrong_relation": f"{subject} works in {bridge}, and {bridge} is home to {target}.",
        "missing_hop": f"The passages mention {subject} and {target}, but do not establish the intermediate birthplace link.",
        "reasoning_error": f"{subject} was born in {bridge}, therefore the answer is City {i+1}, not the institution located there.",
        "semantic_equivalent": f"The source confirms that {bridge} is the birthplace of {subject} and that {target} is there.",
    }
    return base + [anomalies[level]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--groups", type=int, default=30)
    p.add_argument("--threshold", type=float, default=0.82)
    args = p.parse_args()
    embed = LocalTransformerEmbedder(args.embedding_model)
    rows = []
    for level in LEVELS:
        for i in range(args.groups):
            texts = group(level, i)
            matrix = np.vstack([embed(text) for text in texts])
            entropy, clusters, labels = semantic_entropy(matrix, args.threshold)
            rows.append({"level": level, "level_index": LEVELS.index(level), "group": i,
                         "entropy": entropy, "clusters": clusters, "labels": labels})
    summary = {}
    for level in LEVELS:
        values = [r["entropy"] for r in rows if r["level"] == level]
        summary[level] = {"mean": statistics.mean(values), "median": statistics.median(values),
                          "min": min(values), "max": max(values),
                          "mean_clusters": statistics.mean(r["clusters"] for r in rows if r["level"] == level)}
    means = [summary[level]["mean"] for level in LEVELS]
    summary_meta = {"groups_per_level": args.groups, "threshold": args.threshold,
                    "mean_sequence": means,
                    "monotonic_increasing": all(a <= b for a, b in zip(means, means[1:])),
                    "monotonic_decreasing": all(a >= b for a, b in zip(means, means[1:]))}
    payload = {"levels": list(LEVELS), "summary": summary, "meta": summary_meta, "rows": rows}
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(summary_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
