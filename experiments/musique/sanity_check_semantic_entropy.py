"""Synthetic sanity check for within-step semantic entropy."""
from __future__ import annotations

import argparse
import json

import numpy as np

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.entropy import semantic_entropy
from entroflow.reporting import write_json


SETS = {
    "on_topic": [
        "The evidence states that Marie Curie was born in Warsaw.",
        "Marie Curie was born in Warsaw, according to the passage.",
        "The passage identifies Warsaw as Marie Curie's birthplace.",
        "Marie Curie's birthplace is Warsaw.",
        "The source confirms Marie Curie was born in Warsaw.",
    ],
    "one_off_topic": [
        "The evidence states that Marie Curie was born in Warsaw.",
        "Marie Curie was born in Warsaw, according to the passage.",
        "The passage identifies Warsaw as Marie Curie's birthplace.",
        "Marie Curie's birthplace is Warsaw.",
        "The Pacific Ocean is the largest ocean on Earth.",
    ],
    "mixed_two_topics": [
        "The evidence states that Marie Curie was born in Warsaw.",
        "Marie Curie was born in Warsaw, according to the passage.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "The Pacific Ocean covers more area than any other ocean.",
        "Warsaw is the birthplace of Marie Curie.",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.82, 0.85, 0.90])
    args = parser.parse_args()
    embed = LocalTransformerEmbedder(args.embedding_model)
    vectors = {name: np.vstack([embed(text) for text in texts]) for name, texts in SETS.items()}
    rows = []
    for threshold in args.thresholds:
        for name, matrix in vectors.items():
            entropy, clusters, labels = semantic_entropy(matrix, similarity_threshold=threshold)
            rows.append({"threshold": threshold, "condition": name, "entropy": entropy,
                         "cluster_count": clusters, "labels": labels})
    by_threshold = {}
    for threshold in args.thresholds:
        current = {r["condition"]: r for r in rows if r["threshold"] == threshold}
        by_threshold[str(threshold)] = {
            "one_off_topic_detected": current["one_off_topic"]["entropy"] > current["on_topic"]["entropy"],
            "mixed_detected": current["mixed_two_topics"]["entropy"] > current["on_topic"]["entropy"],
        }
    payload = {"rows": rows, "checks": by_threshold}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
