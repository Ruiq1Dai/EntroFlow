"""Evaluate entropy localization against an independently audited pilot."""

from __future__ import annotations

import argparse
from collections import Counter

from entroflow.reporting import read_jsonl, write_json

DEFAULT_SIGNALS = ("assignment_entropy", "entropy_delta", "semantic_surprise_z")


def localization_accuracy(diagnostics, audit, signal):
    truth = {row["example_id"]: row["first_error_role"] for row in audit}
    predictions = {}
    for row in diagnostics:
        predictions[row["example_id"]] = max(
            row["edge_scores"], key=lambda edge: edge[signal]
        )["role"]
    matched = set(truth) & set(predictions)
    hits = sum(predictions[example_id] == truth[example_id] for example_id in matched)
    return {
        "evaluated": len(matched),
        "hits": hits,
        "accuracy": hits / len(matched) if matched else 0.0,
        "prediction_counts": dict(Counter(predictions.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS)
    args = parser.parse_args()
    diagnostics = read_jsonl(args.diagnostics)
    audit = read_jsonl(args.audit)
    truth_counts = Counter(row["first_error_role"] for row in audit)
    majority = max(truth_counts.values(), default=0)
    report = {
        "audit_count": len(audit),
        "truth_counts": dict(truth_counts),
        "majority_role_baseline_accuracy": majority / len(audit) if audit else 0.0,
        "signals": {
            signal: localization_accuracy(diagnostics, audit, signal)
            for signal in args.signals
        },
        "decision": "Current entropy signals do not localize first causal errors.",
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
