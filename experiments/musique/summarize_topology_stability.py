"""Aggregate repeated topology-entropy localization trials."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from entroflow.reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    values = defaultdict(list)
    ranks = defaultdict(list)
    winners = Counter()
    for report in reports:
        ranking = report["ranking"]
        winners[ranking[0]["unit"]] += 1
        for rank, row in enumerate(ranking, start=1):
            ranks[row["unit"]].append(rank)
            values[row["unit"]].append(row)

    summary = []
    for unit, rows in values.items():
        summary.append(
            {
                "unit": unit,
                "mean_rank": float(np.mean(ranks[unit])),
                "top1_count": winners[unit],
                "top3_count": sum(rank <= 3 for rank in ranks[unit]),
                "mean_entropy_delta": float(np.mean([row["entropy_delta"] for row in rows])),
                "mean_js_divergence": float(np.mean([row["js_divergence"] for row in rows])),
                "mean_failure_ood_rate": float(
                    np.mean([row["failure_ood_rate"] for row in rows])
                ),
                "significant_js_count": sum(
                    row["js_permutation_p_value"] < 0.05 for row in rows
                ),
            }
        )
    summary.sort(key=lambda row: (row["mean_rank"], -row["mean_js_divergence"]))
    write_json(
        args.output,
        {
            "trial_count": len(reports),
            "ranking": summary,
            "decision_rule": "Advance only units with stable top-3 rank and repeated distribution shift.",
        },
    )


if __name__ == "__main__":
    main()
