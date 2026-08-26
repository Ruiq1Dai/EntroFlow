"""Evaluate repair policies on the exact same examples as a baseline.

Overall accuracy hides whether a controller actually rescued a failure or
merely changed which examples failed. This report treats rescue, regression,
and intervention cost as the primary paired outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from entroflow.reporting import read_jsonl, write_json


def load_rows(path):
    return {row["example_id"]: row for row in read_jsonl(path)}


def compare(baseline, candidate):
    ids = sorted(set(baseline) & set(candidate))
    rescued = [i for i in ids if not baseline[i].get("em") and candidate[i].get("em")]
    regressed = [i for i in ids if baseline[i].get("em") and not candidate[i].get("em")]
    unchanged = [i for i in ids if baseline[i].get("em") == candidate[i].get("em")]
    token_deltas = [
        int(candidate[i].get("total_tokens", 0)) - int(baseline[i].get("total_tokens", 0))
        for i in ids
    ]
    by_hop = {}
    for hop in (2, 3, 4):
        hop_ids = [i for i in ids if i.startswith(f"{hop}hop")]
        by_hop[str(hop)] = {
            "n": len(hop_ids),
            "rescued": sum(i in rescued for i in hop_ids),
            "regressed": sum(i in regressed for i in hop_ids),
        }
    return {
        "paired_count": len(ids),
        "rescued": len(rescued),
        "regressed": len(regressed),
        "net_gain": len(rescued) - len(regressed),
        "unchanged": len(unchanged),
        "rescue_rate_on_baseline_failures": (
            len(rescued) / sum(not baseline[i].get("em") for i in ids)
            if any(not baseline[i].get("em") for i in ids) else 0.0
        ),
        "regression_rate_on_baseline_successes": (
            len(regressed) / sum(bool(baseline[i].get("em")) for i in ids)
            if any(baseline[i].get("em") for i in ids) else 0.0
        ),
        "mean_token_delta": sum(token_deltas) / len(token_deltas) if token_deltas else 0.0,
        "rescued_ids": rescued,
        "regressed_ids": regressed,
        "by_hop": by_hop,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = load_rows(args.baseline)
    reports = {}
    for path in args.candidate:
        candidate_path = Path(path)
        name = candidate_path.parent.name or candidate_path.stem
        reports[name] = compare(baseline, load_rows(path))
    write_json(args.output, {"baseline": args.baseline, "candidates": reports})


if __name__ == "__main__":
    main()
