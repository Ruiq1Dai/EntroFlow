"""Evaluate earliest message-contract violations against a blinded audit.

This is an exploratory diagnostic. It deliberately does not read entropy
fields or repair actions, and it keeps unrepairable audit rows separate.
"""

from __future__ import annotations

import argparse
from collections import Counter

from entroflow.messages import RoleResponse
from entroflow.reporting import read_jsonl, write_json
from entroflow.routing import contract_violation_score

ROLES = ("planner", "evidence", "reasoner", "aggregator")


def _response(event):
    metadata = event.get("metadata", {})
    return RoleResponse(
        run_id=event.get("run_id", ""),
        example_id=event.get("example_id", ""),
        role=event.get("sender", ""),
        content=event.get("content", ""),
        step=int(event.get("step", 0)),
        status=str(metadata.get("status", "ok")),
        evidence_ids=tuple(metadata.get("evidence_ids", ())),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures")
    parser.add_argument("--diagnostics")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    if not args.failures and not args.diagnostics:
        parser.error("one of --failures or --diagnostics is required")
    audit = {row["example_id"]: row for row in read_jsonl(args.audit)}
    predictions = {}
    details = []
    input_rows = read_jsonl(args.failures or args.diagnostics)
    for row in input_rows:
        if row["example_id"] not in audit:
            continue
        if args.failures:
            responses = {
                event["sender"]: _response(event)
                for event in row["events"]
                if event.get("sender") in ROLES
            }
        else:
            responses = {
                item["role"]: RoleResponse(
                    run_id="",
                    example_id=row["example_id"],
                    role=item["role"],
                    content=item.get("content", ""),
                    step=0,
                )
                for item in row["edge_scores"]
            }
        scores = {role: contract_violation_score(responses[role]) for role in ROLES if role in responses}
        candidate_roles = [role for role in ROLES if scores.get(role, 0.0) >= args.threshold]
        predicted = candidate_roles[0] if candidate_roles else None
        predictions[row["example_id"]] = predicted
        details.append({"example_id": row["example_id"], "predicted_role": predicted, "scores": scores})

    repairable = [
        row for row in audit.values()
        if row.get("repairable", row.get("first_error_role") is not None)
    ]
    evaluated = [row for row in repairable if row["example_id"] in predictions]
    hits = sum(
        predictions[row["example_id"]]
        == row.get("first_recoverable_role", row.get("first_error_role"))
        for row in evaluated
    )
    result = {
        "audit_count": len(audit),
        "repairable_count": len(repairable),
        "evaluated_repairable_count": len(evaluated),
        "hits": hits,
        "accuracy": hits / len(evaluated) if evaluated else 0.0,
        "prediction_counts": dict(Counter(predictions.values())),
        "warning": "Exploratory contract diagnostic; threshold and markers require held-out calibration.",
        "details": details,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
