"""Experiment-level metrics and failed-trajectory export."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize MuSiQue results without conflating accuracy and token F1."""
    materialized = list(rows)
    count = len(materialized)
    correct = sum(bool(row.get("em", False)) for row in materialized)
    f1_values = [float(row.get("f1", 0.0)) for row in materialized]
    completed = sum(bool(row.get("completed", False)) for row in materialized)
    total_tokens = sum(int(row.get("total_tokens", 0)) for row in materialized)
    return {
        "num_examples": count,
        "num_correct": correct,
        "accuracy": correct / count if count else 0.0,
        "accuracy_percent": 100.0 * correct / count if count else 0.0,
        "macro_f1": sum(f1_values) / count if count else 0.0,
        "macro_f1_percent": 100.0 * sum(f1_values) / count if count else 0.0,
        "completion_rate": completed / count if count else 0.0,
        "total_tokens": total_tokens,
        "mean_tokens": total_tokens / count if count else 0.0,
    }


def failed_trajectories(
    results: Iterable[dict[str, Any]], traces: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join incorrect results to their complete ordered communication traces."""
    events_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in traces:
        run_id = str(event.get("run_id", ""))
        if run_id:
            events_by_run[run_id].append(event)

    failures = []
    for result in results:
        if bool(result.get("em", False)):
            continue
        run_id = str(result.get("run_id", ""))
        events = sorted(
            events_by_run.get(run_id, []),
            key=lambda event: (int(event.get("step", 0)), event.get("sender", "")),
        )
        failures.append(
            {
                "run_id": run_id,
                "example_id": result.get("example_id", ""),
                "question": result.get("question", ""),
                "prediction": result.get("prediction", ""),
                "gold_answer": result.get("gold_answer", ""),
                "f1": float(result.get("f1", 0.0)),
                "failure_reason": "incomplete" if not result.get("completed", False) else "incorrect_answer",
                "trace_complete": bool(events),
                "events": events,
            }
        )
    return failures


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(destination)
