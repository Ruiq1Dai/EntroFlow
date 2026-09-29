"""Evaluation and table generation for the compositional topology pilot."""

from __future__ import annotations

import csv
import json
from collections import Counter, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from entroflow.execution import ActivatedGraph
from entroflow.topology_optimizer import StepContract, diagnose_activated

from .fault_injection import InjectionRecord

METHODS = ("A0_raw", "A1_activation_only", "A2_full_opportunity")
EXPECTED_OBSERVED_TYPE = {
    "upstream_generation_fault": "generation_failure",
    "routing_fault": "routing_failure",
    "branch_local_fault": "generation_failure",
    "aggregation_destruction": "aggregation_failure",
    "repair_local_fault": "repair_failure",
    "no_recovery_opportunity": "verification_feedback_failure",
}


def load_contracts(path: Path) -> dict[str, StepContract]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: StepContract.from_dict(key, value) for key, value in payload.items()}


def graph_distance(graph: ActivatedGraph, source: str, target: str | None) -> int | None:
    if target is None:
        return None
    if source == target:
        return 0
    adjacency: dict[str, list[str]] = {}
    for edge in graph.traversed_edges():
        adjacency.setdefault(edge.source_execution_id, []).append(edge.target_execution_id)
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, distance = queue.popleft()
        for next_node in adjacency.get(node, ()):
            if next_node == target:
                return distance + 1
            if next_node not in seen:
                seen.add(next_node)
                queue.append((next_node, distance + 1))
    return None


def evaluate_injections(
    corrupted_graphs: dict[str, ActivatedGraph],
    injections: Iterable[InjectionRecord],
    contracts: dict[str, StepContract],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    for injection in injections:
        graph = corrupted_graphs[injection.corrupted_execution_id]
        for method in METHODS:
            diagnosis = diagnose_activated(graph, contracts, method=method)
            predicted = diagnosis.predicted_source_execution
            event = graph.execution(predicted) if predicted else None
            component = next(
                (item for item in diagnosis.opportunity_components if item.execution_id == predicted),
                None,
            )
            distance = graph_distance(graph, injection.true_fault_execution, predicted)
            case_rows.append(
                {
                    "injection_id": injection.injection_id,
                    "topology_id": injection.topology_id,
                    "sample_id": injection.sample_id,
                    "fault_type": injection.true_fault_type,
                    "method": method,
                    "true_source": injection.true_fault_execution,
                    "predicted_source": predicted or "",
                    "source_correct": predicted == injection.true_fault_execution,
                    "predicted_fault_type": diagnosis.predicted_fault_type,
                    "fault_type_correct": diagnosis.predicted_fault_type
                    == EXPECTED_OBSERVED_TYPE[injection.true_fault_type],
                    "predicted_activated": event.activated if event else None,
                    "predicted_opportunity": component.opportunity if component else False,
                    "inactive_false_blame": bool(event and event.activated is not True),
                    "opportunity_conditioned_false_blame": bool(component and not component.opportunity),
                    "propagation_distance": distance if distance is not None else "unreachable",
                    "downstream_false_blame": distance is not None and distance > 0,
                    "candidate_executions": list(diagnosis.candidate_executions),
                }
            )
    aggregate: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in case_rows if row["method"] == method]
        numeric_distances = [
            int(row["propagation_distance"])
            for row in rows
            if isinstance(row["propagation_distance"], int)
        ]
        counts = Counter(
            "distance_0"
            if distance == 0
            else "distance_1"
            if distance == 1
            else "distance_ge_2"
            for distance in numeric_distances
        )
        aggregate.append(
            {
                "method": method,
                "n": len(rows),
                "source_localization_accuracy": sum(row["source_correct"] for row in rows) / len(rows),
                "fault_type_localization_accuracy": sum(row["fault_type_correct"] for row in rows) / len(rows),
                "downstream_false_blame_rate": sum(row["downstream_false_blame"] for row in rows) / len(rows),
                "inactive_false_blame_rate": sum(row["inactive_false_blame"] for row in rows) / len(rows),
                "opportunity_conditioned_false_blame_rate": sum(row["opportunity_conditioned_false_blame"] for row in rows) / len(rows),
                "mean_blame_distance": sum(numeric_distances) / len(numeric_distances) if numeric_distances else "",
                "distance_0": counts["distance_0"],
                "distance_1": counts["distance_1"],
                "distance_ge_2": counts["distance_ge_2"],
                "unreachable": sum(row["propagation_distance"] == "unreachable" for row in rows),
            }
        )
    return case_rows, aggregate


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def diagnosis_records(
    graphs: Iterable[ActivatedGraph], contracts: dict[str, StepContract]
) -> list[dict[str, Any]]:
    records = []
    for graph in graphs:
        for method in METHODS:
            result = diagnose_activated(graph, contracts, method=method)
            records.append(
                {
                    "run_id": graph.run_id,
                    "topology_id": graph.topology.topology_id,
                    **result.to_dict(),
                }
            )
    return records

