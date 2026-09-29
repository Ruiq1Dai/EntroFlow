"""Ground-truth fault plans kept separate from diagnosis-visible traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256

FAULT_TYPES = {
    "F1": "upstream_generation_fault",
    "F2": "routing_fault",
    "F3": "branch_local_fault",
    "F4": "aggregation_destruction",
    "F5": "repair_local_fault",
    "F6": "no_recovery_opportunity",
}

MUTATION_OPERATORS = {
    "F1": "replace_upstream_candidate_with_raising_body",
    "F2": "invert_oracle_route_decision",
    "F3": "replace_solver_b_candidate_with_raising_body",
    "F4": "force_judge_to_select_known_wrong_visible_candidate",
    "F5": "replace_repair_output_after_actionable_feedback",
    "F6": "hide_verifier_feedback_from_repair",
}

PILOT_MATRIX = {
    "p1_sequence_v1": ("F1",),
    "p2_fork_join_v1": ("F3", "F4"),
    "p3a_oracle_gated_select_v1": ("F2",),
    "p4_bounded_feedback_v1": ("F5", "F6"),
    "c1_parallel_conditional_repair_v1": ("F1", "F2", "F3", "F4", "F5", "F6"),
}


@dataclass(frozen=True)
class InjectionRecord:
    injection_id: str
    clean_execution_id: str
    corrupted_execution_id: str
    topology_id: str
    sample_id: str
    true_fault_execution: str
    true_fault_type: str
    mutation_operator: str
    counterfactual_pair_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def stable_id(*parts: str) -> str:
    digest = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"ctg-{digest}"


def true_fault_execution(topology_id: str, fault: str) -> str:
    if fault == "F1":
        return "solver_a@0" if topology_id.startswith("c1_") else "generator@0"
    return {
        "F2": "route_gate@0",
        "F3": "solver_b@0",
        "F4": "judge@0",
        "F5": "repair@0",
        "F6": "verifier@0",
    }[fault]


def make_injection_record(
    topology_id: str,
    sample_id: str,
    fault: str,
    clean_run_id: str,
    corrupted_run_id: str,
) -> InjectionRecord:
    if fault not in FAULT_TYPES:
        raise ValueError(f"unsupported fault: {fault!r}")
    pair_key = "repair-opportunity" if fault in {"F5", "F6"} else fault
    return InjectionRecord(
        injection_id=stable_id("injection", topology_id, sample_id, fault),
        clean_execution_id=clean_run_id,
        corrupted_execution_id=corrupted_run_id,
        topology_id=topology_id,
        sample_id=sample_id,
        true_fault_execution=true_fault_execution(topology_id, fault),
        true_fault_type=FAULT_TYPES[fault],
        mutation_operator=MUTATION_OPERATORS[fault],
        counterfactual_pair_id=stable_id("counterfactual", topology_id, sample_id, pair_key),
    )

