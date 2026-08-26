"""Routing policies for static and runtime-repaired AutoGen workflows."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .entropy import EdgeEntropyEvent, RollingEdgeEntropy
from .messages import RoleResponse
from .repairs import RepairAction

ROLES = ("planner", "evidence", "reasoner", "validator", "aggregator")

_CONTRACT_MARKERS = re.compile(
    r"\b(?:no passage|not explicitly|unsupported|missing|incomplete|cannot|"
    r"uncertain|contradict|insufficient|not identified|not stated|does not)\b",
    re.IGNORECASE,
)


def contract_violation_score(response: RoleResponse) -> float:
    """Estimate whether a role failed to preserve a required handoff contract.

    This is a runtime diagnostic, not a correctness label. It combines the
    agent's structured status with conservative lexical evidence that the
    response itself reports an unresolved hop. The score is intentionally
    bounded so it can be combined with normalized entropy signals.
    """
    score = {
        "unparsed": 1.0,
        "insufficient": 1.0,
        "uncertain": 0.9,
        "invalid": 0.9,
        "retry": 0.7,
    }.get(response.status.lower(), 0.0)
    marker_count = len(_CONTRACT_MARKERS.findall(response.content))
    score = max(score, min(1.0, marker_count / 3.0))
    if (
        response.role in {"evidence", "reasoner"}
        and not response.evidence_ids
        and marker_count
    ):
        score = max(score, 0.6)
    return min(1.0, score)


@dataclass
class WorkflowState:
    run_id: str
    example_id: str
    history: list[RoleResponse] = field(default_factory=list)
    evidence_retries: int = 0

    @property
    def last(self) -> RoleResponse | None:
        return self.history[-1] if self.history else None

    def latest(self, role: str) -> RoleResponse | None:
        return next((row for row in reversed(self.history) if row.role == role), None)


@dataclass(frozen=True)
class RoutingDecision:
    next_role: str | None
    action: RepairAction = RepairAction.NONE
    visible_roles: tuple[str, ...] = ()
    entropy_event: EdgeEntropyEvent | None = None
    # An intervention is only scientifically useful when its hypothesis is
    # explicit and can be checked against the next trace events.
    diagnosis: str = ""
    repair_target: str = ""
    repair_instruction: str = ""
    confidence: float | None = None


class RoutingPolicy(Protocol):
    def select(self, state: WorkflowState) -> RoutingDecision: ...


class StaticPolicy:
    """Fixed Planner -> Evidence -> Reasoner -> Aggregator baseline."""

    def select(self, state: WorkflowState) -> RoutingDecision:
        if state.last is None:
            return RoutingDecision("planner")
        transitions = {
            "planner": RoutingDecision("evidence", visible_roles=("planner",)),
            "evidence": RoutingDecision("reasoner", visible_roles=("planner", "evidence")),
            "reasoner": RoutingDecision("aggregator", visible_roles=("reasoner",)),
            "validator": RoutingDecision("reasoner", visible_roles=("evidence", "validator")),
            "aggregator": RoutingDecision(None),
        }
        return transitions[state.last.role]


class TraceOnlyPolicy(StaticPolicy):
    """Repair explicit failure statuses without using entropy."""

    def select(self, state: WorkflowState) -> RoutingDecision:
        last = state.last
        if (
            last
            and last.role == "evidence"
            and last.status in {"insufficient", "retry"}
            and state.evidence_retries < 1
        ):
            return RoutingDecision(
                "evidence",
                RepairAction.RETRY_WITH_CONSTRAINTS,
                visible_roles=("planner", "evidence"),
            )
        if last and last.role == "reasoner" and last.status in {"uncertain", "invalid"}:
            return RoutingDecision(
                "validator",
                RepairAction.INSERT_VALIDATION,
                visible_roles=("evidence", "reasoner"),
            )
        return super().select(state)


class AlwaysValidatePolicy(StaticPolicy):
    """Insert Validator after Evidence for every example."""

    def select(self, state: WorkflowState) -> RoutingDecision:
        if state.last and state.last.role == "evidence":
            return RoutingDecision(
                "validator",
                RepairAction.INSERT_VALIDATION,
                visible_roles=("planner", "evidence"),
            )
        return super().select(state)


class AlwaysRetryEvidencePolicy(StaticPolicy):
    """Retry Evidence once with the original planning constraints."""

    def select(self, state: WorkflowState) -> RoutingDecision:
        if state.last and state.last.role == "evidence" and state.evidence_retries < 1:
            return RoutingDecision(
                "evidence",
                RepairAction.RETRY_WITH_CONSTRAINTS,
                visible_roles=("planner", "evidence"),
            )
        return super().select(state)


class AlwaysEvidenceBypassPolicy(StaticPolicy):
    """Expose both Evidence and Reasoner directly to Aggregator."""

    def select(self, state: WorkflowState) -> RoutingDecision:
        if state.last and state.last.role == "reasoner":
            return RoutingDecision(
                "aggregator",
                RepairAction.EVIDENCE_BYPASS,
                visible_roles=("evidence", "reasoner"),
            )
        return super().select(state)


class RandomRepairPolicy(StaticPolicy):
    """Apply the same repair library without using trace semantics."""

    def __init__(self, probability: float = 0.25, seed: int = 42):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self._random = random.Random(seed)

    def select(self, state: WorkflowState) -> RoutingDecision:
        last = state.last
        if last and self._random.random() < self.probability:
            if last.role == "evidence":
                return RoutingDecision(
                    "validator",
                    RepairAction.INSERT_VALIDATION,
                    visible_roles=("planner", "evidence"),
                )
            if last.role == "reasoner":
                return RoutingDecision(
                    "aggregator",
                    RepairAction.EVIDENCE_BYPASS,
                    visible_roles=("evidence", "reasoner"),
                )
        return super().select(state)


class EntropyPolicy(StaticPolicy):
    """Use rolling edge entropy to select a constrained repair action."""

    def __init__(
        self,
        embedder: Callable[[str], np.ndarray],
        monitor: RollingEdgeEntropy | None = None,
        threshold: float = 0.15,
    ):
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.embedder = embedder
        self.monitor = monitor or RollingEdgeEntropy()
        self.threshold = threshold

    def select(self, state: WorkflowState) -> RoutingDecision:
        baseline = super().select(state)
        last = state.last
        if last is None or baseline.next_role is None:
            return baseline
        contract_score = contract_violation_score(last)
        # A broken handoff is a stronger local signal than a high-entropy
        # message that may simply contain multiple valid alternatives.
        if contract_score >= 0.6 and last.role in {"evidence", "reasoner"}:
            return self._contract_repair_decision(last, baseline, contract_score)
        if last.semantic_entropy is not None:
            previous_values = [
                row.semantic_entropy for row in state.history[:-1]
                if row.semantic_entropy is not None
            ]
            previous = previous_values[-1] if previous_values else None
            event = EdgeEntropyEvent(
                edge=f"{last.role}_to_{baseline.next_role}",
                sample_count=last.semantic_cluster_count,
                cluster_count=last.semantic_cluster_count,
                entropy=last.semantic_entropy,
                delta=None if previous is None else last.semantic_entropy - previous,
                ready=True,
            )
            anomalous = last.semantic_entropy >= 0.6 or (
                event.delta is not None and abs(event.delta) >= self.threshold
            )
            return self._repair_decision(last.role, baseline, event, anomalous)
        edge = f"{last.role}_to_{baseline.next_role}"
        event = self.monitor.observe(edge, self.embedder(last.content))
        anomalous = event.ready and event.delta is not None and abs(event.delta) >= self.threshold
        return self._repair_decision(last.role, baseline, event, anomalous)

    @staticmethod
    def _contract_repair_decision(last, baseline, score):
        if last.role == "evidence":
            return RoutingDecision(
                "evidence",
                RepairAction.EXPAND_EVIDENCE,
                visible_roles=("planner", "evidence"),
                diagnosis="The evidence handoff reports an unresolved hop or constraint.",
                repair_target="evidence_chain",
                repair_instruction=(
                    "Search for the missing bridge relation, preserve the planner's "
                    "entity and answer-type constraints, and return a complete cited chain."
                ),
                confidence=score,
            )
        return RoutingDecision(
            "validator",
            RepairAction.INSERT_VALIDATION,
            visible_roles=("evidence", "reasoner"),
            diagnosis="The reasoning handoff reports an unsupported or inconsistent hop.",
            repair_target="reasoning_chain",
            repair_instruction=(
                "Validate each inferred hop against the cited evidence and identify the "
                "first unsupported bridge before aggregation."
            ),
            confidence=score,
        )

    def _repair_decision(self, role, baseline, event, anomalous):
        if not anomalous:
            return RoutingDecision(
                baseline.next_role,
                baseline.action,
                baseline.visible_roles,
                event,
            )
        delta = event.delta or 0.0
        # Use both magnitude and direction: a sharp rise means disagreement,
        # while a sharp fall after an unstable window favors consolidation.
        severe = abs(delta) >= self.threshold * 2
        if role == "evidence" and delta > 0 and severe:
            return RoutingDecision(
                "validator",
                RepairAction.INSERT_VALIDATION,
                visible_roles=("planner", "evidence"),
                entropy_event=event,
                diagnosis="Evidence messages became sharply more semantically dispersed.",
                repair_target="evidence_chain",
                repair_instruction=(
                    "Re-check every hop in order, quote one supporting passage per hop, "
                    "and reject chains with an unresolved bridge entity."
                ),
                confidence=min(0.95, 0.55 + abs(delta)),
            )
        if role == "evidence" and delta > 0:
            return RoutingDecision(
                "validator", RepairAction.CROSS_CHECK,
                visible_roles=("planner", "evidence"), entropy_event=event,
                diagnosis="Evidence messages show a moderate semantic split.",
                repair_target="evidence_discriminator",
                repair_instruction=(
                    "Compare the competing evidence chains and state the missing "
                    "discriminator before selecting one."
                ),
                confidence=min(0.9, 0.5 + abs(delta)),
            )
        if role == "evidence" and delta < 0:
            return RoutingDecision(
                "validator", RepairAction.REORDER_VALIDATION,
                visible_roles=("evidence",), entropy_event=event,
                diagnosis="Evidence dispersion fell after an unstable evidence window.",
                repair_target="evidence_consistency",
                repair_instruction=(
                    "Validate the selected chain before reasoning; preserve entity, "
                    "relation, and answer-type constraints."
                ),
                confidence=min(0.85, 0.45 + abs(delta)),
            )
        if role == "reasoner":
            return RoutingDecision(
                "aggregator",
                RepairAction.EVIDENCE_BYPASS if severe else RepairAction.COMPRESS_CONTEXT,
                visible_roles=("evidence", "reasoner"),
                entropy_event=event,
                diagnosis="Reasoning messages became semantically unstable.",
                repair_target="reasoning_chain",
                repair_instruction=(
                    "Re-derive the answer from the ordered evidence chain, mark each "
                    "step as supported or unsupported, and do not use an ungrounded shortcut."
                ),
                confidence=min(0.9, 0.5 + abs(delta)),
            )
        return RoutingDecision(
            baseline.next_role,
            baseline.action,
            baseline.visible_roles,
            event,
        )
