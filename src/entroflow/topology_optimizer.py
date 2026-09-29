"""Framework-neutral trajectory diagnosis and local topology rewrite selection.

The optimizer never participates in task-time routing.  Task-specific answer
evaluation is injected as an offline callback after trajectories are complete.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from .execution import ActivatedGraph, ExecutionInstance

AnswerEvaluator = Callable[[str, Mapping[str, Any]], bool]


@dataclass(frozen=True)
class WorkflowGraph:
    workflow_id: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    phases: tuple[tuple[str, ...], ...]
    roles: Mapping[str, str]
    visibility: Mapping[str, tuple[str, ...]]
    routing: str
    shared_state: bool
    topology_type: str
    logging_coverage: Mapping[str, float]
    repeated_call_rate: Mapping[str, float]

    def predecessors(self, node: str) -> tuple[str, ...]:
        return tuple(source for source, target in self.edges if target == node)

    def successors(self, node: str) -> tuple[str, ...]:
        return tuple(target for source, target in self.edges if source == node)


@dataclass(frozen=True)
class NodeSpec:
    kind: str
    answer_label: str | None = None
    answer_source: str = "output"
    required_fields: tuple[str, ...] = ()
    candidates: Mapping[str, str] = field(default_factory=dict)
    opportunity: str = "always"
    opportunity_sources: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NodeSpec:
        return cls(
            kind=str(payload["kind"]),
            answer_label=payload.get("answer_label"),
            answer_source=str(payload.get("answer_source", "output")),
            required_fields=tuple(payload.get("required_fields", ())),
            candidates=dict(payload.get("candidates", {})),
            opportunity=str(payload.get("opportunity", "always")),
            opportunity_sources=tuple(payload.get("opportunity_sources", ())),
        )


@dataclass(frozen=True)
class AccessRequirement:
    """One topology-independent causal-access requirement.

    All requirements on a contract must pass.  Group modes make aggregation
    and recovery expressible without naming a topology in diagnosis code.
    """

    name: str
    inputs: tuple[str, ...]
    mode: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AccessRequirement:
        return cls(
            name=str(payload["name"]),
            inputs=tuple(map(str, payload.get("inputs", ()))),
            mode=str(payload.get("mode", "all_observed")),
        )


@dataclass(frozen=True)
class StepContract:
    contract_id: str
    required_inputs: tuple[str, ...] = ()
    access_requirements: tuple[AccessRequirement, ...] = ()
    recovery_inputs: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, contract_id: str, payload: Mapping[str, Any]) -> StepContract:
        return cls(
            contract_id=contract_id,
            required_inputs=tuple(map(str, payload.get("required_inputs", ()))),
            access_requirements=tuple(
                AccessRequirement.from_dict(item)
                for item in payload.get("access_requirements", ())
            ),
            recovery_inputs=tuple(map(str, payload.get("recovery_inputs", ()))),
        )


@dataclass(frozen=True)
class OpportunityComponents:
    execution_id: str
    activated: bool | None
    required_inputs_present: bool
    required_inputs_visible: bool
    contract_applicable: bool
    recovery_information_available: bool
    opportunity: bool
    opportunity_reason: str


@dataclass(frozen=True)
class ActivatedDiagnosis:
    method: str
    sample_id: str
    predicted_source_execution: str | None
    predicted_fault_type: str
    opportunity_components: tuple[OpportunityComponents, ...]
    candidate_executions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NodeStatistics:
    node: str
    kind: str
    n: int
    failure_count: int
    failure_rate: float
    opportunity_count: int
    opportunity_failure_count: int
    opportunity_failure_rate: float
    downstream_failure_when_healthy: float
    downstream_failure_when_failed: float
    downstream_impact: float
    recovery_count: int
    recovery_opportunity_count: int
    weakness_score: float
    representative_failures: tuple[str, ...]


@dataclass(frozen=True)
class EdgeStatistics:
    source: str
    target: str
    n: int
    source_healthy_count: int
    degradation_count: int
    degradation_rate: float
    source_failed_count: int
    propagation_count: int
    propagation_rate: float
    recovery_count: int
    recovery_rate: float
    weakness_score: float
    representative_failures: tuple[str, ...]


@dataclass(frozen=True)
class CommonModeStatistics:
    nodes: tuple[str, str]
    n: int
    both_failed_count: int
    both_failed_rate: float
    one_failed_count: int
    one_failed_rate: float
    covariance: float
    phi: float
    weakness_score: float
    representative_failures: tuple[str, ...]


@dataclass(frozen=True)
class Diagnosis:
    workflow_id: str
    topology_type: str
    sample_count: int
    final_accuracy: float
    node_statistics: tuple[NodeStatistics, ...]
    edge_statistics: tuple[EdgeStatistics, ...]
    common_mode_statistics: tuple[CommonModeStatistics, ...]
    top_weak_node: str | None
    top_weak_edge: str | None
    top_weak_region: str | None
    weakness_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewriteProposal:
    rewrite_type: str
    target_nodes: tuple[str, ...]
    target_edges: tuple[tuple[str, str], ...]
    source_signal: str
    source_score: float
    rationale: str
    operations: tuple[Mapping[str, Any], ...]
    expected_topology_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMetrics:
    workflow_id: str
    sample_count: int
    accuracy: float
    total_tokens: int
    mean_tokens: float
    mean_latency_seconds: float
    median_latency_seconds: float
    total_agent_calls: int
    mean_agent_calls: float


@dataclass(frozen=True)
class Comparison:
    baseline: RunMetrics
    candidate: RunMetrics
    delta_accuracy: float
    delta_total_tokens: int
    token_ratio: float
    delta_mean_latency_seconds: float
    latency_ratio: float
    delta_agent_calls: int
    weakness_before: float
    weakness_after: float
    weakness_reduction: float
    error_propagation_before: float
    error_propagation_after: float
    delta_error_propagation: float
    accepted: bool
    decision_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_EVENT_FIELDS = (
    "role",
    "input",
    "actual_visible_context",
    "output",
    "incoming_edges",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_workflow(rows: Sequence[Mapping[str, Any]]) -> WorkflowGraph:
    """Infer a graph and logging contract without assuming role names."""
    if not rows:
        raise ValueError("at least one trajectory is required")
    first = rows[0]
    topology = first.get("topology") or {}
    nodes = tuple(str(node) for node in topology.get("nodes", ()))
    edges = tuple(tuple(map(str, edge)) for edge in topology.get("edges", ()))
    phases = tuple(tuple(map(str, phase)) for phase in topology.get("execution_phases", ()))
    if not nodes:
        nodes = tuple(dict.fromkeys(str(event.get("role") or event.get("agent")) for event in first["trajectory"]))
    if not edges:
        inferred: list[tuple[str, str]] = []
        for event in first["trajectory"]:
            inferred.extend(tuple(map(str, edge)) for edge in event.get("incoming_edges", ()))
        edges = tuple(dict.fromkeys(inferred))
    node_set = set(nodes)
    invalid = [edge for edge in edges if len(edge) != 2 or edge[0] not in node_set or edge[1] not in node_set]
    if invalid:
        raise ValueError(f"topology contains invalid edges: {invalid}")
    for row in rows[1:]:
        other = row.get("topology") or {}
        other_nodes = tuple(map(str, other.get("nodes", nodes)))
        other_edges = tuple(tuple(map(str, edge)) for edge in other.get("edges", edges))
        if other_nodes != nodes or other_edges != edges:
            raise ValueError("all trajectories in one diagnosis must share a topology")

    roles = {str(node): str((first.get("role_contracts") or {}).get(node, "unspecified")) for node in nodes}
    visibility: dict[str, tuple[str, ...]] = {}
    for node in nodes:
        senders: list[str] = []
        for row in rows:
            for event in row.get("trajectory", ()):
                if str(event.get("role") or event.get("agent")) == node:
                    senders.extend(str(edge[0]) for edge in event.get("incoming_edges", ()) if len(edge) == 2)
        visibility[node] = tuple(dict.fromkeys(senders)) or tuple(
            source for source, target in edges if target == node
        )
    total_events = sum(len(row.get("trajectory", ())) for row in rows)
    logging_coverage = {
        field_name: (
            sum(field_name in event for row in rows for event in row.get("trajectory", ())) / total_events
            if total_events
            else 0.0
        )
        for field_name in REQUIRED_EVENT_FIELDS
    }
    repeated_call_rate = {
        node: _rate(
            [
                sum(
                    str(event.get("role") or event.get("agent")) == node
                    for event in row.get("trajectory", ())
                )
                > 1
                for row in rows
            ]
        )
        for node in nodes
    }
    shared_state = bool(topology.get("shared_state")) or any(
        bool(event.get("shared_state") or event.get("metadata", {}).get("shared_state"))
        for row in rows
        for event in row.get("trajectory", ())
    )
    return WorkflowGraph(
        workflow_id=str(first.get("workflow_id") or first.get("condition") or "workflow"),
        nodes=nodes,
        edges=edges,
        phases=phases,
        roles=roles,
        visibility=visibility,
        routing="phase_order" if phases else "event_order",
        shared_state=shared_state,
        topology_type=classify_topology(nodes, edges, phases, shared_state, roles),
        logging_coverage=logging_coverage,
        repeated_call_rate=repeated_call_rate,
    )


def inspect_topology_config(config: Mapping[str, Any]) -> WorkflowGraph:
    """Inspect an AutoGen/GPTSwarm-style or hand-authored graph config.

    The accepted minimal shape is ``{"nodes": [...], "edges": [...]}``; a
    nested ``topology`` object and optional role contracts/phases are also
    supported. Logging coverage is zero until trajectories are supplied.
    """
    topology = config.get("topology", config)
    row = {
        "workflow_id": config.get("workflow_id", topology.get("name", "configured_workflow")),
        "topology": {
            "nodes": topology.get("nodes", ()),
            "edges": topology.get("edges", ()),
            "execution_phases": topology.get("execution_phases", topology.get("phases", ())),
            "shared_state": topology.get("shared_state", config.get("shared_state", False)),
        },
        "role_contracts": config.get("role_contracts", config.get("roles", {})),
        "trajectory": (),
    }
    return inspect_workflow([row])


def classify_topology(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    phases: Sequence[Sequence[str]],
    shared_state: bool,
    roles: Mapping[str, str],
) -> str:
    if shared_state:
        return "shared_state"
    indegree = Counter(target for _, target in edges)
    outdegree = Counter(source for source, _ in edges)
    role_values = " ".join(roles.values()).lower()
    has_judge = any(token in role_values for token in ("audit", "verify", "critic", "judge"))
    has_parallel = any(len(phase) > 1 for phase in phases)
    has_fanout = any(outdegree[node] > 1 for node in nodes)
    has_merge = any(indegree[node] > 1 for node in nodes)
    phase_index = {node: index for index, phase in enumerate(phases) for node in phase}
    has_peer_edge = any(phase_index.get(a) == phase_index.get(b) for a, b in edges)
    if has_peer_edge or ("debate" in role_values or "adversarial" in role_values):
        return "debate_adversarial"
    if has_judge and has_parallel:
        return "generate_verify"
    if has_fanout and has_merge:
        return "fan_out_merge"
    if has_parallel and any(indegree[node] > 1 for node in nodes):
        return "coordinator_worker"
    return "sequential_pipeline"


def _event_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(event.get("role") or event.get("agent")): event
        for event in row.get("trajectory", ())
    }


def _extract_labeled(text: str, label: str | None) -> str:
    if not label:
        return text.strip()
    match = re.search(rf"{re.escape(label)}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _verdict(text: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}\s*:\s*(VALID|INVALID|UNCERTAIN)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else "MISSING"


def _opportunity(spec: NodeSpec, states: Mapping[str, bool | None]) -> bool:
    values = [states.get(source) for source in spec.opportunity_sources]
    if spec.opportunity == "always":
        return True
    if spec.opportunity == "all_sources_healthy":
        return bool(values) and all(value is True for value in values)
    if spec.opportunity == "any_source_healthy":
        return any(value is True for value in values)
    if spec.opportunity == "all_sources_observed":
        return bool(values) and all(value is not None for value in values)
    raise ValueError(f"unknown opportunity rule: {spec.opportunity}")


def observe_nodes(
    row: Mapping[str, Any],
    graph: WorkflowGraph,
    specs: Mapping[str, NodeSpec],
    answer_evaluator: AnswerEvaluator,
) -> tuple[dict[str, bool | None], dict[str, bool]]:
    events = _event_map(row)
    states: dict[str, bool | None] = {}
    opportunities: dict[str, bool] = {}
    unresolved = set(graph.nodes)
    while unresolved:
        progressed = False
        for node in tuple(unresolved):
            spec = specs.get(node)
            event = events.get(node)
            if spec is None or event is None:
                states[node] = None
                opportunities[node] = False
                unresolved.remove(node)
                progressed = True
                continue
            dependencies = set(spec.opportunity_sources) | set(spec.candidates)
            if any(source in unresolved for source in dependencies):
                continue
            text = str(event.get("output", ""))
            eligible = _opportunity(spec, states)
            if spec.kind == "answer":
                raw = str(row.get("final_answer", "")) if spec.answer_source == "final_answer" else text
                answer = _extract_labeled(raw, spec.answer_label)
                state = bool(answer) and bool(answer_evaluator(answer, row))
            elif spec.kind == "contract":
                state = bool(text.strip()) and all(field.lower() in text.lower() for field in spec.required_fields)
            elif spec.kind == "judge":
                judgments = []
                for source, label in spec.candidates.items():
                    source_state = states.get(source)
                    if source_state is None:
                        continue
                    expected = "VALID" if source_state else "INVALID"
                    judgments.append(_verdict(text, label) == expected)
                state = bool(judgments) and all(judgments)
            else:
                raise ValueError(f"unknown node kind {spec.kind!r} for {node}")
            states[node] = state
            opportunities[node] = eligible
            unresolved.remove(node)
            progressed = True
        if not progressed:
            raise ValueError(f"cyclic or unresolved diagnostic dependencies: {sorted(unresolved)}")
    return states, opportunities


def _access_satisfied(requirement: AccessRequirement, event: ExecutionInstance) -> bool:
    inputs = {item.name: item for item in event.inputs}
    selected = [inputs.get(name) for name in requirement.inputs]
    if not selected or any(item is None or not item.present or not item.visible for item in selected):
        return False
    states = [item.state for item in selected if item is not None]
    if requirement.mode == "all_healthy":
        return all(state is True for state in states)
    if requirement.mode == "any_healthy":
        return any(state is True for state in states)
    if requirement.mode == "all_observed":
        return all(state is not None for state in states)
    if requirement.mode == "any_observed":
        return any(state is not None for state in states)
    raise ValueError(f"unknown access requirement mode: {requirement.mode!r}")


def opportunity_components(
    event: ExecutionInstance,
    contract: StepContract | None,
) -> OpportunityComponents:
    """Explain opportunity from activation, visibility, contract and access.

    This function deliberately receives neither topology identity nor fault
    injection metadata.  Task adapters may declare contracts, but diagnosis
    applies the same composition rules to every topology.
    """
    if event.trace_status == "missing_trace":
        return OpportunityComponents(
            event.execution_id, None, False, False, False, False, False, "missing_trace"
        )
    if event.activated is not True:
        return OpportunityComponents(
            event.execution_id,
            False,
            False,
            False,
            event.contract_applicable,
            False,
            False,
            event.skip_reason or "inactive_by_policy",
        )
    if contract is None:
        return OpportunityComponents(
            event.execution_id, True, False, False, False, False, False, "missing_contract"
        )
    inputs = {item.name: item for item in event.inputs}
    required = [inputs.get(name) for name in contract.required_inputs]
    present = all(item is not None and item.present for item in required)
    visible = present and all(item is not None and item.visible for item in required)
    recovery_items = [inputs.get(name) for name in contract.recovery_inputs]
    recovery = all(
        item is not None
        and item.present
        and item.visible
        and item.actionable is not False
        for item in recovery_items
    )
    access = all(_access_satisfied(requirement, event) for requirement in contract.access_requirements)
    opportunity = present and visible and event.contract_applicable and recovery and access
    if not present:
        reason = "required_inputs_absent"
    elif not visible:
        reason = "required_inputs_not_visible"
    elif not event.contract_applicable:
        reason = "contract_not_applicable"
    elif not recovery:
        reason = "insufficient_recovery_information"
    elif not access:
        reason = "insufficient_causal_access"
    else:
        reason = "causal_opportunity_available"
    return OpportunityComponents(
        execution_id=event.execution_id,
        activated=True,
        required_inputs_present=present,
        required_inputs_visible=visible,
        contract_applicable=event.contract_applicable,
        recovery_information_available=recovery,
        opportunity=opportunity,
        opportunity_reason=reason,
    )


def diagnose_activated(
    graph: ActivatedGraph,
    contracts: Mapping[str, StepContract | Mapping[str, Any]],
    *,
    method: str = "A2_full_opportunity",
) -> ActivatedDiagnosis:
    """Diagnose one unfolded execution graph at execution-instance level.

    ``A0_raw`` emulates last-failure attribution and may confuse skipped output
    with failure. ``A1_activation_only`` filters to executed instances.  The
    full method additionally requires declared causal opportunity.
    """
    graph.validate()
    parsed = {
        key: value if isinstance(value, StepContract) else StepContract.from_dict(key, value)
        for key, value in contracts.items()
    }
    components = tuple(
        opportunity_components(event, parsed.get(graph.topology.node(event.static_node_id).contract_id))
        for event in graph.executions
    )
    component_by_id = {item.execution_id: item for item in components}
    ordered = sorted(graph.executions, key=lambda item: item.ordinal)
    if method == "A0_raw":
        candidates = [event for event in ordered if event.local_state is not True]
        predicted = candidates[-1] if candidates else None
    elif method == "A1_activation_only":
        candidates = [event for event in ordered if event.activated is True and event.local_state is False]
        predicted = candidates[-1] if candidates else None
    elif method == "A2_full_opportunity":
        candidates = [
            event
            for event in ordered
            if event.activated is True
            and event.local_state is False
            and component_by_id[event.execution_id].opportunity
        ]
        # Pure propagation has already been removed by the opportunity gate.
        # Among remaining independent local failures, the latest one is the
        # last failed causal opportunity before the observed outcome (for
        # example, an aggregator that discards a correct sibling candidate or
        # a repair that fails despite actionable feedback).
        predicted = candidates[-1] if candidates else None
    else:
        raise ValueError(f"unknown attribution method: {method!r}")
    return ActivatedDiagnosis(
        method=method,
        sample_id=graph.sample_id,
        predicted_source_execution=predicted.execution_id if predicted else None,
        predicted_fault_type=predicted.local_failure_type if predicted else "",
        opportunity_components=components,
        candidate_executions=tuple(event.execution_id for event in candidates),
    )


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def diagnose(
    rows: Sequence[Mapping[str, Any]],
    specs: Mapping[str, NodeSpec | Mapping[str, Any]],
    answer_evaluator: AnswerEvaluator,
) -> Diagnosis:
    graph = inspect_workflow(rows)
    parsed_specs = {
        node: value if isinstance(value, NodeSpec) else NodeSpec.from_dict(value)
        for node, value in specs.items()
    }
    observations = []
    for row in rows:
        states, opportunities = observe_nodes(row, graph, parsed_specs, answer_evaluator)
        observations.append((row, states, opportunities))
    final_values = [bool(row.get("correct")) for row in rows]

    node_stats: list[NodeStatistics] = []
    for node in graph.nodes:
        known = [(row, states[node], opportunities[node]) for row, states, opportunities in observations if states[node] is not None]
        failures = [not bool(state) for _, state, _ in known]
        eligible = [(row, bool(state)) for row, state, opportunity in known if opportunity]
        opportunity_failures = [not state for _, state in eligible]
        healthy_final_fail = [not bool(row.get("correct")) for row, state, _ in known if state]
        failed_final_fail = [not bool(row.get("correct")) for row, state, _ in known if not state]
        impact = max(0.0, _rate(failed_final_fail) - _rate(healthy_final_fail))
        support = len(eligible) / len(rows) if rows else 0.0
        weakness = _rate(opportunity_failures) * support * (0.5 + 0.5 * impact)
        recovery_pool = [row for row, state, opportunity in known if not opportunity and state]
        recovery_opportunities = [row for row, _, opportunity in known if not opportunity]
        reps = tuple(
            str(row.get("sample_id"))
            for row, state, opportunity in known
            if opportunity and not state and not row.get("correct")
        )[:5]
        node_stats.append(
            NodeStatistics(
                node=node,
                kind=parsed_specs.get(node, NodeSpec("unknown")).kind,
                n=len(known),
                failure_count=sum(failures),
                failure_rate=_rate(failures),
                opportunity_count=len(eligible),
                opportunity_failure_count=sum(opportunity_failures),
                opportunity_failure_rate=_rate(opportunity_failures),
                downstream_failure_when_healthy=_rate(healthy_final_fail),
                downstream_failure_when_failed=_rate(failed_final_fail),
                downstream_impact=impact,
                recovery_count=len(recovery_pool),
                recovery_opportunity_count=len(recovery_opportunities),
                weakness_score=weakness,
                representative_failures=reps,
            )
        )

    edge_stats: list[EdgeStatistics] = []
    for source, target in graph.edges:
        pairs = [
            (row, states.get(source), states.get(target))
            for row, states, _ in observations
            if states.get(source) is not None and states.get(target) is not None
        ]
        source_healthy = [(row, bool(target_state)) for row, source_state, target_state in pairs if source_state]
        source_failed = [(row, bool(target_state)) for row, source_state, target_state in pairs if not source_state]
        degradation = [not state for _, state in source_healthy]
        propagation = [not state for _, state in source_failed]
        recovery = [state for _, state in source_failed]
        final_harm = _rate([not bool(row.get("correct")) for row, state in source_healthy if not state])
        weakness = _rate(degradation) * (len(source_healthy) / len(rows) if rows else 0.0) * (0.5 + 0.5 * final_harm)
        reps = tuple(str(row.get("sample_id")) for row, state in source_healthy if not state and not row.get("correct"))[:5]
        edge_stats.append(
            EdgeStatistics(
                source=source,
                target=target,
                n=len(pairs),
                source_healthy_count=len(source_healthy),
                degradation_count=sum(degradation),
                degradation_rate=_rate(degradation),
                source_failed_count=len(source_failed),
                propagation_count=sum(propagation),
                propagation_rate=_rate(propagation),
                recovery_count=sum(recovery),
                recovery_rate=_rate(recovery),
                weakness_score=weakness,
                representative_failures=reps,
            )
        )

    answer_nodes = [node for node, spec in parsed_specs.items() if spec.kind == "answer"]
    common_stats: list[CommonModeStatistics] = []
    for left, right in combinations(answer_nodes, 2):
        same_phase = any(left in phase and right in phase for phase in graph.phases)
        # Common-mode and disagreement statistics are meaningful for peer
        # candidates, not for causally linked answer nodes later in the graph.
        # Requiring a declared parallel phase prevents a downstream refiner
        # from being misdiagnosed as a sibling worker merely because it shares
        # one ancestor with that worker.
        if not same_phase:
            continue
        pairs = [
            (row, bool(states[left]), bool(states[right]))
            for row, states, _ in observations
            if states.get(left) is not None and states.get(right) is not None
        ]
        if not pairs:
            continue
        left_errors = [not l for _, l, _ in pairs]
        right_errors = [not r for _, _, r in pairs]
        both = [l and r for l, r in zip(left_errors, right_errors)]
        one = [l != r for l, r in zip(left_errors, right_errors)]
        covariance = _rate(both) - _rate(left_errors) * _rate(right_errors)
        table = Counter((l, r) for _, l, r in pairs)
        cc, cw, wc, ww = table[(True, True)], table[(True, False)], table[(False, True)], table[(False, False)]
        denominator = math.sqrt((cc + cw) * (wc + ww) * (cc + wc) * (cw + ww))
        phi = (cc * ww - cw * wc) / denominator if denominator else 0.0
        weakness = _rate(both) * max(0.0, phi)
        reps = tuple(str(row.get("sample_id")) for (row, _, _), failed in zip(pairs, both) if failed and not row.get("correct"))[:5]
        common_stats.append(
            CommonModeStatistics(
                nodes=(left, right),
                n=len(pairs),
                both_failed_count=sum(both),
                both_failed_rate=_rate(both),
                one_failed_count=sum(one),
                one_failed_rate=_rate(one),
                covariance=covariance,
                phi=phi,
                weakness_score=weakness,
                representative_failures=reps,
            )
        )

    node_stats.sort(key=lambda item: (-item.weakness_score, item.node))
    edge_stats.sort(key=lambda item: (-item.weakness_score, item.source, item.target))
    common_stats.sort(key=lambda item: (-item.weakness_score, item.nodes))
    candidates: list[tuple[float, str]] = []
    if node_stats:
        candidates.append((node_stats[0].weakness_score, f"node:{node_stats[0].node}"))
    if edge_stats:
        candidates.append((edge_stats[0].weakness_score, f"edge:{edge_stats[0].source}->{edge_stats[0].target}"))
    if common_stats:
        candidates.append((common_stats[0].weakness_score, "subgraph:" + "+".join(common_stats[0].nodes)))
    top_score, top_region = max(candidates, default=(0.0, None))
    return Diagnosis(
        workflow_id=graph.workflow_id,
        topology_type=graph.topology_type,
        sample_count=len(rows),
        final_accuracy=_rate(final_values),
        node_statistics=tuple(node_stats),
        edge_statistics=tuple(edge_stats),
        common_mode_statistics=tuple(common_stats),
        top_weak_node=node_stats[0].node if node_stats else None,
        top_weak_edge=f"{edge_stats[0].source}->{edge_stats[0].target}" if edge_stats else None,
        top_weak_region=top_region,
        weakness_score=top_score,
    )


def select_local_rewrite(
    diagnosis: Diagnosis,
    graph: WorkflowGraph,
    attempted_rewrites: Sequence[str] = (),
) -> RewriteProposal:
    """Select one evidence-backed local rewrite; never inspect gold answers."""
    attempted = set(attempted_rewrites)
    candidates: list[tuple[float, RewriteProposal]] = []
    if "parallel_workers_to_adversarial_debate" not in attempted:
        for common in diagnosis.common_mode_statistics:
            left, right = common.nodes
            conflict_score = 0.5 * common.one_failed_rate
            source_score = max(common.weakness_score, conflict_score)
            if conflict_score > common.weakness_score:
                source_signal = "candidate_conflict"
                rationale = (
                    f"{left} and {right} disagree in {common.one_failed_count}/{common.n} "
                    "trajectories; expose one result to the other for adversarial conflict resolution."
                )
            else:
                source_signal = "common_mode_failure"
                rationale = (
                    f"{left} and {right} fail together in {common.both_failed_count}/{common.n} "
                    f"trajectories with phi={common.phi:.3f}; expose one result to the other for "
                    "an adversarial second pass."
                )
            candidates.append(
                (
                    source_score,
                    RewriteProposal(
                        rewrite_type="parallel_workers_to_adversarial_debate",
                        target_nodes=common.nodes,
                        target_edges=(),
                        source_signal=source_signal,
                        source_score=source_score,
                        rationale=rationale,
                        operations=(
                            {"op": "add_edge", "source": left, "target": right},
                            {"op": "sequentialize", "first": left, "second": right},
                            {"op": "set_role_mode", "node": right, "mode": "adversarial_debate"},
                        ),
                        expected_topology_type="debate_adversarial",
                    ),
                )
            )
    judge_nodes = [row for row in diagnosis.node_statistics if row.kind == "judge"]
    if judge_nodes and "single_verifier_to_multiple_verifiers" not in attempted:
        judge = judge_nodes[0]
        candidates.append(
            (
                judge.weakness_score,
                RewriteProposal(
                    rewrite_type="single_verifier_to_multiple_verifiers",
                    target_nodes=(judge.node,),
                    target_edges=tuple(edge for edge in graph.edges if edge[1] == judge.node),
                    source_signal="opportunity_conditioned_judge_failure",
                    source_score=judge.weakness_score,
                    rationale=(
                        f"{judge.node} fails on {judge.opportunity_failure_count}/{judge.opportunity_count} "
                        "eligible trajectories; add one heterogeneous peer verifier and preserve the existing verifier."
                    ),
                    operations=(
                        {"op": "clone_as_peer", "node": judge.node, "suffix": "_Peer"},
                        {"op": "fanout_incoming_edges", "node": judge.node, "suffix": "_Peer"},
                        {"op": "connect_peer_to_successors", "node": judge.node, "suffix": "_Peer"},
                    ),
                    expected_topology_type="generate_verify",
                ),
            )
        )
    kind_by_node = {row.node: row.kind for row in diagnosis.node_statistics}
    merge_nodes = [
        row
        for row in diagnosis.node_statistics
        if row.kind == "answer"
        and len(graph.predecessors(row.node)) >= 2
        and sum(
            kind_by_node.get(source) == "answer"
            for source in graph.predecessors(row.node)
        ) >= 2
    ]
    if merge_nodes and "simple_merge_to_coordinator" not in attempted:
        merge = merge_nodes[0]
        candidates.append(
            (
                merge.weakness_score,
                RewriteProposal(
                    rewrite_type="simple_merge_to_coordinator",
                    target_nodes=(merge.node,),
                    target_edges=tuple(edge for edge in graph.edges if edge[1] == merge.node),
                    source_signal="merge_degradation",
                    source_score=merge.weakness_score,
                    rationale=f"{merge.node} degrades an available correct upstream state; replace implicit merge with explicit scored selection.",
                    operations=({"op": "set_role_mode", "node": merge.node, "mode": "coordinator_scoring"},),
                    expected_topology_type="coordinator_worker",
                ),
            )
        )
    generator_nodes = [
        row
        for row in diagnosis.node_statistics
        if row.kind == "answer"
        and graph.successors(row.node)
        and (
            any(kind_by_node.get(target) == "judge" for target in graph.successors(row.node))
            or any(
                row.node in phase
                and any(other != row.node and kind_by_node.get(other) == "answer" for other in phase)
                for phase in graph.phases
            )
        )
    ]
    if generator_nodes and "single_worker_to_fanout" not in attempted:
        generator = generator_nodes[0]
        candidates.append(
            (
                generator.weakness_score,
                RewriteProposal(
                    rewrite_type="single_worker_to_fanout",
                    target_nodes=(generator.node,),
                    target_edges=(),
                    source_signal="generator_failure",
                    source_score=generator.weakness_score,
                    rationale=f"{generator.node} has high opportunity-conditioned failure; add one heterogeneous peer.",
                    operations=({"op": "clone_as_heterogeneous_worker", "node": generator.node},),
                    expected_topology_type="fan_out_merge",
                ),
            )
        )
    if graph.topology_type == "sequential_pipeline" and diagnosis.edge_statistics:
        edge = diagnosis.edge_statistics[0]
        score = edge.propagation_rate * (edge.source_failed_count / edge.n if edge.n else 0.0)
        if score > 0 and "long_chain_to_generate_verify" not in attempted:
            candidates.append(
                (
                    score,
                    RewriteProposal(
                        rewrite_type="long_chain_to_generate_verify",
                        target_nodes=(edge.source, edge.target),
                        target_edges=((edge.source, edge.target),),
                        source_signal="downstream_error_propagation",
                        source_score=score,
                        rationale=(
                            f"{edge.source}->{edge.target} propagates {edge.propagation_count}/"
                            f"{edge.source_failed_count} observed source failures; add one local verification stage."
                        ),
                        operations=(
                            {"op": "insert_verifier_on_edge", "source": edge.source, "target": edge.target},
                        ),
                        expected_topology_type="generate_verify",
                    ),
                )
            )
    repeated_nodes = tuple(
        node for node, rate in graph.repeated_call_rate.items() if rate >= 0.2
    )
    if (
        repeated_nodes
        and not graph.shared_state
        and "independent_branches_to_shared_state" not in attempted
    ):
        score = max(graph.repeated_call_rate[node] for node in repeated_nodes)
        if score > 0:
            candidates.append(
                (
                    score,
                    RewriteProposal(
                        rewrite_type="independent_branches_to_shared_state",
                        target_nodes=repeated_nodes,
                        target_edges=tuple(
                            edge
                            for edge in graph.edges
                            if edge[0] in repeated_nodes or edge[1] in repeated_nodes
                        ),
                        source_signal="repeated_intermediate_state_exchange",
                        source_score=score,
                        rationale=(
                            "Repeated calls to branch nodes indicate iterative intermediate-state exchange; "
                            "expose a bounded shared state."
                        ),
                        operations=(
                            {"op": "enable_shared_state", "nodes": repeated_nodes},
                        ),
                        expected_topology_type="shared_state",
                    ),
                )
            )
    if not candidates:
        raise ValueError("no supported local rewrite remains for the diagnosed topology")
    return max(candidates, key=lambda item: item[0])[1]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_metrics(rows: Sequence[Mapping[str, Any]]) -> RunMetrics:
    if not rows:
        raise ValueError("at least one trajectory is required")
    latencies = []
    calls = 0
    for row in rows:
        events = list(row.get("trajectory", ()))
        calls += len(events)
        if events:
            started = min(_parse_time(str(event["started_at"])) for event in events)
            finished = max(_parse_time(str(event["finished_at"])) for event in events)
            latencies.append((finished - started).total_seconds())
    ordered = sorted(latencies)
    middle = len(ordered) // 2
    median = (
        (ordered[middle - 1] + ordered[middle]) / 2
        if len(ordered) % 2 == 0
        else ordered[middle]
    ) if ordered else 0.0
    total_tokens = sum(int(row.get("token_usage", {}).get("total_tokens", 0)) for row in rows)
    return RunMetrics(
        workflow_id=str(rows[0].get("workflow_id") or rows[0].get("condition") or "workflow"),
        sample_count=len(rows),
        accuracy=_rate([bool(row.get("correct")) for row in rows]),
        total_tokens=total_tokens,
        mean_tokens=total_tokens / len(rows),
        mean_latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
        median_latency_seconds=median,
        total_agent_calls=calls,
        mean_agent_calls=calls / len(rows),
    )


def compare_runs(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_diagnosis: Diagnosis,
    candidate_diagnosis: Diagnosis,
    *,
    max_token_ratio: float = 1.35,
    max_latency_ratio: float = 1.50,
    min_weakness_reduction: float = 0.05,
) -> Comparison:
    baseline_id_list = [str(row.get("sample_id")) for row in baseline_rows]
    candidate_id_list = [str(row.get("sample_id")) for row in candidate_rows]
    if len(set(baseline_id_list)) != len(baseline_id_list):
        raise ValueError("baseline contains duplicate sample IDs")
    if len(set(candidate_id_list)) != len(candidate_id_list):
        raise ValueError("candidate contains duplicate sample IDs")
    baseline_ids = set(baseline_id_list)
    candidate_ids = set(candidate_id_list)
    if baseline_ids != candidate_ids:
        raise ValueError("baseline and candidate must use exactly the same sample IDs")

    def evaluation_contract(row: Mapping[str, Any]) -> tuple[Any, ...]:
        dataset = row.get("dataset") or {}
        final_evaluation = row.get("final_evaluation") or {}
        return (
            str(row.get("sample_id")),
            dataset.get("name"),
            dataset.get("revision"),
            dataset.get("split"),
            dataset.get("dataset_index"),
            final_evaluation.get("evaluator_version"),
        )

    baseline_contracts = {evaluation_contract(row) for row in baseline_rows}
    candidate_contracts = {evaluation_contract(row) for row in candidate_rows}
    if baseline_contracts != candidate_contracts:
        raise ValueError(
            "baseline and candidate must use the same dataset revision, indices, and evaluator version"
        )
    baseline = run_metrics(baseline_rows)
    candidate = run_metrics(candidate_rows)
    token_ratio = (
        candidate.total_tokens / baseline.total_tokens
        if baseline.total_tokens
        else (1.0 if candidate.total_tokens == 0 else math.inf)
    )
    latency_ratio = (
        candidate.mean_latency_seconds / baseline.mean_latency_seconds
        if baseline.mean_latency_seconds
        else (1.0 if candidate.mean_latency_seconds == 0 else math.inf)
    )
    delta_accuracy = candidate.accuracy - baseline.accuracy
    weakness_reduction = baseline_diagnosis.weakness_score - candidate_diagnosis.weakness_score
    baseline_failed_edges = sum(
        edge.source_failed_count for edge in baseline_diagnosis.edge_statistics
    )
    candidate_failed_edges = sum(
        edge.source_failed_count for edge in candidate_diagnosis.edge_statistics
    )
    propagation_before = (
        sum(edge.propagation_count for edge in baseline_diagnosis.edge_statistics)
        / baseline_failed_edges
        if baseline_failed_edges
        else 0.0
    )
    propagation_after = (
        sum(edge.propagation_count for edge in candidate_diagnosis.edge_statistics)
        / candidate_failed_edges
        if candidate_failed_edges
        else 0.0
    )
    cost_ok = token_ratio <= max_token_ratio and latency_ratio <= max_latency_ratio
    improved = delta_accuracy > 0 or weakness_reduction >= min_weakness_reduction
    accepted = cost_ok and improved
    if not cost_ok:
        reason = "rejected: token or latency budget exceeded"
    elif delta_accuracy > 0:
        reason = "accepted: task accuracy improved within budget"
    elif weakness_reduction >= min_weakness_reduction:
        reason = "accepted: weakness score decreased materially within budget"
    else:
        reason = "rejected: neither accuracy nor weakness score improved enough"
    return Comparison(
        baseline=baseline,
        candidate=candidate,
        delta_accuracy=delta_accuracy,
        delta_total_tokens=candidate.total_tokens - baseline.total_tokens,
        token_ratio=token_ratio,
        delta_mean_latency_seconds=candidate.mean_latency_seconds - baseline.mean_latency_seconds,
        latency_ratio=latency_ratio,
        delta_agent_calls=candidate.total_agent_calls - baseline.total_agent_calls,
        weakness_before=baseline_diagnosis.weakness_score,
        weakness_after=candidate_diagnosis.weakness_score,
        weakness_reduction=weakness_reduction,
        error_propagation_before=propagation_before,
        error_propagation_after=propagation_after,
        delta_error_propagation=propagation_after - propagation_before,
        accepted=accepted,
        decision_reason=reason,
    )


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
