"""Per-sample activated execution graphs.

Execution instances use ``static_node_id@iteration`` identities.  Feedback is
therefore represented as an unfolded DAG even when the static topology has a
bounded feedback edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .topology import StaticTopology

TRACE_SCHEMA_VERSION = "activated-execution-v1"


@dataclass(frozen=True)
class ExecutionInput:
    name: str
    source_execution_id: str | None
    present: bool
    visible: bool
    state: bool | None
    actionable: bool | None = None
    value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionInstance:
    static_node_id: str
    execution_id: str
    role: str
    iteration: int
    ordinal: int
    activated: bool | None
    activation_reason: str = ""
    skip_reason: str = ""
    trace_status: str = "activated"
    inputs: tuple[ExecutionInput, ...] = ()
    output: Any = None
    local_state: bool | None = None
    local_failure_type: str = ""
    contract_applicable: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    call_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = f"{self.static_node_id}@{self.iteration}"
        if self.execution_id != expected:
            raise ValueError(f"execution_id must be {expected!r}, got {self.execution_id!r}")
        if self.iteration < 0 or self.ordinal < 0:
            raise ValueError("iteration and ordinal must be non-negative")
        valid = {"activated", "inactive_by_policy", "missing_trace"}
        if self.trace_status not in valid:
            raise ValueError(f"invalid trace_status: {self.trace_status!r}")
        if self.trace_status == "activated" and self.activated is not True:
            raise ValueError("activated trace events require activated=True")
        if self.trace_status == "inactive_by_policy" and self.activated is not False:
            raise ValueError("inactive trace events require activated=False")
        if self.trace_status == "missing_trace" and self.activated is not None:
            raise ValueError("missing trace events require activated=None")
        if self.call_count < 0:
            raise ValueError("call_count cannot be negative")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ActivatedEdge:
    static_edge_id: str
    source_execution_id: str
    target_execution_id: str
    traversed: bool
    condition: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ControlDecision:
    decision_id: str
    execution_id: str
    decision_type: str
    value: str
    condition_observed: str
    correct: bool | None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActivatedGraph:
    run_id: str
    sample_id: str
    topology: StaticTopology
    executions: tuple[ExecutionInstance, ...]
    edges: tuple[ActivatedEdge, ...]
    decisions: tuple[ControlDecision, ...] = ()
    schema_version: str = TRACE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported activated trace schema: {self.schema_version!r}")
        execution_ids = [event.execution_id for event in self.executions]
        if len(set(execution_ids)) != len(execution_ids):
            raise ValueError("execution IDs must be unique")
        ordinals = [event.ordinal for event in self.executions]
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("execution ordinals must be unique")
        static_nodes = set(self.topology.node_ids)
        unknown = [event.execution_id for event in self.executions if event.static_node_id not in static_nodes]
        if unknown:
            raise ValueError(f"activated graph references unknown static nodes: {unknown}")
        by_id = {event.execution_id: event for event in self.executions}
        static_edges = {edge.edge_id for edge in self.topology.edges}
        for edge in self.edges:
            if edge.static_edge_id not in static_edges:
                raise ValueError(f"unknown static edge: {edge.static_edge_id}")
            if edge.source_execution_id not in by_id or edge.target_execution_id not in by_id:
                raise ValueError("activated edge endpoint is not an execution instance")
            if edge.traversed and by_id[edge.source_execution_id].ordinal >= by_id[edge.target_execution_id].ordinal:
                raise ValueError("unfolded activated graph must be acyclic and ordinal-forward")
        for decision in self.decisions:
            if decision.execution_id not in by_id:
                raise ValueError("control decision references an unknown execution")
        rounds = [event.iteration for event in self.executions if event.activated and event.static_node_id == "repair"]
        if rounds and max(rounds) + 1 > self.topology.max_iterations:
            raise ValueError("activated repair rounds exceed the configured maximum")

    def execution(self, execution_id: str) -> ExecutionInstance:
        return next(event for event in self.executions if event.execution_id == execution_id)

    def activated_executions(self) -> tuple[ExecutionInstance, ...]:
        return tuple(event for event in self.executions if event.activated is True)

    def traversed_edges(self) -> tuple[ActivatedEdge, ...]:
        return tuple(edge for edge in self.edges if edge.traversed)

    def descriptors(self) -> dict[str, Any]:
        active = self.activated_executions()
        traversed = self.traversed_edges()
        adjacency: dict[str, list[str]] = {}
        indegree = {event.execution_id: 0 for event in active}
        for edge in traversed:
            adjacency.setdefault(edge.source_execution_id, []).append(edge.target_execution_id)
            indegree[edge.target_execution_id] = indegree.get(edge.target_execution_id, 0) + 1
        roots = [node for node, degree in indegree.items() if degree == 0]
        depths = {node: 0 for node in roots}
        for event in sorted(active, key=lambda item: item.ordinal):
            for target in adjacency.get(event.execution_id, ()):
                depths[target] = max(depths.get(target, 0), depths.get(event.execution_id, 0) + 1)
        terminal_ids = {
            event.execution_id
            for event in active
            if event.static_node_id in self.topology.terminal_nodes
            or not adjacency.get(event.execution_id)
        }
        lengths = [depths.get(node, 0) for node in terminal_ids]
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "topology_id": self.topology.topology_id,
            "activated_node_count": len(active),
            "activated_edge_count": len(traversed),
            "mean_activated_path_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "maximum_activated_path_length": max(lengths, default=0),
            "llm_calls": sum(event.call_count for event in active),
            "tokens": sum(event.total_tokens for event in active),
            "latency_ms": sum(event.latency_ms for event in active),
            "iterations_used": 1 + max((event.iteration for event in active), default=-1),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivatedGraphBuilder:
    """Small append-only builder used by task adapters and deterministic tests."""

    def __init__(self, run_id: str, sample_id: str, topology: StaticTopology, metadata: Mapping[str, Any] | None = None):
        self.run_id = run_id
        self.sample_id = sample_id
        self.topology = topology
        self.metadata = dict(metadata or {})
        self._executions: list[ExecutionInstance] = []
        self._edges: list[ActivatedEdge] = []
        self._decisions: list[ControlDecision] = []

    def add_execution(self, event: ExecutionInstance) -> None:
        if any(existing.execution_id == event.execution_id for existing in self._executions):
            raise ValueError(f"duplicate execution instance: {event.execution_id}")
        self._executions.append(event)

    def add_edge(self, edge: ActivatedEdge) -> None:
        self._edges.append(edge)

    def add_decision(self, decision: ControlDecision) -> None:
        self._decisions.append(decision)

    def build(self) -> ActivatedGraph:
        graph = ActivatedGraph(
            run_id=self.run_id,
            sample_id=self.sample_id,
            topology=self.topology,
            executions=tuple(sorted(self._executions, key=lambda item: item.ordinal)),
            edges=tuple(self._edges),
            decisions=tuple(self._decisions),
            metadata=self.metadata,
        )
        graph.validate()
        return graph

