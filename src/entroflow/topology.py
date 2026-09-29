"""Declarative topology primitives for compositional workflow experiments.

This module describes the *possible* graph only.  Per-sample execution belongs
in :mod:`entroflow.execution`; keeping the two separate prevents a skipped
conditional branch from being mistaken for a missing trace event.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

TOPOLOGY_SCHEMA_VERSION = "compositional-topology-v1"
PRIMITIVES = frozenset({"sequence", "fork", "join", "select", "iterate", "terminate", "delegate"})
EDGE_KINDS = frozenset({"data", "control", "feedback"})


@dataclass(frozen=True)
class StaticNode:
    node_id: str
    role: str
    contract_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StaticNode:
        node_id = str(value["id"])
        return cls(
            node_id=node_id,
            role=str(value.get("role", node_id)),
            contract_id=str(value.get("contract_id", node_id)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class StaticEdge:
    edge_id: str
    source: str
    target: str
    kind: str = "data"
    condition: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], ordinal: int) -> StaticEdge:
        source, target = str(value["source"]), str(value["target"])
        condition = value.get("condition")
        edge_id = str(value.get("id") or f"e{ordinal}:{source}->{target}")
        return cls(
            edge_id=edge_id,
            source=source,
            target=target,
            kind=str(value.get("kind", "data")),
            condition=None if condition is None else str(condition),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class StaticTopology:
    """A registry entry describing all paths a workflow may activate."""

    topology_id: str
    nodes: tuple[StaticNode, ...]
    edges: tuple[StaticEdge, ...]
    entry_nodes: tuple[str, ...]
    terminal_nodes: tuple[str, ...]
    primitives: tuple[str, ...]
    max_iterations: int = 0
    schema_version: str = TOPOLOGY_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StaticTopology:
        topology = cls(
            topology_id=str(value["topology_id"]),
            nodes=tuple(StaticNode.from_dict(node) for node in value.get("nodes", ())),
            edges=tuple(
                StaticEdge.from_dict(edge, index)
                for index, edge in enumerate(value.get("edges", ()))
            ),
            entry_nodes=tuple(map(str, value.get("entry_nodes", ()))),
            terminal_nodes=tuple(map(str, value.get("terminal_nodes", ()))),
            primitives=tuple(map(str, value.get("primitives", ()))),
            max_iterations=int(value.get("max_iterations", 0)),
            schema_version=str(value.get("schema_version", TOPOLOGY_SCHEMA_VERSION)),
            metadata=dict(value.get("metadata", {})),
        )
        topology.validate()
        return topology

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def node(self, node_id: str) -> StaticNode:
        return next(node for node in self.nodes if node.node_id == node_id)

    def validate(self) -> None:
        if self.schema_version != TOPOLOGY_SCHEMA_VERSION:
            raise ValueError(f"unsupported topology schema: {self.schema_version!r}")
        if not self.topology_id or not self.nodes:
            raise ValueError("topology_id and at least one node are required")
        node_ids = self.node_ids
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("static node IDs must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("static edge IDs must be unique")
        unknown_edges = [
            edge.edge_id
            for edge in self.edges
            if edge.source not in node_ids or edge.target not in node_ids
        ]
        if unknown_edges:
            raise ValueError(f"edges reference unknown nodes: {unknown_edges}")
        bad_kinds = [edge.kind for edge in self.edges if edge.kind not in EDGE_KINDS]
        if bad_kinds:
            raise ValueError(f"unsupported edge kinds: {bad_kinds}")
        unknown_primitives = set(self.primitives) - PRIMITIVES
        if unknown_primitives:
            raise ValueError(f"unsupported primitives: {sorted(unknown_primitives)}")
        if set(self.entry_nodes + self.terminal_nodes) - set(node_ids):
            raise ValueError("entry_nodes and terminal_nodes must be declared nodes")
        if "iterate" in self.primitives and self.max_iterations < 1:
            raise ValueError("iterate topologies require max_iterations >= 1")
        if self.max_iterations > 2:
            raise ValueError("the pilot bounds feedback to at most two iterations")
        self._validate_non_feedback_dag()

    def _validate_non_feedback_dag(self) -> None:
        adjacency: dict[str, list[str]] = defaultdict(list)
        indegree = {node: 0 for node in self.node_ids}
        for edge in self.edges:
            if edge.kind == "feedback":
                continue
            adjacency[edge.source].append(edge.target)
            indegree[edge.target] += 1
        queue = deque(node for node, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            for target in adjacency[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(self.nodes):
            raise ValueError("non-feedback static edges must form a DAG")

    def descriptors(self) -> dict[str, Any]:
        """Return raw structural descriptors; no scalar complexity is imposed."""
        indegree = Counter(edge.target for edge in self.edges if edge.kind != "feedback")
        outdegree = Counter(edge.source for edge in self.edges if edge.kind != "feedback")
        adjacency: dict[str, list[str]] = defaultdict(list)
        degree = {node: 0 for node in self.node_ids}
        for edge in self.edges:
            if edge.kind == "feedback":
                continue
            adjacency[edge.source].append(edge.target)
            degree[edge.target] += 1
        queue = deque(node for node, value in degree.items() if value == 0)
        depth = {node: 0 for node in self.node_ids}
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                depth[target] = max(depth[target], depth[source] + 1)
                degree[target] -= 1
                if degree[target] == 0:
                    queue.append(target)
        return {
            "topology_id": self.topology_id,
            "schema_version": self.schema_version,
            "num_nodes": len(self.nodes),
            "num_possible_edges": len(self.edges),
            "maximum_depth_without_feedback": max(depth.values(), default=0),
            "maximum_branching_factor": max(outdegree.values(), default=0),
            "fan_in_points": sum(value > 1 for value in indegree.values()),
            "conditional_decisions": sum(edge.condition is not None for edge in self.edges),
            "feedback_edges": sum(edge.kind == "feedback" for edge in self.edges),
            "maximum_iterations": self.max_iterations,
            "primitives": list(self.primitives),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_topology_registry(payload: Mapping[str, Any]) -> dict[str, StaticTopology]:
    version = str(payload.get("schema_version", TOPOLOGY_SCHEMA_VERSION))
    if version != TOPOLOGY_SCHEMA_VERSION:
        raise ValueError(f"unsupported registry schema: {version!r}")
    result = {
        str(item["topology_id"]): StaticTopology.from_dict({**item, "schema_version": version})
        for item in payload.get("topologies", ())
    }
    if len(result) != len(tuple(payload.get("topologies", ()))):
        raise ValueError("topology IDs in one registry must be unique")
    return result
