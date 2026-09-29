import json
from pathlib import Path

import pytest

from entroflow.topology import StaticTopology, load_topology_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments/compositional_topology/topology_registry.json"


def registry():
    return load_topology_registry(json.loads(REGISTRY.read_text(encoding="utf-8")))


def test_registry_contains_only_scoped_pilot_topologies():
    assert set(registry()) == {
        "p1_sequence_v1",
        "p2_fork_join_v1",
        "p3a_oracle_gated_select_v1",
        "p4_bounded_feedback_v1",
        "c1_parallel_conditional_repair_v1",
    }


def test_feedback_static_graph_is_valid_when_feedback_edge_is_explicit():
    topology = registry()["p4_bounded_feedback_v1"]
    assert topology.max_iterations == 2
    assert sum(edge.kind == "feedback" for edge in topology.edges) == 1
    assert topology.descriptors()["maximum_iterations"] == 2


def test_non_feedback_cycle_is_rejected():
    with pytest.raises(ValueError, match="DAG"):
        StaticTopology.from_dict(
            {
                "topology_id": "bad",
                "primitives": ["sequence"],
                "nodes": [{"id": "a"}, {"id": "b"}],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "a"},
                ],
                "entry_nodes": ["a"],
                "terminal_nodes": ["b"],
            }
        )


def test_shared_state_and_c2_are_deferred():
    text = REGISTRY.read_text(encoding="utf-8").casefold()
    assert "sharedstate" not in text
    assert '"c2_' not in text

