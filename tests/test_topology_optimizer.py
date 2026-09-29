import copy

from entroflow.topology_optimizer import (
    NodeSpec,
    compare_runs,
    diagnose,
    inspect_topology_config,
    inspect_workflow,
    select_local_rewrite,
)


def row(sample_id, left, right, judge, final, correct):
    topology = {
        "nodes": ["alpha", "beta", "arbiter", "finish"],
        "edges": [
            ["alpha", "arbiter"],
            ["beta", "arbiter"],
            ["arbiter", "finish"],
        ],
        "execution_phases": [["alpha", "beta"], ["arbiter"], ["finish"]],
    }
    outputs = {
        "alpha": f"Candidate: {left}",
        "beta": f"Candidate: {right}",
        "arbiter": judge,
        "finish": f"Final: {final}",
    }
    incoming = {
        "alpha": [],
        "beta": [],
        "arbiter": [["alpha", "arbiter"], ["beta", "arbiter"]],
        "finish": [["arbiter", "finish"]],
    }
    events = []
    for index, role in enumerate(topology["nodes"]):
        events.append(
            {
                "role": role,
                "input": "problem",
                "actual_visible_context": [{"role": "user", "content": "problem"}],
                "output": outputs[role],
                "incoming_edges": incoming[role],
                "started_at": f"2026-01-01T00:00:0{index}+00:00",
                "finished_at": f"2026-01-01T00:00:0{index + 1}+00:00",
            }
        )
    return {
        "workflow_id": "arbitrary_names",
        "sample_id": sample_id,
        "topology": topology,
        "role_contracts": {
            "alpha": "answer",
            "beta": "answer",
            "arbiter": "judge",
            "finish": "answer",
        },
        "trajectory": events,
        "final_answer": final,
        "correct": correct,
        "gold": "1",
        "token_usage": {"total_tokens": 100},
    }


SPECS = {
    "alpha": NodeSpec("answer", answer_label="Candidate"),
    "beta": NodeSpec("answer", answer_label="Candidate"),
    "arbiter": NodeSpec(
        "judge",
        candidates={"alpha": "A verdict", "beta": "B verdict"},
        opportunity="all_sources_observed",
        opportunity_sources=("alpha", "beta"),
    ),
    "finish": NodeSpec(
        "answer",
        answer_source="final_answer",
        opportunity="all_sources_healthy",
        opportunity_sources=("arbiter",),
    ),
}


def evaluate(answer, payload):
    return answer == payload["gold"]


def fixture_rows():
    return [
        row("one", "1", "1", "A verdict: VALID\nB verdict: VALID", "1", True),
        row("two", "0", "0", "A verdict: VALID\nB verdict: VALID", "0", False),
        row("three", "0", "0", "A verdict: INVALID\nB verdict: INVALID", "0", False),
        row("four", "1", "1", "A verdict: VALID\nB verdict: VALID", "1", True),
    ]


def test_inspector_uses_declared_graph_not_fixed_agent_names():
    graph = inspect_workflow(fixture_rows())
    assert graph.nodes == ("alpha", "beta", "arbiter", "finish")
    assert graph.visibility["arbiter"] == ("alpha", "beta")
    assert graph.logging_coverage["output"] == 1.0


def test_hand_authored_config_is_supported_without_trajectories():
    graph = inspect_topology_config(
        {
            "workflow_id": "manual",
            "nodes": ["x", "y"],
            "edges": [["x", "y"]],
            "phases": [["x"], ["y"]],
        }
    )
    assert graph.visibility["y"] == ("x",)
    assert graph.topology_type == "sequential_pipeline"
    assert graph.logging_coverage["output"] == 0.0


def test_diagnosis_finds_common_mode_and_not_just_final_node():
    diagnosis = diagnose(fixture_rows(), SPECS, evaluate)
    assert diagnosis.common_mode_statistics[0].nodes == ("alpha", "beta")
    assert diagnosis.common_mode_statistics[0].both_failed_count == 2
    assert diagnosis.top_weak_region.startswith("subgraph:")
    assert diagnosis.node_statistics[0].node != "finish"


def test_selector_maps_correlated_workers_to_one_local_debate_rewrite():
    rows = fixture_rows()
    graph = inspect_workflow(rows)
    diagnosis = diagnose(rows, SPECS, evaluate)
    proposal = select_local_rewrite(diagnosis, graph)
    assert proposal.rewrite_type == "parallel_workers_to_adversarial_debate"
    assert proposal.target_nodes == ("alpha", "beta")
    assert sum(operation["op"] == "add_edge" for operation in proposal.operations) == 1


def test_comparison_requires_same_samples_and_applies_budget():
    baseline = fixture_rows()
    candidate = fixture_rows()
    for item in candidate:
        item["workflow_id"] = "candidate"
        item["token_usage"]["total_tokens"] = 110
    diagnosis = diagnose(baseline, SPECS, evaluate)
    comparison = compare_runs(baseline, candidate, diagnosis, diagnosis)
    assert not comparison.accepted
    assert comparison.token_ratio == 1.1


def test_comparison_rejects_changed_evaluation_contract():
    baseline = fixture_rows()
    candidate = fixture_rows()
    for item in baseline:
        item["dataset"] = {
            "name": "Omni-MATH",
            "revision": "frozen-v1",
            "split": "test",
            "dataset_index": item["sample_id"],
        }
        item["final_evaluation"] = {"evaluator_version": "math-v1"}
    for item in candidate:
        item["dataset"] = {
            "name": "Omni-MATH",
            "revision": "changed-v2",
            "split": "test",
            "dataset_index": item["sample_id"],
        }
        item["final_evaluation"] = {"evaluator_version": "math-v1"}
    diagnosis = diagnose(baseline, SPECS, evaluate)
    try:
        compare_runs(baseline, candidate, diagnosis, diagnosis)
    except ValueError as exc:
        assert "same dataset revision" in str(exc)
    else:
        raise AssertionError("comparison accepted a changed dataset revision")


def test_repeated_branch_calls_can_trigger_shared_state_rewrite():
    rows = fixture_rows()
    for item in rows:
        repeated = copy.deepcopy(item["trajectory"][0])
        repeated["output"] = item["trajectory"][0]["output"]
        item["trajectory"].insert(1, repeated)
    graph = inspect_workflow(rows)
    assert graph.repeated_call_rate["alpha"] == 1.0
    diagnosis = diagnose(rows, SPECS, evaluate)
    proposal = select_local_rewrite(
        diagnosis,
        graph,
        attempted_rewrites=(
            "parallel_workers_to_adversarial_debate",
            "single_verifier_to_multiple_verifiers",
            "simple_merge_to_coordinator",
            "single_worker_to_fanout",
        ),
    )
    assert proposal.rewrite_type == "independent_branches_to_shared_state"
    assert proposal.target_nodes == ("alpha",)
