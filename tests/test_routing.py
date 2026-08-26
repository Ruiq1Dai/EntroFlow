import numpy as np

from entroflow.entropy import RollingEdgeEntropy
from entroflow.messages import RoleResponse
from entroflow.repairs import RepairAction
from entroflow.routing import (
    AlwaysEvidenceBypassPolicy,
    AlwaysRetryEvidencePolicy,
    AlwaysValidatePolicy,
    EntropyPolicy,
    StaticPolicy,
    WorkflowState,
    contract_violation_score,
)


def response(role, content="message", status="ok"):
    return RoleResponse("run", "example", role, content, step=1, status=status)


def test_static_route_reaches_aggregator():
    policy = StaticPolicy()
    state = WorkflowState("run", "example")
    assert policy.select(state).next_role == "planner"
    state.history.append(response("planner"))
    assert policy.select(state).next_role == "evidence"
    state.history.append(response("evidence"))
    assert policy.select(state).next_role == "reasoner"
    state.history.append(response("reasoner"))
    assert policy.select(state).next_role == "aggregator"
    state.history.append(response("aggregator"))
    assert policy.select(state).next_role is None


def test_entropy_policy_inserts_validation_after_anomalous_evidence():
    vectors = {
        "first": np.array([0.0, 0.0]),
        "second": np.array([1.0, 1.0]),
        "third": np.array([0.0, 0.0]),
    }
    policy = EntropyPolicy(
        embedder=vectors.__getitem__,
        monitor=RollingEdgeEntropy(window_size=3, min_samples=2, max_clusters=2),
        threshold=0.05,
    )
    decisions = []
    for content in ("first", "second", "third"):
        state = WorkflowState("run", content, history=[response("evidence", content)])
        decisions.append(policy.select(state))
    assert decisions[-1].action == RepairAction.INSERT_VALIDATION
    assert decisions[-1].next_role == "validator"
    assert decisions[-1].repair_target == "evidence_chain"
    assert decisions[-1].repair_instruction
    assert decisions[-1].confidence is not None


def test_fixed_candidate_repairs_target_the_expected_weak_region():
    state = WorkflowState("run", "example", history=[response("evidence")])
    assert AlwaysValidatePolicy().select(state).next_role == "validator"
    assert AlwaysRetryEvidencePolicy().select(state).next_role == "evidence"
    reasoner_state = WorkflowState(
        "run", "example", history=[response("reasoner")]
    )
    decision = AlwaysEvidenceBypassPolicy().select(reasoner_state)
    assert decision.next_role == "aggregator"


def test_contract_violation_preempts_entropy_for_unresolved_evidence():
    unresolved = response(
        "evidence",
        "The required bridge is missing and no passage states the league.",
        status="insufficient",
    )
    assert contract_violation_score(unresolved) == 1.0
    policy = EntropyPolicy(
        embedder=lambda _: np.array([1.0, 0.0]),
        threshold=0.15,
    )
    decision = policy.select(WorkflowState("run", "example", history=[unresolved]))
    assert decision.action == RepairAction.EXPAND_EVIDENCE
    assert decision.next_role == "evidence"
    assert "missing bridge" in decision.repair_instruction
