import copy
import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("topology_optimization.py")
SPEC = importlib.util.spec_from_file_location("omni_topology_optimization", MODULE_PATH)
assert SPEC and SPEC.loader
OPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OPT
SPEC.loader.exec_module(OPT)


class OmniTopologyRewriteTests(unittest.TestCase):
    def test_debate_rewrite_is_local_and_preserves_all_existing_nodes(self):
        base = OPT.baseline_workflow_config()
        proposal = OPT.RewriteProposal(
            rewrite_type="parallel_workers_to_adversarial_debate",
            target_nodes=("DerivationSolver", "IndependentSolver"),
            target_edges=(),
            source_signal="common_mode_failure",
            source_score=0.5,
            rationale="fixture",
            operations=(),
            expected_topology_type="debate_adversarial",
        )
        candidate = OPT.apply_omni_rewrite(base, proposal, "G1")
        self.assertEqual(base["topology"]["nodes"], candidate["topology"]["nodes"])
        self.assertNotIn(
            ["DerivationSolver", "IndependentSolver"],
            base["topology"]["edges"],
        )
        self.assertIn(
            ["DerivationSolver", "IndependentSolver"],
            candidate["topology"]["edges"],
        )
        self.assertEqual(base["workflow_id"], "omni_heterogeneous_v1")

    def test_peer_critic_rewrite_adds_only_one_node(self):
        base = OPT.baseline_workflow_config()
        proposal = OPT.RewriteProposal(
            rewrite_type="single_verifier_to_multiple_verifiers",
            target_nodes=("Critic",),
            target_edges=(),
            source_signal="judge_failure",
            source_score=0.3,
            rationale="fixture",
            operations=(),
            expected_topology_type="generate_verify",
        )
        candidate = OPT.apply_omni_rewrite(copy.deepcopy(base), proposal, "G2")
        self.assertEqual(set(candidate["topology"]["nodes"]) - set(base["topology"]["nodes"]), {"Critic_Peer"})
        self.assertIn(["Critic_Peer", "Refiner"], candidate["topology"]["edges"])
        self.assertIn("Critic_Peer", candidate["diagnostic_spec"])

    def test_worker_fanout_clones_only_the_target_worker_connections(self):
        base = OPT.baseline_workflow_config()
        proposal = OPT.RewriteProposal(
            rewrite_type="single_worker_to_fanout",
            target_nodes=("IndependentSolver",),
            target_edges=(),
            source_signal="generator_failure",
            source_score=0.4,
            rationale="fixture",
            operations=(),
            expected_topology_type="fan_out_merge",
        )
        candidate = OPT.apply_omni_rewrite(base, proposal, "G1")
        peer = "IndependentSolver_Peer"
        self.assertEqual(set(candidate["topology"]["nodes"]) - set(base["topology"]["nodes"]), {peer})
        self.assertIn(["Decomposer", peer], candidate["topology"]["edges"])
        self.assertIn([peer, "Critic"], candidate["topology"]["edges"])
        self.assertIn([peer, "Refiner"], candidate["topology"]["edges"])
        self.assertIn(peer, candidate["diagnostic_spec"]["Critic"]["candidates"])

    def test_merge_rewrite_changes_role_mode_without_replacing_the_graph(self):
        base = OPT.baseline_workflow_config()
        proposal = OPT.RewriteProposal(
            rewrite_type="simple_merge_to_coordinator",
            target_nodes=("Refiner",),
            target_edges=(),
            source_signal="merge_degradation",
            source_score=0.2,
            rationale="fixture",
            operations=(),
            expected_topology_type="coordinator_worker",
        )
        candidate = OPT.apply_omni_rewrite(base, proposal, "G1")
        self.assertEqual(candidate["topology"]["nodes"], base["topology"]["nodes"])
        self.assertEqual(candidate["topology"]["edges"], base["topology"]["edges"])
        self.assertEqual(candidate["role_contracts"]["Refiner"], "coordinator_based_selection")
        self.assertIn("Score every candidate explicitly", candidate["role_prompts"]["Refiner"])


if __name__ == "__main__":
    unittest.main()
