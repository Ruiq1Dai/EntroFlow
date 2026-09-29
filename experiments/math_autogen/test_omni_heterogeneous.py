import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

MODULE_PATH = Path(__file__).with_name("omni_heterogeneous.py")
SPEC = importlib.util.spec_from_file_location("omni_heterogeneous", MODULE_PATH)
assert SPEC and SPEC.loader
HET = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HET
SPEC.loader.exec_module(HET)


class FakeCompletionClient:
    RESPONSES: ClassVar[dict[str, str]] = {
        "ProblemAnalyzer": "Target: x\nProblem type: algebra\nConstraints: x>0\nRisk flags: sign",
        "Decomposer": "Step 1: isolate x\nStep 2: verify\nValidation checkpoints: substitute",
        "DerivationSolver": "Step 1 gives x=2.\nProposed answer: 2",
        "IndependentSolver": "Substitution gives x=2.\nProposed answer: 2",
        "Critic": (
            "Derivation verdict: VALID\nIndependent verdict: VALID\nConflict: NONE\n"
            "Revision target: NONE\nNo defect found."
        ),
        "Refiner": "Preserved: both checks\nRepaired: none\nRefined answer: 2",
        "Finalizer": r"<FINAL_ANSWER>\boxed{2}</FINAL_ANSWER>",
    }

    async def create(self, messages, **kwargs):
        system = messages[0].content
        role = next(role for role, prompt in HET.ROLE_PROMPTS.items() if prompt == system)
        return SimpleNamespace(
            content=self.RESPONSES[role],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class ConfigCompletionClient:
    def __init__(self, config):
        self.prompt_roles = {prompt: role for role, prompt in config["role_prompts"].items()}

    async def create(self, messages, **kwargs):
        role = self.prompt_roles[messages[0].content]
        if role.endswith("_Peer") and "Solver" in role:
            content = "Proposed answer: 2\nIndependent substitution confirms the result."
        else:
            content = FakeCompletionClient.RESPONSES[role]
        return SimpleNamespace(
            content=content,
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def sample(index: int = 0):
    return {
        "dataset_index": index,
        "dataset_revision": "test-revision",
        "problem": "Solve x+1=3.",
        "solution": r"Subtract one, so \boxed{2}.",
        "level": "1.5",
        "type": "Mathematics -> Algebra",
    }


class OmniHeterogeneousContractTests(unittest.TestCase):
    def test_old_topology_is_not_imported_or_modified(self):
        self.assertEqual(len(HET.HETEROGENEOUS_TOPOLOGY["nodes"]), 7)
        self.assertNotIn("Planner", HET.HETEROGENEOUS_TOPOLOGY["nodes"])
        self.assertNotIn("Solver_A", HET.HETEROGENEOUS_TOPOLOGY["nodes"])

    def test_solver_visibility_is_parallel_and_peer_isolated(self):
        messages = [("User", "problem"), ("Decomposer", "plan")]
        derivation = HET.make_visible_context("DerivationSolver", messages)
        independent = HET.make_visible_context("IndependentSolver", messages)
        self.assertNotIn("IndependentSolver", derivation[-1]["content"])
        self.assertNotIn("DerivationSolver", independent[-1]["content"])
        self.assertNotEqual(derivation[0]["content"], independent[0]["content"])

    def test_each_prompt_enforces_a_distinct_contract(self):
        self.assertIn("Do not solve", HET.ROLE_PROMPTS["ProblemAnalyzer"])
        self.assertIn("Do not execute", HET.ROLE_PROMPTS["Decomposer"])
        self.assertIn("following the Decomposer steps", HET.ROLE_PROMPTS["DerivationSolver"])
        self.assertIn("genuinely independent", HET.ROLE_PROMPTS["IndependentSolver"])
        self.assertIn("do not give a replacement", HET.ROLE_PROMPTS["Critic"])
        self.assertIn("Do not start a fresh third", HET.ROLE_PROMPTS["Refiner"])
        self.assertIn("Perform no mathematical reasoning", HET.ROLE_PROMPTS["Finalizer"])

    def test_declared_edges_match_realized_contexts(self):
        self.assertEqual(
            HET.validate_message_edges(
                "Refiner",
                [
                    ("User", "problem"),
                    ("DerivationSolver", "a"),
                    ("IndependentSolver", "b"),
                    ("Critic", "c"),
                ],
            ),
            [
                ["DerivationSolver", "Refiner"],
                ["IndependentSolver", "Refiner"],
                ["Critic", "Refiner"],
            ],
        )

    def test_three_sample_smoke_run_records_complete_trajectories(self):
        async def run_three():
            workflow = HET.OmniHeterogeneousWorkflow(FakeCompletionClient(), "fake-model")
            return await asyncio.gather(*(workflow.run_sample(sample(i)) for i in range(3)))

        rows = asyncio.run(run_three())
        expected_roles = HET.HETEROGENEOUS_TOPOLOGY["nodes"]
        expected_edges = HET.HETEROGENEOUS_TOPOLOGY["edges"]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual([event["role"] for event in row["trajectory"]], expected_roles)
            self.assertEqual(row["realized_agent_edges"], expected_edges)
            self.assertTrue(row["run_id"])
            self.assertEqual(row["final_answer"], "2")
            self.assertTrue(row["correct"])
            for event in row["trajectory"]:
                self.assertTrue(event["actual_visible_context"])
                self.assertTrue(event["input"])
                self.assertTrue(event["output"])
                self.assertEqual(event["receiver"], event["role"])
                self.assertIn("incoming_edges", event)

    def test_phase_driven_runner_executes_a_new_worker_without_name_specific_code(self):
        config = HET.baseline_workflow_config()
        config = {
            **config,
            "topology": {
                **config["topology"],
                "nodes": list(config["topology"]["nodes"]),
                "edges": [list(edge) for edge in config["topology"]["edges"]],
                "execution_phases": [list(phase) for phase in config["topology"]["execution_phases"]],
            },
            "role_contracts": dict(config["role_contracts"]),
            "role_prompts": dict(config["role_prompts"]),
            "role_max_tokens": dict(config["role_max_tokens"]),
        }
        peer = "IndependentSolver_Peer"
        config["topology"]["nodes"].insert(4, peer)
        config["topology"]["edges"].extend(
            [["Decomposer", peer], [peer, "Critic"], [peer, "Refiner"]]
        )
        config["topology"]["execution_phases"][2].append(peer)
        config["role_contracts"][peer] = "heterogeneous_peer_answer"
        config["role_prompts"][peer] = "peer prompt"
        config["role_max_tokens"][peer] = 512

        async def run_one():
            workflow = HET.OmniHeterogeneousWorkflow(ConfigCompletionClient(config), "fake", config)
            return await workflow.run_sample(sample())

        result = asyncio.run(run_one())
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(len(result["trajectory"]), 8)
        peer_event = next(event for event in result["trajectory"] if event["role"] == peer)
        self.assertEqual(peer_event["incoming_edges"], [["Decomposer", peer]])
        critic_event = next(event for event in result["trajectory"] if event["role"] == "Critic")
        self.assertIn([peer, "Critic"], critic_event["incoming_edges"])


if __name__ == "__main__":
    unittest.main()
