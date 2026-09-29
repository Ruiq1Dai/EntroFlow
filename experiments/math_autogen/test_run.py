import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("math_autogen_run", MODULE_PATH)
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN
SPEC.loader.exec_module(RUN)


class AnswerEvaluationTests(unittest.TestCase):
    def test_extracts_last_nested_boxed_answer(self):
        solution = r"First \boxed{wrong}. Finally \boxed{\frac{3}{x+1}}."
        self.assertEqual(RUN.extract_gold_answer(solution), r"\frac{3}{x+1}")

    def test_normalized_fraction_match(self):
        self.assertTrue(RUN.answers_equal("1/2", r"\frac{1}{2}"))

    def test_strips_box_inside_final_tag(self):
        output = r"<FINAL_ANSWER>\boxed{17}</FINAL_ANSWER>"
        self.assertEqual(RUN.extract_final_answer(output), "17")

    def test_currency_marker_is_ignored(self):
        self.assertTrue(RUN.answers_equal("7.50", r"\$7.50"))
        self.assertTrue(RUN.answers_equal("0.5", r"\$0.50"))

    def test_thousands_separators_are_ignored(self):
        self.assertTrue(RUN.answers_equal("900000000", r"900,\!000,\!000"))

    def test_multiple_choice_parentheses_are_ignored(self):
        self.assertTrue(RUN.answers_equal("E", r"\text{(E)}"))

    def test_empty_prediction_is_incorrect(self):
        self.assertFalse(RUN.answers_equal("", "0"))

    def test_matrix_component_equivalence_is_official(self):
        prediction = r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
        gold = r"\begin{bmatrix}1&2\\3&4\end{bmatrix}"
        self.assertTrue(RUN.answers_equal(prediction, gold))

    def test_multi_answer_gold_is_unordered_and_complete(self):
        gold = RUN.build_gold_spec(
            "Find all values of x.",
            r"One value is \boxed{-2}; the other is \boxed{3}.",
        )
        self.assertTrue(RUN.evaluate_answer("3,-2", gold).correct)
        self.assertFalse(RUN.evaluate_answer("3", gold).correct)

    def test_omitted_unit_is_tolerated_but_conflicting_unit_is_not(self):
        self.assertTrue(RUN.answers_equal("12", r"12\text{ centimeters}"))
        self.assertFalse(RUN.answers_equal(r"12\text{ meters}", r"12\text{ centimeters}"))
        self.assertTrue(RUN.answers_equal("9", r"9\text{ a.m.}"))
        self.assertTrue(RUN.answers_equal(r"440\text{ cm}^2", "440"))
        self.assertFalse(RUN.answers_equal("2", r"2\text{ million}"))

    def test_stratified_sampling_covers_and_is_deterministic(self):
        dataset = [
            {"type": subject, "level": level}
            for subject in ("A", "B")
            for level in ("Level 1", "Level 2")
            for _ in range(10)
        ]
        first = RUN.stratified_sample_indices(dataset, 20, 42)
        second = RUN.stratified_sample_indices(dataset, 20, 42)
        self.assertEqual(first, second)
        self.assertEqual(
            {(dataset[index]["type"], dataset[index]["level"]) for index in first},
            {("A", "Level 1"), ("A", "Level 2"), ("B", "Level 1"), ("B", "Level 2")},
        )

    def test_solver_contexts_do_not_expose_peer_outputs(self):
        problem = "problem"
        plan = "plan"
        context_a = RUN.make_visible_context("Solver_A", [("User", problem), ("Planner", plan)])
        context_b = RUN.make_visible_context("Solver_B", [("User", problem), ("Planner", plan)])
        self.assertNotIn("Solver_B", context_a[-1]["content"])
        self.assertNotIn("Solver_A", context_b[-1]["content"])
        self.assertEqual(context_a[-1]["content"], context_b[-1]["content"])

    def test_topology_is_fixed(self):
        self.assertEqual(
            RUN.TOPOLOGY["edges"],
            [
                ["Planner", "Solver_A"],
                ["Planner", "Solver_B"],
                ["Solver_A", "Verifier"],
                ["Solver_B", "Verifier"],
                ["Verifier", "Aggregator"],
            ],
        )

    def test_planner_bypass_removes_only_planner_from_solver_input(self):
        baseline = RUN.solver_input_messages("problem", "plan", planner_bypass=False)
        bypass = RUN.solver_input_messages("problem", "plan", planner_bypass=True)
        self.assertEqual(baseline, [("User", "problem"), ("Planner", "plan")])
        self.assertEqual(bypass, [("User", "problem")])
        baseline_context = RUN.make_visible_context("Solver_A", baseline)
        bypass_context = RUN.make_visible_context("Solver_A", bypass)
        self.assertEqual(baseline_context[0], bypass_context[0])
        self.assertNotIn("plan", bypass_context[-1]["content"])

    def test_solver_diversification_changes_only_solver_b_system_prompt(self):
        messages = [("User", "problem"), ("Planner", "plan")]
        baseline_a = RUN.make_visible_context("Solver_A", messages)
        diversified_a = RUN.make_visible_context("Solver_A", messages, RUN.ROLE_PROMPTS["Solver_A"])
        baseline_b = RUN.make_visible_context("Solver_B", messages)
        diversified_b = RUN.make_visible_context(
            "Solver_B",
            messages,
            RUN.DIVERSIFIED_SOLVER_B_PROMPT,
        )
        self.assertEqual(baseline_a, diversified_a)
        self.assertNotEqual(baseline_b[0], diversified_b[0])
        self.assertEqual(baseline_b[1], diversified_b[1])
        self.assertIn("reverse-verify", diversified_b[0]["content"])

    def test_active_probe_flags_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            RUN.parse_args(["--planner-bypass", "--solver-diversification"])

    def test_solver_b_planner_mask_changes_only_one_input_edge(self):
        problem = "problem"
        plan = "plan"
        solver_a = RUN.solver_input_messages(problem, plan, planner_bypass=False)
        solver_b = RUN.solver_input_messages(problem, plan, planner_bypass=True)
        self.assertEqual(solver_a, [("User", problem), ("Planner", plan)])
        self.assertEqual(solver_b, [("User", problem)])
        self.assertEqual(
            RUN.SOLVER_B_PLANNER_MASK_TOPOLOGY["bypassed_edges"],
            [["Planner", "Solver_B"]],
        )

    def test_solver_b_self_review_preserves_baseline_draft_prompt(self):
        workflow = RUN.MathWorkflow(object(), solver_b_self_review=True)
        self.assertEqual(workflow.role_prompts["Solver_B_Draft"], RUN.ROLE_PROMPTS["Solver_B"])
        self.assertEqual(workflow.role_prompts["Solver_A"], RUN.ROLE_PROMPTS["Solver_A"])
        self.assertEqual(workflow.role_prompts["Solver_B"], RUN.SOLVER_B_SELF_REVIEW_PROMPT)
        self.assertIn(["Solver_B_Draft", "Solver_B"], RUN.SOLVER_B_SELF_REVIEW_TOPOLOGY["edges"])

    def test_verifier_rederivation_changes_only_verifier_prompt(self):
        workflow = RUN.MathWorkflow(object(), verifier_rederivation=True)
        for agent in ("Planner", "Solver_A", "Solver_B", "Aggregator"):
            self.assertEqual(workflow.role_prompts[agent], RUN.ROLE_PROMPTS[agent])
        self.assertEqual(workflow.role_prompts["Verifier"], RUN.VERIFIER_REDERIVATION_PROMPT)


if __name__ == "__main__":
    unittest.main()
