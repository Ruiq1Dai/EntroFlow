import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("localize_topology.py")
SPEC = importlib.util.spec_from_file_location("math_topology_localizer", MODULE_PATH)
assert SPEC and SPEC.loader
LOCALIZER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LOCALIZER
SPEC.loader.exec_module(LOCALIZER)


class MathEquivalenceTests(unittest.TestCase):
    def test_symbolic_commutativity(self):
        equivalent, method = LOCALIZER.math_equivalent(r"2\sqrt{3}+1", r"1+2\sqrt3")
        self.assertTrue(equivalent)
        self.assertEqual(method, "symbolic")

    def test_componentwise_matrix_fractions(self):
        prediction = r"\begin{pmatrix}\frac{2}{5} \\ -\frac{1}{5} \\ 0\end{pmatrix}"
        gold = r"\begin{pmatrix}2/5 \\ -1/5 \\ 0\end{pmatrix}"
        self.assertEqual(LOCALIZER.math_equivalent(prediction, gold), (True, "componentwise_matrix"))

    def test_different_values_are_not_equivalent(self):
        self.assertFalse(LOCALIZER.math_equivalent("18", "17")[0])

    def test_nested_huge_exponent_is_bounded(self):
        self.assertEqual(
            LOCALIZER.math_equivalent(r"3^{2^{32}}", "561"),
            (False, "normalized_mismatch"),
        )

    def test_solver_answer_falls_back_to_boxed(self):
        answer, method = LOCALIZER.extract_agent_answer("Solver_A", r"Work. Therefore \boxed{12}.")
        self.assertEqual((answer, method), ("12", "last_boxed_fallback"))


class StatisticsTests(unittest.TestCase):
    def test_js_divergence_identical_distributions(self):
        self.assertAlmostEqual(LOCALIZER.js_divergence([0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0]), 0.0)

    def test_transition_names_cover_all_states(self):
        self.assertEqual(set(LOCALIZER.TRANSITIONS), set(LOCALIZER.TRANSITION_NAMES))

    def test_common_mode_covariance(self):
        rows = []
        for solver_a_z, solver_b_z in [(0, 0), (0, 0), (1, 1), (1, 1)]:
            rows.append(
                {
                    "node_states": {
                        "Solver_A": {"Z": solver_a_z},
                        "Solver_B": {"Z": solver_b_z},
                    }
                }
            )
        stats = LOCALIZER.common_mode_statistics(rows, bootstrap_samples=20, seed=1)
        self.assertEqual(stats["both_error_count"], 2)
        self.assertAlmostEqual(stats["C_AB"], 0.25)
        self.assertAlmostEqual(stats["phi_correlation"], 1.0)

    def test_zero_failure_wilson_interval_has_positive_upper_bound(self):
        low, high = LOCALIZER.wilson_interval(0, 77)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)


if __name__ == "__main__":
    unittest.main()
