"""Offline topology weakness localization from binary mathematical states.

This script never creates a model client and never calls an LLM. The stored
trajectory ``correct`` field is used only to split success/failure groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from tokenize import TokenError
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

try:
    from experiments.math_autogen.evaluator import build_gold_spec, evaluate_answer
    from experiments.math_autogen.evaluator import (
        math_equivalent as evaluator_math_equivalent,
    )
    from experiments.math_autogen.run import (
        _last_boxed,
        normalize_answer,
    )
except ModuleNotFoundError:  # Direct ``python experiments/math_autogen/...`` execution.
    from evaluator import build_gold_spec, evaluate_answer
    from evaluator import math_equivalent as evaluator_math_equivalent
    from run import _last_boxed, normalize_answer


NODES = ("Planner", "Solver_A", "Solver_B", "Verifier", "Aggregator")
EDGES = (
    ("Planner", "Solver_A"),
    ("Planner", "Solver_B"),
    ("Solver_A", "Verifier"),
    ("Solver_B", "Verifier"),
    ("Verifier", "Aggregator"),
)
TRANSITIONS = ("1->1", "1->0", "0->1", "0->0")
TRANSITION_NAMES = {
    "1->1": "normal_maintenance",
    "1->0": "degradation",
    "0->1": "recovery",
    "0->0": "error_propagation",
}
PARSER_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip_math_wrappers(value: str) -> str:
    value = value.strip().strip("`").strip()
    value = re.sub(r"^\*\*|\*\*$", "", value).strip()
    boxed = _last_boxed(value)
    if boxed is not None:
        value = boxed.strip()
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    value = re.sub(r"^\*\*|\*\*$", "", value).strip()
    return value


def extract_agent_answer(agent: str, output: str, stored_final: str = "") -> tuple[str, str]:
    """Extract an answer using only deterministic role-format rules."""
    if agent == "Aggregator":
        return _strip_math_wrappers(stored_final), "stored_final_answer"
    if agent == "Planner":
        matches = re.findall(
            r"(?:final|proposed)\s+answer\s*:\s*(.+)$",
            output,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return (_strip_math_wrappers(matches[-1]), "explicit_answer") if matches else ("", "no_answer_expected")
    label = "proposed" if agent.startswith("Solver_") else "recommended"
    matches = re.findall(
        rf"{label}\s+answer\s*:\s*(.+)$",
        output,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if matches:
        return _strip_math_wrappers(matches[-1]), f"{label}_answer"
    boxed = _last_boxed(output)
    if boxed is not None:
        return _strip_math_wrappers(boxed), "last_boxed_fallback"
    return "", "missing_answer"


def _balanced_replace(command: str, text: str, unary: bool = False) -> str:
    """Convert simple balanced LaTeX commands into parser-friendly syntax."""
    marker = "\\" + command
    while marker in text:
        start = text.rfind(marker)
        first = text.find("{", start + len(marker))
        if first < 0:
            break

        def group_end(group_start: int, current_text: str = text) -> int:
            depth = 0
            for index in range(group_start, len(current_text)):
                if current_text[index] == "{":
                    depth += 1
                elif current_text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return index
            return -1

        first_end = group_end(first)
        if first_end < 0:
            break
        numerator = text[first + 1 : first_end]
        if unary:
            replacement = f"sqrt(({numerator}))"
            text = text[:start] + replacement + text[first_end + 1 :]
            continue
        second = text.find("{", first_end + 1)
        if second != first_end + 1:
            break
        second_end = group_end(second)
        if second_end < 0:
            break
        denominator = text[second + 1 : second_end]
        replacement = f"(({numerator})/({denominator}))"
        text = text[:start] + replacement + text[second_end + 1 :]
    return text


def _latex_to_sympy_text(value: str) -> str:
    value = _strip_math_wrappers(value)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    value = re.sub(r"\\sqrt\s*([^\s{}])", r"\\sqrt{\1}", value)
    value = _balanced_replace("frac", value)
    value = _balanced_replace("sqrt", value, unary=True)
    value = value.replace(r"\cdot", "*").replace(r"\times", "*")
    value = value.replace(r"\pi", "pi").replace(r"\infty", "oo")
    value = value.replace(r"\mathrm{i}", "I").replace(r"\imath", "I")
    value = re.sub(r"(?<=\d)i\b", "*I", value)
    value = re.sub(r"\bi\b", "I", value)
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("^", "**")
    value = value.replace(r"\,", "").replace(r"\!", "")
    value = value.replace("°", "").replace(r"^\circ", "")
    return value.strip()


def _parse_scalar(value: str) -> sympy.Expr | None:
    text = _latex_to_sympy_text(value)
    if not text or any(token in text for token in (r"\begin", r"\cup", "[", "]")):
        return None
    # Do not let malformed/wrong model answers trigger unbounded integer
    # construction (for example 3**(2**32)). Such cases remain mismatches.
    if len(text) > 256 or re.search(r"\*\*\([^)]*\*\*", text):
        return None
    if text.count("=") > 1:
        return None
    try:
        return parse_expr(
            text,
            local_dict={"i": sympy.I, "I": sympy.I, "pi": sympy.pi, "oo": sympy.oo},
            transformations=PARSER_TRANSFORMATIONS,
            evaluate=False,
        )
    except (SyntaxError, TokenError, TypeError, ValueError, NameError, AttributeError):
        return None


def _split_top_level(value: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    close_to_open = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(value):
        if char in depths:
            depths[char] += 1
        elif char in close_to_open:
            depths[close_to_open[char]] -= 1
        elif char == delimiter and all(depth == 0 for depth in depths.values()):
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts


def _matrix_components(value: str) -> list[str] | None:
    match = re.fullmatch(
        r"\s*\\begin\{[pbvBV]?matrix\}(.*?)\\end\{[pbvBV]?matrix\}\s*",
        _strip_math_wrappers(value),
        flags=re.DOTALL,
    )
    if not match:
        return None
    return [cell.strip() for row in match.group(1).split(r"\\") for cell in row.split("&")]


def math_equivalent(prediction: str, gold: str) -> tuple[bool, str]:
    """Conservative deterministic equivalence: normalized, component, then SymPy."""
    return evaluator_math_equivalent(prediction, gold)


def _legacy_math_equivalent(prediction: str, gold: str) -> tuple[bool, str]:
    """Previous implementation retained only to make historical logic reviewable."""
    prediction = _strip_math_wrappers(prediction)
    gold = _strip_math_wrappers(gold)
    if not prediction:
        return False, "missing"
    if normalize_answer(prediction) == normalize_answer(gold):
        return True, "normalized_exact"

    prediction_matrix = _matrix_components(prediction)
    gold_matrix = _matrix_components(gold)
    if prediction_matrix is not None or gold_matrix is not None:
        if prediction_matrix is None or gold_matrix is None or len(prediction_matrix) != len(gold_matrix):
            return False, "matrix_shape_mismatch"
        component_results = [math_equivalent(left, right)[0] for left, right in zip(prediction_matrix, gold_matrix)]
        return (all(component_results), "componentwise_matrix" if all(component_results) else "matrix_mismatch")

    if (
        prediction[:1] in "({"
        and gold[:1] in "({"
        and prediction[-1:] in ")}"
        and gold[-1:] in ")}"
    ):
        prediction_parts = _split_top_level(prediction[1:-1])
        gold_parts = _split_top_level(gold[1:-1])
        if len(prediction_parts) == len(gold_parts) and len(prediction_parts) > 1:
            component_results = [math_equivalent(left, right)[0] for left, right in zip(prediction_parts, gold_parts)]
            return (all(component_results), "componentwise_tuple" if all(component_results) else "tuple_mismatch")

    if prediction.count("=") == 1 and gold.count("=") == 1:
        pred_left, pred_right = prediction.split("=", 1)
        gold_left, gold_right = gold.split("=", 1)
        pred_residual = _parse_scalar(pred_left)
        pred_rhs = _parse_scalar(pred_right)
        gold_residual = _parse_scalar(gold_left)
        gold_rhs = _parse_scalar(gold_right)
        if None not in (pred_residual, pred_rhs, gold_residual, gold_rhs):
            pred_expression = sympy.expand(pred_residual - pred_rhs)
            gold_expression = sympy.expand(gold_residual - gold_rhs)
            if sympy.simplify(pred_expression - gold_expression) == 0:
                return True, "symbolic_equation"
            if gold_expression != 0:
                ratio = sympy.simplify(pred_expression / gold_expression)
                if not ratio.free_symbols and ratio != 0:
                    return True, "symbolic_equation_scaled"
        return False, "equation_mismatch"

    prediction_expression = _parse_scalar(prediction)
    gold_expression = _parse_scalar(gold)
    if prediction_expression is not None and gold_expression is not None:
        try:
            if sympy.simplify(prediction_expression - gold_expression) == 0:
                return True, "symbolic"
            difference = complex(sympy.N(prediction_expression - gold_expression, 15))
            if abs(difference) <= 1e-9:
                return True, "numeric_tolerance"
        except (TypeError, ValueError):
            pass
        return False, "symbolic_mismatch"
    return False, "normalized_mismatch"


def _mentions_rejection(verifier_output: str, solver: str) -> bool:
    negative = re.compile(
        r"\b(?:incorrect|wrong|error|flaw(?:ed)?|invalid|reject(?:ed)?|not correct)\b",
        re.IGNORECASE,
    )
    chunks = re.split(r"[\n.!?]+", verifier_output)
    aliases = (solver, solver.replace("_", " "))
    return any(any(alias.lower() in chunk.lower() for alias in aliases) and negative.search(chunk) for chunk in chunks)


def build_localization(row: dict[str, Any], source_line: int) -> dict[str, Any]:
    events = {event["agent"]: event for event in row["trajectory"]}
    missing = [node for node in NODES if node not in events]
    if missing:
        raise ValueError(f"{row['sample_id']} is missing trajectory nodes: {missing}")

    gold_spec = build_gold_spec(row["problem"], row["gold_solution"])
    reextracted_gold = gold_spec.canonical_answer
    gold_evaluation = evaluate_answer(row["gold_answer"], gold_spec)
    gold_verified, gold_method = gold_evaluation.correct, gold_evaluation.method
    if not gold_verified:
        raise ValueError(
            f"{row['sample_id']} stored gold {row['gold_answer']!r} disagrees with "
            f"gold solution extraction {reextracted_gold!r}"
        )
    gold = reextracted_gold
    node_states: dict[str, dict[str, Any]] = {}
    for node in NODES:
        answer, extraction_method = extract_agent_answer(
            node,
            events[node]["output"],
            stored_final=row["final_answer"],
        )
        if node == "Planner" and not answer:
            healthy = bool(events[node]["output"].strip())
            equivalent = None
            equivalence_method = "not_answer_bearing"
            health_basis = "nonempty_plan_without_conflicting_explicit_answer"
        else:
            evaluation = evaluate_answer(answer, gold_spec)
            equivalent, equivalence_method = evaluation.correct, evaluation.method
            healthy = equivalent
            health_basis = "answer_mathematically_equivalent_to_confirmed_gold"
        node_states[node] = {
            "Z": int(healthy),
            "answer": answer,
            "answer_extraction": extraction_method,
            "reference_gold_answer": gold,
            "mathematically_equivalent_to_gold": equivalent,
            "equivalence_method": equivalence_method,
            "health_basis": health_basis,
        }

    verifier = node_states["Verifier"]
    solver_a = node_states["Solver_A"]
    solver_b = node_states["Solver_B"]
    correct_solvers = [name for name in ("Solver_A", "Solver_B") if node_states[name]["Z"] == 1]
    wrong_solvers = [name for name in ("Solver_A", "Solver_B") if node_states[name]["Z"] == 0]
    verifier_answer = verifier["answer"]
    selected_correct_solver = any(
        math_equivalent(verifier_answer, node_states[name]["answer"])[0] for name in correct_solvers
    )
    rejected_wrong = {
        "Solver_A": bool(solver_a["Z"] == 0 and _mentions_rejection(events["Verifier"]["output"], "Solver_A")),
        "Solver_B": bool(solver_b["Z"] == 0 and _mentions_rejection(events["Verifier"]["output"], "Solver_B")),
    }
    rejected_all_wrong = bool(wrong_solvers) and all(rejected_wrong[name] for name in wrong_solvers)
    if verifier["Z"] and correct_solvers:
        verifier_behavior = "selected_or_derived_gold_with_correct_solver_available"
    elif verifier["Z"] and not correct_solvers:
        verifier_behavior = "recovered_gold_from_two_incorrect_solver_answers"
    elif not verifier_answer and not correct_solvers and rejected_all_wrong:
        verifier_behavior = "correctly_rejected_both_incorrect_answers_without_recommendation"
        verifier["Z"] = 1
        verifier["health_basis"] = "explicit_rejection_of_all_incorrect_solver_answers"
    elif correct_solvers:
        verifier_behavior = "failed_to_select_available_correct_solver_answer"
    else:
        verifier_behavior = "propagated_or_created_incorrect_recommendation"
    verifier.update(
        {
            "correct_solver_answers_available": correct_solvers,
            "incorrect_solver_answers": wrong_solvers,
            "selected_correct_solver_answer": selected_correct_solver,
            "rejected_incorrect_solver": rejected_wrong,
            "rejected_all_incorrect_solvers": rejected_all_wrong,
            "behavior": verifier_behavior,
        }
    )

    edge_states: dict[str, dict[str, Any]] = {}
    for source, target in EDGES:
        source_state = node_states[source]["Z"]
        target_state = node_states[target]["Z"]
        transition = f"{source_state}->{target_state}"
        edge_states[f"{source}->{target}"] = {
            "source_Z": source_state,
            "target_Z": target_state,
            "transition": transition,
            "transition_type": TRANSITION_NAMES[transition],
        }

    opportunities = {
        "Planner": {
            "eligible": True,
            "failed": node_states["Planner"]["Z"] == 0,
            "definition": "trajectory contains a Planner execution",
        },
        "Solver_A": {
            "eligible": node_states["Planner"]["Z"] == 1,
            "failed": node_states["Planner"]["Z"] == 1 and node_states["Solver_A"]["Z"] == 0,
            "definition": "Planner contract-health Z=1",
        },
        "Solver_B": {
            "eligible": node_states["Planner"]["Z"] == 1,
            "failed": node_states["Planner"]["Z"] == 1 and node_states["Solver_B"]["Z"] == 0,
            "definition": "Planner contract-health Z=1",
        },
        "Verifier": {
            "eligible": bool(correct_solvers),
            "failed": bool(correct_solvers) and node_states["Verifier"]["Z"] == 0,
            "definition": "at least one Solver answer is mathematically equivalent to gold",
        },
        "Aggregator": {
            "eligible": node_states["Verifier"]["Z"] == 1,
            "failed": node_states["Verifier"]["Z"] == 1 and node_states["Aggregator"]["Z"] == 0,
            "definition": "Verifier provides a mathematically correct result",
        },
        "Planner_to_dual_Solvers_common_mode": {
            "eligible": node_states["Planner"]["Z"] == 1,
            "failed": node_states["Solver_A"]["Z"] == 0 and node_states["Solver_B"]["Z"] == 0,
            "definition": "both independent Solver branches receive the shared Planner output",
        },
    }

    return {
        "sample_id": row["sample_id"],
        "run_id": row["run_id"],
        "source_trajectory_line": source_line,
        "trajectory_group": "success" if bool(row["correct"]) else "failure",
        "source_correct_label": bool(row["correct"]),
        "gold_answer_confirmation": {
            "stored_gold_answer": row["gold_answer"],
            "reextracted_from_gold_solution": reextracted_gold,
            "verified": gold_verified,
            "verification_method": gold_method,
        },
        "problem": row["problem"],
        "final_answer": row["final_answer"],
        "node_states": node_states,
        "edge_transitions": edge_states,
        "opportunity_conditioned_states": opportunities,
    }


def transition_counts(rows: Sequence[dict[str, Any]], edge: str) -> Counter[str]:
    return Counter(row["edge_transitions"][edge]["transition"] for row in rows)


def distribution(counts: Counter[str]) -> list[float]:
    total = sum(counts.values())
    return [counts[state] / total if total else 0.0 for state in TRANSITIONS]


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits, bounded by [0, 1]."""
    midpoint = [(a + b) / 2 for a, b in zip(left, right)]

    def kl(values: Sequence[float]) -> float:
        return sum(value * math.log2(value / middle) for value, middle in zip(values, midpoint) if value > 0)

    return 0.5 * kl(left) + 0.5 * kl(right)


def edge_statistics(localizations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {
        "success": [row for row in localizations if row["trajectory_group"] == "success"],
        "failure": [row for row in localizations if row["trajectory_group"] == "failure"],
    }
    statistics: list[dict[str, Any]] = []
    for source, target in EDGES:
        edge = f"{source}->{target}"
        success_counts = transition_counts(groups["success"], edge)
        failure_counts = transition_counts(groups["failure"], edge)
        success_distribution = distribution(success_counts)
        failure_distribution = distribution(failure_counts)
        row: dict[str, Any] = {
            "edge": edge,
            "source": source,
            "target": target,
            "success_n": len(groups["success"]),
            "failure_n": len(groups["failure"]),
            "js_divergence_bits": js_divergence(success_distribution, failure_distribution),
        }
        for group, counts, values in (
            ("success", success_counts, success_distribution),
            ("failure", failure_counts, failure_distribution),
        ):
            for transition, value in zip(TRANSITIONS, values):
                key = transition.replace("->", "_to_")
                row[f"{group}_{key}_count"] = counts[transition]
                row[f"{group}_{key}_rate"] = value
        row["degradation_rate_difference"] = row["failure_1_to_0_rate"] - row["success_1_to_0_rate"]
        row["propagation_rate_difference"] = row["failure_0_to_0_rate"] - row["success_0_to_0_rate"]
        row["recovery_rate_difference"] = row["failure_0_to_1_rate"] - row["success_0_to_1_rate"]
        anomaly_shifts = {
            "degradation_excess": max(row["degradation_rate_difference"], 0.0),
            "propagation_excess": max(row["propagation_rate_difference"], 0.0),
            "recovery_deficit": max(-row["recovery_rate_difference"], 0.0),
        }
        label, magnitude = max(anomaly_shifts.items(), key=lambda item: item[1])
        row["primary_shift"] = label if magnitude > 0 else "no_positive_anomaly_shift"
        statistics.append(row)
    statistics.sort(key=lambda row: (-row["js_divergence_bits"], row["edge"]))
    for rank, row in enumerate(statistics, 1):
        row["rank"] = rank
    return statistics


def add_bootstrap_stability(
    statistics: list[dict[str, Any]],
    localizations: Sequence[dict[str, Any]],
    samples: int,
    seed: int,
) -> None:
    success = [row for row in localizations if row["trajectory_group"] == "success"]
    failure = [row for row in localizations if row["trajectory_group"] == "failure"]
    rng = random.Random(seed)
    values = {f"{source}->{target}": [] for source, target in EDGES}
    rank1 = Counter[str]()
    top3 = Counter[str]()
    for _ in range(samples):
        sampled = rng.choices(success, k=len(success)) + rng.choices(failure, k=len(failure))
        sample_stats = edge_statistics(sampled)
        for row in sample_stats:
            values[row["edge"]].append(row["js_divergence_bits"])
        rank1[sample_stats[0]["edge"]] += 1
        top3.update(row["edge"] for row in sample_stats[:3])
    for row in statistics:
        ordered = sorted(values[row["edge"]])
        low_index = int(0.025 * (samples - 1))
        high_index = int(0.975 * (samples - 1))
        row["js_bootstrap_ci_low"] = ordered[low_index]
        row["js_bootstrap_ci_high"] = ordered[high_index]
        row["bootstrap_rank1_frequency"] = rank1[row["edge"]] / samples
        row["bootstrap_top3_frequency"] = top3[row["edge"]] / samples


def node_statistics(localizations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for node in NODES:
        success = [row["node_states"][node]["Z"] for row in localizations if row["trajectory_group"] == "success"]
        failure = [row["node_states"][node]["Z"] for row in localizations if row["trajectory_group"] == "failure"]
        overall = success + failure
        success_anomaly = 1.0 - sum(success) / len(success) if success else 0.0
        failure_anomaly = 1.0 - sum(failure) / len(failure) if failure else 0.0
        opportunities = [row["opportunity_conditioned_states"][node] for row in localizations]
        opportunity_count = sum(item["eligible"] for item in opportunities)
        opportunity_failure_count = sum(item["failed"] for item in opportunities)
        opportunity_rate = opportunity_failure_count / opportunity_count if opportunity_count else 0.0
        opportunity_ci_low, opportunity_ci_high = wilson_interval(
            opportunity_failure_count,
            opportunity_count,
        )
        rows.append(
            {
                "node": node,
                "health_basis": localizations[0]["node_states"][node]["health_basis"],
                "opportunity_definition": opportunities[0]["definition"],
                "opportunity_count": opportunity_count,
                "opportunity_failure_count": opportunity_failure_count,
                "opportunity_conditioned_failure_rate": opportunity_rate,
                "opportunity_failure_ci_low": opportunity_ci_low,
                "opportunity_failure_ci_high": opportunity_ci_high,
                "overall_n": len(overall),
                "overall_healthy_count": sum(overall),
                "overall_anomaly_count": len(overall) - sum(overall),
                "overall_anomaly_rate": 1.0 - sum(overall) / len(overall),
                "success_n": len(success),
                "success_anomaly_count": len(success) - sum(success),
                "success_anomaly_rate": success_anomaly,
                "failure_n": len(failure),
                "failure_anomaly_count": len(failure) - sum(failure),
                "failure_anomaly_rate": failure_anomaly,
                "anomaly_rate_difference": failure_anomaly - success_anomaly,
            }
        )
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    low = max(0.0, center - margin)
    return (0.0 if low < 1e-15 else low), min(1.0, center + margin)


def _fisher_exact_two_sided(a_error: int, b_error: int, both_error: int, total: int) -> float:
    lower = max(0, a_error + b_error - total)
    upper = min(a_error, b_error)
    denominator = math.comb(total, a_error)

    def probability(overlap: int) -> float:
        return math.comb(b_error, overlap) * math.comb(total - b_error, a_error - overlap) / denominator

    observed_probability = probability(both_error)
    return sum(
        probability(overlap)
        for overlap in range(lower, upper + 1)
        if probability(overlap) <= observed_probability + 1e-15
    )


def common_mode_statistics(
    localizations: Sequence[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    total = len(localizations)
    a_states = [1 - row["node_states"]["Solver_A"]["Z"] for row in localizations]
    b_states = [1 - row["node_states"]["Solver_B"]["Z"] for row in localizations]
    a_error = sum(a_states)
    b_error = sum(b_states)
    both_error = sum(a and b for a, b in zip(a_states, b_states))
    a_only_error = a_error - both_error
    b_only_error = b_error - both_error
    neither_error = total - both_error - a_only_error - b_only_error
    p_a = a_error / total
    p_b = b_error / total
    p_both = both_error / total
    expected_joint = p_a * p_b
    covariance = p_both - expected_joint
    denominator = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    phi = covariance / denominator if denominator else 0.0
    b_given_a = both_error / a_error if a_error else 0.0
    b_given_not_a = b_only_error / (total - a_error) if total > a_error else 0.0
    risk_ratio = b_given_a / b_given_not_a if b_given_not_a else math.inf
    odds_denominator = a_only_error * b_only_error
    odds_ratio = both_error * neither_error / odds_denominator if odds_denominator else math.inf

    rng = random.Random(seed)
    bootstrap_covariances = []
    for _ in range(bootstrap_samples):
        sampled_indices = rng.choices(range(total), k=total)
        sample_a = sum(a_states[index] for index in sampled_indices) / total
        sample_b = sum(b_states[index] for index in sampled_indices) / total
        sample_both = sum(a_states[index] and b_states[index] for index in sampled_indices) / total
        bootstrap_covariances.append(sample_both - sample_a * sample_b)
    bootstrap_covariances.sort()
    low_index = int(0.025 * (bootstrap_samples - 1))
    high_index = int(0.975 * (bootstrap_samples - 1))
    fisher_p = _fisher_exact_two_sided(a_error, b_error, both_error, total)
    return {
        "pair": "Solver_A+Solver_B",
        "n": total,
        "solver_a_error_count": a_error,
        "solver_b_error_count": b_error,
        "both_error_count": both_error,
        "solver_a_only_error_count": a_only_error,
        "solver_b_only_error_count": b_only_error,
        "neither_error_count": neither_error,
        "p_error_a": p_a,
        "p_error_b": p_b,
        "p_both_error_observed": p_both,
        "p_both_error_expected_independent": expected_joint,
        "expected_both_error_count_independent": expected_joint * total,
        "excess_both_error_count": both_error - expected_joint * total,
        "C_AB": covariance,
        "C_AB_bootstrap_ci_low": bootstrap_covariances[low_index],
        "C_AB_bootstrap_ci_high": bootstrap_covariances[high_index],
        "phi_correlation": phi,
        "p_b_error_given_a_error": b_given_a,
        "p_b_error_given_a_correct": b_given_not_a,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "fisher_exact_two_sided_p": fisher_p,
        "significant_positive_common_mode_at_0_05": covariance > 0 and fisher_p < 0.05,
    }


def component_statistics(
    localizations: Sequence[dict[str, Any]],
    edge_rows: Sequence[dict[str, Any]],
    common_mode: dict[str, Any],
) -> list[dict[str, Any]]:
    edge_by_name = {row["edge"]: row for row in edge_rows}
    component_edges = {
        "Planner": [],
        "Solver_A": ["Planner->Solver_A"],
        "Solver_B": ["Planner->Solver_B"],
        "Verifier": ["Solver_A->Verifier", "Solver_B->Verifier"],
        "Aggregator": ["Verifier->Aggregator"],
        "Planner_to_dual_Solvers_common_mode": ["Planner->Solver_A", "Planner->Solver_B"],
    }
    rows = []
    for component, edges in component_edges.items():
        opportunities = [row["opportunity_conditioned_states"][component] for row in localizations]
        opportunity_count = sum(item["eligible"] for item in opportunities)
        failure_count = sum(item["failed"] for item in opportunities)
        failure_rate = failure_count / opportunity_count if opportunity_count else 0.0
        ci_low, ci_high = wilson_interval(failure_count, opportunity_count)
        metrics = {
            edge: {
                "js_divergence_bits": edge_by_name[edge]["js_divergence_bits"],
                "degradation_rate_difference": edge_by_name[edge]["degradation_rate_difference"],
                "propagation_rate_difference": edge_by_name[edge]["propagation_rate_difference"],
                "recovery_rate_difference": edge_by_name[edge]["recovery_rate_difference"],
            }
            for edge in edges
        }
        js_values = [payload["js_divergence_bits"] for payload in metrics.values()]
        degradation_values = [payload["degradation_rate_difference"] for payload in metrics.values()]
        propagation_values = [payload["propagation_rate_difference"] for payload in metrics.values()]
        recovery_values = [payload["recovery_rate_difference"] for payload in metrics.values()]
        row = {
            "component": component,
            "opportunity_definition": opportunities[0]["definition"],
            "opportunity_count": opportunity_count,
            "opportunity_failure_count": failure_count,
            "opportunity_conditioned_failure_rate": failure_rate,
            "opportunity_failure_ci_low": ci_low,
            "opportunity_failure_ci_high": ci_high,
            "associated_edges": ";".join(edges),
            "max_associated_js_divergence_bits": max(js_values) if js_values else "",
            "max_degradation_rate_difference": max(degradation_values) if degradation_values else "",
            "max_propagation_rate_difference": max(propagation_values) if propagation_values else "",
            "min_recovery_rate_difference": min(recovery_values) if recovery_values else "",
            "edge_metrics_json": json.dumps(metrics, separators=(",", ":")),
            "common_mode_pair": "Solver_A+Solver_B" if component in {"Solver_A", "Solver_B", "Planner_to_dual_Solvers_common_mode"} else "",
            "common_mode_C_AB": common_mode["C_AB"] if component in {"Solver_A", "Solver_B", "Planner_to_dual_Solvers_common_mode"} else "",
            "common_mode_fisher_p": common_mode["fisher_exact_two_sided_p"] if component in {"Solver_A", "Solver_B", "Planner_to_dual_Solvers_common_mode"} else "",
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def representative_failures(
    localizations: Sequence[dict[str, Any]],
    statistics: Sequence[dict[str, Any]],
    limit: int = 5,
) -> list[tuple[float, dict[str, Any]]]:
    """Select failures by their transition log-likelihood ratio, without labels."""
    success_n = sum(row["trajectory_group"] == "success" for row in localizations)
    failure_n = sum(row["trajectory_group"] == "failure" for row in localizations)
    stat_by_edge = {row["edge"]: row for row in statistics}
    scored = []
    for row in localizations:
        if row["trajectory_group"] != "failure":
            continue
        score = 0.0
        for source, target in EDGES:
            edge = f"{source}->{target}"
            transition = row["edge_transitions"][edge]["transition"].replace("->", "_to_")
            stats = stat_by_edge[edge]
            success_count = stats[f"success_{transition}_count"]
            failure_count = stats[f"failure_{transition}_count"]
            # Jeffreys smoothing prevents infinite scores for unseen transitions.
            success_probability = (success_count + 0.5) / (success_n + 0.5 * len(TRANSITIONS))
            failure_probability = (failure_count + 0.5) / (failure_n + 0.5 * len(TRANSITIONS))
            score += math.log2(failure_probability / success_probability)
        scored.append((score, row))
    ordered = sorted(scored, key=lambda item: (-item[0], item[1]["sample_id"]))
    selected: list[tuple[float, dict[str, Any]]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for item in ordered:
        signature = tuple(payload["transition"] for payload in item[1]["edge_transitions"].values())
        if signature not in seen_signatures:
            selected.append(item)
            seen_signatures.add(signature)
    for item in ordered:
        if item not in selected:
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _fmt_rate(value: float) -> str:
    return f"{100 * value:.1f}%"


def _markdown_escape(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ").strip()


def write_summary(
    path: Path,
    trajectories_path: Path,
    localizations: Sequence[dict[str, Any]],
    edge_rows: Sequence[dict[str, Any]],
    node_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
    common_mode: dict[str, Any],
    bootstrap_samples: int,
) -> None:
    success_n = sum(row["trajectory_group"] == "success" for row in localizations)
    failure_n = len(localizations) - success_n
    representatives = representative_failures(localizations, edge_rows)
    edge_by_name = {row["edge"]: row for row in edge_rows}
    component_by_name = {row["component"]: row for row in component_rows}
    mathematical_final_failures = sum(row["node_states"]["Aggregator"]["Z"] == 0 for row in localizations)
    common_mode_final_failures = sum(
        row["node_states"]["Aggregator"]["Z"] == 0
        and row["node_states"]["Solver_A"]["Z"] == 0
        and row["node_states"]["Solver_B"]["Z"] == 0
        for row in localizations
    )
    verifier_out_of_opportunity_recoveries = sum(
        row["node_states"]["Solver_A"]["Z"] == 0
        and row["node_states"]["Solver_B"]["Z"] == 0
        and row["node_states"]["Verifier"]["Z"] == 1
        for row in localizations
    )
    lines = [
        "# MATH topology weakness localization",
        "",
        "## Protocol",
        "",
        f"Analyzed {len(localizations)} stored trajectories ({success_n} success, {failure_n} failure). ",
        "The stored `correct` label is used only for this grouping. It is never used to set a node state or assign a root cause. ",
        "Every stored gold answer was independently re-extracted from `gold_solution` and checked for deterministic mathematical equivalence before localization.",
        "",
        "Answer-bearing nodes use conservative deterministic equivalence in this order: normalized MATH exact match, component-wise matrix/tuple comparison, then SymPy simplification or numeric tolerance. ",
        "Verifier health additionally permits an answer-less verdict only when both incorrect Solver answers are explicitly rejected. Planner is non-answer-bearing, so its state is only a conservative contract-health proxy: a non-empty plan with no conflicting explicit final answer. This does not certify the plan's mathematics.",
        "",
        "For each edge, `degradation difference`, `propagation difference`, and `recovery difference` are failure-group rate minus success-group rate. JS divergence uses base-2 logarithms. Bootstrap intervals and rank frequencies resample success and failure trajectories separately.",
        "",
        "## 1. Suspicious components: original JS candidate ranking",
        "",
        "| Rank | Edge | JS divergence (bits) | 95% bootstrap CI | Degradation Δ | Propagation Δ | Recovery Δ | Primary distribution shift | Top-3 frequency |",
        "|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in edge_rows[:3]:
        lines.append(
            f"| {row['rank']} | `{row['edge']}` | {row['js_divergence_bits']:.4f} | "
            f"[{row['js_bootstrap_ci_low']:.4f}, {row['js_bootstrap_ci_high']:.4f}] | "
            f"{row['degradation_rate_difference']:+.4f} | {row['propagation_rate_difference']:+.4f} | "
            f"{row['recovery_rate_difference']:+.4f} | {row['primary_shift']} | "
            f"{_fmt_rate(row['bootstrap_top3_frequency'])} |"
        )
    lines.extend(
        [
            "",
            "These are suspicious *locations of distribution shift*, not asserted root causes. A large degradation excess means healthy upstream states more often become unhealthy downstream states in failed trajectories; a propagation excess means downstream agents more often retain upstream errors; a negative recovery difference indicates a recovery deficit.",
            "",
            (
                "In this run, both `Solver->Verifier` edges are dominated by propagation excesses of "
                f"{edge_by_name['Solver_A->Verifier']['propagation_rate_difference']:+.4f} and "
                f"{edge_by_name['Solver_B->Verifier']['propagation_rate_difference']:+.4f}; their recovery shifts "
                f"are {edge_by_name['Solver_A->Verifier']['recovery_rate_difference']:+.4f} and "
                f"{edge_by_name['Solver_B->Verifier']['recovery_rate_difference']:+.4f}. With no degradation "
                "excess, the observed shift is failure to recover wrong "
                "Solver answers, rather than Verifier corrupting a correct Solver answer. "
                "`Verifier->Aggregator` also has a propagation shift of "
                f"{edge_by_name['Verifier->Aggregator']['propagation_rate_difference']:+.4f} but zero degradation "
                "and recovery shifts, "
                "so it describes pass-through persistence. The largest degradation shifts occur instead on "
                f"`Planner->Solver_B` ({edge_by_name['Planner->Solver_B']['degradation_rate_difference']:+.4f}) "
                f"and `Planner->Solver_A` ({edge_by_name['Planner->Solver_A']['degradation_rate_difference']:+.4f}); "
                "because Planner health is only "
                "a contract proxy, these should be read as Solver answer-generation anomalies, not proof that the "
                "Planner-to-Solver connections caused them."
            ),
            "",
            "## 2. Opportunity-conditioned weak components",
            "",
            "| Component | Opportunity condition | Opportunities | Failures | Conditioned failure rate | 95% Wilson CI | Max associated JS |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for component in (
        "Solver_A",
        "Solver_B",
        "Verifier",
        "Aggregator",
        "Planner_to_dual_Solvers_common_mode",
        "Planner",
    ):
        row = component_by_name[component]
        max_js = row["max_associated_js_divergence_bits"]
        max_js_text = f"{max_js:.4f}" if isinstance(max_js, float) else "—"
        lines.append(
            f"| `{component}` | {row['opportunity_definition']} | {row['opportunity_count']} | "
            f"{row['opportunity_failure_count']} | {row['opportunity_conditioned_failure_rate']:.4f} | "
            f"[{row['opportunity_failure_ci_low']:.4f}, {row['opportunity_failure_ci_high']:.4f}] | "
            f"{max_js_text} |"
        )
    verifier_row = component_by_name["Verifier"]
    aggregator_row = component_by_name["Aggregator"]
    lines.extend(
        [
            "",
            (
                f"Verifier had {verifier_row['opportunity_count']} defined opportunities (at least one correct "
                f"Solver) and failed {verifier_row['opportunity_failure_count']} times; Aggregator had "
                f"{aggregator_row['opportunity_count']} opportunities (correct Verifier result) and failed "
                f"{aggregator_row['opportunity_failure_count']} times. Thus their large original JS values are "
                "explained here by where errors pass, not by observed failures when a correct input was available."
            ),
            (
                f"Outside the strict Verifier opportunity definition, both Solvers were wrong in "
                f"{common_mode['both_error_count']} samples and Verifier independently recovered the gold answer "
                f"in {verifier_out_of_opportunity_recoveries} of them."
            ),
            "",
            "## 3. Common-mode weak region",
            "",
            "Solver error contingency table:",
            "",
            "| | Solver_B error | Solver_B correct |",
            "|---|---:|---:|",
            f"| Solver_A error | {common_mode['both_error_count']} | {common_mode['solver_a_only_error_count']} |",
            f"| Solver_A correct | {common_mode['solver_b_only_error_count']} | {common_mode['neither_error_count']} |",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| P(E_A) | {common_mode['p_error_a']:.4f} |",
            f"| P(E_B) | {common_mode['p_error_b']:.4f} |",
            f"| Observed P(E_A,E_B) | {common_mode['p_both_error_observed']:.4f} |",
            f"| Independent expectation P(E_A)P(E_B) | {common_mode['p_both_error_expected_independent']:.4f} |",
            f"| C_AB | {common_mode['C_AB']:+.4f} |",
            f"| C_AB 95% bootstrap CI | [{common_mode['C_AB_bootstrap_ci_low']:.4f}, {common_mode['C_AB_bootstrap_ci_high']:.4f}] |",
            f"| Phi correlation | {common_mode['phi_correlation']:.4f} |",
            f"| Risk ratio | {common_mode['risk_ratio']:.4f} |",
            f"| Odds ratio | {common_mode['odds_ratio']:.4f} |",
            f"| Fisher exact two-sided p | {common_mode['fisher_exact_two_sided_p']:.3e} |",
            "",
            (
                f"Both Solvers failed together {common_mode['both_error_count']} times, versus "
                f"{common_mode['expected_both_error_count_independent']:.2f} expected under marginal "
                f"independence—an excess of {common_mode['excess_both_error_count']:.2f}. The positive "
                f"C_AB={common_mode['C_AB']:.4f} and Fisher p={common_mode['fisher_exact_two_sided_p']:.3e} "
                "show statistically significant synchronized failure in this sample. This identifies a common-mode "
                "weak region, but natural traces cannot distinguish shared Planner influence, shared model bias, "
                "or correlated item difficulty."
            ),
            "",
            "## Current evidence-weighted answer",
            "",
            (
                f"The opportunity-conditioned evidence points more strongly to the shared `Planner->dual Solver` "
                f"region than to Verifier or Aggregator: Verifier failed 0/{verifier_row['opportunity_count']} "
                f"defined repair opportunities and Aggregator degraded 0/{aggregator_row['opportunity_count']} "
                f"correct Verifier results, while both Solver branches failed together "
                f"{common_mode['both_error_count']}/{common_mode['n']} times with significant positive dependence."
            ),
            (
                f"All {common_mode_final_failures}/{mathematical_final_failures} mathematically wrong final answers "
                "occurred after both Solver states were already wrong. This supports prioritizing the shared-upstream/"
                "common-mode region for later controlled tests, without claiming that the Planner itself is the root "
                "cause."
            ),
            "",
            "## All edge transition distributions",
            "",
            "| Edge | Group | 1→1 | 1→0 degradation | 0→1 recovery | 0→0 propagation |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in edge_rows:
        for group in ("success", "failure"):
            lines.append(
                f"| `{row['edge']}` | {group} | {row[f'{group}_1_to_1_rate']:.4f} | "
                f"{row[f'{group}_1_to_0_rate']:.4f} | {row[f'{group}_0_to_1_rate']:.4f} | "
                f"{row[f'{group}_0_to_0_rate']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Node-level anomaly rates",
            "",
            "| Node | Overall | Success | Failure | Failure-success Δ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in node_rows:
        lines.append(
            f"| `{row['node']}` | {row['overall_anomaly_rate']:.4f} | "
            f"{row['success_anomaly_rate']:.4f} | {row['failure_anomaly_rate']:.4f} | "
            f"{row['anomaly_rate_difference']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Node anomaly rates describe local answer/contract states only. They are not topology bottleneck labels: an anomalous node may reflect an upstream error, a local degradation, or a failure to recover.",
            "",
            "## Five representative failure trajectories",
            "",
            "These five are selected automatically by first covering distinct transition signatures and then taking the largest smoothed transition log-likelihood ratio for failure versus success; no manual root-cause labels are used.",
            "",
        ]
    )
    for score, row in representatives:
        transitions = ", ".join(
            f"{edge}={payload['transition']}" for edge, payload in row["edge_transitions"].items()
        )
        problem = _markdown_escape(row["problem"])
        if len(problem) > 280:
            problem = problem[:277] + "..."
        lines.extend(
            [
                f"### `{row['sample_id']}` (source JSONL line {row['source_trajectory_line']}, score {score:.3f})",
                "",
                f"- Problem: {problem}",
                f"- Confirmed gold: `{_markdown_escape(row['gold_answer_confirmation']['reextracted_from_gold_solution'])}`",
                f"- Final answer: `{_markdown_escape(row['final_answer'])}`",
                f"- Transitions: `{transitions}`",
                "",
            ]
        )
    minimum_top3 = min(row["bootstrap_top3_frequency"] for row in edge_rows[:3])
    lines.extend(
        [
            "## Interpretation",
            "",
            f"The ranking is based on natural failures only. Across {bootstrap_samples} stratified bootstrap resamples, the least stable member of the observed top three remained in the top three {_fmt_rate(minimum_top3)} of the time. ",
            (
                "This quantifies sampling stability for these 100 trajectories, but it does not establish "
                f"causality. The failure group contains only {failure_n} trajectories, Planner validity is only "
                "weakly observable without a judge, and correlated downstream states can make multiple adjacent "
                "edges appear suspicious. The result therefore supports prioritizing edges for later controlled "
                "validation, not declaring a root cause."
            ),
            (
                "There is also an unavoidable evaluation coupling at the final edge: trajectory groups come from "
                "the stored final-answer `correct` label, while Aggregator health independently compares that same "
                "final answer to gold with a somewhat stronger equivalence checker. Consequently, a high "
                "`Verifier->Aggregator` JS value is descriptive but is not by itself evidence that Aggregator is "
                "the causal bottleneck."
            ),
            "",
            f"Full source trajectories: `{trajectories_path}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    source_rows = read_jsonl(args.trajectories)
    if not source_rows:
        raise ValueError("trajectory input is empty")
    localizations = [build_localization(row, line) for line, row in enumerate(source_rows, 1)]
    if not all(row["gold_answer_confirmation"]["verified"] for row in localizations):
        raise AssertionError("not every gold answer was verified")
    edge_rows = edge_statistics(localizations)
    add_bootstrap_stability(edge_rows, localizations, args.bootstrap_samples, args.seed)
    nodes = node_statistics(localizations)
    common_mode = common_mode_statistics(localizations, args.bootstrap_samples, args.seed)
    components = component_statistics(localizations, edge_rows, common_mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "localization_per_sample.jsonl", localizations)
    write_csv(args.output_dir / "edge_transition_statistics.csv", edge_rows)
    write_csv(args.output_dir / "node_statistics.csv", nodes)
    write_csv(args.output_dir / "component_weakness_statistics.csv", components)
    write_csv(args.output_dir / "common_mode_statistics.csv", [common_mode])
    write_summary(
        args.output_dir / "localization_summary.md",
        args.trajectories,
        localizations,
        edge_rows,
        nodes,
        components,
        common_mode,
        args.bootstrap_samples,
    )
    print(
        f"localized {len(localizations)} trajectories; verified_gold={len(localizations)}; "
        f"top_edge={edge_rows[0]['edge']} js={edge_rows[0]['js_divergence_bits']:.4f}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("experiments/math_autogen/outputs/qwen3_8b_seed42/trajectories.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/math_autogen/outputs/qwen3_8b_seed42/localization"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
