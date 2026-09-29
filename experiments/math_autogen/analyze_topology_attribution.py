"""Offline attribution analysis for the completed Omni-MATH topology study.

This script only reads saved trajectories/reports.  It does not call a model,
change a workflow, or run an evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.math_autogen.evaluator import (
    build_gold_spec,
    evaluate_answer,
    normalize_answer,
)

NODE_ORDER = (
    "DerivationSolver",
    "IndependentSolver",
    "Critic",
    "Refiner",
    "Finalizer",
)

BASE_DIAGNOSTIC_SPEC = {
    "ProblemAnalyzer": {
        "kind": "contract",
        "required_fields": ["Target:", "Problem type:", "Constraints:", "Risk flags:"],
    },
    "Decomposer": {
        "kind": "contract",
        "required_fields": ["Step 1", "Validation checkpoints:"],
    },
    "DerivationSolver": {"kind": "answer", "answer_label": "Proposed answer"},
    "IndependentSolver": {"kind": "answer", "answer_label": "Proposed answer"},
    "Critic": {
        "kind": "judge",
        "candidates": {
            "DerivationSolver": "Derivation verdict",
            "IndependentSolver": "Independent verdict",
        },
    },
    "Refiner": {"kind": "answer", "answer_label": "Refined answer"},
    "Finalizer": {"kind": "answer", "answer_source": "final_answer"},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in rows}


def event_for(row: dict[str, Any], role: str) -> dict[str, Any]:
    return next(event for event in row["trajectory"] if event["role"] == role)


def labeled_answer(text: str) -> str:
    matches = re.findall(
        r"(?im)^\s*(?:Proposed answer|Refined answer|FINAL_ANSWER|ANSWER)\s*:\s*(.+?)\s*$",
        text,
    )
    return normalize_answer(matches[-1]) if matches else ""


def answer_is_correct(answer: str, row: dict[str, Any]) -> bool:
    if not answer:
        return False
    try:
        return bool(
            evaluate_answer(
                answer,
                build_gold_spec(row["problem"], row["gold_solution"]),
            ).correct
        )
    # SymPy can raise AttributeError for syntactically parseable but invalid
    # expression trees (for example, a Tuple used as a Pow base).  Such a
    # malformed candidate is simply not a correct mathematical answer.
    except (TypeError, ValueError, SyntaxError, IndexError, AttributeError):
        return False


def extract_labeled(text: str, label: str | None) -> str:
    if not label:
        return text.strip()
    match = re.search(
        rf"{re.escape(label)}\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def extract_verdict(text: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}\s*:\s*(VALID|INVALID|UNCERTAIN)",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else "MISSING"


def observation_map(row: dict[str, Any], topology: dict[str, Any]) -> dict[str, bool]:
    specs = row.get("diagnostic_spec") or BASE_DIAGNOSTIC_SPEC
    states: dict[str, bool] = {}
    for role in topology["nodes"]:
        spec = specs[role]
        output = str(event_for(row, role).get("output", ""))
        if spec["kind"] == "answer":
            raw = str(row["final_answer"]) if spec.get("answer_source") == "final_answer" else output
            state = answer_is_correct(extract_labeled(raw, spec.get("answer_label")), row)
        elif spec["kind"] == "contract":
            state = bool(output.strip()) and all(
                field.lower() in output.lower() for field in spec["required_fields"]
            )
        elif spec["kind"] == "judge":
            judgments = [
                extract_verdict(output, label) == ("VALID" if states[source] else "INVALID")
                for source, label in spec["candidates"].items()
            ]
            state = bool(judgments) and all(judgments)
        else:
            raise ValueError(f"unsupported diagnostic kind: {spec['kind']}")
        states[role] = state
    return {role: not healthy for role, healthy in states.items()}


def transition(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    rescues: list[str] = []
    regressions: list[str] = []
    for sample_id in sorted(before):
        b_correct = bool(before[sample_id]["correct"])
        a_correct = bool(after[sample_id]["correct"])
        if not b_correct and a_correct:
            rescues.append(sample_id)
        elif b_correct and not a_correct:
            regressions.append(sample_id)
    return rescues, regressions


def exact_mcnemar_p(rescues: int, regressions: int) -> float:
    discordant = rescues + regressions
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, k) * 0.5**discordant
        for k in range(min(rescues, regressions) + 1)
    )
    return min(1.0, 2.0 * tail)


def binary_phi(table: dict[str, int]) -> float:
    a = table["both_healthy"]
    b = table["left_healthy_right_failed"]
    c = table["left_failed_right_healthy"]
    d = table["both_failed"]
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return ((a * d - b * c) / denominator) if denominator else 0.0


def token_words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def jaccard(left: str, right: str) -> float:
    a, b = token_words(left), token_words(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def pair_metrics(
    rows: list[dict[str, Any]],
    topology: dict[str, Any],
    left_role: str,
    right_role: str,
    *,
    verifier: bool = False,
) -> dict[str, Any]:
    table = Counter()
    answer_agreement = 0
    correctness_disagreement = 0
    reasoning_similarity: list[float] = []
    left_failed_total = 0
    right_failed_total = 0
    right_rescues_left = 0
    left_rescues_right = 0
    for row in rows:
        observations = observation_map(row, topology)
        left_failed = observations[left_role]
        right_failed = observations[right_role]
        if not left_failed and not right_failed:
            table["both_healthy"] += 1
        elif not left_failed and right_failed:
            table["left_healthy_right_failed"] += 1
        elif left_failed and not right_failed:
            table["left_failed_right_healthy"] += 1
        else:
            table["both_failed"] += 1
        correctness_disagreement += left_failed != right_failed
        left_failed_total += left_failed
        right_failed_total += right_failed
        right_rescues_left += left_failed and not right_failed
        left_rescues_right += right_failed and not left_failed

        left_output = str(event_for(row, left_role).get("output", ""))
        right_output = str(event_for(row, right_role).get("output", ""))
        if verifier:
            specs = row.get("diagnostic_spec") or BASE_DIAGNOSTIC_SPEC
            left_value = tuple(
                extract_verdict(left_output, label)
                for label in specs[left_role]["candidates"].values()
            )
            right_value = tuple(
                extract_verdict(right_output, label)
                for label in specs[right_role]["candidates"].values()
            )
        else:
            left_value = labeled_answer(left_output)
            right_value = labeled_answer(right_output)
        answer_agreement += left_value == right_value
        reasoning_similarity.append(jaccard(left_output, right_output))

    total = len(rows)
    result = {
        "n": total,
        "agreement_rate": answer_agreement / total,
        "disagreement_rate": 1.0 - answer_agreement / total,
        "correctness_disagreement_rate": correctness_disagreement / total,
        "joint_failure_rate": table["both_failed"] / total,
        "failure_table": dict(table),
        "failure_phi": binary_phi(table),
        "right_healthy_given_left_failed": (
            right_rescues_left / left_failed_total if left_failed_total else 0.0
        ),
        "left_healthy_given_right_failed": (
            left_rescues_right / right_failed_total if right_failed_total else 0.0
        ),
        "mean_reasoning_token_set_jaccard": sum(reasoning_similarity) / total,
    }
    return result


def state_pattern(
    row: dict[str, Any], topology: dict[str, Any], roles: tuple[str, ...]
) -> str:
    observations = observation_map(row, topology)
    return "(" + ",".join("F" if observations[role] else "T" for role in roles) + ")"


def role_token_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        for event in row["trajectory"]:
            result[event["role"]] += int(event.get("token_usage", {}).get("total_tokens", 0))
    return dict(result)


def pair_category(
    row: dict[str, Any], topology: dict[str, Any], left: str, right: str
) -> str:
    failed = observation_map(row, topology)
    if failed[left] and failed[right]:
        return "both_failed"
    if failed[left]:
        return "left_failed_right_healthy"
    if failed[right]:
        return "left_healthy_right_failed"
    return "both_healthy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-root",
        type=Path,
        default=ROOT
        / "experiments/math_autogen/outputs/topology_optimization_omni",
    )
    parser.add_argument(
        "--g0",
        type=Path,
        default=ROOT
        / "experiments/math_autogen/outputs/omni_heterogeneous_seed42"
        / "trajectories.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    study = args.study_root
    output_dir = args.output_dir or study / "attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "G0": args.g0,
        "G1": study / "G1/trajectories.jsonl",
        "G2": study / "versions/G2/evaluation/trajectories.jsonl",
        "G3": study / "versions/G3/evaluation/trajectories.jsonl",
    }
    configs = {
        key: read_json(study / f"versions/{key}/config.json")
        for key in ("G0", "G1", "G2", "G3")
    }
    rows = {key: read_jsonl(path) for key, path in paths.items()}
    by_id = {key: indexed(value) for key, value in rows.items()}
    topologies = {key: configs[key]["topology"] for key in configs}

    transitions: dict[str, dict[str, Any]] = {}
    for before_key, after_key in (("G0", "G1"), ("G1", "G2"), ("G1", "G3")):
        rescues, regressions = transition(by_id[before_key], by_id[after_key])
        transitions[f"{before_key}_to_{after_key}"] = {
            "rescues": rescues,
            "regressions": regressions,
            "net_gain": len(rescues) - len(regressions),
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                len(rescues), len(regressions)
            ),
        }

    g01 = transitions["G0_to_G1"]
    g01_rescue_patterns = Counter(
        state_pattern(by_id["G0"][sample_id], topologies["G0"], NODE_ORDER)
        for sample_id in g01["rescues"]
    )
    g01_regression_patterns = Counter(
        state_pattern(by_id["G0"][sample_id], topologies["G0"], NODE_ORDER)
        for sample_id in g01["regressions"]
    )
    baseline_error_ids = [
        sample_id
        for sample_id, row in by_id["G0"].items()
        if not bool(row["correct"])
    ]
    both_solver_failed_errors = [
        sample_id
        for sample_id in baseline_error_ids
        if state_pattern(
            by_id["G0"][sample_id],
            topologies["G0"],
            ("DerivationSolver", "IndependentSolver"),
        )
        == "(F,F)"
    ]
    targeted_rescues = [
        sample_id for sample_id in g01["rescues"] if sample_id in both_solver_failed_errors
    ]

    g2_pair = pair_metrics(
        rows["G2"],
        topologies["G2"],
        "IndependentSolver",
        "IndependentSolver_Peer",
    )
    g3_pair = pair_metrics(
        rows["G3"],
        topologies["G3"],
        "Critic",
        "Critic_Peer",
        verifier=True,
    )
    g0_pair = pair_metrics(
        rows["G0"],
        topologies["G0"],
        "DerivationSolver",
        "IndependentSolver",
    )
    g1_pair = pair_metrics(
        rows["G1"],
        topologies["G1"],
        "DerivationSolver",
        "IndependentSolver",
    )
    g2_transition_categories = {
        category: dict(
            Counter(
                pair_category(
                    by_id["G2"][sample_id],
                    topologies["G2"],
                    "IndependentSolver",
                    "IndependentSolver_Peer",
                )
                for sample_id in transitions["G1_to_G2"][category]
            )
        )
        for category in ("rescues", "regressions")
    }
    g3_transition_categories = {
        category: dict(
            Counter(
                pair_category(
                    by_id["G3"][sample_id],
                    topologies["G3"],
                    "Critic",
                    "Critic_Peer",
                )
                for sample_id in transitions["G1_to_G3"][category]
            )
        )
        for category in ("rescues", "regressions")
    }

    critic_contingency = Counter()
    for row in rows["G1"]:
        critic_failed = observation_map(row, topologies["G1"])["Critic"]
        critic_contingency[
            f"critic_{'failed' if critic_failed else 'healthy'}_"
            f"final_{'correct' if row['correct'] else 'wrong'}"
        ] += 1
    risk_when_failed = (
        critic_contingency["critic_failed_final_wrong"]
        / (
            critic_contingency["critic_failed_final_wrong"]
            + critic_contingency["critic_failed_final_correct"]
        )
    )
    risk_when_healthy = (
        critic_contingency["critic_healthy_final_wrong"]
        / (
            critic_contingency["critic_healthy_final_wrong"]
            + critic_contingency["critic_healthy_final_correct"]
        )
    )
    g1_critic_failure_ids = {
        row["sample_id"]
        for row in rows["G1"]
        if observation_map(row, topologies["G1"])["Critic"]
    }
    peer_healthy_on_g1_critic_failures = [
        sample_id
        for sample_id in sorted(g1_critic_failure_ids)
        if not observation_map(by_id["G3"][sample_id], topologies["G3"])[
            "Critic_Peer"
        ]
    ]
    g1_critic_failed_final_wrong = {
        row["sample_id"]
        for row in rows["G1"]
        if observation_map(row, topologies["G1"])["Critic"] and not row["correct"]
    }
    converted_critic_failures = [
        sample_id
        for sample_id in sorted(g1_critic_failed_final_wrong)
        if by_id["G3"][sample_id]["correct"]
    ]
    peer_override_regressions = [
        sample_id
        for sample_id in transitions["G1_to_G3"]["regressions"]
        if pair_category(
            by_id["G3"][sample_id], topologies["G3"], "Critic", "Critic_Peer"
        )
        == "left_healthy_right_failed"
    ]

    round_rows = [
        {
            "round": 1,
            "diagnosed_component": "DerivationSolver + IndependentSolver common-mode subgraph",
            "weakness_score": 0.5661805555555556,
            "rewrite_type": "parallel_workers_to_adversarial_debate",
            "rescues": 9,
            "regressions": 6,
            "net_gain": 3,
            "accuracy_delta": 0.03,
            "token_delta": 32906,
            "weakness_delta": 0.43 - 0.5661805555555556,
            "accepted": True,
        },
        {
            "round": 2,
            "diagnosed_component": "IndependentSolver",
            "weakness_score": 0.3964141414141414,
            "rewrite_type": "single_worker_to_fanout",
            "rescues": 10,
            "regressions": 9,
            "net_gain": 1,
            "accuracy_delta": 0.01,
            "token_delta": 337830,
            "weakness_delta": 0.0,
            "accepted": False,
        },
        {
            "round": 3,
            "diagnosed_component": "Critic",
            "weakness_score": 0.295,
            "rewrite_type": "single_verifier_to_multiple_verifiers",
            "rescues": 4,
            "regressions": 7,
            "net_gain": -3,
            "accuracy_delta": -0.03,
            "token_delta": 165602,
            "weakness_delta": 0.435 - 0.43,
            "accepted": False,
        },
    ]

    summary = {
        "scope": "offline_saved_trajectory_attribution_only",
        "transitions": transitions,
        "g0_to_g1": {
            "baseline_error_count": len(baseline_error_ids),
            "baseline_errors_with_both_solvers_failed": len(
                both_solver_failed_errors
            ),
            "target_aligned_rescues": len(targeted_rescues),
            "target_aligned_rescue_rate_among_rescues": (
                len(targeted_rescues) / len(g01["rescues"])
            ),
            "rescue_rate_within_targeted_baseline_errors": (
                len(targeted_rescues) / len(both_solver_failed_errors)
            ),
            "rescue_patterns_before": dict(g01_rescue_patterns),
            "regression_patterns_before": dict(g01_regression_patterns),
            "weakness_before": 0.5661805555555556,
            "weakness_after": 0.43,
            "error_propagation_before": 0.6821917808219178,
            "error_propagation_after": 0.6780487804878049,
            "baseline_worker_pair": g0_pair,
            "debate_worker_pair": g1_pair,
        },
        "g2_worker_pair": g2_pair,
        "g2_final_transition_pair_categories": g2_transition_categories,
        "g2_role_token_totals": role_token_totals(rows["G2"]),
        "g1_role_token_totals": role_token_totals(rows["G1"]),
        "g3_critic_localization": {
            "contingency": dict(critic_contingency),
            "final_failure_risk_given_critic_failure": risk_when_failed,
            "final_failure_risk_given_critic_healthy": risk_when_healthy,
            "risk_ratio": risk_when_failed / risk_when_healthy,
            "risk_difference": risk_when_failed - risk_when_healthy,
        },
        "g3_verifier_pair": g3_pair,
        "g3_final_transition_pair_categories": g3_transition_categories,
        "g3_peer_on_original_critic_failures": {
            "g1_critic_failure_count": len(g1_critic_failure_ids),
            "peer_healthy_count": len(peer_healthy_on_g1_critic_failures),
            "peer_healthy_sample_ids": peer_healthy_on_g1_critic_failures,
            "g1_critic_failed_and_final_wrong_count": len(
                g1_critic_failed_final_wrong
            ),
            "converted_to_final_correct_count": len(converted_critic_failures),
            "converted_sample_ids": converted_critic_failures,
            "peer_override_regression_ids": peer_override_regressions,
        },
        "g3_role_token_totals": role_token_totals(rows["G3"]),
        "rounds": round_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "rewrite_effectiveness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(round_rows[0]))
        writer.writeheader()
        writer.writerows(round_rows)

    with (output_dir / "paired_transitions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for transition_name, transition_data in transitions.items():
            before_key, after_key = transition_name.split("_to_")
            for category in ("rescues", "regressions"):
                for sample_id in transition_data[category]:
                    record = {
                        "transition": transition_name,
                        "category": category[:-1],
                        "sample_id": sample_id,
                        "before_correct": by_id[before_key][sample_id]["correct"],
                        "after_correct": by_id[after_key][sample_id]["correct"],
                        "before_final_answer": by_id[before_key][sample_id][
                            "final_answer"
                        ],
                        "after_final_answer": by_id[after_key][sample_id][
                            "final_answer"
                        ],
                        "before_node_state": state_pattern(
                            by_id[before_key][sample_id],
                            topologies[before_key],
                            NODE_ORDER,
                        ),
                        "after_node_state": state_pattern(
                            by_id[after_key][sample_id],
                            topologies[after_key],
                            NODE_ORDER,
                        ),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    representatives = {
        "G0_to_G1_targeted_rescue": ("omni-math-test-1168", "G0", "G1"),
        "G0_to_G1_debate_regression": ("omni-math-test-2854", "G0", "G1"),
        "G1_to_G2_peer_rescue": ("omni-math-test-2662", "G1", "G2"),
        "G1_to_G2_peer_regression": ("omni-math-test-1168", "G1", "G2"),
        "G1_to_G3_peer_override": ("omni-math-test-2551", "G1", "G3"),
        "G1_to_G3_non_unique_rescue": ("omni-math-test-2686", "G1", "G3"),
    }
    with (output_dir / "representative_trajectories.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for label, (sample_id, before_key, after_key) in representatives.items():
            record = {
                "label": label,
                "sample_id": sample_id,
                "problem": by_id[before_key][sample_id]["problem"],
                "before": by_id[before_key][sample_id],
                "after": by_id[after_key][sample_id],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
