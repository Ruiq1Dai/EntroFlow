"""Omni-MATH adapter for the framework-neutral topology optimizer plug-in."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from entroflow.topology_optimizer import (
        Diagnosis,
        RewriteProposal,
        compare_runs,
        diagnose,
        inspect_workflow,
        read_jsonl,
        select_local_rewrite,
        write_json,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from entroflow.topology_optimizer import (
        Diagnosis,
        RewriteProposal,
        compare_runs,
        diagnose,
        inspect_workflow,
        read_jsonl,
        select_local_rewrite,
        write_json,
    )

try:
    from experiments.math_autogen.evaluator import build_gold_spec, evaluate_answer
    from experiments.math_autogen.omni_heterogeneous import (
        DIAGNOSTIC_SPEC,
        baseline_workflow_config,
    )
except ModuleNotFoundError:
    from evaluator import build_gold_spec, evaluate_answer
    from omni_heterogeneous import DIAGNOSTIC_SPEC, baseline_workflow_config


RESULT_COLUMNS = [
    "round",
    "workflow_id",
    "topology_type",
    "weak_node",
    "weak_edge",
    "rewrite_type",
    "accuracy_before",
    "accuracy_after",
    "delta_accuracy",
    "tokens_before",
    "tokens_after",
    "token_cost_delta",
    "token_ratio",
    "latency_before",
    "latency_after",
    "latency_delta",
    "agent_calls_before",
    "agent_calls_after",
    "weakness_before",
    "weakness_after",
    "error_propagation_before",
    "error_propagation_after",
    "delta_error_propagation",
    "accepted",
]


def answer_evaluator(answer: str, row: dict[str, Any]) -> bool:
    """Offline-only task evaluation. This callback is never available to routing."""
    gold = build_gold_spec(row["problem"], row["gold_solution"])
    try:
        return evaluate_answer(answer, gold).correct
    except (TypeError, ValueError, SyntaxError, IndexError):
        return False


def specs_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows and rows[0].get("diagnostic_spec"):
        return rows[0]["diagnostic_spec"]
    return DIAGNOSTIC_SPEC


def diagnose_rows(rows: list[dict[str, Any]]) -> Diagnosis:
    return diagnose(rows, specs_for(rows), answer_evaluator)


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return baseline_workflow_config()
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_phase_with_sequence(topology: dict[str, Any], first: str, second: str) -> None:
    phases = topology["execution_phases"]
    for index, phase in enumerate(phases):
        if first in phase and second in phase:
            others = [node for node in phase if node not in (first, second)]
            replacement = [[first], [second]]
            if others:
                replacement.insert(0, others)
            phases[index : index + 1] = replacement
            return
    raise ValueError(f"cannot sequentialize {first!r} and {second!r}: no shared phase")


def _unique_node_name(nodes: list[str], stem: str) -> str:
    candidate = stem
    suffix = 2
    while candidate in nodes:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _append_edge(edges: list[list[str]], source: str, target: str) -> None:
    edge = [source, target]
    if edge not in edges:
        edges.append(edge)


def _judge_prompt(candidates: dict[str, str]) -> str:
    verdict_lines = ", ".join(f"`{label}: VALID|INVALID|UNCERTAIN`" for label in candidates.values())
    return (
        "You are an adversarial verifier. Audit every candidate against the original problem without using "
        "agreement or writing a replacement answer. Locate the earliest unsupported step, calculation error, "
        "or omitted constraint. Start with one verdict line per candidate in this exact format: "
        f"{verdict_lines}. Then report `Conflict: NONE|ANSWER|REASONING` and identify concrete revision targets."
    )


def apply_omni_rewrite(
    base_config: dict[str, Any],
    proposal: RewriteProposal,
    workflow_id: str,
) -> dict[str, Any]:
    """Apply one supported local graph diff to a copied Omni workflow config."""
    config = copy.deepcopy(base_config)
    config["parent_workflow_id"] = base_config["workflow_id"]
    config["workflow_id"] = workflow_id
    config["topology"]["name"] = workflow_id
    config["rewrite"] = proposal.to_dict()
    topology = config["topology"]
    edges = topology["edges"]

    if proposal.rewrite_type == "parallel_workers_to_adversarial_debate":
        first, second = proposal.target_nodes
        edge = [first, second]
        if edge not in edges:
            edges.append(edge)
        _replace_phase_with_sequence(topology, first, second)
        config["role_contracts"][second] = "adversarial_debate_answer"
        config["role_prompts"][second] = (
            "You are the adversarial second Solver in a debate. Your FIRST line must be exactly "
            "`Proposed answer: <answer>`. You see the original Omni-MATH problem, the Decomposer checklist, "
            "and the first Solver's full derivation. Treat that derivation as an untrusted claim: identify its "
            "earliest invalid assumption or calculation, actively seek a counterexample or independent invariant, "
            "and preserve it only if every decisive step survives. Develop a corrected alternative when needed. "
            "Do not decide by agreement or style. Stay under 700 words."
        )
    elif proposal.rewrite_type == "single_worker_to_fanout":
        (worker,) = proposal.target_nodes
        peer = _unique_node_name(topology["nodes"], f"{worker}_Peer")
        worker_index = topology["nodes"].index(worker)
        topology["nodes"].insert(worker_index + 1, peer)
        incoming = [edge for edge in list(edges) if edge[1] == worker]
        outgoing = [edge for edge in list(edges) if edge[0] == worker]
        for source, _ in incoming:
            _append_edge(edges, source, peer)
        for _, target in outgoing:
            _append_edge(edges, peer, target)
        for phase in topology["execution_phases"]:
            if worker in phase:
                phase.append(peer)
                break
        source_spec = copy.deepcopy(config["diagnostic_spec"][worker])
        answer_label = source_spec.get("answer_label", "Proposed answer")
        config["role_contracts"][peer] = "heterogeneous_peer_answer"
        config["role_prompts"][peer] = (
            "You are a heterogeneous peer problem solver. Solve the original problem independently from the "
            "other candidate workers, using a materially different representation, theorem, invariant, or "
            f"enumeration strategy. Your FIRST line must be exactly `{answer_label}: <answer>`. Reverse-check "
            "all original constraints and stay under 700 words."
        )
        config["role_max_tokens"][peer] = config["role_max_tokens"][worker]
        config["diagnostic_spec"][peer] = source_spec
        for target in (edge[1] for edge in outgoing):
            target_spec = config["diagnostic_spec"].get(target, {})
            if target_spec.get("kind") == "judge":
                candidates = target_spec.setdefault("candidates", {})
                candidates[peer] = f"{peer} verdict"
                config["role_prompts"][target] = _judge_prompt(candidates)
            elif target_spec.get("kind") == "answer":
                config["role_prompts"][target] += (
                    f" Treat {peer} as an additional candidate and score it by explicit constraint checks."
                )
    elif proposal.rewrite_type == "single_verifier_to_multiple_verifiers":
        (judge,) = proposal.target_nodes
        peer = f"{judge}_Peer"
        if peer in topology["nodes"]:
            raise ValueError(f"peer verifier already exists: {peer}")
        judge_index = topology["nodes"].index(judge)
        topology["nodes"].insert(judge_index + 1, peer)
        incoming = [edge for edge in list(edges) if edge[1] == judge]
        outgoing = [edge for edge in list(edges) if edge[0] == judge]
        edges.extend([[source, peer] for source, _ in incoming])
        edges.extend([[peer, target] for _, target in outgoing])
        for phase in topology["execution_phases"]:
            if judge in phase:
                phase.append(peer)
                break
        config["role_contracts"][peer] = "candidate_audit_only"
        config["role_prompts"][peer] = _judge_prompt(
            config["diagnostic_spec"][judge].get("candidates", {})
        )
        config["role_max_tokens"][peer] = config["role_max_tokens"][judge]
        config["diagnostic_spec"][peer] = copy.deepcopy(config["diagnostic_spec"][judge])
        for _, target in outgoing:
            config["role_prompts"][target] += (
                " You receive multiple independent verifier reports. Reconcile their concrete checks; do not use "
                "majority vote, and explicitly resolve any verdict disagreement before editing the solver work."
            )
    elif proposal.rewrite_type == "simple_merge_to_coordinator":
        (merge_node,) = proposal.target_nodes
        predecessors = [source for source, target in edges if target == merge_node]
        if len(predecessors) < 2:
            raise ValueError("coordinator rewrite requires a merge node with at least two predecessors")
        answer_label = config["diagnostic_spec"][merge_node].get("answer_label", "Selected answer")
        config["role_contracts"][merge_node] = "coordinator_based_selection"
        config["role_prompts"][merge_node] = (
            "You are a coordinator selecting among upstream candidate answers and verifier reports. Score every "
            "candidate explicitly for original-constraint coverage, derivation validity, independent checks, "
            "units, and requested format. Do not use majority vote, confidence language, or stylistic quality. "
            "Repair only a concrete localized defect and explain the decisive score. End with exactly one line "
            f"`{answer_label}: <answer>`."
        )
    elif proposal.rewrite_type == "long_chain_to_generate_verify":
        source, target = proposal.target_nodes
        verifier = _unique_node_name(topology["nodes"], f"{source}_Verifier")
        topology["nodes"].insert(topology["nodes"].index(target), verifier)
        _append_edge(edges, source, verifier)
        _append_edge(edges, verifier, target)
        target_phase = next(
            index for index, phase in enumerate(topology["execution_phases"]) if target in phase
        )
        topology["execution_phases"].insert(target_phase, [verifier])
        label = f"{source} verdict"
        config["role_contracts"][verifier] = "local_verification_only"
        config["role_prompts"][verifier] = (
            f"Verify {source}'s output against the original problem. Do not create a replacement answer. "
            f"Begin with `{label}: VALID|INVALID|UNCERTAIN`, then identify the earliest concrete defect."
        )
        config["role_max_tokens"][verifier] = 640
        config["diagnostic_spec"][verifier] = {
            "kind": "judge",
            "candidates": {source: label},
            "opportunity": "all_sources_observed",
            "opportunity_sources": [source],
        }
        config["role_prompts"][target] += (
            f" Use {verifier}'s local verdict to prevent propagation of a demonstrated upstream error."
        )
    elif proposal.rewrite_type == "independent_branches_to_shared_state":
        topology["shared_state"] = True
    else:
        raise ValueError(f"Omni adapter does not support rewrite {proposal.rewrite_type!r}")
    return config


def prepare_round(args: argparse.Namespace) -> None:
    if args.round < 1 or args.round > 3:
        raise ValueError("optimization round must be in [1, 3]")
    state_path = args.work_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state.get("stopped") and not getattr(args, "allow_after_stop", False):
        raise ValueError(f"optimization loop already stopped: {state.get('stop_reason')}")
    rows = read_jsonl(args.base_trajectories)
    graph = inspect_workflow(rows)
    diagnosis = diagnose_rows(rows)
    attempted = list(args.attempted_rewrite)
    for path in sorted(args.work_dir.glob("round_*/rewrite.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        attempted.append(payload["rewrite_type"])
    proposal = select_local_rewrite(diagnosis, graph, attempted)
    base_config = read_config(args.base_config)
    config = apply_omni_rewrite(
        base_config,
        proposal,
        args.candidate_workflow_id,
    )
    round_dir = args.work_dir / f"round_{args.round:02d}"
    base_version = args.work_dir / "versions" / "G0"
    candidate_version = args.work_dir / "versions" / f"G{args.round}"
    base_snapshot = round_dir / "base_config.json"
    candidate_snapshot = candidate_version / "config.json"
    write_json(base_snapshot, base_config)
    write_json(candidate_snapshot, config)
    initial_config = base_version / "config.json"
    initial_manifest = base_version / "manifest.json"
    if args.round == 1 and not initial_config.exists():
        write_json(initial_config, base_config)
    if args.round == 1 and not initial_manifest.exists():
        write_json(
            initial_manifest,
            {
                "version": "G0",
                "workflow_id": graph.workflow_id,
                "config": str(initial_config),
                "trajectories": str(args.base_trajectories),
                "status": "baseline",
            },
        )
    write_json(
        candidate_version / "manifest.json",
        {
            "version": f"G{args.round}",
            "workflow_id": args.candidate_workflow_id,
            "parent_workflow_id": graph.workflow_id,
            "config": str(candidate_snapshot),
            "status": "pending_evaluation",
            "rewrite_type": proposal.rewrite_type,
        },
    )
    write_json(round_dir / "diagnosis_before.json", diagnosis.to_dict())
    write_json(round_dir / "rewrite.json", proposal.to_dict())
    write_json(round_dir / "candidate_config.json", config)
    write_json(
        round_dir / "plan.json",
        {
            "round": args.round,
            "continuation_override": bool(state.get("stopped")),
            "base_workflow_id": graph.workflow_id,
            "base_trajectories": str(args.base_trajectories),
            "base_config": str(base_snapshot),
            "candidate_workflow_id": args.candidate_workflow_id,
            "candidate_config": str(candidate_snapshot),
            "rewrite_type": proposal.rewrite_type,
            "topology_diff": list(proposal.operations),
            "topology_before": " ; ".join(f"{source} -> {target}" for source, target in graph.edges),
            "topology_after": " ; ".join(
                f"{source} -> {target}" for source, target in config["topology"]["edges"]
            ),
            "reason": proposal.rationale,
        },
    )
    print(json.dumps({
        "top_weak_node": diagnosis.top_weak_node,
        "top_weak_edge": diagnosis.top_weak_edge,
        "top_weak_region": diagnosis.top_weak_region,
        "weakness_score": diagnosis.weakness_score,
        "rewrite": proposal.to_dict(),
        "candidate_config": str(candidate_snapshot),
    }, ensure_ascii=False, indent=2))


def _upsert_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            previous = [item for item in csv.DictReader(handle) if item.get("round") != str(row["round"])]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(previous)
        writer.writerow(row)


def _write_round_summary(
    path: Path,
    before: Diagnosis,
    after: Diagnosis,
    comparison: Any,
    rewrite: dict[str, Any],
    plan: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    weak_node = before.node_statistics[0] if before.node_statistics else None
    weak_edge = before.edge_statistics[0] if before.edge_statistics else None
    lines = [
        f"# Topology optimization round {decision['round']}",
        "",
        "## Diagnosis",
        "",
        f"- Workflow: `{before.workflow_id}` (`{before.topology_type}`)",
        f"- Top weak node: `{before.top_weak_node}`",
        f"- Top weak edge: `{before.top_weak_edge}`",
        f"- Weakness score: {before.weakness_score:.6f}",
    ]
    if weak_node:
        lines.extend(
            [
                f"- Node opportunity failures: {weak_node.opportunity_failure_count}/{weak_node.opportunity_count}",
                f"- Node downstream impact: {weak_node.downstream_impact:.6f}",
                f"- Representative node failures: {', '.join(weak_node.representative_failures) or 'none'}",
            ]
        )
    if weak_edge:
        lines.extend(
            [
                f"- Edge degradation: {weak_edge.degradation_count}/{weak_edge.source_healthy_count}",
                f"- Edge propagation: {weak_edge.propagation_count}/{weak_edge.source_failed_count}",
                f"- Representative edge failures: {', '.join(weak_edge.representative_failures) or 'none'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Local rewrite",
            "",
            f"- Type: `{rewrite['rewrite_type']}`",
            f"- Reason: {rewrite['rationale']}",
            f"- Before: `{plan['topology_before']}`",
            f"- After: `{plan['topology_after']}`",
            "",
            "## Paired result",
            "",
            (
                f"- Accuracy: {comparison.baseline.accuracy:.6f} -> {comparison.candidate.accuracy:.6f} "
                f"(delta {comparison.delta_accuracy:+.6f})"
            ),
            (
                f"- Weakness: {comparison.weakness_before:.6f} -> {comparison.weakness_after:.6f} "
                f"(reduction {comparison.weakness_reduction:+.6f})"
            ),
            (
                f"- Error propagation: {comparison.error_propagation_before:.6f} -> "
                f"{comparison.error_propagation_after:.6f} "
                f"(delta {comparison.delta_error_propagation:+.6f})"
            ),
            (
                f"- Tokens: {comparison.baseline.total_tokens} -> {comparison.candidate.total_tokens} "
                f"(ratio {comparison.token_ratio:.6f})"
            ),
            (
                f"- Mean latency: {comparison.baseline.mean_latency_seconds:.6f}s -> "
                f"{comparison.candidate.mean_latency_seconds:.6f}s (ratio {comparison.latency_ratio:.6f})"
            ),
            f"- Agent calls: {comparison.baseline.total_agent_calls} -> {comparison.candidate.total_agent_calls}",
            (
                f"- Decision: **{'accepted' if decision['accepted'] else 'rejected / rollback'}** — "
                f"{decision['decision_reason']}"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def finalize_round(args: argparse.Namespace) -> None:
    base_rows = read_jsonl(args.base_trajectories)
    candidate_rows = read_jsonl(args.candidate_trajectories)
    before = diagnose_rows(base_rows)
    after = diagnose_rows(candidate_rows)
    comparison = compare_runs(
        base_rows,
        candidate_rows,
        before,
        after,
        max_token_ratio=args.max_token_ratio,
        max_latency_ratio=args.max_latency_ratio,
        min_weakness_reduction=args.min_weakness_reduction,
    )
    rewrite = json.loads(args.rewrite.read_text(encoding="utf-8"))
    round_dir = args.work_dir / f"round_{args.round:02d}"
    plan_path = round_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # Backfill version snapshots for plans produced by the earlier two-step
    # prototype. New plans already contain these fields.
    if "base_config" not in plan:
        base_snapshot = round_dir / "base_config.json"
        base_payload = read_config(None)
        if base_payload["workflow_id"] != comparison.baseline.workflow_id:
            raise ValueError("legacy plan is missing its non-default base config")
        write_json(base_snapshot, base_payload)
        plan["base_config"] = str(base_snapshot)
    candidate_snapshot = args.work_dir / "versions" / f"G{args.round}" / "config.json"
    if not candidate_snapshot.exists():
        old_candidate = Path(plan["candidate_config"])
        write_json(candidate_snapshot, json.loads(old_candidate.read_text(encoding="utf-8")))
    plan["candidate_config"] = str(candidate_snapshot)
    if "topology_before" not in plan:
        before_graph = inspect_workflow(base_rows)
        candidate_graph = inspect_workflow(candidate_rows)
        plan["topology_before"] = " ; ".join(
            f"{source} -> {target}" for source, target in before_graph.edges
        )
        plan["topology_after"] = " ; ".join(
            f"{source} -> {target}" for source, target in candidate_graph.edges
        )
    write_json(plan_path, plan)
    initial_version = args.work_dir / "versions" / "G0"
    if not (initial_version / "config.json").exists():
        write_json(
            initial_version / "config.json",
            json.loads(Path(plan["base_config"]).read_text(encoding="utf-8")),
        )
    if not (initial_version / "manifest.json").exists():
        write_json(
            initial_version / "manifest.json",
            {
                "version": "G0",
                "workflow_id": comparison.baseline.workflow_id,
                "config": str(initial_version / "config.json"),
                "trajectories": str(args.base_trajectories),
                "status": "baseline",
            },
        )
    write_json(round_dir / "diagnosis_after.json", after.to_dict())
    write_json(round_dir / "comparison.json", comparison.to_dict())
    decision = {
        "round": args.round,
        "accepted": comparison.accepted,
        "decision_reason": comparison.decision_reason,
        "best_workflow_id": comparison.candidate.workflow_id if comparison.accepted else comparison.baseline.workflow_id,
        "best_trajectories": str(args.candidate_trajectories if comparison.accepted else args.base_trajectories),
        "performance_delta": comparison.delta_accuracy,
        "cost_delta": comparison.delta_total_tokens,
        "rewrite_type": rewrite["rewrite_type"],
        "best_config": plan["candidate_config"] if comparison.accepted else plan["base_config"],
    }
    write_json(round_dir / "decision.json", decision)
    _write_round_summary(
        round_dir / "round_summary.md",
        before,
        after,
        comparison,
        rewrite,
        plan,
        decision,
    )
    _upsert_result(
        args.work_dir / "round_results.csv",
        {
            "round": args.round,
            "workflow_id": comparison.candidate.workflow_id,
            "topology_type": after.topology_type,
            "weak_node": before.top_weak_node,
            "weak_edge": before.top_weak_edge,
            "rewrite_type": rewrite["rewrite_type"],
            "accuracy_before": comparison.baseline.accuracy,
            "accuracy_after": comparison.candidate.accuracy,
            "delta_accuracy": comparison.delta_accuracy,
            "tokens_before": comparison.baseline.total_tokens,
            "tokens_after": comparison.candidate.total_tokens,
            "token_cost_delta": comparison.delta_total_tokens,
            "token_ratio": comparison.token_ratio,
            "latency_before": comparison.baseline.mean_latency_seconds,
            "latency_after": comparison.candidate.mean_latency_seconds,
            "latency_delta": comparison.delta_mean_latency_seconds,
            "agent_calls_before": comparison.baseline.total_agent_calls,
            "agent_calls_after": comparison.candidate.total_agent_calls,
            "weakness_before": comparison.weakness_before,
            "weakness_after": comparison.weakness_after,
            "error_propagation_before": comparison.error_propagation_before,
            "error_propagation_after": comparison.error_propagation_after,
            "delta_error_propagation": comparison.delta_error_propagation,
            "accepted": comparison.accepted,
        },
    )
    state_path = args.work_dir / "state.json"
    previous_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    no_accuracy_gain = int(previous_state.get("consecutive_no_accuracy_gain", 0))
    no_accuracy_gain = 0 if comparison.delta_accuracy > 0 else no_accuracy_gain + 1
    budget_exceeded = comparison.token_ratio > args.max_token_ratio or comparison.latency_ratio > args.max_latency_ratio
    weakness_stalled = comparison.weakness_reduction < args.min_weakness_reduction
    stopped = False
    stop_reason = None
    if budget_exceeded:
        stopped, stop_reason = True, "token_or_latency_budget_exceeded"
    elif no_accuracy_gain >= 2:
        stopped, stop_reason = True, "two_consecutive_rounds_without_accuracy_gain"
    elif weakness_stalled:
        stopped, stop_reason = True, "weakness_score_did_not_decrease_materially"
    elif args.round >= 3:
        stopped, stop_reason = True, "maximum_rounds_reached"
    state = {
        "last_completed_round": args.round,
        "current_best_workflow_id": decision["best_workflow_id"],
        "current_best_trajectories": decision["best_trajectories"],
        "current_best_config": decision["best_config"],
        "consecutive_no_accuracy_gain": no_accuracy_gain,
        "last_weakness_reduction": comparison.weakness_reduction,
        "stopped": stopped,
        "stop_reason": stop_reason,
    }
    write_json(state_path, state)
    candidate_manifest = args.work_dir / "versions" / f"G{args.round}" / "manifest.json"
    write_json(
        candidate_manifest,
        {
            "version": f"G{args.round}",
            "workflow_id": comparison.candidate.workflow_id,
            "parent_workflow_id": comparison.baseline.workflow_id,
            "config": plan["candidate_config"],
            "trajectories": str(args.candidate_trajectories),
            "status": "accepted" if comparison.accepted else "rejected",
            "rewrite_type": rewrite["rewrite_type"],
            "delta_accuracy": comparison.delta_accuracy,
            "delta_total_tokens": comparison.delta_total_tokens,
        },
    )
    after_node = after.node_statistics[0] if after.node_statistics else None
    after_edge = after.edge_statistics[0] if after.edge_statistics else None
    print(json.dumps({
        "round": args.round,
        "diagnosis_after": {
            "top_weak_node": after.top_weak_node,
            "top_weak_edge": after.top_weak_edge,
            "weakness_score": after.weakness_score,
            "node_support": asdict(after_node) if after_node else None,
            "edge_support": asdict(after_edge) if after_edge else None,
        },
        "comparison": comparison.to_dict(),
        "decision": decision,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "round_summary": str(round_dir / "round_summary.md"),
    }, ensure_ascii=False, indent=2))


def _candidate_run_command(
    args: argparse.Namespace,
    summary: dict[str, Any],
    config_path: Path,
    reference_trajectories: Path,
    output_dir: Path,
) -> list[str]:
    difficulty = summary.get("difficulty_range") or [None, None]
    command = [
        str(args.runner_python),
        str(Path(__file__).with_name("omni_heterogeneous.py")),
        "--limit",
        str(summary["num_samples"]),
        "--seed",
        str(summary["seed"]),
        "--concurrency",
        str(args.concurrency),
        "--output-dir",
        str(output_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--dataset-split",
        str(summary.get("split", "test")),
        "--workflow-config",
        str(config_path),
        "--reference-trajectories",
        str(reference_trajectories),
        "--env-file",
        str(args.env_file),
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
    ]
    if summary.get("sampling_strategy") == "proportional_subject_x_difficulty":
        command.append("--stratified")
    if difficulty[0] is not None:
        command.extend(("--difficulty-min", str(difficulty[0])))
    if difficulty[1] is not None:
        command.extend(("--difficulty-max", str(difficulty[1])))
    if summary.get("dataset_revision"):
        command.extend(("--dataset-revision", str(summary["dataset_revision"])))
    for name in ("base_url", "api_key", "model"):
        value = getattr(args, name)
        if value:
            command.extend((f"--{name.replace('_', '-')}", str(value)))
    return command


def run_loop(args: argparse.Namespace) -> None:
    """Run diagnose -> one rewrite -> paired evaluation -> decision, sequentially."""
    if args.max_rounds < 1 or args.max_rounds > 3:
        raise ValueError("--max-rounds must be in [1, 3]")
    current_trajectories = args.base_trajectories
    current_config = args.base_config
    summary_path = current_trajectories.parent / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"baseline run summary is required: {summary_path}")
    evaluation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if evaluation_summary.get("num_samples") != len(read_jsonl(current_trajectories)):
        raise ValueError("baseline run summary and trajectory count disagree")

    state_path = args.work_dir / "state.json"
    start_round = 1
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("stopped") and not args.allow_after_stop:
            print(json.dumps({"stopped": True, "reason": state.get("stop_reason")}, ensure_ascii=False))
            return
        start_round = int(state.get("last_completed_round", 0)) + 1
        current_trajectories = Path(state["current_best_trajectories"])
        current_config = Path(state["current_best_config"])

    for round_number in range(start_round, args.max_rounds + 1):
        candidate_id = f"{args.workflow_prefix}_G{round_number}"
        plan_path = args.work_dir / f"round_{round_number:02d}" / "plan.json"
        if not plan_path.exists():
            prepare_args = argparse.Namespace(
                round=round_number,
                base_trajectories=current_trajectories,
                base_config=current_config,
                candidate_workflow_id=candidate_id,
                work_dir=args.work_dir,
                attempted_rewrite=[],
                allow_after_stop=args.allow_after_stop,
            )
            prepare_round(prepare_args)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        candidate_config = Path(plan["candidate_config"])
        candidate_dir = args.work_dir / "versions" / f"G{round_number}" / "evaluation"
        candidate_trajectories = candidate_dir / "trajectories.jsonl"
        candidate_rows = read_jsonl(candidate_trajectories) if candidate_trajectories.exists() else []
        if len(candidate_rows) != evaluation_summary["num_samples"]:
            command = _candidate_run_command(
                args,
                evaluation_summary,
                candidate_config,
                current_trajectories,
                candidate_dir,
            )
            if candidate_rows:
                command.append("--resume")
            print(json.dumps({"round": round_number, "phase": "evaluate", "command": command}, ensure_ascii=False))
            runner_env = dict(os.environ)
            runner_env["PYTHONNOUSERSITE"] = "1"
            subprocess.run(command, check=True, env=runner_env)
        finalize_args = argparse.Namespace(
            round=round_number,
            base_trajectories=current_trajectories,
            candidate_trajectories=candidate_trajectories,
            rewrite=args.work_dir / f"round_{round_number:02d}" / "rewrite.json",
            work_dir=args.work_dir,
            max_token_ratio=args.max_token_ratio,
            max_latency_ratio=args.max_latency_ratio,
            min_weakness_reduction=args.min_weakness_reduction,
        )
        finalize_round(finalize_args)
        state = json.loads((args.work_dir / "state.json").read_text(encoding="utf-8"))
        if state["current_best_trajectories"] == str(candidate_trajectories):
            current_trajectories = candidate_trajectories
            current_config = candidate_config
        if state["stopped"]:
            print(json.dumps({"stopped": True, "reason": state["stop_reason"]}, ensure_ascii=False))
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-round")
    prepare.add_argument("--round", type=int, required=True)
    prepare.add_argument("--base-trajectories", type=Path, required=True)
    prepare.add_argument("--base-config", type=Path)
    prepare.add_argument("--candidate-workflow-id", required=True)
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument("--attempted-rewrite", action="append", default=[])
    prepare.add_argument(
        "--allow-after-stop",
        action="store_true",
        help="explicitly prepare one exploratory round after a recorded stop condition",
    )
    prepare.set_defaults(function=prepare_round)

    finalize = subparsers.add_parser("finalize-round")
    finalize.add_argument("--round", type=int, required=True)
    finalize.add_argument("--base-trajectories", type=Path, required=True)
    finalize.add_argument("--candidate-trajectories", type=Path, required=True)
    finalize.add_argument("--rewrite", type=Path, required=True)
    finalize.add_argument("--work-dir", type=Path, required=True)
    finalize.add_argument("--max-token-ratio", type=float, default=1.35)
    finalize.add_argument("--max-latency-ratio", type=float, default=1.50)
    finalize.add_argument("--min-weakness-reduction", type=float, default=0.05)
    finalize.set_defaults(function=finalize_round)

    loop = subparsers.add_parser("run-loop")
    loop.add_argument("--base-trajectories", type=Path, required=True)
    loop.add_argument("--base-config", type=Path)
    loop.add_argument("--workflow-prefix", default="omni_topology")
    loop.add_argument("--work-dir", type=Path, required=True)
    loop.add_argument("--max-rounds", type=int, default=3)
    loop.add_argument("--max-token-ratio", type=float, default=1.35)
    loop.add_argument("--max-latency-ratio", type=float, default=1.50)
    loop.add_argument("--min-weakness-reduction", type=float, default=0.05)
    loop.add_argument("--runner-python", type=Path, default=Path(sys.executable))
    loop.add_argument("--concurrency", type=int, default=4)
    loop.add_argument("--cache-dir", type=Path, default=Path("experiments/math_autogen/.cache_omni"))
    loop.add_argument("--env-file", type=Path, default=Path(".env"))
    loop.add_argument("--base-url")
    loop.add_argument("--api-key")
    loop.add_argument("--model")
    loop.add_argument("--timeout", type=float, default=240.0)
    loop.add_argument("--max-retries", type=int, default=2)
    loop.add_argument(
        "--allow-after-stop",
        action="store_true",
        help="explicitly continue remaining rounds after a recorded early-stop condition",
    )
    loop.set_defaults(function=run_loop)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
