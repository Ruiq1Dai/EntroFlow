"""Run a fixed-backbone AutoGen workflow on MuSiQue JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import asdict
from pathlib import Path

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.evaluation import exact_match_any, token_f1_any
from entroflow.musique import load_musique
from entroflow.reporting import (
    failed_trajectories,
    read_jsonl,
    summarize_results,
    write_json,
    write_jsonl,
)
from entroflow.routing import (
    AlwaysEvidenceBypassPolicy,
    AlwaysRetryEvidencePolicy,
    AlwaysValidatePolicy,
    EntropyPolicy,
    RandomRepairPolicy,
    StaticPolicy,
    TraceOnlyPolicy,
)


def load_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE settings without overriding the process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_policy(args):
    if args.policy == "static":
        return StaticPolicy()
    if args.policy == "random":
        return RandomRepairPolicy(probability=args.repair_probability, seed=args.seed)
    if args.policy == "trace":
        return TraceOnlyPolicy()
    if args.policy == "validate":
        return AlwaysValidatePolicy()
    if args.policy == "retry":
        return AlwaysRetryEvidencePolicy()
    if args.policy == "bypass":
        return AlwaysEvidenceBypassPolicy()
    if not args.embedding_model:
        raise ValueError("--embedding-model is required for entropy policy")
    return EntropyPolicy(
        embedder=LocalTransformerEmbedder(args.embedding_model),
        threshold=args.entropy_threshold,
    )


def stratified_hop_sample(examples, size: int, seed: int):
    """Select a near-balanced, reproducible sample across 2/3/4-hop questions."""
    if size < 1:
        raise ValueError("stratified sample size must be positive")
    groups = {2: [], 3: [], 4: []}
    for example in examples:
        match = re.match(r"([234])hop", example.example_id)
        if match:
            groups[int(match.group(1))].append(example)
    base, remainder = divmod(size, len(groups))
    quotas = {hop: base + int(index < remainder) for index, hop in enumerate(groups)}
    rng = random.Random(seed)
    selected = []
    for hop, candidates in groups.items():
        if len(candidates) < quotas[hop]:
            raise ValueError(f"not enough {hop}-hop examples for stratified sample")
        selected.extend(rng.sample(candidates, quotas[hop]))
    rng.shuffle(selected)
    return selected


def hop_count(example_id: str) -> int:
    match = re.match(r"([234])hop", example_id)
    if not match:
        raise ValueError(f"cannot determine hop count from example ID: {example_id}")
    return int(match.group(1))


def shuffled_by_hop(examples, seed: int):
    """Shuffle within hop groups and interleave them for failure-quota collection."""
    groups = {2: [], 3: [], 4: []}
    for example in examples:
        groups[hop_count(example.example_id)].append(example)
    rng = random.Random(seed)
    for candidates in groups.values():
        rng.shuffle(candidates)
    interleaved = []
    while any(groups.values()):
        for hop in (2, 3, 4):
            if groups[hop]:
                interleaved.append(groups[hop].pop())
    return interleaved


def cap_failures_by_hop(failures, targets):
    if not targets:
        return failures
    selected = []
    counts = {2: 0, 3: 0, 4: 0}
    for failure in failures:
        hop = hop_count(failure["example_id"])
        if counts[hop] < targets[hop]:
            selected.append(failure)
            counts[hop] += 1
    return selected


async def run(args) -> None:
    from entroflow.autogen_workflow import AutoGenMuSiQueWorkflow
    from entroflow.model_client import create_model_client

    load_env()
    examples = load_musique(args.data, limit=None if args.stratified_hops else args.limit)
    if args.selection_file:
        selected_rows = read_jsonl(args.selection_file)
        selected_ids = [row["example_id"] for row in selected_rows[: args.limit]]
        examples_by_id = {example.example_id: example for example in load_musique(args.data)}
        examples = [examples_by_id[example_id] for example_id in selected_ids]
    if args.stratified_hops:
        examples = stratified_hop_sample(examples, args.stratified_hops, args.seed)
    failure_targets = None
    if args.target_failures_by_hop:
        failure_targets = dict(zip((2, 3, 4), args.target_failures_by_hop))
        examples = shuffled_by_hop(load_musique(args.data), args.seed)
    policy = build_policy(args)
    model_client = create_model_client()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; pass --resume to append")
    completed_ids = set()
    failure_count = 0
    failures_by_hop = {2: 0, 3: 0, 4: 0}
    if args.resume and output.exists():
        existing_rows = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completed_ids = {row["example_id"] for row in existing_rows}
        failure_count = sum(not bool(row.get("em", False)) for row in existing_rows)
        for row in existing_rows:
            if not bool(row.get("em", False)):
                failures_by_hop[hop_count(row["example_id"])] += 1
    semantic_embedder = getattr(policy, "embedder", None) if args.policy == "entropy" else None
    workflow = AutoGenMuSiQueWorkflow(
        model_client=model_client,
        policy=policy,
        trace_path=args.traces,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        semantic_samples=args.semantic_samples if args.policy == "entropy" else 1,
        semantic_embedder=semantic_embedder,
    )

    async def evaluate(example):
        result = await workflow.run(example)
        accepted_answers = (example.answer, *example.answer_aliases)
        return {
            **asdict(result),
            "policy": args.policy,
            "question": example.question,
            "gold_answer": example.answer,
            "answer_aliases": list(example.answer_aliases),
            "em": exact_match_any(result.prediction, accepted_answers),
            "f1": token_f1_any(result.prediction, accepted_answers),
        }

    try:
        remaining = [example for example in examples if example.example_id not in completed_ids]
        while remaining:
            eligible = [
                example
                for example in remaining
                if not failure_targets
                or failures_by_hop[hop_count(example.example_id)]
                < failure_targets[hop_count(example.example_id)]
            ]
            if not eligible:
                break
            batch = eligible[: args.concurrency]
            batch_ids = {example.example_id for example in batch}
            remaining = [example for example in remaining if example.example_id not in batch_ids]
            rows = await asyncio.gather(*(evaluate(example) for example in batch))
            for example, row in zip(batch, rows):
                _append_jsonl(output, row)
                example_hop = hop_count(example.example_id)
                if not row["em"]:
                    failure_count += 1
                    failures_by_hop[example_hop] += 1
                print(
                    f"{example.example_id}: completed={row['completed']} "
                    f"em={row['em']} f1={row['f1']:.3f} tokens={row['total_tokens']} "
                    f"failures={failure_count}",
                    flush=True,
                )
            if args.target_failures is not None and failure_count >= args.target_failures:
                break
            if failure_targets and all(
                failures_by_hop[hop] >= target for hop, target in failure_targets.items()
            ):
                break
    finally:
        await model_client.close()

    result_rows = read_jsonl(output)
    summary = summarize_results(result_rows)
    summary["failures_by_hop"] = {
        str(hop): sum(
            not bool(row.get("em", False)) and hop_count(row["example_id"]) == hop
            for row in result_rows
        )
        for hop in (2, 3, 4)
    }
    summary.update({"policy": args.policy, "results": str(output), "traces": str(args.traces)})
    failures = failed_trajectories(result_rows, read_jsonl(args.traces))
    analysis_failures = cap_failures_by_hop(failures, failure_targets)
    summary["analysis_failure_count"] = len(analysis_failures)
    summary["analysis_failures_by_hop"] = {
        str(hop): sum(hop_count(row["example_id"]) == hop for row in analysis_failures)
        for hop in (2, 3, 4)
    }
    write_json(args.summary, summary)
    write_jsonl(args.failures, analysis_failures)
    print(
        f"summary: n={summary['num_examples']} accuracy={summary['accuracy_percent']:.2f}% "
        f"macro_f1={summary['macro_f1_percent']:.2f}%",
        flush=True,
    )


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--summary", default="artifacts/musique_summary.json")
    parser.add_argument("--failures", default="artifacts/musique_failures.jsonl")
    parser.add_argument(
        "--policy",
        choices=("static", "random", "trace", "entropy", "validate", "retry", "bypass"),
        default="static",
    )
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--entropy-threshold", type=float, default=0.15)
    parser.add_argument("--semantic-samples", type=int, default=5)
    parser.add_argument("--repair-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-file")
    parser.add_argument(
        "--stratified-hops",
        type=int,
        metavar="N",
        help="sample N examples near-equally across 2/3/4-hop groups",
    )
    parser.add_argument(
        "--target-failures",
        type=int,
        help="stop after this many cumulative non-EM results (including resumed rows)",
    )
    parser.add_argument(
        "--target-failures-by-hop",
        type=int,
        nargs=3,
        metavar=("HOP2", "HOP3", "HOP4"),
        help="collect exact cumulative failure quotas for 2/3/4-hop examples",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.semantic_samples < 2 and args.policy == "entropy":
        parser.error("--semantic-samples must be at least 2 for entropy policy")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
