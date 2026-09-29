"""Run the functionally heterogeneous Omni-MATH AutoGen workflow.

This module intentionally lives beside, rather than inside, ``run.py`` so the
historical Planner/dual-Solver/Verifier/Aggregator topology remains frozen.
Gold solutions are never exposed to an agent. They are used only after the
workflow finishes to evaluate the Finalizer output.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

try:
    from experiments.math_autogen.evaluator import (
        EVALUATOR_VERSION,
        build_gold_spec,
        evaluate_answer,
        gold_metadata,
        normalize_answer,
    )
    from experiments.math_autogen.run import (
        OMNI_DATASET_NAME,
        append_jsonl,
        extract_final_answer,
        load_env,
        load_samples,
        read_jsonl,
    )
except ModuleNotFoundError:  # Direct script execution.
    from evaluator import (
        EVALUATOR_VERSION,
        build_gold_spec,
        evaluate_answer,
        gold_metadata,
        normalize_answer,
    )
    from run import (
        OMNI_DATASET_NAME,
        append_jsonl,
        extract_final_answer,
        load_env,
        load_samples,
        read_jsonl,
    )


WORKFLOW_ID = "omni_heterogeneous_v1"

HETEROGENEOUS_TOPOLOGY = {
    "name": WORKFLOW_ID,
    "nodes": [
        "ProblemAnalyzer",
        "Decomposer",
        "DerivationSolver",
        "IndependentSolver",
        "Critic",
        "Refiner",
        "Finalizer",
    ],
    "edges": [
        ["ProblemAnalyzer", "Decomposer"],
        ["Decomposer", "DerivationSolver"],
        ["Decomposer", "IndependentSolver"],
        ["DerivationSolver", "Critic"],
        ["IndependentSolver", "Critic"],
        ["DerivationSolver", "Refiner"],
        ["IndependentSolver", "Refiner"],
        ["Critic", "Refiner"],
        ["Refiner", "Finalizer"],
    ],
    "execution_phases": [
        ["ProblemAnalyzer"],
        ["Decomposer"],
        ["DerivationSolver", "IndependentSolver"],
        ["Critic"],
        ["Refiner"],
        ["Finalizer"],
    ],
}

ROLE_CONTRACTS = {
    "ProblemAnalyzer": "analysis_only",
    "Decomposer": "decomposition_only",
    "DerivationSolver": "plan_dependent_answer",
    "IndependentSolver": "independent_method_answer",
    "Critic": "candidate_audit_only",
    "Refiner": "critique_conditioned_revision",
    "Finalizer": "format_only",
}

ROLE_PROMPTS = {
    "ProblemAnalyzer": (
        "You are ProblemAnalyzer for Omni-MATH. Analyze only: identify the requested object, mathematical "
        "domain, explicit and implicit constraints, ambiguous interpretations, and likely failure risks. "
        "Do not solve, calculate the requested value, propose a method sequence, or state any candidate/final "
        "answer. Use exactly these headings: `Target:`, `Problem type:`, `Constraints:`, `Risk flags:`."
    ),
    "Decomposer": (
        "You are Decomposer. Convert the original problem and ProblemAnalyzer report into an executable "
        "dependency-ordered plan. State subgoals, what each step consumes and produces, and checkpoints that "
        "would falsify a step. Do not execute algebra/arithmetic, prove a subgoal, or state any candidate/final "
        "answer. Use numbered `Step 1`, `Step 2`, ... entries followed by `Validation checkpoints:`."
    ),
    "DerivationSolver": (
        "You are DerivationSolver. Solve the Omni-MATH problem by following the Decomposer steps in order. "
        "Your FIRST line must be exactly `Proposed answer: <answer>` so the candidate survives any later "
        "reasoning truncation. Then label each derivation with the corresponding step number, show auditable mathematical work, and "
        "explicitly report any plan step that must be repaired before continuing. Do not discuss the unseen "
        "IndependentSolver. Never repeat a failed derivation: identify its first invalid step once, repair it "
        "locally, and continue. Stay under 700 words."
    ),
    "IndependentSolver": (
        "You are IndependentSolver. Produce a genuinely independent solution to the original Omni-MATH "
        "problem. Your FIRST line must be exactly `Proposed answer: <answer>` so the candidate survives any "
        "later reasoning truncation. You may use the Decomposer text only as a checklist of targets and constraints; do not adopt "
        "its derivation route or imitate the likely direct derivation. Re-formulate the problem with a materially "
        "different representation, theorem, construction, invariant, or enumeration strategy, then reverse-check "
        "the result. You cannot see DerivationSolver. Stay under 700 words."
    ),
    "Critic": (
        "You are Critic. Audit the two solver reports against the original problem. Do not solve the whole "
        "problem again and do not give a replacement or final answer. For each solver, locate the earliest "
        "unsupported logical step, calculation error, omitted condition, or formatting issue; if none is found, "
        "say so. Explicitly analyze agreement or conflict between their answers. Begin with exactly four lines: "
        "`Derivation verdict: VALID|INVALID|UNCERTAIN`, `Independent verdict: VALID|INVALID|UNCERTAIN`, "
        "`Conflict: NONE|ANSWER|REASONING`, and `Revision target: DERIVATION|INDEPENDENT|BOTH|NONE`."
    ),
    "Refiner": (
        "You are Refiner. Revise the existing solver work using the Critic report. Do not start a fresh third "
        "solution and do not ignore the Critic. Preserve valid steps, repair only identified defects, reconcile "
        "conflicting candidates, and re-check all original constraints. Briefly list `Preserved:` and `Repaired:` "
        "items, then end with exactly one line `Refined answer: <answer>`."
    ),
    "Finalizer": (
        "You are Finalizer. Perform no mathematical reasoning and do not alter the Refiner's mathematical "
        "content. Extract only its refined answer, normalize required units and LaTeX presentation, and return "
        "exactly `<FINAL_ANSWER>\\boxed{answer}</FINAL_ANSWER>` with no other text."
    ),
}

ROLE_MAX_TOKENS = {
    "ProblemAnalyzer": 512,
    "Decomposer": 640,
    "DerivationSolver": 2048,
    "IndependentSolver": 2048,
    "Critic": 896,
    "Refiner": 1280,
    "Finalizer": 192,
}

DIAGNOSTIC_SPEC = {
    "ProblemAnalyzer": {
        "kind": "contract",
        "required_fields": ["Target:", "Problem type:", "Constraints:", "Risk flags:"],
    },
    "Decomposer": {
        "kind": "contract",
        "required_fields": ["Step 1", "Validation checkpoints:"],
        "opportunity": "all_sources_healthy",
        "opportunity_sources": ["ProblemAnalyzer"],
    },
    "DerivationSolver": {
        "kind": "answer",
        "answer_label": "Proposed answer",
        "opportunity": "all_sources_healthy",
        "opportunity_sources": ["Decomposer"],
    },
    "IndependentSolver": {
        "kind": "answer",
        "answer_label": "Proposed answer",
        "opportunity": "all_sources_healthy",
        "opportunity_sources": ["Decomposer"],
    },
    "Critic": {
        "kind": "judge",
        "candidates": {
            "DerivationSolver": "Derivation verdict",
            "IndependentSolver": "Independent verdict",
        },
        "opportunity": "all_sources_observed",
        "opportunity_sources": ["DerivationSolver", "IndependentSolver"],
    },
    "Refiner": {
        "kind": "answer",
        "answer_label": "Refined answer",
        "opportunity": "any_source_healthy",
        "opportunity_sources": ["DerivationSolver", "IndependentSolver"],
    },
    "Finalizer": {
        "kind": "answer",
        "answer_source": "final_answer",
        "opportunity": "all_sources_healthy",
        "opportunity_sources": ["Refiner"],
    },
}

_TOPOLOGY_EDGES = {tuple(edge) for edge in HETEROGENEOUS_TOPOLOGY["edges"]}


@dataclass(frozen=True)
class AgentRequest:
    run_id: str
    sample_id: str
    agent: str
    input_text: str
    actual_visible_context: list[dict[str, str]]


@dataclass(frozen=True)
class AgentResponse:
    agent: str
    output: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class HeterogeneousMathAgent(RoutedAgent):
    """One role-scoped agent; visibility is supplied explicitly per request."""

    def __init__(
        self,
        role: str,
        model_client: ChatCompletionClient,
        role_prompt: str,
        max_tokens: int,
    ):
        super().__init__(f"Omni-MATH {role}")
        self.role = role
        self.model_client = model_client
        self.role_prompt = role_prompt
        self.max_tokens = max_tokens

    @message_handler
    async def handle(self, message: AgentRequest, ctx: MessageContext) -> AgentResponse:
        if message.agent != self.role:
            raise ValueError(f"{self.role} received request for {message.agent}")
        result = await self.model_client.create(
            [
                SystemMessage(content=self.role_prompt),
                UserMessage(content=message.input_text, source="experiment_orchestrator"),
            ],
            extra_create_args={
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
            cancellation_token=ctx.cancellation_token,
        )
        if not isinstance(result.content, str):
            raise TypeError(f"{self.role} returned non-text content")
        return AgentResponse(
            agent=self.role,
            output=result.content,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_visible_context(
    role: str,
    messages: Sequence[tuple[str, str]],
    role_prompts: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build the exact system/user messages delivered to one role."""
    user_content = "\n\n".join(f"[{sender}]\n{content}" for sender, content in messages)
    return [
        {
            "role": "system",
            "sender": "system",
            "content": (role_prompts or ROLE_PROMPTS)[role],
        },
        {
            "role": "user",
            "sender": "+".join(sender for sender, _ in messages),
            "content": user_content,
        },
    ]


def validate_message_edges(
    role: str,
    messages: Sequence[tuple[str, str]],
    topology_edges: set[tuple[str, str]] | None = None,
) -> list[list[str]]:
    """Return and validate the agent-to-agent edges realized by a call."""
    incoming = [
        [sender, role]
        for sender, _ in messages
        if sender not in {"User", "SharedState"}
    ]
    declared_edges = topology_edges or _TOPOLOGY_EDGES
    unknown = [edge for edge in incoming if tuple(edge) not in declared_edges]
    if unknown:
        raise ValueError(f"messages realize undeclared topology edges: {unknown}")
    return incoming


def baseline_workflow_config() -> dict[str, Any]:
    return copy.deepcopy({
        "workflow_id": WORKFLOW_ID,
        "parent_workflow_id": None,
        "topology": HETEROGENEOUS_TOPOLOGY,
        "role_contracts": ROLE_CONTRACTS,
        "role_prompts": ROLE_PROMPTS,
        "role_max_tokens": ROLE_MAX_TOKENS,
        "diagnostic_spec": DIAGNOSTIC_SPEC,
        "rewrite": None,
    })


def load_workflow_config(path: Path | None) -> dict[str, Any]:
    config = baseline_workflow_config()
    if path is None:
        return config
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("workflow_id", "topology", "role_contracts", "role_prompts", "role_max_tokens"):
        if key not in payload:
            raise ValueError(f"workflow config is missing {key!r}")
    config.update(payload)
    nodes = set(config["topology"]["nodes"])
    for mapping_name in ("role_contracts", "role_prompts", "role_max_tokens"):
        missing = nodes - set(config[mapping_name])
        if missing:
            raise ValueError(f"{mapping_name} is missing nodes: {sorted(missing)}")
    return config


class OmniHeterogeneousWorkflow:
    def __init__(
        self,
        model_client: ChatCompletionClient,
        model_name: str,
        config: dict[str, Any] | None = None,
    ):
        self.model_client = model_client
        self.model_name = model_name
        self.config = config or baseline_workflow_config()
        self.workflow_id = str(self.config["workflow_id"])
        self.topology = self.config["topology"]
        self.role_contracts = self.config["role_contracts"]
        self.role_prompts = self.config["role_prompts"]
        self.role_max_tokens = self.config["role_max_tokens"]
        self.topology_edges = {tuple(edge) for edge in self.topology["edges"]}
        self.predecessors = {
            role: [source for source, target in self.topology["edges"] if target == role]
            for role in self.topology["nodes"]
        }
        sinks = [
            role
            for role in self.topology["nodes"]
            if not any(source == role for source, _ in self.topology["edges"])
        ]
        configured_final = self.topology.get("final_node")
        if configured_final is not None:
            if configured_final not in self.topology["nodes"]:
                raise ValueError(f"configured final_node is not a topology node: {configured_final!r}")
            self.final_node = str(configured_final)
        elif len(sinks) == 1:
            self.final_node = sinks[0]
        else:
            raise ValueError(
                "workflow topology must declare final_node when it does not have exactly one sink; "
                f"found sinks={sinks}"
            )

    async def _register(self, runtime: SingleThreadedAgentRuntime) -> None:
        for role in self.topology["nodes"]:
            await HeterogeneousMathAgent.register(
                runtime,
                role.lower(),
                lambda role=role: HeterogeneousMathAgent(
                    role,
                    self.model_client,
                    self.role_prompts[role],
                    int(self.role_max_tokens[role]),
                ),
            )

    async def run_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid4())
        sample_id = f"omni-math-test-{sample['dataset_index']:04d}"
        events: list[dict[str, Any]] = []
        runtime = SingleThreadedAgentRuntime()
        await self._register(runtime)
        runtime.start()

        async def invoke(
            role: str,
            messages: Sequence[tuple[str, str]],
            phase: int,
            launch_order: int,
        ) -> AgentResponse:
            context = make_visible_context(role, messages, self.role_prompts)
            incoming_edges = validate_message_edges(role, messages, self.topology_edges)
            request = AgentRequest(
                run_id=run_id,
                sample_id=sample_id,
                agent=role,
                input_text=context[-1]["content"],
                actual_visible_context=context,
            )
            started_at = utc_now()
            response = await runtime.send_message(request, AgentId(role.lower(), run_id))
            finished_at = utc_now()
            if not isinstance(response, AgentResponse):
                raise TypeError(f"unexpected {role} response: {type(response).__name__}")
            events.append(
                {
                    "agent": role,
                    "role": role,
                    "role_contract": self.role_contracts[role],
                    "sender": [sender for sender, _ in messages],
                    "receiver": role,
                    "incoming_edges": incoming_edges,
                    "shared_state": bool(self.topology.get("shared_state")),
                    "model": self.model_name,
                    "execution_phase": phase,
                    "launch_order": launch_order,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "actual_visible_context": context,
                    "input": request.input_text,
                    "output": response.output,
                    "token_usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                    },
                }
            )
            return response

        run_status = "completed"
        error = None
        final_answer = ""
        try:
            problem = sample["problem"]
            outputs: dict[str, AgentResponse] = {}
            launch_order = 0
            scheduled: set[str] = set()
            for phase, phase_roles in enumerate(self.topology["execution_phases"], start=1):
                calls = []
                for role in phase_roles:
                    if role in scheduled:
                        raise ValueError(f"topology schedules node more than once: {role!r}")
                    missing = [source for source in self.predecessors[role] if source not in outputs]
                    if missing:
                        raise ValueError(
                            f"phase order invokes {role!r} before predecessors are available: {missing}"
                        )
                    scheduled.add(role)
                    launch_order += 1
                    messages = [("User", problem)] + [
                        (source, outputs[source].output) for source in self.predecessors[role]
                    ]
                    if self.topology.get("shared_state"):
                        shared_outputs = [
                            (source, response.output)
                            for source, response in outputs.items()
                            if source not in self.predecessors[role]
                        ]
                        if shared_outputs:
                            messages.append(
                                (
                                    "SharedState",
                                    "\n\n".join(
                                        f"[{source}]\n{output}"
                                        for source, output in shared_outputs
                                    ),
                                )
                            )
                    calls.append((role, invoke(role, messages, phase, launch_order)))
                phase_outputs = await asyncio.gather(*(call for _, call in calls))
                outputs.update(
                    (role, response)
                    for (role, _), response in zip(calls, phase_outputs)
                )
            unscheduled = set(self.topology["nodes"]) - scheduled
            if unscheduled:
                raise ValueError(f"topology nodes are not scheduled in execution_phases: {sorted(unscheduled)}")
            final_answer = extract_final_answer(outputs[self.final_node].output)
        except Exception as exc:  # noqa: BLE001 - preserve partial traces on model/runtime failure.
            run_status = "runtime_error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            await runtime.stop_when_idle()

        events.sort(key=lambda event: event["launch_order"])

        # Evaluation begins only after every agent call has finished. Neither
        # gold_solution nor gold_answer appears in any visible context above.
        gold = build_gold_spec(sample["problem"], sample["solution"])
        evaluation = evaluate_answer(final_answer, gold)
        correct = run_status == "completed" and evaluation.correct
        total_prompt = sum(event["token_usage"]["prompt_tokens"] for event in events)
        total_completion = sum(event["token_usage"]["completion_tokens"] for event in events)
        observed_edges = {
            tuple(edge)
            for event in events
            for edge in event["incoming_edges"]
        }
        realized_edges = [
            list(edge)
            for edge in self.topology["edges"]
            if tuple(edge) in observed_edges
        ]
        return {
            "schema_version": "2.0",
            "workflow_id": self.workflow_id,
            "run_id": run_id,
            "sample_id": sample_id,
            "dataset": {
                "name": OMNI_DATASET_NAME,
                "revision": sample["dataset_revision"],
                "split": "test",
                "dataset_index": sample["dataset_index"],
                "level": sample["level"],
                "type": sample["type"],
            },
            "problem": sample["problem"],
            "condition": self.workflow_id,
            "topology": self.topology,
            "role_contracts": self.role_contracts,
            "diagnostic_spec": self.config.get("diagnostic_spec", {}),
            "rewrite": self.config.get("rewrite"),
            "realized_agent_edges": realized_edges,
            "execution_order": self.topology["execution_phases"],
            "trajectory": events,
            "gold_solution": sample["solution"],
            "gold_answer": gold.canonical_answer,
            "gold_evaluation": gold_metadata(gold),
            "final_answer": final_answer,
            "normalized_gold_answer": normalize_answer(gold.canonical_answer),
            "normalized_final_answer": normalize_answer(final_answer),
            "final_evaluation": {
                "evaluator_version": EVALUATOR_VERSION,
                "method": evaluation.method,
                "component_methods": list(evaluation.component_methods),
            },
            "correct": correct,
            "run_status": run_status,
            "error": error,
            "token_usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
        }


CSV_COLUMNS = [
    "sample_id",
    "dataset_index",
    "level",
    "type",
    "run_status",
    "correct",
    "gold_answer",
    "final_answer",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "dataset_index": row["dataset"]["dataset_index"],
                    "level": row["dataset"]["level"],
                    "type": row["dataset"]["type"],
                    "run_status": row["run_status"],
                    "correct": row["correct"],
                    "gold_answer": row["gold_answer"],
                    "final_answer": row["final_answer"],
                    "prompt_tokens": row["token_usage"]["prompt_tokens"],
                    "completion_tokens": row["token_usage"]["completion_tokens"],
                    "total_tokens": row["token_usage"]["total_tokens"],
                }
            )


def build_summary(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    completed = sum(row["run_status"] == "completed" for row in rows)
    correct = sum(bool(row["correct"]) for row in rows)
    workflow_id = rows[0]["workflow_id"] if rows else WORKFLOW_ID
    topology = rows[0]["topology"] if rows else HETEROGENEOUS_TOPOLOGY
    role_contracts = rows[0]["role_contracts"] if rows else ROLE_CONTRACTS
    return {
        "workflow_id": workflow_id,
        "dataset": OMNI_DATASET_NAME,
        "split": args.dataset_split,
        "dataset_revision": args.dataset_revision,
        "seed": args.seed,
        "sampling_strategy": "proportional_subject_x_difficulty" if args.stratified else "simple_random",
        "difficulty_range": [args.difficulty_min, args.difficulty_max],
        "sample_strata": dict(
            sorted(Counter(f"{row['dataset']['type']} | {row['dataset']['level']}" for row in rows).items())
        ),
        "requested_samples": args.limit,
        "num_samples": len(rows),
        "completed_count": completed,
        "runtime_error_count": len(rows) - completed,
        "success_count": correct,
        "failure_count": len(rows) - correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "accuracy_percent": 100.0 * correct / len(rows) if rows else 0.0,
        "evaluator_version": EVALUATOR_VERSION,
        "model": args.model,
        "base_url": args.base_url,
        "topology": topology,
        "role_contracts": role_contracts,
        "token_usage": {
            key: sum(row["token_usage"][key] for row in rows)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "trajectories_jsonl": str(args.output_dir / "trajectories.jsonl"),
        "summary_csv": str(args.output_dir / "summary.csv"),
    }


def write_summary_files(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    write_summary_csv(output_dir / "summary.csv", rows)
    (output_dir / "run_summary.json").write_text(
        json.dumps(build_summary(rows, args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def samples_from_reference(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct exactly the previously evaluated inputs from saved traces.

    This is only a sample-set lock.  Gold fields are retained in the in-memory
    sample for post-run evaluation, but are never included in any agent message.
    """
    samples: list[dict[str, Any]] = []
    for row in rows:
        dataset = row["dataset"]
        samples.append(
            {
                "dataset_index": dataset["dataset_index"],
                "dataset_revision": dataset.get("revision", ""),
                "problem": row["problem"],
                "solution": row["gold_solution"],
                "level": dataset.get("level", ""),
                "type": dataset.get("type", ""),
            }
        )
    return samples


async def run(args: argparse.Namespace) -> None:
    import httpx
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    load_env(args.env_file)
    args.api_key = args.api_key or os.getenv("MODEL_API_KEY", "dummy_key")
    args.base_url = args.base_url or os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    args.model = args.model or os.getenv("MODEL_NAME", "Qwen3-8B")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "trajectories.jsonl"
    if trajectory_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {trajectory_path}; pass --resume")

    if args.reference_trajectories:
        reference_rows = read_jsonl(args.reference_trajectories)
        if not reference_rows:
            raise ValueError(f"reference trajectory file is empty: {args.reference_trajectories}")
        samples = samples_from_reference(reference_rows)
        if len(samples) != args.limit:
            raise ValueError(
                f"--limit={args.limit} does not match reference sample count={len(samples)}"
            )
        reference_indices = {row["dataset"]["dataset_index"] for row in reference_rows}
        selected_indices = {sample["dataset_index"] for sample in samples}
        if selected_indices != reference_indices:
            raise ValueError(
                "candidate sample indices differ from --reference-trajectories: "
                f"selected_only={sorted(selected_indices - reference_indices)[:5]} "
                f"reference_only={sorted(reference_indices - selected_indices)[:5]}"
            )
    else:
        samples = load_samples(
            args.limit,
            args.seed,
            args.cache_dir,
            args.dataset_revision,
            stratified=args.stratified,
            dataset_name=OMNI_DATASET_NAME,
            split=args.dataset_split,
            difficulty_min=args.difficulty_min,
            difficulty_max=args.difficulty_max,
        )
    existing = read_jsonl(trajectory_path) if args.resume else []
    completed_indices = {row["dataset"]["dataset_index"] for row in existing}
    remaining = [sample for sample in samples if sample["dataset_index"] not in completed_indices]

    http_client = httpx.AsyncClient(trust_env=False)
    model_client = OpenAIChatCompletionClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        model_info={
            "function_calling": False,
            "json_output": False,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
        parallel_tool_calls=False,
        timeout=args.timeout,
        max_retries=args.max_retries,
        http_client=http_client,
    )
    workflow_config = load_workflow_config(args.workflow_config)
    workflow = OmniHeterogeneousWorkflow(model_client, args.model, workflow_config)
    try:
        for offset in range(0, len(remaining), args.concurrency):
            batch = remaining[offset : offset + args.concurrency]
            batch_rows = await asyncio.gather(*(workflow.run_sample(sample) for sample in batch))
            for row in batch_rows:
                append_jsonl(trajectory_path, row)
                existing.append(row)
                print(
                    f"[{len(existing)}/{args.limit}] {row['sample_id']} "
                    f"status={row['run_status']} correct={row['correct']} "
                    f"answer={row['final_answer']!r} tokens={row['token_usage']['total_tokens']}",
                    flush=True,
                )
            write_summary_files(args.output_dir, existing, args)
    finally:
        await model_client.close()

    selected_indices = {sample["dataset_index"] for sample in samples}
    rows = [row for row in existing if row["dataset"]["dataset_index"] in selected_indices]
    write_summary_files(args.output_dir, rows, args)
    summary = build_summary(rows, args)
    print(
        f"done: n={summary['num_samples']} accuracy={summary['accuracy_percent']:.2f}% "
        f"success={summary['success_count']} failure={summary['failure_count']} "
        f"runtime_errors={summary['runtime_error_count']}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/math_autogen/outputs/omni_heterogeneous_seed42"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/math_autogen/.cache_omni"),
    )
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-revision", default="")
    parser.add_argument("--difficulty-min", type=float)
    parser.add_argument("--difficulty-max", type=float)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument(
        "--workflow-config",
        type=Path,
        help="JSON workflow copy produced by the topology optimizer",
    )
    parser.add_argument(
        "--reference-trajectories",
        type=Path,
        help="fail unless the selected dataset indices exactly match this baseline JSONL",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if (
        args.difficulty_min is not None
        and args.difficulty_max is not None
        and args.difficulty_min > args.difficulty_max
    ):
        parser.error("--difficulty-min cannot exceed --difficulty-max")
    return args


def main() -> None:
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("interrupted; use --resume to continue", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
