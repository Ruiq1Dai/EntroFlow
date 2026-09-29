"""Generate self-contained MATH trajectories with a fixed AutoGen topology.

The experiment is deliberately independent from EntroFlow's MuSiQue workflow.
It does not import or modify any routing, fault-injection, debugging, scoring,
or repair code from the rest of the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
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

__all__ = ["answers_equal", "extract_gold_answer"]

try:
    from experiments.math_autogen.evaluator import (
        EVALUATOR_VERSION,
        answers_equal,
        build_gold_spec,
        evaluate_answer,
        extract_gold_answer,
        gold_metadata,
        last_boxed,
        normalize_answer,
    )
except ModuleNotFoundError:  # Direct script execution.
    from evaluator import (
        EVALUATOR_VERSION,
        answers_equal,
        build_gold_spec,
        evaluate_answer,
        extract_gold_answer,
        gold_metadata,
        last_boxed,
        normalize_answer,
    )

DATASET_NAME = "qwedsacf/competition_math"
OMNI_DATASET_NAME = "KbsdJames/Omni-MATH"
DATASET_REVISION = "509c45f9e8254bd0070c1e9a84fb4a54a03b0b8e"
TOPOLOGY = {
    "nodes": ["Planner", "Solver_A", "Solver_B", "Verifier", "Aggregator"],
    "edges": [
        ["Planner", "Solver_A"],
        ["Planner", "Solver_B"],
        ["Solver_A", "Verifier"],
        ["Solver_B", "Verifier"],
        ["Verifier", "Aggregator"],
    ],
    "execution_phases": [
        ["Planner"],
        ["Solver_A", "Solver_B"],
        ["Verifier"],
        ["Aggregator"],
    ],
}
PLANNER_BYPASS_TOPOLOGY = {
    "nodes": TOPOLOGY["nodes"],
    "edges": [
        ["Solver_A", "Verifier"],
        ["Solver_B", "Verifier"],
        ["Verifier", "Aggregator"],
    ],
    "bypassed_edges": [
        ["Planner", "Solver_A"],
        ["Planner", "Solver_B"],
    ],
    "execution_phases": TOPOLOGY["execution_phases"],
}
SOLVER_B_PLANNER_MASK_TOPOLOGY = {
    "nodes": TOPOLOGY["nodes"],
    "edges": [
        ["Planner", "Solver_A"],
        ["Solver_A", "Verifier"],
        ["Solver_B", "Verifier"],
        ["Verifier", "Aggregator"],
    ],
    "bypassed_edges": [["Planner", "Solver_B"]],
    "execution_phases": TOPOLOGY["execution_phases"],
}
SOLVER_B_SELF_REVIEW_TOPOLOGY = {
    "nodes": ["Planner", "Solver_A", "Solver_B_Draft", "Solver_B", "Verifier", "Aggregator"],
    "edges": [
        ["Planner", "Solver_A"],
        ["Planner", "Solver_B_Draft"],
        ["Planner", "Solver_B"],
        ["Solver_B_Draft", "Solver_B"],
        ["Solver_A", "Verifier"],
        ["Solver_B", "Verifier"],
        ["Verifier", "Aggregator"],
    ],
    "execution_phases": [
        ["Planner"],
        ["Solver_A", "Solver_B_Draft"],
        ["Solver_B"],
        ["Verifier"],
        ["Aggregator"],
    ],
}

ROLE_PROMPTS = {
    "Planner": (
        "You are the Planner in a competition-math team. Analyze the problem, identify the target, "
        "list the key facts and constraints, and propose a concise step-by-step solution plan. Do not "
        "carry out the full calculation and do not claim a final answer. Your plan will be sent independently "
        "to two solvers. Be precise, challenge tempting but unjustified heuristics, and stay under 180 words."
    ),
    "Solver_A": (
        "You are Solver_A. Independently solve the competition-math problem using the planner's plan as "
        "optional guidance, but correct it if needed. Prefer a direct symbolic derivation. Show enough work for "
        "a verifier to audit, stay under 450 words, and end with exactly one line of the form "
        "`Proposed answer: <answer>`. You cannot see the other solver's work."
    ),
    "Solver_B": (
        "You are Solver_B. Independently solve the competition-math problem using the planner's plan as "
        "optional guidance, but audit every claimed heuristic. Seek an alternative derivation or a decisive "
        "sanity check; inspect algebra, edge cases, and interpretation carefully. Stay under 450 words and end "
        "with exactly one line of the form `Proposed answer: <answer>`. You cannot see the other solver's work."
    ),
    "Verifier": (
        "You are the Verifier. Simultaneously audit both independent solver reports against the original "
        "problem. Check derivations, constraints, arithmetic, and answer formatting. Resolve disagreements by "
        "re-deriving the disputed step. Your FIRST line must be `Recommended answer: <answer>`. Then give a "
        "concise audit under 350 words, stating which reasoning is valid."
    ),
    "Aggregator": (
        "You are the Aggregator. Use the verifier report to give the final answer to the original problem. "
        "Do not add a solution or commentary. Return exactly `<FINAL_ANSWER>answer</FINAL_ANSWER>`, preserving "
        "any needed LaTeX."
    ),
}

DIVERSIFIED_SOLVER_B_PROMPT = (
    "You are Solver_B. Independently rebuild the mathematical model from the original competition-math "
    "problem before solving it. Treat the planner's plan only as optional orientation: do not simply follow "
    "or paraphrase its proposed route. Prefer a materially different representation, theorem, construction, "
    "or computational route from the most obvious approach; if no distinct route is viable, derive the "
    "method again from first principles. Before answering, reverse-verify your result by substituting it back, "
    "reconstructing the required quantity from the answer, or checking an independent invariant and all domain "
    "constraints. Stay under 450 words and end with exactly one line of the form "
    "`Proposed answer: <answer>`. You cannot see or reference Solver_A's work."
)
SOLVER_B_SELF_REVIEW_PROMPT = (
    "You are Solver_B performing a mandatory second pass over your own draft. You cannot see Solver_A. "
    "Re-read the original problem, Planner plan, and Solver_B_Draft. Find the first unsupported assumption or "
    "calculation error, independently recompute the decisive quantities, and check all constraints. Then "
    "reverse-verify the candidate answer by substitution, reconstruction, or an invariant. If the draft survives "
    "these checks, preserve it; otherwise correct it. Give a concise auditable derivation under 450 words and "
    "end with exactly one line of the form `Proposed answer: <answer>`."
)
VERIFIER_REDERIVATION_PROMPT = (
    "You are the Verifier. First solve the original competition-math problem independently from first "
    "principles without adopting either Solver's conclusion. Record a compact internal derivation, then audit "
    "Solver_A and Solver_B against it, checking constraints, algebra, arithmetic, and formatting. On disagreement, "
    "trust only a derivation you can reproduce and reverse-check by substitution, reconstruction, or an invariant. "
    "Your FIRST line must be `Recommended answer: <answer>`. Then give a concise audit under 350 words stating "
    "which reasoning, if any, agrees with the independent derivation."
)
VERIFIER_DISAGREEMENT_PROBE_PROMPT = (
    "You are the Verifier. If Solver_A and Solver_B give different final answers, independently verify "
    "each candidate separately before choosing: substitute it into the original conditions, reverse-derive "
    "the requested quantity, or check all constraints. Do not decide by superficial plausibility or majority. "
    "If the candidates agree, use the ordinary audit logic. Your FIRST line must be `Recommended answer: <answer>`. "
    "Then give a concise audit under 350 words stating the checks performed and which candidate is valid."
)

ROLE_MAX_TOKENS = {
    "Planner": 384,
    "Solver_A": 1024,
    "Solver_B": 1024,
    "Solver_B_Draft": 1024,
    "Verifier": 768,
    "Aggregator": 192,
}


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


class MathAgent(RoutedAgent):
    """One AutoGen Core agent whose visibility is fully specified by its request."""

    def __init__(self, role: str, model_client: ChatCompletionClient, role_prompt: str):
        super().__init__(f"MATH {role}")
        self.role = role
        self.model_client = model_client
        self.role_prompt = role_prompt

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
                "max_tokens": ROLE_MAX_TOKENS[self.role],
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
    agent: str,
    messages: Sequence[tuple[str, str]],
    role_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Return the exact system/user context passed to the model for an agent."""
    user_content = "\n\n".join(f"[{sender}]\n{content}" for sender, content in messages)
    return [
        {"role": "system", "sender": "system", "content": role_prompt or ROLE_PROMPTS[agent]},
        {"role": "user", "sender": "+".join(sender for sender, _ in messages), "content": user_content},
    ]


def _input_from_context(context: Sequence[dict[str, str]]) -> str:
    return context[-1]["content"]


_last_boxed = last_boxed


def extract_final_answer(output: str) -> str:
    matches = re.findall(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", output, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        answer = matches[-1].strip()
        boxed = _last_boxed(answer)
        return boxed.strip() if boxed is not None else answer
    boxed = _last_boxed(output)
    if boxed is not None:
        return boxed.strip()
    match = re.search(r"(?:final|proposed)\s+answer\s*:\s*(.+)$", output, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else output.strip()


def solver_input_messages(
    problem: str,
    planner_output: str,
    planner_bypass: bool,
) -> list[tuple[str, str]]:
    if planner_bypass:
        return [("User", problem)]
    return [("User", problem), ("Planner", planner_output)]


class MathWorkflow:
    def __init__(
        self,
        model_client: ChatCompletionClient,
        planner_bypass: bool = False,
        solver_diversification: bool = False,
        solver_b_planner_mask: bool = False,
        solver_b_self_review: bool = False,
        verifier_rederivation: bool = False,
        verifier_disagreement_probe: bool = False,
        model_name: str | None = None,
        solver_b_model_client: ChatCompletionClient | None = None,
        solver_b_model_name: str | None = None,
    ):
        if sum(
            (
                planner_bypass,
                solver_diversification,
                solver_b_planner_mask,
                solver_b_self_review,
                verifier_rederivation,
                verifier_disagreement_probe,
            )
        ) > 1:
            raise ValueError("active probes cannot be combined")
        self.model_client = model_client
        self.planner_bypass = planner_bypass
        self.solver_diversification = solver_diversification
        self.solver_b_planner_mask = solver_b_planner_mask
        self.solver_b_self_review = solver_b_self_review
        self.verifier_rederivation = verifier_rederivation
        self.verifier_disagreement_probe = verifier_disagreement_probe
        self.solver_b_model_client = solver_b_model_client
        self.solver_b_model_name = solver_b_model_name
        self.agent_models = {node: model_name for node in TOPOLOGY["nodes"]}
        if solver_b_model_name:
            self.agent_models["Solver_B"] = solver_b_model_name
        self.role_prompts = dict(ROLE_PROMPTS)
        if solver_diversification:
            self.role_prompts["Solver_B"] = DIVERSIFIED_SOLVER_B_PROMPT
        if solver_b_self_review:
            self.role_prompts["Solver_B_Draft"] = ROLE_PROMPTS["Solver_B"]
            self.role_prompts["Solver_B"] = SOLVER_B_SELF_REVIEW_PROMPT
        if verifier_rederivation:
            self.role_prompts["Verifier"] = VERIFIER_REDERIVATION_PROMPT
        if verifier_disagreement_probe:
            self.role_prompts["Verifier"] = VERIFIER_DISAGREEMENT_PROBE_PROMPT

    async def _register(self, runtime: SingleThreadedAgentRuntime) -> None:
        roles = list(TOPOLOGY["nodes"])
        if self.solver_b_self_review:
            roles.append("Solver_B_Draft")
        for role in roles:
            model_client = (
                self.solver_b_model_client
                if role == "Solver_B" and self.solver_b_model_client is not None
                else self.model_client
            )
            await MathAgent.register(
                runtime,
                role.lower(),
                lambda role=role, model_client=model_client: MathAgent(
                    role, model_client, self.role_prompts[role]
                ),
            )

    async def run_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        run_id = str(uuid4())
        sample_id = f"math-test-{sample['dataset_index']:04d}"
        events: list[dict[str, Any]] = []
        runtime = SingleThreadedAgentRuntime()
        await self._register(runtime)
        runtime.start()

        async def invoke(
            agent: str,
            messages: Sequence[tuple[str, str]],
            sender: str | list[str],
            phase: int,
            launch_order: int,
        ) -> AgentResponse:
            context = make_visible_context(agent, messages, self.role_prompts[agent])
            request = AgentRequest(
                run_id=run_id,
                sample_id=sample_id,
                agent=agent,
                input_text=_input_from_context(context),
                actual_visible_context=context,
            )
            started_at = utc_now()
            response = await runtime.send_message(request, AgentId(agent.lower(), run_id))
            finished_at = utc_now()
            if not isinstance(response, AgentResponse):
                raise TypeError(f"unexpected {agent} response: {type(response).__name__}")
            events.append(
                {
                    "agent": agent,
                    "sender": sender,
                    "receiver": agent,
                    "model": self.agent_models.get(agent),
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
            planner = await invoke("Planner", [("User", problem)], "User", 1, 1)
            solver_a_messages = solver_input_messages(problem, planner.output, self.planner_bypass)
            solver_b_messages = solver_input_messages(
                problem,
                planner.output,
                self.planner_bypass or self.solver_b_planner_mask,
            )
            solver_a_sender = "User" if self.planner_bypass else "Planner"
            solver_b_sender = "User" if self.planner_bypass or self.solver_b_planner_mask else "Planner"
            solver_a_task = asyncio.create_task(
                invoke(
                    "Solver_A",
                    solver_a_messages,
                    solver_a_sender,
                    2,
                    2,
                )
            )
            if self.solver_b_self_review:
                solver_b_draft_task = asyncio.create_task(
                    invoke(
                        "Solver_B_Draft",
                        solver_b_messages,
                        solver_b_sender,
                        2,
                        3,
                    )
                )
                solver_a, solver_b_draft = await asyncio.gather(solver_a_task, solver_b_draft_task)
                solver_b = await invoke(
                    "Solver_B",
                    [
                        ("User", problem),
                        ("Planner", planner.output),
                        ("Solver_B_Draft", solver_b_draft.output),
                    ],
                    ["Planner", "Solver_B_Draft"],
                    3,
                    4,
                )
                verifier_phase = 4
                verifier_order = 5
                aggregator_phase = 5
                aggregator_order = 6
            else:
                solver_b_task = asyncio.create_task(
                    invoke(
                        "Solver_B",
                        solver_b_messages,
                        solver_b_sender,
                        2,
                        3,
                    )
                )
                solver_a, solver_b = await asyncio.gather(solver_a_task, solver_b_task)
                verifier_phase = 3
                verifier_order = 4
                aggregator_phase = 4
                aggregator_order = 5
            verifier = await invoke(
                "Verifier",
                [("User", problem), ("Solver_A", solver_a.output), ("Solver_B", solver_b.output)],
                ["Solver_A", "Solver_B"],
                verifier_phase,
                verifier_order,
            )
            aggregator = await invoke(
                "Aggregator",
                [("User", problem), ("Verifier", verifier.output)],
                "Verifier",
                aggregator_phase,
                aggregator_order,
            )
            final_answer = extract_final_answer(aggregator.output)
        except Exception as exc:  # noqa: BLE001 - partial service-failure traces must survive.
            run_status = "runtime_error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            await runtime.stop_when_idle()

        events.sort(key=lambda event: event["launch_order"])
        gold = build_gold_spec(sample["problem"], sample["solution"])
        gold_answer = gold.canonical_answer
        total_prompt = sum(event["token_usage"]["prompt_tokens"] for event in events)
        total_completion = sum(event["token_usage"]["completion_tokens"] for event in events)
        evaluation = evaluate_answer(final_answer, gold)
        correct = run_status == "completed" and evaluation.correct
        return {
            "schema_version": "1.1",
            "run_id": run_id,
            "sample_id": sample_id,
            "dataset": {
                "name": DATASET_NAME,
                "revision": sample["dataset_revision"],
                "split": "test",
                "dataset_index": sample["dataset_index"],
                "level": sample["level"],
                "type": sample["type"],
            },
            "problem": sample["problem"],
            "condition": (
                "model_diversity"
                if self.solver_b_model_name
                else
                "planner_bypass"
                if self.planner_bypass
                else "solver_diversification"
                if self.solver_diversification
                else "solver_b_planner_mask"
                if self.solver_b_planner_mask
                else "solver_b_self_review"
                if self.solver_b_self_review
                else "verifier_rederivation"
                if self.verifier_rederivation
                else "verifier_disagreement_probe"
                if self.verifier_disagreement_probe
                else "baseline"
            ),
            "intervention": {
                "name": (
                    "model_diversity"
                    if self.solver_b_model_name
                    else
                    "planner_bypass"
                    if self.planner_bypass
                    else "solver_diversification"
                    if self.solver_diversification
                    else "solver_b_planner_mask"
                    if self.solver_b_planner_mask
                    else "solver_b_self_review"
                    if self.solver_b_self_review
                    else "verifier_rederivation"
                    if self.verifier_rederivation
                    else "verifier_disagreement_probe"
                    if self.verifier_disagreement_probe
                    else "none"
                ),
                "description": (
                    "Only Solver_B uses the configured alternate model; all prompts, topology, and other agents are unchanged."
                    if self.solver_b_model_name
                    else
                    "Solver_A and Solver_B receive only the original problem; Planner output is not delivered."
                    if self.planner_bypass
                    else "Only Solver_B uses an independent-remodeling, different-method, reverse-verification prompt."
                    if self.solver_diversification
                    else "Only Planner-to-Solver_B delivery is masked; Solver_A still receives Planner output."
                    if self.solver_b_planner_mask
                    else "Solver_B adds a separate second-pass review of its own baseline-prompt draft."
                    if self.solver_b_self_review
                    else "Only Verifier changes to independent re-derivation before auditing both Solver reports."
                    if self.verifier_rederivation
                    else "Only Verifier independently checks each candidate when Solver answers disagree."
                    if self.verifier_disagreement_probe
                    else "none"
                ),
            },
            "topology": (
                PLANNER_BYPASS_TOPOLOGY
                if self.planner_bypass
                else SOLVER_B_PLANNER_MASK_TOPOLOGY
                if self.solver_b_planner_mask
                else SOLVER_B_SELF_REVIEW_TOPOLOGY
                if self.solver_b_self_review
                else TOPOLOGY
            ),
            "execution_order": (
                SOLVER_B_SELF_REVIEW_TOPOLOGY["execution_phases"]
                if self.solver_b_self_review
                else TOPOLOGY["execution_phases"]
            ),
            "trajectory": events,
            "gold_solution": sample["solution"],
            "gold_answer": gold_answer,
            "gold_evaluation": gold_metadata(gold),
            "final_answer": final_answer,
            "normalized_gold_answer": normalize_answer(gold_answer),
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


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def stratified_sample_indices(dataset: Sequence[dict[str, Any]], limit: int, seed: int) -> list[int]:
    """Allocate proportionally across every subject×difficulty stratum."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        if "type" in row and "level" in row:
            key = (str(row["type"]), str(row["level"]))
        else:
            domain = " | ".join(row["domain"]) if isinstance(row["domain"], list) else str(row["domain"])
            key = (domain, str(row["difficulty"]))
        groups[key].append(index)
    raw_allocations = {
        key: limit * len(indices) / len(dataset) for key, indices in groups.items()
    }
    allocations = {key: math.floor(value) for key, value in raw_allocations.items()}
    remainder = limit - sum(allocations.values())
    order = sorted(
        groups,
        key=lambda key: (-(raw_allocations[key] - allocations[key]), key),
    )
    for key in order[:remainder]:
        allocations[key] += 1
    if limit >= len(groups) and any(count == 0 for count in allocations.values()):
        raise AssertionError("proportional allocation unexpectedly omitted an available stratum")
    rng = random.Random(seed)
    selected = [
        index
        for key in sorted(groups)
        for index in rng.sample(groups[key], allocations[key])
    ]
    rng.shuffle(selected)
    return selected


def load_samples(
    limit: int,
    seed: int,
    cache_dir: Path,
    revision: str,
    stratified: bool = False,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        revision=revision or None,
        cache_dir=str(cache_dir),
    )
    eligible_indices = list(range(len(dataset)))
    if dataset_name == OMNI_DATASET_NAME:
        eligible_indices = []
        for i, row in enumerate(dataset):
            answer = str(row.get("answer", "")).strip()
            if not answer or "no final answer" in answer.lower() or "does not contain a solution" in answer.lower():
                continue
            try:
                build_gold_spec(row["problem"], f"{row['solution']}\n\\boxed{{{answer}}}")
            except (TypeError, ValueError):
                continue
            eligible_indices.append(i)
        if difficulty_min is not None or difficulty_max is not None:
            eligible_indices = [
                i for i in eligible_indices
                if (difficulty_min is None or float(dataset[i]["difficulty"]) >= difficulty_min)
                and (difficulty_max is None or float(dataset[i]["difficulty"]) <= difficulty_max)
            ]
    if limit < 1 or limit > len(eligible_indices):
        raise ValueError(f"--limit must be in [1, {len(eligible_indices)} eligible samples]")
    sampling_dataset = dataset.select(eligible_indices) if hasattr(dataset, "select") else [dataset[i] for i in eligible_indices]
    indices = (
        stratified_sample_indices(sampling_dataset, limit, seed)
        if stratified
        else random.Random(seed).sample(range(len(eligible_indices)), limit)
    )
    indices = [eligible_indices[index] for index in indices]
    rows = []
    for index in indices:
        item = dataset[index]
        if dataset_name == OMNI_DATASET_NAME:
            category = " | ".join(item["domain"]) if isinstance(item["domain"], list) else str(item["domain"])
            level = str(item["difficulty"])
            solution = item["solution"]
            answer = item.get("answer", "")
            # Omni-MATH's solution formatting is heterogeneous; append the
            # explicit answer as a final canonical box so evaluator v2 can
            # deterministically parse even malformed/missing LaTeX boxes.
            if answer:
                solution = f"{solution}\n\\boxed{{{answer}}}"
        else:
            category = item["type"]
            level = item["level"]
            solution = item["solution"]
        rows.append({
            "dataset_index": index,
            "dataset_revision": revision,
            "problem": item["problem"],
            "solution": solution,
            "level": level,
            "type": category,
        })
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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
    return {
        "dataset": args.dataset_name,
        "dataset_revision": args.dataset_revision,
        "split": "test",
        "seed": args.seed,
        "sampling_strategy": "proportional_subject_x_difficulty" if args.stratified else "simple_random",
        "difficulty_range": [args.difficulty_min, args.difficulty_max],
        "sample_strata": dict(
            sorted(Counter(f"{row['dataset']['type']} | {row['dataset']['level']}" for row in rows).items())
        ),
        "evaluator_version": EVALUATOR_VERSION,
        "requested_samples": args.limit,
        "num_samples": len(rows),
        "completed_count": completed,
        "runtime_error_count": len(rows) - completed,
        "success_count": correct,
        "failure_count": len(rows) - correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "accuracy_percent": 100.0 * correct / len(rows) if rows else 0.0,
        "token_usage": {
            key: sum(row["token_usage"][key] for row in rows)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "model": args.model,
        "solver_b_model": args.solver_b_model,
        "solver_b_base_url": args.solver_b_base_url,
        "base_url": args.base_url,
        "condition": (
            "model_diversity"
            if args.solver_b_model
            else
            "planner_bypass"
            if args.planner_bypass
            else "solver_diversification"
            if args.solver_diversification
            else "solver_b_planner_mask"
            if args.solver_b_planner_mask
            else "solver_b_self_review"
            if args.solver_b_self_review
            else "verifier_rederivation"
            if args.verifier_rederivation
            else "verifier_disagreement_probe"
            if args.verifier_disagreement_probe
            else "baseline"
        ),
        "topology": (
            PLANNER_BYPASS_TOPOLOGY
            if args.planner_bypass
            else SOLVER_B_PLANNER_MASK_TOPOLOGY
            if args.solver_b_planner_mask
            else SOLVER_B_SELF_REVIEW_TOPOLOGY
            if args.solver_b_self_review
            else TOPOLOGY
        ),
        "trajectories_jsonl": str(args.output_dir / "trajectories.jsonl"),
        "summary_csv": str(args.output_dir / "summary.csv"),
    }


def write_summary_files(output_dir: Path, rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    write_csv(output_dir / "summary.csv", rows)
    summary = build_summary(rows, args)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reevaluate_output_dir(args: argparse.Namespace) -> None:
    """Reapply the current deterministic answer evaluator to saved trajectories."""
    trajectory_path = args.output_dir / "trajectories.jsonl"
    rows = read_jsonl(trajectory_path)
    if not rows:
        raise FileNotFoundError(f"no trajectories found at {trajectory_path}")
    args.limit = len(rows)
    for row in rows:
        gold = build_gold_spec(row["problem"], row["gold_solution"])
        evaluation = evaluate_answer(row["final_answer"], gold)
        row["schema_version"] = "1.1"
        row["gold_answer"] = gold.canonical_answer
        row["gold_evaluation"] = gold_metadata(gold)
        row["normalized_gold_answer"] = normalize_answer(row["gold_answer"])
        row["normalized_final_answer"] = normalize_answer(row["final_answer"])
        row["final_evaluation"] = {
            "evaluator_version": EVALUATOR_VERSION,
            "method": evaluation.method,
            "component_methods": list(evaluation.component_methods),
        }
        row["correct"] = row["run_status"] == "completed" and evaluation.correct
    temporary_path = trajectory_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(trajectory_path)
    write_summary_files(args.output_dir, rows, args)
    summary = build_summary(rows, args)
    print(
        f"reevaluated: n={summary['num_samples']} accuracy={summary['accuracy_percent']:.2f}% "
        f"success={summary['success_count']} failure={summary['failure_count']}",
        flush=True,
    )


async def run(args: argparse.Namespace) -> None:
    import httpx
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    load_env(args.env_file)
    args.api_key = args.api_key or os.getenv("MODEL_API_KEY", "dummy_key")
    args.base_url = args.base_url or os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    args.model = args.model or os.getenv("MODEL_NAME", "Qwen3-8B")
    args.solver_b_model = args.solver_b_model or os.getenv("SOLVER_B_MODEL")
    args.solver_b_base_url = args.solver_b_base_url or os.getenv(
        "SOLVER_B_BASE_URL", "http://127.0.0.1:8001/v1"
    )
    args.solver_b_api_key = args.solver_b_api_key or os.getenv(
        "SOLVER_B_API_KEY", args.api_key
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "trajectories.jsonl"
    if trajectory_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {trajectory_path}; pass --resume")

    samples = load_samples(
        args.limit,
        args.seed,
        args.cache_dir,
        args.dataset_revision,
        stratified=args.stratified,
        dataset_name=args.dataset_name,
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
    solver_b_model_client = None
    if args.solver_b_model:
        solver_b_model_client = OpenAIChatCompletionClient(
            model=args.solver_b_model,
            api_key=args.solver_b_api_key,
            base_url=args.solver_b_base_url,
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
            http_client=httpx.AsyncClient(trust_env=False),
        )
    workflow = MathWorkflow(
        model_client,
        planner_bypass=args.planner_bypass,
        solver_diversification=args.solver_diversification,
        solver_b_planner_mask=args.solver_b_planner_mask,
        solver_b_self_review=args.solver_b_self_review,
        verifier_rederivation=args.verifier_rederivation,
        verifier_disagreement_probe=args.verifier_disagreement_probe,
        model_name=args.model,
        solver_b_model_client=solver_b_model_client,
        solver_b_model_name=args.solver_b_model,
    )
    try:
        for offset in range(0, len(remaining), args.concurrency):
            batch = remaining[offset : offset + args.concurrency]
            batch_rows = await asyncio.gather(*(workflow.run_sample(sample) for sample in batch))
            for row in batch_rows:
                append_jsonl(trajectory_path, row)
                print(
                    f"[{len(existing) + 1}/{args.limit}] {row['sample_id']} "
                    f"status={row['run_status']} correct={row['correct']} "
                    f"answer={row['final_answer']!r} gold={row['gold_answer']!r} "
                    f"tokens={row['token_usage']['total_tokens']}",
                    flush=True,
                )
                existing.append(row)
            write_summary_files(args.output_dir, existing, args)
    finally:
        await model_client.close()
        if solver_b_model_client is not None:
            await solver_b_model_client.close()

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
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/math_autogen/outputs/qwen3_8b_seed42"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/math_autogen/.cache"))
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--difficulty-min", type=float)
    parser.add_argument("--difficulty-max", type=float)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--solver-b-model")
    parser.add_argument("--solver-b-base-url")
    parser.add_argument("--solver-b-api-key")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reevaluate-only", action="store_true")
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="sample proportionally from every available subject×difficulty stratum",
    )
    parser.add_argument(
        "--planner-bypass",
        action="store_true",
        help="run Planner but do not expose its output to either Solver",
    )
    parser.add_argument(
        "--solver-diversification",
        action="store_true",
        help="change only Solver_B to an independent-remodeling and reverse-verification strategy",
    )
    parser.add_argument(
        "--solver-b-planner-mask",
        action="store_true",
        help="mask only Planner-to-Solver_B while preserving Planner-to-Solver_A",
    )
    parser.add_argument(
        "--solver-b-self-review",
        action="store_true",
        help="add a separate Solver_B review pass over its own baseline-prompt draft",
    )
    parser.add_argument(
        "--verifier-rederivation",
        action="store_true",
        help="change only Verifier to independently re-derive before auditing Solver reports",
    )
    parser.add_argument("--verifier-disagreement-probe", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if sum(
        (
            args.planner_bypass,
            args.solver_diversification,
            args.solver_b_planner_mask,
            args.solver_b_self_review,
            args.verifier_rederivation,
            args.verifier_disagreement_probe,
        )
    ) > 1:
        parser.error("active probe flags cannot be combined")
    return args


def main() -> None:
    try:
        args = parse_args()
        if args.reevaluate_only:
            load_env(args.env_file)
            args.base_url = args.base_url or os.getenv("MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
            args.model = args.model or os.getenv("MODEL_NAME", "Qwen3-8B")
            reevaluate_output_dir(args)
        else:
            asyncio.run(run(args))
    except KeyboardInterrupt:
        print("interrupted; use --resume to continue", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
