"""AutoGen Core workflow controlled by an EntroFlow routing policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from autogen_core import AgentId, SingleThreadedAgentRuntime
from autogen_core.models import ChatCompletionClient

from .agents import WorkflowRoleAgent
from .messages import RoleRequest, RoleResponse
from .musique import MuSiQueExample
from .repairs import RepairAction
from .routing import ROLES, RoutingDecision, RoutingPolicy, WorkflowState
from .tracing import JsonlTraceRecorder, TraceEvent


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    example_id: str
    prediction: str
    total_tokens: int
    turns: int
    repairs: tuple[str, ...]
    completed: bool


class AutoGenMuSiQueWorkflow:
    def __init__(
        self,
        model_client: ChatCompletionClient,
        policy: RoutingPolicy,
        trace_path: str | Path,
        max_turns: int = 8,
        max_tokens: int = 20000,
        semantic_samples: int = 1,
        semantic_embedder=None,
    ):
        self.model_client = model_client
        self.policy = policy
        self.recorder = JsonlTraceRecorder(trace_path)
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.semantic_samples = semantic_samples
        self.semantic_embedder = semantic_embedder

    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        for role in ROLES:
            await WorkflowRoleAgent.register(
                runtime,
                role,
                lambda role=role: WorkflowRoleAgent(role, self.model_client, self.semantic_samples, self.semantic_embedder),
            )

    @staticmethod
    def _visible_history(state: WorkflowState, decision: RoutingDecision) -> str:
        if not state.history:
            return "No previous collaboration messages."
        visible = set(decision.visible_roles)
        rows = [row for row in state.history if not visible or row.role in visible]
        return "\n\n".join(
            f"{row.role.upper()} [status={row.status}, evidence_ids={list(row.evidence_ids)}]:\n{row.content}"
            for row in rows
        )

    def _build_context(
        self,
        example: MuSiQueExample,
        state: WorkflowState,
        decision: RoutingDecision,
    ) -> str:
        role = decision.next_role
        parts = [f"Original question:\n{example.question}"]
        if role == "evidence":
            parts.append(f"Candidate passages:\n{example.formatted_paragraphs()}")
        parts.append(f"Visible collaboration history:\n{self._visible_history(state, decision)}")
        if decision.action != RepairAction.NONE:
            parts.append(
                "Runtime repair recommendation:\n"
                f"- action: {decision.action.value}\n"
                f"- diagnosis: {decision.diagnosis or 'anomaly detected'}\n"
                f"- target: {decision.repair_target or 'workflow state'}\n"
                f"- instruction: {decision.repair_instruction or 'inspect the preceding trace'}\n"
                f"- confidence: {decision.confidence if decision.confidence is not None else 'unknown'}"
            )
        return "\n\n".join(parts)

    async def run(self, example: MuSiQueExample) -> WorkflowResult:
        run_id = str(uuid4())
        state = WorkflowState(run_id=run_id, example_id=example.example_id)
        repairs = []
        total_tokens = 0
        runtime = SingleThreadedAgentRuntime()
        await self._register_agents(runtime)
        runtime.start()
        try:
            for step in range(1, self.max_turns + 1):
                decision = self.policy.select(state)
                if decision.next_role is None or total_tokens >= self.max_tokens:
                    break
                if decision.action == RepairAction.RETRY_WITH_CONSTRAINTS:
                    state.evidence_retries += 1
                if decision.action != RepairAction.NONE:
                    repairs.append(decision.action.value)
                entropy_metadata = (
                    asdict(decision.entropy_event) if decision.entropy_event is not None else {}
                )
                request = RoleRequest(
                    run_id=run_id,
                    example_id=example.example_id,
                    role=decision.next_role,
                    question=example.question,
                    context=self._build_context(example, state, decision),
                    step=step,
                    metadata={
                        "repair_action": decision.action.value,
                        "entropy": entropy_metadata,
                        "diagnosis": decision.diagnosis,
                        "repair_target": decision.repair_target,
                        "repair_instruction": decision.repair_instruction,
                        "confidence": decision.confidence,
                    },
                )
                self.recorder.append(
                    TraceEvent(
                        run_id=run_id,
                        example_id=example.example_id,
                        step=step,
                        sender="entroflow_router",
                        receiver=decision.next_role,
                        content=request.context,
                        repair_action=decision.action.value,
                    metadata={
                        "entropy": entropy_metadata,
                        "diagnosis": decision.diagnosis,
                        "repair_target": decision.repair_target,
                        "repair_instruction": decision.repair_instruction,
                        "confidence": decision.confidence,
                    },
                )
                )
                response = await runtime.send_message(
                    request,
                    AgentId(decision.next_role, example.example_id),
                )
                if not isinstance(response, RoleResponse):
                    raise TypeError(f"unexpected response type: {type(response).__name__}")
                state.history.append(response)
                total_tokens += response.total_tokens
                self.recorder.append(
                    TraceEvent(
                        run_id=run_id,
                        example_id=example.example_id,
                        step=step,
                        sender=response.role,
                        receiver="entroflow_router",
                        content=response.content,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        repair_action=decision.action.value,
                        metadata={
                            "status": response.status,
                            "evidence_ids": list(response.evidence_ids),
                            "answer": response.answer,
                            "semantic_entropy": response.semantic_entropy,
                            "semantic_cluster_count": response.semantic_cluster_count,
                        },
                    )
                )
        finally:
            await runtime.stop_when_idle()

        final = state.latest("aggregator")
        return WorkflowResult(
            run_id=run_id,
            example_id=example.example_id,
            prediction=final.answer if final else "",
            total_tokens=total_tokens,
            turns=len(state.history),
            repairs=tuple(repairs),
            completed=final is not None,
        )
