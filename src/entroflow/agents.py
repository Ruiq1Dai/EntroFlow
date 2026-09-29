"""AutoGen role agents used by the MuSiQue workflow."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from .entropy import semantic_entropy
from .messages import RoleRequest, RoleResponse

ROLE_PROMPTS = {
    "planner": (
        "You are the Planner in a multi-hop QA workflow. Decompose the original question into an ordered chain "
        "of 2-4 atomic subquestions. For every hop, state the subject entity, relation, and expected object type. "
        "Preserve exact entity, relation, temporal, and comparison constraints. Resolve referring expressions and "
        "typed descriptions before asking for a downstream attribute. When an expression has multiple plausible "
        "interpretations, keep the alternatives explicit and state which later relation or evidence would distinguish "
        "them. State how the answer to each hop becomes the subject of the next hop. Do not answer the hops."
    ),
    "evidence": (
        "You are the Evidence agent. Execute the Planner's full ordered chain over the candidate passages. Search "
        "for bridge entities across passages and for lexical or structural variants of the same relation. Use passage "
        "titles, descriptions, and sentences jointly as evidence; a title is a useful type or identity cue but is not "
        "by itself sufficient proof. Select the smallest complete evidence chain, return every numeric passage ID in "
        "chain order, and quote the supporting sentence for each hop in content. Do not reject a chain merely because "
        "no single passage states the final answer directly. If multiple candidates remain, report the competing "
        "chains and the missing discriminator. Use status=insufficient only when a required hop is unsupported."
    ),
    "reasoner": (
        "You are the Reasoner. Follow the Planner's hop order and derive the answer only from the Evidence agent's "
        "quoted chain. Explicitly connect the bridge entity between adjacent passage IDs, preserve the original "
        "relation and answer type, and distinguish direct evidence from inference. If candidates conflict, compare "
        "their full chains and reject unsupported shortcuts. Give a concise derived answer only when one chain is "
        "supported; use status=uncertain when a hop is missing or ambiguous."
    ),
    "validator": (
        "You are the Validator. Check evidence support, entity consistency, and preserved constraints. Use status=valid "
        "or status=invalid and state the minimal correction required."
    ),
    "aggregator": (
        "You are the Aggregator. Answer the original question concisely using only visible validated findings. "
        "Put only the answer span in the answer field; do not put explanation, labels, or punctuation there. "
        "Put a short explanation in content. If the evidence is insufficient, set answer to an empty string."
    ),
}

OUTPUT_INSTRUCTION = """
Return one JSON object with this schema:
{"content": "...", "answer": "...", "status": "ok", "evidence_ids": [0, 1]}
Put ordered subquestions, evidence quotations, or reasoning in content as required by your role.
Do not wrap the JSON in Markdown.
""".strip()

ROLE_MAX_TOKENS = {
    "planner": 256,
    "evidence": 768,
    "reasoner": 768,
    "validator": 512,
    "aggregator": 256,
}


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return {"content": text, "status": "unparsed", "evidence_ids": []}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"content": text, "status": "unparsed", "evidence_ids": []}
    return payload if isinstance(payload, dict) else {"content": text, "status": "unparsed"}


class WorkflowRoleAgent(RoutedAgent):
    def __init__(self, role: str, model_client: ChatCompletionClient, semantic_samples: int = 1, embedder=None):
        if role not in ROLE_PROMPTS:
            raise ValueError(f"unsupported role: {role}")
        super().__init__(f"EntroFlow {role} agent")
        self.role = role
        self.model_client = model_client
        self.semantic_samples = max(1, semantic_samples)
        self.embedder = embedder
        self.system_message = SystemMessage(content=f"{ROLE_PROMPTS[role]}\n\n{OUTPUT_INSTRUCTION}")

    @message_handler
    async def handle_request(self, message: RoleRequest, ctx: MessageContext) -> RoleResponse:
        results = []
        for _ in range(self.semantic_samples):
            result = await self.model_client.create(
                [self.system_message, UserMessage(content=message.context, source="entroflow_router")],
                extra_create_args={
                    "max_tokens": ROLE_MAX_TOKENS[self.role],
                    "temperature": 0.7 if self.semantic_samples > 1 else 0,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                }, cancellation_token=ctx.cancellation_token,
            )
            if not isinstance(result.content, str):
                raise TypeError(f"{self.role} returned tool calls; text JSON was required")
            results.append(result)
        raw_content = results[0].content
        entropy_value, cluster_count = None, 0
        if self.semantic_samples > 1 and self.embedder is not None:
            vectors = np.vstack([self.embedder(r.content) for r in results])
            entropy_value, cluster_count, labels = semantic_entropy(vectors)
            # Continue with a representative (medoid) candidate, not an arbitrary sample.
            counts = np.bincount(labels)
            chosen = int(np.argmax([counts[label] for label in labels]))
            raw_content = results[chosen].content
        if not isinstance(raw_content, str):
            raise TypeError(f"{self.role} returned tool calls; text JSON was required")
        payload = _parse_json_object(raw_content)
        content = payload.get("content", raw_content)
        status = payload.get("status", "ok")
        raw_ids = payload.get("evidence_ids", [])
        evidence_ids = tuple(int(value) for value in raw_ids if isinstance(value, (int, str)) and str(value).isdigit())
        return RoleResponse(
            run_id=message.run_id,
            example_id=message.example_id,
            role=self.role,
            content=str(content),
            answer=str(payload.get("answer", "")) if self.role == "aggregator" else "",
            status=str(status).lower(),
            evidence_ids=evidence_ids,
            step=message.step,
            prompt_tokens=sum(r.usage.prompt_tokens for r in results),
            completion_tokens=sum(r.usage.completion_tokens for r in results),
            semantic_entropy=entropy_value,
            semantic_cluster_count=cluster_count,
        )
