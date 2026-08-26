"""Shared workflow messages independent of any specific AutoGen team API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoleRequest:
    run_id: str
    example_id: str
    role: str
    question: str
    context: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleResponse:
    run_id: str
    example_id: str
    role: str
    content: str
    step: int
    answer: str = ""
    status: str = "ok"
    evidence_ids: tuple[int, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    semantic_entropy: float | None = None
    semantic_cluster_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
