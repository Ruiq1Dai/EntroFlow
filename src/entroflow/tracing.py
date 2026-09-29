"""Append-only collaboration trace storage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .execution import ActivatedGraph


@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    example_id: str
    step: int
    sender: str
    receiver: str
    content: str
    event_type: str = "communication"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    repair_action: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


class JsonlTraceRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event: TraceEvent) -> None:
        payload = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


class ActivatedGraphRecorder:
    """Append activated graphs without changing the legacy event recorder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, graph: ActivatedGraph) -> None:
        graph.validate()
        payload = json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
