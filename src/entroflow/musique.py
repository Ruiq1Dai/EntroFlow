"""Strict loader for MuSiQue answerable JSONL data."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MuSiQueParagraph:
    idx: int
    title: str
    text: str
    is_supporting: bool | None = None


@dataclass(frozen=True)
class MuSiQueExample:
    example_id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    paragraphs: tuple[MuSiQueParagraph, ...]
    decomposition: tuple[dict[str, Any], ...] = ()

    def formatted_paragraphs(self) -> str:
        return "\n\n".join(
            f"[{paragraph.idx}] {paragraph.title}\n{paragraph.text}"
            for paragraph in self.paragraphs
        )


def _required_string(row: dict[str, Any], key: str, source: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: missing non-empty '{key}'")
    return value.strip()


def parse_musique_row(row: dict[str, Any], source: str = "row") -> MuSiQueExample:
    example_id = row.get("id", row.get("example_id"))
    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"{source}: missing non-empty 'id'")
    raw_paragraphs = row.get("paragraphs")
    if not isinstance(raw_paragraphs, list) or not raw_paragraphs:
        raise ValueError(f"{source}: missing non-empty 'paragraphs'")

    paragraphs = []
    for position, item in enumerate(raw_paragraphs):
        if not isinstance(item, dict):
            raise TypeError(f"{source}: paragraph {position} must be an object")
        text = item.get("paragraph_text", item.get("paragraph", item.get("text")))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source}: paragraph {position} has no text")
        raw_idx = item.get("idx", item.get("paragraph_idx", position))
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: paragraph {position} has invalid idx") from exc
        paragraphs.append(
            MuSiQueParagraph(
                idx=idx,
                title=str(item.get("title", "")),
                text=text.strip(),
                is_supporting=item.get("is_supporting"),
            )
        )

    decomposition = row.get("question_decomposition", row.get("decomposition", []))
    if decomposition is None:
        decomposition = []
    if not isinstance(decomposition, list):
        raise TypeError(f"{source}: decomposition must be a list")
    raw_aliases = row.get("answer_aliases", [])
    if raw_aliases is None:
        raw_aliases = []
    if not isinstance(raw_aliases, list):
        raise TypeError(f"{source}: answer_aliases must be a list")
    answer_aliases = tuple(value.strip() for value in raw_aliases if isinstance(value, str) and value.strip())

    return MuSiQueExample(
        example_id=example_id.strip(),
        question=_required_string(row, "question", source),
        answer=_required_string(row, "answer", source),
        answer_aliases=answer_aliases,
        paragraphs=tuple(paragraphs),
        decomposition=tuple(decomposition),
    )


def load_musique(path: str | Path, limit: int | None = None) -> list[MuSiQueExample]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    examples = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{source}:{line_number}: expected a JSON object")
            examples.append(parse_musique_row(row, f"{source}:{line_number}"))
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"{source}: no examples found")
    return examples


def iter_musique(path: str | Path) -> Iterable[MuSiQueExample]:
    yield from load_musique(path)
