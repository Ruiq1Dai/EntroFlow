import json

import pytest

from entroflow.musique import load_musique, parse_musique_row
from experiments.musique.run import (
    cap_failures_by_hop,
    hop_count,
    shuffled_by_hop,
    stratified_hop_sample,
)


def valid_row():
    return {
        "id": "sample-1",
        "question": "Which entity connects A and B?",
        "answer": "Entity C",
        "answer_aliases": ["C"],
        "paragraphs": [
            {"idx": 0, "title": "A", "paragraph_text": "A is linked to C.", "is_supporting": True},
            {"idx": 1, "title": "B", "paragraph_text": "C is linked to B.", "is_supporting": True},
        ],
        "question_decomposition": [{"question": "What is linked to A?", "answer": "C"}],
    }


def test_parse_musique_row():
    example = parse_musique_row(valid_row())
    assert example.example_id == "sample-1"
    assert example.answer_aliases == ("C",)
    assert len(example.paragraphs) == 2
    assert "[0] A" in example.formatted_paragraphs()


def test_loader_reads_jsonl(tmp_path):
    path = tmp_path / "musique.jsonl"
    path.write_text(json.dumps(valid_row()) + "\n", encoding="utf-8")
    assert load_musique(path)[0].answer == "Entity C"


def test_loader_rejects_missing_passages():
    row = valid_row()
    row["paragraphs"] = []
    with pytest.raises(ValueError, match="paragraphs"):
        parse_musique_row(row)


def test_stratified_hop_sample_is_balanced_and_reproducible():
    examples = []
    for hop in (2, 3, 4):
        for index in range(10):
            row = valid_row()
            row["id"] = f"{hop}hop__{index}"
            examples.append(parse_musique_row(row))
    first = stratified_hop_sample(examples, 20, seed=42)
    second = stratified_hop_sample(examples, 20, seed=42)
    assert [row.example_id for row in first] == [row.example_id for row in second]
    counts = {
        hop: sum(row.example_id.startswith(f"{hop}hop") for row in first)
        for hop in (2, 3, 4)
    }
    assert counts == {2: 7, 3: 7, 4: 6}


def test_shuffled_by_hop_interleaves_all_hop_groups():
    examples = []
    for hop in (2, 3, 4):
        for index in range(2):
            row = valid_row()
            row["id"] = f"{hop}hop__{index}"
            examples.append(parse_musique_row(row))
    selected = shuffled_by_hop(examples, seed=42)
    assert [hop_count(row.example_id) for row in selected] == [2, 3, 4, 2, 3, 4]
    assert {row.example_id for row in selected} == {row.example_id for row in examples}


def test_cap_failures_by_hop_enforces_exact_analysis_quotas():
    failures = [
        {"example_id": f"{hop}hop__{index}"}
        for index in range(3)
        for hop in (2, 3, 4)
    ]
    selected = cap_failures_by_hop(failures, {2: 2, 3: 1, 4: 3})
    assert [hop_count(row["example_id"]) for row in selected].count(2) == 2
    assert [hop_count(row["example_id"]) for row in selected].count(3) == 1
    assert [hop_count(row["example_id"]) for row in selected].count(4) == 3
