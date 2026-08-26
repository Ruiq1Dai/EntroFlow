import json

from entroflow.reporting import (
    failed_trajectories,
    summarize_results,
    write_json,
    write_jsonl,
)


def test_summary_keeps_accuracy_and_f1_separate():
    summary = summarize_results(
        [
            {"em": True, "f1": 1.0, "completed": True, "total_tokens": 10},
            {"em": False, "f1": 0.5, "completed": True, "total_tokens": 20},
        ]
    )
    assert summary["accuracy_percent"] == 50.0
    assert summary["macro_f1_percent"] == 75.0
    assert summary["mean_tokens"] == 15.0


def test_failed_trajectories_join_by_run_id_and_preserve_only_failures():
    results = [
        {"run_id": "ok", "example_id": "1", "em": True},
        {
            "run_id": "bad",
            "example_id": "2",
            "question": "q",
            "prediction": "p",
            "gold_answer": "g",
            "em": False,
            "f1": 0.0,
            "completed": True,
        },
    ]
    traces = [
        {"run_id": "bad", "step": 2, "sender": "evidence", "content": "e"},
        {"run_id": "bad", "step": 1, "sender": "planner", "content": "p"},
        {"run_id": "ok", "step": 1, "sender": "planner", "content": "ignored"},
    ]
    failures = failed_trajectories(results, traces)
    assert len(failures) == 1
    assert failures[0]["example_id"] == "2"
    assert [event["step"] for event in failures[0]["events"]] == [1, 2]


def test_atomic_report_writers(tmp_path):
    summary_path = tmp_path / "summary.json"
    failures_path = tmp_path / "failures.jsonl"
    write_json(summary_path, {"accuracy": 0.159})
    write_jsonl(failures_path, [{"example_id": "failed"}])
    assert json.loads(summary_path.read_text())["accuracy"] == 0.159
    assert json.loads(failures_path.read_text())["example_id"] == "failed"
