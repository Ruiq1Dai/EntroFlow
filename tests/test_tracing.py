import json

from entroflow.tracing import JsonlTraceRecorder, TraceEvent


def test_jsonl_trace_recorder(tmp_path):
    path = tmp_path / "trace.jsonl"
    recorder = JsonlTraceRecorder(path)
    recorder.append(TraceEvent("run", "example", 1, "planner", "evidence", "query"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sender"] == "planner"
    assert payload["receiver"] == "evidence"
