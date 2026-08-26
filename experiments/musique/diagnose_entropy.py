"""Diagnose whether rolling communication entropy predicts failures.

The default analysis labels every trace point by the example's final EM. For
causal hop-level AUROC, pass --hop-labels with rows containing
{"example_id": ..., "error_hops": [2, ...]}.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import matplotlib.pyplot as plt
except ImportError:  # plotting is optional; metrics must remain runnable
    plt = None

from entroflow.embeddings import LocalTransformerEmbedder
from entroflow.entropy import RollingEdgeEntropy
from entroflow.reporting import read_jsonl, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--traces", required=True)
    p.add_argument("--embedding-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--plot", required=True)
    p.add_argument("--hop-labels", help="JSONL: {example_id, error_hops: [hop indices]}")
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--min-samples", type=int, default=8)
    p.add_argument("--max-clusters", type=int, default=4)
    args = p.parse_args()

    results = {r["run_id"]: r for r in read_jsonl(args.results)}
    events = read_jsonl(args.traces)
    monitor = RollingEdgeEntropy(args.window_size, args.min_samples, args.max_clusters)
    embedder = LocalTransformerEmbedder(args.embedding_model)
    rows = []
    ordered = sorted(events, key=lambda x: (x["run_id"], x["step"], x["sender"]))
    requests = {(e["run_id"], e["step"]): e for e in ordered if e["sender"] == "entroflow_router"}
    for event in ordered:
        if event["receiver"] != "entroflow_router" or event["sender"] == "entroflow_router":
            continue
        next_request = requests.get((event["run_id"], event["step"] + 1))
        if next_request is None:
            continue
        edge = f"{event['sender']}_to_{next_request['receiver']}"
        outcome = results.get(event["run_id"], {})
        signal = monitor.observe(edge, embedder(event["content"]))
        rows.append({
            "run_id": event["run_id"], "example_id": event["example_id"],
            "step": event["step"], "edge": edge,
            "entropy": signal.entropy, "delta": signal.delta,
            "abs_delta": abs(signal.delta) if signal.delta is not None else None,
            "ready": signal.ready, "em": bool(outcome.get("em", False)),
            "hop": int(event["example_id"][0]),
        })
    ready = [r for r in rows if r["ready"] and r["delta"] is not None]
    summary = {"points": len(rows), "ready_points": len(ready), "edges": {}}
    for edge in sorted({r["edge"] for r in ready}):
        group = [r for r in ready if r["edge"] == edge]
        summary["edges"][edge] = _metrics(group, "final_em")
    hop_labels = {}
    if args.hop_labels:
        hop_labels = {r["example_id"]: set(r.get("error_hops", [])) for r in read_jsonl(args.hop_labels)}
        hop_rows = [r for r in ready if r["example_id"] in hop_labels]
        for r in hop_rows:
            r["error_hop"] = int(r["hop"]) in hop_labels[r["example_id"]]
        summary["hop_level"] = _metrics(hop_rows, "error_hop") if hop_rows else {"error": "no labeled points"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"summary": summary, "points": rows})
    _plot(rows, args.plot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _metrics(rows, label_name):
    labels = np.asarray([not bool(r["em"]) if label_name == "final_em" else bool(r[label_name]) for r in rows])
    result = {"n": len(rows), "positive": int(labels.sum())}
    for key in ("delta", "abs_delta"):
        values = np.asarray([r[key] for r in rows], dtype=float)
        result[f"{key}_mean_positive"] = float(values[labels].mean()) if labels.any() else None
        result[f"{key}_mean_negative"] = float(values[~labels].mean()) if (~labels).any() else None
        result[f"{key}_auroc"] = float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else None
    return result


def _plot(rows, path):
    if plt is None:
        print("warning: matplotlib is not installed; skipped plot generation")
        return
    ready = [r for r in rows if r["ready"] and r["delta"] is not None]
    if not ready:
        return
    by_edge = defaultdict(lambda: {True: [], False: []})
    for r in ready:
        by_edge[r["edge"]][bool(r["em"])].append((r["step"], r["delta"]))
    fig, axes = plt.subplots(len(by_edge), 1, squeeze=False, figsize=(10, 3 * len(by_edge)))
    for ax, (edge, groups) in zip(axes[:, 0], sorted(by_edge.items())):
        for em, values in groups.items():
            if values:
                x, y = zip(*values)
                ax.scatter(x, y, s=10, alpha=.35, label="correct" if em else "wrong")
        ax.axhline(0, color="black", linewidth=.5)
        ax.set_title(edge); ax.set_xlabel("trace step"); ax.set_ylabel("delta"); ax.legend()
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
