"""NLI-based graded entropy sanity check on the shared 180 synthetic samples.

Each group contains four paraphrases of the same answer plus one graded
anomaly.  Two statements are in the same semantic cluster when entailment is
supported in both directions.  Neutral is intentionally *not* treated as
equivalence: this preserves distinctions common in multi-hop QA.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LEVELS = ("off_topic", "wrong_entity", "wrong_relation", "missing_hop", "reasoning_error", "semantic_equivalent")


def group(level: str, i: int) -> list[str]:
    subject, bridge, target = f"Person {i}", f"City {i}", f"Institution {i}"
    base = [
        f"{subject} was born in {bridge}, and {bridge} is home to {target}.",
        f"The evidence links {subject} to birthplace {bridge}; {target} is located there.",
        f"According to the passages, {subject}'s birthplace is {bridge}, which contains {target}.",
        f"The chain is: {subject} -> born in -> {bridge} -> contains -> {target}.",
    ]
    anomalies = {
        "off_topic": "The Pacific Ocean is the largest ocean on Earth in area.",
        "wrong_entity": f"Person {i+1000} was born in {bridge}, and {bridge} is home to {target}.",
        "wrong_relation": f"{subject} works in {bridge}, and {bridge} is home to {target}.",
        "missing_hop": f"The passages mention {subject} and {target}, but do not establish the intermediate birthplace link.",
        "reasoning_error": f"{subject} was born in {bridge}, therefore the answer is City {i+1}, not the institution located there.",
        "semantic_equivalent": f"The source confirms that {bridge} is the birthplace of {subject} and that {target} is there.",
    }
    return base + [anomalies[level]]


def entropy(labels: list[int]) -> float:
    counts = Counter(labels)
    n = len(labels)
    return float(-sum((c / n) * math.log(c / n) for c in counts.values()))


def components(adj: list[list[bool]]) -> list[int]:
    n, out, seen = len(adj), [-1] * len(adj), set()
    for start in range(n):
        if start in seen:
            continue
        stack, seen = [start], seen | {start}
        while stack:
            node = stack.pop()
            out[node] = start
            for nxt in range(n):
                if nxt not in seen and adj[node][nxt]:
                    seen.add(nxt); stack.append(nxt)
    remap = {root: idx for idx, root in enumerate(sorted(set(out)))}
    return [remap[x] for x in out]


class NLI:
    def __init__(self, model_name: str, batch_size: int, device: str | None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        # Hugging Face stores id2label as {class_index: label}; preserve that
        # direction when resolving the class indices (older code inverted it).
        self.labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next((k for k, v in self.labels.items() if "entail" in v), 1)
        self.contra_idx = next((k for k, v in self.labels.items() if "contrad" in v), 0)

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        result = []
        with torch.inference_mode():
            for pos in range(0, len(pairs), self.batch_size):
                a, b = zip(*pairs[pos:pos + self.batch_size])
                toks = self.tokenizer(list(a), list(b), padding=True, truncation=True, return_tensors="pt").to(self.device)
                probs = torch.softmax(self.model(**toks).logits, dim=-1).cpu().numpy()
                for row in probs:
                    result.append({"entailment": float(row[self.entail_idx]), "contradiction": float(row[self.contra_idx]), "neutral": float(max(0.0, 1 - row[self.entail_idx] - row[self.contra_idx]))})
        return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/deberta-v3-large-mnli")
    p.add_argument("--output", required=True)
    p.add_argument("--groups", type=int, default=30)
    p.add_argument("--entail-threshold", type=float, default=0.70)
    p.add_argument("--contradiction-threshold", type=float, default=0.50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    model = NLI(args.model, args.batch_size, args.device)
    rows = []
    for level in LEVELS:
        for i in range(args.groups):
            texts = group(level, i); pairs = [(a, b) for a in texts for b in texts if a != b]
            scores = model.score(pairs); by_pair = {(a, b): s for (a, b), s in zip(pairs, scores)}
            adj = [[False] * len(texts) for _ in texts]
            for x in range(len(texts)):
                for y in range(len(texts)):
                    if x != y:
                        ab, ba = by_pair[(texts[x], texts[y])], by_pair[(texts[y], texts[x])]
                        adj[x][y] = ab["entailment"] >= args.entail_threshold and ba["entailment"] >= args.entail_threshold
            labels = components(adj)
            rows.append({"level": level, "group": i, "entropy": entropy(labels), "clusters": len(set(labels)), "labels": labels, "target_scores": [by_pair[(texts[4], texts[j])] for j in range(4)]})
    summary = {level: {"mean": statistics.mean(r["entropy"] for r in rows if r["level"] == level), "median": statistics.median(r["entropy"] for r in rows if r["level"] == level), "mean_clusters": statistics.mean(r["clusters"] for r in rows if r["level"] == level)} for level in LEVELS}
    payload = {"model": args.model, "rules": {"bidirectional_entailment": True, "entailment_threshold": args.entail_threshold, "contradiction_threshold_logged": args.contradiction_threshold, "neutral_is_equivalence": False}, "levels": list(LEVELS), "summary": summary, "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
