# EntroFlow

EntroFlow is a runtime plug-in for multi-agent workflows. It observes
communication traces, estimates rolling edge-level communication entropy, and
selects constrained repairs to communication links, execution order, or
validation pathways.

The research target is a fair comparison under a fixed LLM backbone, task
split, initial AutoGen workflow, and inference budget. EntroFlow is not a new
QA backbone or a replacement for AutoGen.

## Scope

The first experiment uses AutoGen with MuSiQue. The baseline workflow, random
repair controller, trace-based controller without entropy, and EntroFlow must
use the same model, prompts, maximum steps, and token budget.

EntroFlow receives structured message embeddings from the host runtime. Its
initial repair library is deliberately constrained to evidence bypass,
validation insertion, and execution reordering. Results must report accuracy,
total inference cost, intervention cost, regressions, and repair acceptance.

See [the Chinese research scope](docs/research_scope.md) for the current
protocol. Historical MA-Base/V4 code, traces, results, and documents have been
removed and must not be used as evidence for this project.

## Install

```bash
pip install -e '.[dev]'
pip install 'autogen-agentchat>=0.4' 'autogen-core>=0.4'
```

The core package only depends on NumPy and scikit-learn. AutoGen integration
is intentionally kept outside the entropy core so the plug-in can attach to a
supported AutoGen runtime without owning the execution framework.

## Core API

```python
import numpy as np
from entroflow.entropy import RollingEdgeEntropy

monitor = RollingEdgeEntropy(window_size=32, min_samples=8)
event = monitor.observe("retriever_to_reasoner", np.asarray(embedding))
if event.ready and abs(event.delta) >= 0.15:
    # The host controller may choose one allowed repair action.
    pass
```

This repository contains no benchmark data, model weights, API credentials, or
historical experiment outputs.
# EntroFlow
# EntroFlow
