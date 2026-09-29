# EntroFlow

EntroFlow is a framework-neutral diagnostics library for multi-agent workflows.
It reconstructs workflow topology from saved trajectories, distinguishes local
failures from propagated failures using opportunity-aware node contracts, and
evaluates local workflow changes with paired outcomes and cost.

The core package does not require AutoGen or a model provider. Framework
adapters are responsible for recording topology, visible context, outputs,
provenance, and task outcomes.

## Features

- Static topology primitives for sequence, fork/join, conditional routing, and
  bounded feedback.
- Per-sample activated execution graphs with execution-instance identities.
- Opportunity-aware node and edge diagnosis without topology-specific rules.
- Local rewrite proposals and paired comparison under explicit cost budgets.
- Optional AutoGen and embedding integrations.

## Installation

```bash
python -m pip install -e .
```

Development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Optional integrations:

```bash
python -m pip install -e '.[autogen,embedding]'
```

EntroFlow requires Python 3.10 or newer.

## Quick start

Rolling communication entropy remains available for framework adapters:

```python
import numpy as np

from entroflow.entropy import RollingEdgeEntropy

monitor = RollingEdgeEntropy(window_size=32, min_samples=8)
event = monitor.observe("retriever_to_reasoner", np.asarray([0.1, 0.2, 0.3]))

if event.ready:
    print(event.entropy, event.delta)
```

Topology inspection and diagnosis use framework-neutral dictionaries and node
contracts:

```python
from entroflow.topology_optimizer import NodeSpec, diagnose, inspect_workflow

graph = inspect_workflow(saved_trajectories)
diagnosis = diagnose(
    saved_trajectories,
    specs={
        "solver": NodeSpec("answer", answer_label="Candidate"),
        "judge": NodeSpec(
            "judge",
            candidates={"solver": "verdict"},
            opportunity="all_sources_observed",
            opportunity_sources=("solver",),
        ),
    },
    answer_evaluator=evaluate_answer,
)
```

See [the topology optimizer guide](docs/topology_optimizer_plugin.md) for the
trajectory schema, diagnosis contract, and local rewrite loop.

## Repository layout

```text
src/entroflow/   Core library
tests/           Unit and integration tests
docs/            Maintained architecture and research documentation
experiments/     Reproducible experiment runners; generated outputs are ignored
```

Large datasets, checkpoints, model outputs, run logs, and generated reports are
not stored in the repository. Experiment commands create their output
directories locally.

## Development

```bash
python -m pytest -q
ruff check .
python -m build
```

Contribution guidelines are in [CONTRIBUTING.md](CONTRIBUTING.md). The project
is released under the [MIT License](LICENSE).

Chinese documentation: [README_ch.md](README_ch.md).
