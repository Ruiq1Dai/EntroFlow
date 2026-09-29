# Math and Omni-MATH experiments

This directory contains the maintained math workflow runners and deterministic
evaluators. Generated datasets, caches, trajectories, and reports are written
below ignored local output directories.

## Setup

```bash
python -m pip install -r experiments/math_autogen/requirements.txt
```

Configure an OpenAI-compatible endpoint through `.env` or command-line options:

```text
MODEL_API_KEY=dummy_key
MODEL_BASE_URL=http://127.0.0.1:8000/v1
MODEL_NAME=Qwen3-8B
```

## Competition MATH runner

```bash
python experiments/math_autogen/run.py \
  --limit 100 \
  --seed 42 \
  --concurrency 4 \
  --output-dir experiments/math_autogen/outputs/math_seed42
```

## Omni-MATH heterogeneous workflow

The seven-role workflow and its visibility contract are documented in
[the Omni-MATH guide](../../docs/omni_heterogeneous_workflow.md).

```bash
python experiments/math_autogen/omni_heterogeneous.py \
  --limit 100 \
  --seed 42 \
  --concurrency 4 \
  --output-dir experiments/math_autogen/outputs/omni_seed42
```

## Offline analysis

```bash
python experiments/math_autogen/localize_topology.py \
  --trajectories experiments/math_autogen/outputs/math_seed42/trajectories.jsonl \
  --output-dir experiments/math_autogen/outputs/math_seed42/localization
```

`topology_optimization.py` provides the local inspect–diagnose–rewrite–compare
loop described in [the optimizer guide](../../docs/topology_optimizer_plugin.md).

## Tests

```bash
python -m pytest -q \
  experiments/math_autogen/test_run.py \
  experiments/math_autogen/test_localize_topology.py \
  experiments/math_autogen/test_omni_heterogeneous.py \
  experiments/math_autogen/test_topology_optimization.py
```
