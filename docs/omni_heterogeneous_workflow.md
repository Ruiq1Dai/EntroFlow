# Omni-MATH heterogeneous workflow

## Scope and decision log

- **Purpose:** generate trajectories whose agents have distinct functional
  contracts, so later experiments can localize different node, edge, and local
  subgraph weaknesses across datasets and topologies.
- **Decision:** add a separate `omni_heterogeneous_v1` workflow. The historical
  five-node MATH workflow remains unchanged and is the topology control.
- **Assumption:** role, visibility, and reasoning-protocol heterogeneity can
  create distinguishable component behavior even when one base model serves all
  roles. This is an experimental design assumption, not a validated finding.
- **Prediction:** failures can appear at analysis/decomposition, either solver
  path, comparison, repair, or formatting stages instead of collapsing into a
  single symmetric dual-Solver region.
- **Disconfirming observation:** if roles routinely violate their contracts or
  both solver traces remain semantically identical, prompt-level functional
  heterogeneity is insufficient and a model/tool heterogeneous branch is needed.

No weak-point score or gold-conditioned routing is implemented here. Gold is
used only after the complete workflow finishes to evaluate the final answer.

## Topology

```text
ProblemAnalyzer
      |
  Decomposer
    /     \
DerivationSolver   IndependentSolver
    \     /
     Critic
    /  |  \
   solver traces
       |
    Refiner
       |
   Finalizer
```

The precise machine-readable edges are stored in every trajectory. Both
solvers receive the original problem and Decomposer output, run concurrently,
and cannot see one another. DerivationSolver must execute the decomposition;
IndependentSolver must use it only as a target/constraint checklist and choose
a materially different route.

Critic sees both solver reports but is prohibited from producing a replacement
answer. Refiner sees both reports and the critique and must edit existing work.
Finalizer sees only the original problem and Refiner output and is prohibited
from mathematical reasoning.

## Run

```bash
PYTHONNOUSERSITE=1 HF_DATASETS_OFFLINE=1 \
  python -u \
  experiments/math_autogen/omni_heterogeneous.py \
  --limit 100 \
  --seed 42 \
  --concurrency 4 \
  --stratified \
  --difficulty-min 1.5 \
  --difficulty-max 3.5 \
  --output-dir experiments/math_autogen/outputs/omni_heterogeneous_seed42
```

Use `--resume` after interruption. The script refuses to overwrite an existing
`trajectories.jsonl` without it.

## Outputs

- `trajectories.jsonl`: one self-contained row per sample.
- `summary.csv`: final-answer evaluation and token totals per sample.
- `run_summary.json`: run configuration, fixed topology, role contracts, totals,
  and output paths.

Each trajectory row includes `workflow_id`, `run_id`, `sample_id`, dataset
metadata, problem, topology nodes/edges/phases, role contracts,
`realized_agent_edges`, ordered agent events, final evaluation, status, and
token usage. Each event includes role, role contract, senders, receiver,
incoming edges, exact visible system/user context, flattened input, output,
timestamps, execution phase/order, model, and token usage.

## Contract tests

```bash
python -m unittest \
  experiments/math_autogen/test_omni_heterogeneous.py
```
