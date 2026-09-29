# Local topology optimizer plug-in

`entroflow.topology_optimizer` is a framework-neutral, trajectory-driven layer
for diagnosing and locally rewriting multi-agent workflow graphs. It does not
run inside task-time routing and never exposes gold answers to agents.

## Closed loop

```text
G0 -> trajectories -> diagnose -> one local rewrite -> G1
   -> same evaluation IDs -> compare cost/performance/weakness
   -> accept or rollback -> repeat (maximum three rounds)
```

The plug-in separates four responsibilities:

1. `inspect_topology_config` and `inspect_workflow` reconstruct arbitrary node
   names, edges, phases, message visibility, role contracts, shared state, and
   logging coverage from a hand-authored config or trajectories.
2. `diagnose` consumes framework-neutral node contracts plus an offline task
   evaluator. It reports opportunity-conditioned node failures, edge
   degradation/propagation/recovery, sibling common-mode failure, weakness
   scores, and representative failures.
3. `select_local_rewrite` maps the strongest observed signal to one minimal
   rewrite. It does not inspect gold or problem text.
4. `compare_runs` requires identical sample IDs and compares accuracy, weakness,
   error propagation, tokens, latency, and agent calls under explicit budgets. It also rejects
   duplicate IDs or a changed dataset name/revision/split/index/evaluator version.

The core module has no AutoGen or GPTSwarm dependency. Framework adapters only
need to serialize topology and trajectories into the documented fields. The
Omni adapter executes phases and predecessor visibility from the copied graph
configuration rather than a fixed list of agent names, so a local node/edge
rewrite changes the actual AutoGen run.

Supported evidence-to-rewrite mappings are:

| Observed trajectory signal | One local rewrite |
| --- | --- |
| high answer-worker weakness | clone one heterogeneous worker and fan out its local edges |
| correlated parallel failures or candidate conflict | sequential adversarial debate between the two peers |
| opportunity-conditioned judge failure | add one heterogeneous peer verifier |
| merge degradation with a healthy candidate | change that merge node to explicit coordinator scoring |
| propagation on a sequential edge | insert one local verifier on that edge |
| repeated branch calls / intermediate-state exchange | enable bounded shared-state mode |

Only the highest-scoring available rewrite is proposed in a round. A rewrite
type already attempted in the current loop is excluded.

## Required trajectory fields

Each sample needs a stable `sample_id`, `workflow_id`, final `correct` label,
token totals, a declared topology, and an ordered event list. Events should
include role, input, exact visible context, output, incoming edges, timestamps,
and token usage. Missing logging fields are measured rather than silently
assumed.

Node semantics are supplied separately through `NodeSpec`: answer-bearing,
contract-only, or judge; answer label/source; expected fields; and opportunity
conditions. This avoids hard-coding names such as `Solver` or `Verifier`.

## Omni-MATH adapter

`experiments/math_autogen/topology_optimization.py` applies supported graph
diffs to a copied `omni_heterogeneous` configuration and records each round
under `outputs/topology_optimization_omni/`.

Run the complete loop (up to three rounds):

```bash
PYTHONPATH=src PYTHONNOUSERSITE=1 \
  python \
  experiments/math_autogen/topology_optimization.py run-loop \
  --base-trajectories experiments/math_autogen/outputs/omni_heterogeneous_seed42/trajectories.jsonl \
  --workflow-prefix omni_heterogeneous \
  --work-dir experiments/math_autogen/outputs/topology_optimization_omni \
  --max-rounds 3 --concurrency 4
```

`run-loop` reads the frozen evaluation metadata from the baseline
`run_summary.json`, passes the current trajectories as
`--reference-trajectories`, and fails before comparison if IDs or the evaluator
contract changed. It prints and persists each round's diagnosis, topology diff,
comparison, and accept/rollback decision before preparing another round.
If an operator explicitly requests an exploratory continuation after an
early-stop condition, `--allow-after-stop` permits only the remaining rounds;
it preserves the recorded stop, current best workflow, budgets, and evaluator.

For an externally managed model runner, prepare and finalize one round
separately:

```bash
PYTHONPATH=src PYTHONNOUSERSITE=1 \
  python \
  experiments/math_autogen/topology_optimization.py prepare-round \
  --round 1 \
  --base-trajectories experiments/math_autogen/outputs/omni_heterogeneous_seed42/trajectories.jsonl \
  --candidate-workflow-id omni_heterogeneous_G1 \
  --work-dir experiments/math_autogen/outputs/topology_optimization_omni
```

Run the generated candidate with `omni_heterogeneous.py --workflow-config ...`
and `--reference-trajectories` to enforce identical evaluation IDs. Then call
`finalize-round` with both trajectory files and the recorded rewrite. It writes
diagnoses, comparison, decision, loop state, and the unified
`round_results.csv` table.

Default decision gates:

- accept only if token ratio <= 1.35 and mean latency ratio <= 1.50;
- within budget, accept an accuracy increase or weakness reduction >= 0.05;
- otherwise rollback;
- stop after two consecutive rounds without accuracy gain, no material weakness
  reduction, a budget violation, or three completed rounds.

Every attempted version is retained under `versions/G0`, `versions/G1`, ...,
including rejected candidates. Each version has an independent copied config
and a manifest pointing to its trajectories. Round directories contain the before
and after diagnoses, proposal, graph diff, comparison, and decision. The unified
`round_results.csv` is updated idempotently by round number.
