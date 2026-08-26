# EntroFlow Research Scope

## Question

Under the same LLM backbone, MuSiQue split, initial AutoGen workflow, and
inference budget, can a runtime communication-entropy plug-in identify a
**first recoverable failure point** and select a repair recommendation that
rescues the answer at lower cost than matched trace-only or random actions?

This is deliberately weaker than claiming that entropy identifies the true
causal error. A first recoverable failure point is an independently audited
trace location where the prescribed repair can change the downstream answer.
The audit must be completed without reading the entropy score or selected
action.

The current method hypothesis is a two-stage detector: entropy/change signals
identify unstable communication, while an explicit handoff-contract check
localizes which constraint or bridge relation was lost. The repair targets the
lost contract (evidence expansion or validation) rather than applying a generic
repair to the highest-entropy role.

## System boundary

AutoGen is the execution substrate. It owns agents, messages, tools, and the
initial workflow. EntroFlow observes structured collaboration traces and does
not replace the backbone or generate arbitrary workflow code.

The initial agent roles are Planner, Evidence, Reasoner, Validator, and
Aggregator. All roles use the same backbone. Messages must include sender,
receiver, step, content, token count, and an embedding produced by a fixed
encoder.

## Runtime policy

For each directed communication edge, EntroFlow maintains a bounded rolling
window of message embeddings and computes normalized entropy plus its change.
An anomaly may trigger at most one allowed action per example:

- add an evidence bypass to Validator or Aggregator;
- run validation before the next reasoning step;
- reorder an unresolved validation step before aggregation;
- retry one evidence request with preserved constraints.

The policy must log the entropy event, its diagnosis hypothesis, target,
repair instruction, confidence, selected action, added token cost, and outcome.
It must not use gold answers at runtime. A recommendation is considered
implemented only when the instruction changes the receiving agent's context;
an action label in a trace is not sufficient.

## Evaluation

Compare Static AutoGen, Random Repair, Trace-based Repair without entropy, and
EntroFlow. Hold model, prompts, sample split, maximum steps, action library,
and token/call budget fixed. Report EM, F1, total tokens, latency, intervention
cost, first successful repair budget, acceptance rate, regression rate, rescue
rate on baseline failures, and per-failure-stratum outcomes. Every repair
comparison must be paired by `example_id`; overall accuracy alone is not
evidence of repair.

MuSiQue traces are retained in full. Retrieval dead ends are reported as a
separate stratum; they are not silently removed or claimed as entropy failures.

## Falsification

The plug-in hypothesis fails if EntroFlow does not improve paired net rescue
(rescues minus regressions) over Random Repair and the trace-based controller
on held-out samples under the same budget, or if an apparent gain disappears
after intervention cost is counted. Localization is reported as a diagnostic
metric against a blinded audit, not as proof of causality.
