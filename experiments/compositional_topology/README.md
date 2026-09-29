# Compositional topology fixtures

This directory contains the machine-readable topology and contract fixtures
used to exercise EntroFlow's activated-graph diagnosis:

- P1 sequence;
- P2 fork/join;
- P3a oracle-gated conditional selection;
- P4 bounded feedback;
- C1 fork/join followed by verification and conditional repair.

`topology_registry.json` describes static possible graphs. `configs/contracts.json`
contains reusable node contracts. `fault_injection.py` defines deterministic
fault metadata, while `evaluate.py` computes localization and propagation
metrics after ground truth is joined at evaluation time.

The fixtures do not contain benchmark data, model generations, or saved run
outputs. P3a validates routing attribution under a known route; it is not a
learned routing policy.
