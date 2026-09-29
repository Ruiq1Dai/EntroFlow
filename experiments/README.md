# Experiments

This directory contains reproducible experiment runners and task adapters.
Generated trajectories, checkpoints, reports, model caches, and datasets are
intentionally excluded from version control; see the repository `.gitignore`.

Each maintained experiment directory should contain its own usage notes and
write outputs below a local `outputs/`, `artifacts/`, or task-specific ignored
path. Experiment code must not change the public behavior of `src/entroflow`.

The core test suite uses synthetic fixtures and does not require benchmark
downloads or saved model generations.
