# Contributing to EntroFlow

## Development setup

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check .
```

## Changes

- Keep the core package framework-neutral.
- Add tests for behavior changes and preserve backward compatibility unless a
  breaking change is explicitly documented.
- Keep task prompts and dataset adapters outside `src/entroflow`.
- Do not commit datasets, credentials, model outputs, caches, or generated
  experiment reports.
- Record random seeds, model identity, evaluator version, and sample IDs for
  reproducible experiments.

## Pull requests

Describe the user-visible behavior, tests run, and any compatibility or cost
implications. Keep unrelated refactors in separate changes.
