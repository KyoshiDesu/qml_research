# Repository Guidelines

## Project Structure & Module Organization

Core package code lives in `src/qsvm_vm_compare/`. Keep configuration, data preparation, classical modeling, quantum modeling, metrics, and orchestration in their existing focused modules. `scripts/run_experiment.py` is the package-based CLI; `run_one_go_svm_qsvm_experiment.py` and `qsvm_single_file.py` are standalone experiment workflows. Default runtime settings belong in `configs/default_experiment.json`. Research notes and reports are under `docs/` and `infra/`.

Treat `results/`, `outputs/`, `output_gpu/`, and `combined_outputs/` as generated artifacts. Do not manually edit reports, metrics, plots, or serialized models there.

## Build, Test, and Development Commands

Use Python 3.11 or newer. The Conda environment targets Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .
python scripts/run_experiment.py --config configs/default_experiment.json
python -m compileall src scripts
```

The editable install exposes `qsvm_vm_compare` during development. The experiment command writes timestamped JSON into `results/`; quantum simulation can be slow. `compileall` is the current lightweight syntax check. For the broader standalone workflow, install `requirements.txt` and run `python run_one_go_svm_qsvm_experiment.py --dataset_name iris --output_dir outputs`.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation, type hints, and `from __future__ import annotations` in package modules. Use `snake_case` for functions, variables, modules, CLI flags, and JSON keys; use `PascalCase` for classes and `UPPER_CASE` for constants. Prefer small, single-purpose functions and `pathlib.Path` over string path manipulation. No formatter or linter is configured; keep imports grouped as standard library, third-party, then local.

## Testing Guidelines

There is no committed automated test suite or coverage threshold. New logic should add `pytest` tests under `tests/`, named `test_<module>.py`, with functions named `test_<behavior>()`. Keep tests deterministic by setting `random_state`, use small sample limits, and avoid requiring a GPU. Before submitting, run syntax checks and at least one small built-in-dataset smoke experiment.

## Commit & Pull Request Guidelines

History mixes short imperative messages with occasional Conventional Commit prefixes. Prefer focused messages such as `feat: add backend metadata` or `fix: validate qubit count`. Pull requests should explain the research or code change, list commands run, identify configuration changes, and link relevant issues. Include representative metric/report paths for experiment changes and screenshots only when plots or rendered documentation change. Avoid committing large regenerated output trees unless they are intentional deliverables.
