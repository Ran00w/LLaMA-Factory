# Repository Guidelines

## Project Structure & Module Organization
LLaMA Factory is packaged under `src/llamafactory/`, where launchers, training pipelines, and web/API glue live. Top-level entry scripts (`src/train.py`, `src/api.py`, `src/webui.py`) delegate to the package so contributions should target module extensions, not standalone scripts. Tests live in `tests/`, organized by concern (`tests/model`, `tests/train`, `tests/e2e`); `tests_v1/` preserves legacy coverage—add new cases under `tests/` unless you are fixing regressions. Reference experiment templates and LoRA configs in `examples/`, automation helpers in `scripts/`, and container/docker setups under `docker/`. Shared assets (logos, badges) and sample datasets reside in `assets/` and `data/`.

## Build, Test, and Development Commands
- `pip install -e ".[torch,metrics]" --no-build-isolation` sets up a local dev editable install; add extras like `.[vllm]` only when needed.
- `make style` applies `ruff` checks and formatting to `scripts`, `src`, `tests`, and `tests_v1`.
- `make test` runs `pytest -vv tests/` with `CUDA_VISIBLE_DEVICES=` to keep suites CPU-friendly; export GPUs explicitly when required.
- `make commit` installs and executes the pre-commit hooks; use it before pushing.
- `llamafactory-cli train --help` (alias `lmf`) surfaces the supported task/adapter arguments for new experiments.

## Coding Style & Naming Conventions
Python files use 4-space indentation, a 119-character soft limit, and double quotes by default (see `pyproject.toml`). Keep modules `snake_case`, classes `PascalCase`, and constants upper-snake; prefer descriptive names that align with existing adapters (e.g., `*_trainer.py`). `ruff` enforces import ordering, google-style docstrings, and progressive type upgrades—run `make style` after edits and avoid introducing `black` formatting.

## Testing Guidelines
Add unit coverage alongside the code under test, following the folder pattern and `test_*.py` naming. Use `pytest` markers sparingly; skip GPU-heavy flows unless the test guards them via environment checks. Favor deterministic fixtures and reuse helpers under `tests/data` to avoid duplicating large corpora. Include integration smoke tests in `tests/e2e` when touching launchers or CLI behaviors.

## Commit & Pull Request Guidelines
Recent commits use a `[scope] summary (#xxxx)` prefix; match that scope taxonomy (`[model]`, `[misc]`, `[deps]`, etc.) and mention the PR or issue ticket. Keep commits focused and runnable, and note any follow-ups in the body. Pull requests should link related issues, describe user-visible changes, and attach CLI logs or screenshots for UI impacts. Confirm `make style` and `make test` pass and call this out in the PR checklist.
