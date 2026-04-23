# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Ersilia Python package template for developing and distributing AI/ML tools, primarily for antimicrobial drug discovery research. The template provides the scaffold for Ersilia ecosystem packages.

## Setup

Create a Conda environment and install in editable mode with dev dependencies:

```bash
conda create -n my_env python=3.12
conda activate my_env
pip install -e ".[dev]"
```

## Common Commands

```bash
# Run tests
pytest

# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

## Architecture

- **`src/eosquality/`** — package source code using the src-layout convention. Subpackages: `preprocess/` (kind-aware normalization + quantile grids), `scoring/` (support, typicality, run-time scoring), `reference/` (fit state, metadata, diagnostics), `vectorindex/` (Morgan FP kNN backend), `io/` (save/load), `schema/` (column inference), `quality/` (the public `ErsiliaQuality` API), `utils/` (stats, logging, identifiers).
- **`tests/`** — mirrors the package structure; test files named `test_<module>.py`.
- **`data/`** — data files for reproducibility (tracked in git). `data/indices/reference_library/` is the canonical shipped vector index.
- **`pyproject.toml`** — single source of truth for package metadata, dependencies, and tool configuration.

Imports are `from eosquality.<module> import ...`. When adding new functionality, create modules under `src/eosquality/` and mirror test files under `tests/`.

## Documentation Maintenance

Keep `README.md` current in the same pass as the code. When any user-visible change lands — new or renamed score columns, removed features, CLI flag changes, new workflows, modified defaults, versioning rules — update README alongside the implementation. Do not let the README describe removed or deprecated behavior; a stale README is worse than no README.

Scope of "user-visible" for this repo:
- The output columns exposed by `RunResult.scores` and any other public DataFrame/dataclass fields.
- Python API signatures shown in README examples (`ErsiliaQuality.fit(...)`, `run(...)`, `save(...)`, `load(...)`).
- CLI subcommands, flags, and their defaults.
- The workflow narrative (how many steps the user sees, what each produces).
- Concept/math explanations when the underlying formula changes.
- Versioning policy, library identity, and compatibility guarantees.

Prefer editing existing README sections over appending a "Changelog" — the README describes *current* state, not history (git log is authoritative for history). If a removed feature is worth preserving context for, call it out in the relevant PR description or commit message, not in README.

## Interaction Style

Use the `AskUserQuestion` tool extensively before and during any non-trivial task. This includes:

- Clarifying the intent or scope of a request before starting
- Confirming design choices (e.g., module names, function signatures, data formats) before implementing
- Checking assumptions about domain context (e.g., what a model input/output represents biologically)
- Verifying before deleting, refactoring, or restructuring existing code

Prefer asking over assuming, even when the request seems clear.
