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

- **`src/my_package/`** — package source code using the src-layout convention. `core.py` is the main module; add new modules here.
- **`tests/`** — mirrors the package structure; test files named `test_<module>.py`.
- **`data/`** — data files for reproducibility (tracked in git, currently placeholder).
- **`pyproject.toml`** — single source of truth for package metadata, dependencies, and tool configuration.

The package uses a `src/` layout, so imports are `from my_package.<module> import ...`. When adding new functionality, create new modules under `src/my_package/` and corresponding test files under `tests/`.

## Interaction Style

Use the `AskUserQuestion` tool extensively before and during any non-trivial task. This includes:

- Clarifying the intent or scope of a request before starting
- Confirming design choices (e.g., module names, function signatures, data formats) before implementing
- Checking assumptions about domain context (e.g., what a model input/output represents biologically)
- Verifying before deleting, refactoring, or restructuring existing code

Prefer asking over assuming, even when the request seems clear.
