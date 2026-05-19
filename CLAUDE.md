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
# Format code
black src/

# Lint
flake8 src/
```

## Architecture

The package is organized as **per-score components** (one class per score) sitting on top of two **shared upstream layers** (one always present, one only when an index-aware score is fit), plus a thin orchestrator and a few flat infrastructure modules.

### Per-score components — `scores/`

Each score is its own class with `.fit()` / `.run()` / `.save()` / `.load()`. They can be used standalone or composed by `ErsiliaQuality`. Save layout is per-component subfolders under one root.

- **`scores/typicality.py`** — `Typicality`. Density-based per-feature typicality from int8-quantized count LUTs. **Needs only `SharedFitState`; never touches the vector index.**
- **`scores/extremity.py`** — `Extremity`. Position-based per-feature extremity from eosframes-scaled values. Also needs only `SharedFitState`.
- **`scores/support.py`** — `Support`. CDF-based support score using FP-selected kNN Tanimoto distances. Needs `SharedFitState` + `KnnFitState`.
- **`scores/consistency.py`** — `Consistency`. `exp(−mean output-space L1 distance)` over FP-selected neighbors. Same dependencies as Support; shares the same `KnnFitState`.
- **`scores/_helpers.py`** — private module hosting cross-score helpers (`_make_pipeline`, `_resolve_shared_and_knn`, `_resolve_vector_index`, `_query_fp_distances`, `_query_output_distances`, `_component_metadata`). Score classes import from here so no single score "owns" a shared helper.

### Shared upstream layers

- **`shared/`** — `SharedFitState` (schema, eosframes scaler params, binary_class_freq, metadata, reference_ids) + `fit_shared` / `save_shared` / `load_shared` + the `metadata.py` module that defines `FitMetadata` / `compute_metadata`. Persisted under `<root>/shared/`. Required by every score.
- **`knn/`** — `KnnFitState` (ref_repr + k; with fit-time-only `mean_fp_distances` + `reference_knn_indices`) + `fit_knn` / `save_knn` / `load_knn`. Persisted under `<root>/knn/`. `fit_knn` only reads the precomputed FP self-kNN from the vector index — no neighbor search and no output-space arithmetic. Consistency then computes its own output-space means inside `Consistency.fit`, so a Support-only fit pays nothing for output-space distances. The vector-index path is **not** persisted; at run time it is resolved by `shared.metadata.library_id` via `library.identity.reference_library_path()` so saved artifacts are portable.

### Orchestrator + flat infrastructure modules

- **`quality.py`** — `ErsiliaQuality`. Thin orchestrator: fits shared + knn once, then each requested score on top. `fit(..., scores=[...])` selects components — pass `scores=["typicality"]` to skip the vector index entirely. `.save()` writes a top-level `manifest.json` + `shared/` + (optionally) `knn/` + each score's subfolder; `.load()` discovers which subfolders are present and reconstructs accordingly.
- **`vectorindex.py`** — flat module. Morgan FP kNN backend (`.build` for the library, `.load`/`.query`/`.self_knn_indices` for `knn/` fit and run).
- **`preprocess.py`** — flat module. `PreprocessPipeline`, a thin wrapper around `eosframes.fit`/`transform`. Also computes `binary_class_freq` at fit time so the persisted state stays self-contained.
- **`schema/`** — `Schema`/`ColumnSpec` dataclasses and column inference / validation.
- **`library/`** — *maintenance-only, model-independent.* `identity.py` resolves the canonical library by name (env override → repo `data/indices/` → `~/.eosquality/` cache → S3); `download.py` does the S3 fetch. The canonical index ships via `data/indices/` and S3.
- **`cli/`** — unified CLI package. `cli/__init__.py` is the dispatcher (entry point `eosquality.cli:main`). One subcommand handler per module: `cli/build.py` (`index`), `cli/download.py` (`download`), `cli/fit.py` (`fit`), `cli/run.py` (`run`). Each exports a `register_subparsers` function.
- **`config.py`**, **`exceptions.py`**, **`utils/`** — config dataclasses, custom exceptions, and helpers (logging, stats, identifiers, arrays).

### Save layout

```
<root>/
  manifest.json              # informational summary written by ErsiliaQuality.save
  shared/                    # always present
    schema.json
    scaler.json
    binary_class_freq.json
    metadata.json
    reference_ids.json       # JSON list (was joblib)
  knn/                       # iff any of support / consistency was fit
    state.json               # {"k": int} — index path resolved by library_id
    reference_repr.npy
  typicality/                # iff typicality was fit
  extremity/                 # iff extremity was fit
  support/
  consistency/
```

Each score class' `save(root)` writes `<root>/shared/` + (if relevant) `<root>/knn/` + its own subfolder. Each `load(root)` reads the same. `ErsiliaQuality.load(root)` discovers all present subfolders and reconstructs whichever components are present. The top-level `manifest.json` is informational only — the loader does not consult it.

Per-component `metadata.json` files carry only component-specific bookkeeping (`component`, `fit_timestamp`, `fit_duration_seconds`, `k`); shared dataset information (`n_samples`, `n_features`, `eosquality_version`, `library_id`) lives once in `shared/metadata.json`.

### When adding new functionality

1. Decide whether it's a **score component**, **shared upstream state**, or **infrastructure**.
2. New score → add a class under `scores/<name>.py` modeled on `Typicality` (no VI) or `Support` (VI-aware), and wire it into `ErsiliaQuality.fit` / `run` / `save` / `load` + the `DEFAULT_SCORES` tuple.
3. New shared upstream → extend `SharedFitState` (always-on, cheap) or `KnnFitState` (kNN-tier).

## Documentation Maintenance

User-visible docs live in two places: `README.md` (landing page only — install, quick start, links to docs) and `docs/` (everything else). Keep both current in the same pass as the code. When any user-visible change lands, update the corresponding doc alongside the implementation. Do not let docs describe removed or deprecated behavior; stale docs are worse than no docs.

Scope of "user-visible" and where it's documented:
- Output columns exposed by `RunResult.scores` and other public DataFrame/dataclass fields → `docs/api.md`.
- Python API signatures (`ErsiliaQuality.fit(...)`, `run(...)`, `save(...)`, `load(...)`) → `docs/api.md`.
- CLI subcommands, flags, and defaults → `docs/cli.md`.
- Workflow narrative (how many steps the user sees, what each produces) → `docs/cli.md` and `docs/diagram.md`.
- Concept/math explanations when the underlying formula changes → `docs/concepts.md`.
- Versioning policy, library identity, compatibility guarantees, maintainer release steps → `docs/reference-library.md`.
- Install command or top-level pitch → `README.md`.

Prefer editing existing sections over appending a "Changelog" — the docs describe *current* state, not history (git log is authoritative for history). If a removed feature is worth preserving context for, call it out in the relevant PR description or commit message, not in the docs.

## Interaction Style

Use the `AskUserQuestion` tool extensively before and during any non-trivial task. This includes:

- Clarifying the intent or scope of a request before starting
- Confirming design choices (e.g., module names, function signatures, data formats) before implementing
- Checking assumptions about domain context (e.g., what a model input/output represents biologically)
- Verifying before deleting, refactoring, or restructuring existing code

Prefer asking over assuming, even when the request seems clear.
