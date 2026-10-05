# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`eosquality` scores Ersilia Model Hub predictions for how they compare with the same model's predictions on a fixed reference library (about 1.35M molecules). There are five calibrated scores: typicality, extremity, support, consistency and signal. The reference is the model's own behaviour, **not** ground truth. See `docs/concepts.md`.

## Setup

```bash
conda create -n eosquality python=3.12
conda activate eosquality
pip install -e ".[dev]"        # add ",viz" for the figure scripts (stylia)
```

The canonical library lives under `data/indices/ersilia_reference_library_v0/` (gitignored, via eosvc) or `~/.eosquality/` (`eosquality download`). Library resolution looks in `./data/indices/` relative to the **current working directory**. When running from elsewhere, set `EOSQUALITY_REFERENCE_LIBRARY_PATH`.

## Common Commands

```bash
ruff check src tests scripts     # lint (org config + bugbear/pyupgrade)
ruff format src tests scripts    # formatting: ruff format only, no black
pre-commit install               # runs ruff-check and ruff-format on commit
pytest -q                        # ~20 s; builds a tiny custom library from tests/fixtures/
bash scripts/run_all_scores.sh   # refit + score the 5 example models into output/ (~30 min)
conda run -n stylia python scripts/figures/<figure>.py   # writes docs/figures/
```

The package is installed in editable mode. A long `run_all_scores.sh` run imports `src/` at each CLI call, so edits made while it runs leak into later models. Pin a snapshot with `PYTHONPATH=<frozen copy>/src`.

## Architecture

The package is organized as **per-score components** (one class per score) on top of two **shared upstream layers**, plus a thin orchestrator and a few flat infrastructure modules.

### Two modalities

- **reference**: the model's predictions on the reference library. It covers the five scores below and the `shared/` + `knn/` tiers, saved under `<artifacts>/reference_mode/`.
- **training**: the model's per-output-column training sets. It covers the `training/` package (`training_sets/` on disk) and the `training_*` scores, saved under `<artifacts>/training_mode/`.

Each modality is fitted only when its data is given (`fit --reference`, `fit --training-sets`, or both). Training can also be added to existing artifacts (`fit --training-sets DIR --artifacts PATH` / `ErsiliaQuality.add_training`; the Python argument is `training_sets=`). `run` computes every fitted score.

### Per-score components — `scores/`

Each score subclasses `ScoreComponent` (`scores/_base.py`), which handles fit bookkeeping, `metadata.json` and save/load:
- `save(root)` writes `shared/`, then `knn/` (if `USES_KNN`), then the component's own subfolder.
- `save_component(root)` writes only the component's own subfolder.
- `load(root, shared=None, knn=None)` reads `shared/` and `knn/` from disk unless they are passed in.

Subclasses implement `fit`, `run`, `_save_own`, `_load_own` and `is_fitted_`. Class flags select the upstream tiers: `USES_SHARED` (default True), `USES_KNN`, `USES_TRAINING`.

- **`scores/training_distance.py`** — `TrainingDistance` (training modality, `USES_SHARED=False`, **uncalibrated**). Per output column, the distance is 1 − Tanimoto (Morgan) to the nearest training molecule; it is 0 for a query that is a training molecule (flagged `in_training`). The summary is `nanquantile(per-column, 0.66)`. `run` also returns a details table: (query, column) × distance plus the 5 nearest neighbours.

- **`scores/typicality.py`** — `Typicality`. Density-based score: per-column int8 count LUTs, then the Q66 aggregate, then the CDF. Needs only `SharedFitState`.
- **`scores/extremity.py`** — `Extremity`. Position-based score: `min(|scaled|, 1)`, then Q66, then the CDF. Needs only `SharedFitState`.
- **`scores/support.py`** — `Support`. The raw value is the Tanimoto similarity of the **nearest library analogue** (`support_raw`). It is calibrated against the library's own nearest-other-molecule similarities (`vi.self_knn_distances(1)`). Also emits `support_log = −log10(support)`. Needs `SharedFitState` + `KnnFitState`.
- **`scores/_binning.py`** — conditional (binned) CDF calibration used by Consistency: `quantile_bin_edges` (merges ties and small bins), `assign_bins`, `partition_and_sort`, `binned_cdf_score`, and edge/npz save-load helpers.
- **`scores/consistency.py`** — `Consistency`. Output-space L1 distance to the k FP neighbours, calibrated **per FP-distance bin**. The bins start as up to `N_FP_BINS = 10` quantile bins on `mean_fp_distances`. Duplicate edges are merged, and bins smaller than `min(1000, n/20)` are merged into a neighbour. `n_bins_` is the number of bins actually used.
- **`scores/signal.py`** — `Signal` + `SignalLearner`. **Provisional, opt-in** (not in `DEFAULT_SCORES`).
  - **Model.** One XGBoost regressor from a descriptor (`physchem` default, or `maccs`) to the scaled, selected outputs.
  - **Training data.** At most `max_train_samples` rows of `shared.splits.train_indices` (local subset; the shared split is never modified). Early stopping uses `EARLY_STOP_VAL_MAX = 5000` val rows.
  - **Score.** The per-query Gini of `|SHAP|`, calibrated on the full val slice. `SIGNAL_FORMULA_VERSION = "gini_v2"`.
  - **State.** The descriptor is recorded in `umbrella.json`, with no run-time override. Needs the index folder for the descriptor matrices, but not `KnnFitState`.
- **`scores/_descriptors.py`** — `PhyschemBackend` and `MaccsBackend`, which share one interface (`compute_reference_subset`, `query_matrix`, `save_state`, `from_library`, `load_state`).
  - Reference rows are gathered from the library's memory-mapped `physchem_scaled.npy` / `maccs.npy`.
  - Query rows are computed with the same `library/physchem.py` / `library/maccs.py` functions the build used.
  - Physchem checks that RDKit's descriptor list matches the library's.
- **`scores/_helpers.py`** — cross-score helpers:
  - **Calibration.** `_cdf_score` is the single CDF implementation (mid-rank, NaN passes through, `n` comes from the array). `_score_from_aggregates` is its higher-is-higher wrapper. `_sorted_finite` builds a CDF table from finite values. `_nan_aggregate` is the NaN-ignoring Q66 shared by typicality and extremity.
  - **Preprocessing.** `_make_pipeline`, `_make_query_repr`, and `_reference_repr`, which reuses `shared.ref_repr` when it is row-aligned with the reference.
  - **State resolution.** `_resolve_shared`, `_resolve_shared_and_knn`, `_resolve_vector_index` and `_custom_index_path`.
  - **Distances.** `_query_fp_distances` (top-(k+1), dropping the neighbour that is the query molecule itself) and `_query_output_distances` (chunked, NaN-aware L1).

### Shared upstream layers

- **`training/`** — the training-modality tier.
  - `data.py`: `TrainingColumn` and `load_training(folder, output_columns=None, predictions=None)`. One `<column>.csv` per output column (`smiles`, optional `y`, optional `key`). Standardisation via `scores/_helpers._standardize` (largest fragment, canonical isomeric). Duplicates are merged: binary by majority, otherwise by median. Columns with fewer than 20 molecules are skipped.
  - `state.py`: `TrainingFitState` (columns, a per-column `VectorIndex`, `eos_id`, `version`), plus `fit_training`, `save_training_state` and `load_training_state`.
  - Persisted under `<artifacts>/training_mode/training_sets/` with its own `TRAINING_FORMAT_VERSION` (independent of `ARTIFACT_FORMAT_VERSION`).

- **`shared/`** — `SharedFitState` and its `fit_shared` / `save_shared` / `load_shared` functions.
  - **Contents.** schema, eosframes scaler params, binary_class_freq, metadata, reference_ids, splits, selected_columns, and `ref_repr` (the scaled, feature-selected reference matrix, read by Consistency at run time).
  - **`metadata.py`.** Defines `FitMetadata`, which carries `library_id`, `vector_index_path` (custom indices only) and `format_version` (`ARTIFACT_FORMAT_VERSION`, currently 5). `load_shared` rejects other format versions with `ArtifactVersionError`.
  - **`splitter.py`.** Fixed 80/10/10 split with seed 0.
  - **`feature_selection.py`.** Correlation-cluster medoids, capped at `max_features`.
- **`knn/`** — `KnnFitState` holds `k`, plus the fit-time-only `mean_fp_distances` and `reference_knn_indices`. `fit_knn` slices the precomputed self-kNN from the index. Only `{"k": …}` is persisted.

### Orchestrator + flat infrastructure modules

- **`quality.py`** — `ErsiliaQuality`.
  - **`fit(..., scores=[...], vector_index=None)`.** Checks the size, checks unique keys, loads the index once and checks that its SMILES match the reference, then fits shared + knn once, then each requested component in `_SCORE_ORDER`.
  - **`run`.** Validates and scales the query once, runs the FP kNN once, and returns `RunResult(scores, metadata)`. Score columns come in `name, name_raw` pairs, plus `support_log`; metadata keys are prefixed `<component>_`.
  - **`save` / `load` / `add_training`** live in `_artifacts.py`. `save` writes `reference_mode/` (`shared/` and `knn/` once, then `save_component` for each score) and `training_mode/` (`training_sets/` plus training components), each only if fitted, plus `manifest.json`. `load` reads whichever subfolders exist and rejects the old flat layout. The fit and run of the reference modality live in `_reference_modality.py`; score names and classes are in `_registry.py`.
- **`vectorindex.py`** — `VectorIndex`, the Morgan/FPSim2 kNN index.
  - **API.** `build`, `load` (memory-mapped kNN arrays), `query`, `self_knn_indices` / `self_knn_distances`, and the properties `library_name`, `index_dir`, `smiles`, `n_reference`.
  - **Resume.** `build` resumes only when the SMILES digest and parameters match.
  - **Single-threaded queries.** FPSim2 queries run with `n_workers=1` on purpose, because the order of tied neighbours is unstable with more threads.
- **`basic_descriptors.py`** — `BasicDescriptors.build_physchem` / `build_maccs`, used by `eosquality build`.
- **`preprocess.py`** — `PreprocessPipeline`, a thin wrapper around `eosframes.fit` / `transform`.
- **`schema/`** — `Schema` / `ColumnSpec` and column inference / validation.
- **`library/`** — library identity and the descriptor builders.
  - `identity.py` resolves the canonical library locally: env override → `./data/indices/` → `~/.eosquality/`. It never touches the network.
  - `download.py` is used only by `eosquality download`.
  - `physchem.py` / `maccs.py` contain the descriptor functions shared by build and query.
- **`cli/`** — the dispatcher is `cli/__init__.py:main(argv=None)`, which sets the INFO log level. Subcommands: `build`, `download`, `fit` and `run`.
  - `fit` takes `--reference CSV`, `--training-sets DIR`, `--training-predictions CSV`, `-o NEW` or `--artifacts EXISTING`, and `--vector-index`.
  - `run --training-details PATH` (default `<output>.training_details.csv`).
- **`utils/`**
  - `logging.py`: a loguru logger bound with `extra["eosquality"]` and a filtered sink. Quiet (WARNING) as a library, INFO in the CLI, DEBUG with `set_verbosity(True)`.
  - `progress.py`: rich progress bars.
  - `parallel.py`: `map_rows`, serial below 5,000 items, otherwise a process pool.
  - `identifiers.py`: EOS id / version parsing.
- **`config.py`**, **`exceptions.py`** — `ErsiliaQualityConfig(neighbors=NeighborConfig(k))`. Exceptions: `SchemaError`, `NotFittedError`, `IncompatibleArtifactsError`, and its subclass `ArtifactVersionError`.

### Save layout

See `docs/diagram.md`: `<artifacts>/manifest.json`, `reference_mode/`, `training_mode/`. Each component's `metadata.json` carries only `component`, `fit_timestamp`, `fit_duration_seconds` and `k`. Dataset information lives once, in `reference_mode/shared/metadata.json`. `manifest.json` is informational, and the loader does not read it.

### When adding new functionality

1. Decide whether it's a **score component**, **shared upstream state**, or **infrastructure**.
2. For a new score, subclass `ScoreComponent` under `scores/<name>.py`, modelled on `Typicality` (no index) or `Support` (index-aware). Then add it to `_SCORE_ORDER` / `_SCORE_CLASSES` in `quality.py`, to the dispatch in `ErsiliaQuality.run`, and to `DEFAULT_SCORES` / `ALL_SCORES`.
3. For new shared upstream state, extend `SharedFitState` (always on, cheap) or `KnnFitState` (kNN tier).
4. Any change to what saved files mean (formula, layout, calibration) must bump `ARTIFACT_FORMAT_VERSION` in `shared/metadata.py`.
5. Add tests under `tests/`; the `library`, `reference` and `query` fixtures in `conftest.py` build a tiny custom library.

## Code Conventions

- **Formatting and linting** use ruff only (`ruff format`, `ruff check`); black is not used.
- **Dependencies** are pinned to exact versions in `pyproject.toml`; bump them deliberately.
- **CLI** is built with Click (`cli/`); commands raise `CliError` for user-facing errors, and `run_command` turns them into `error:` lines and exit status 1.
- **Size limits:** modules stay under 600 lines and functions under 80; split them before they grow past that.
- **Docstrings:** every public module, class, function and method in `src/` and `scripts/` has a NumPy-style docstring, with `Parameters` and `Returns` sections where they apply. Test functions in `tests/` are exempt from the docstring rules; pytest test names and fixtures are self-describing.

## Documentation Maintenance

User-visible docs live in two places:
- `README.md` is the landing page only: install, quick start, links to docs. Keep it compact, with no diagrams.
- `docs/` holds everything else.

Keep both current in the same pass as the code. When any user-visible change lands, update the corresponding doc alongside the implementation. Do not let docs describe removed or deprecated behavior; stale docs are worse than no docs.

Scope of "user-visible" and where it's documented:
- Output columns exposed by `RunResult.scores` and other public DataFrame/dataclass fields → `docs/api.md`.
- Python API signatures (`ErsiliaQuality.fit(...)`, `run(...)`, `save(...)`, `load(...)`) → `docs/api.md`.
- CLI subcommands, flags, and defaults → `docs/cli.md`.
- Workflow narrative (how many steps the user sees, what each produces) → `docs/cli.md` and `docs/diagram.md`.
- Concept/math explanations when the underlying formula changes → `docs/concepts.md`.
- Versioning policy, library identity, compatibility guarantees, maintainer release steps → `docs/reference-library.md`.
- Current results, figures, limitations, open questions → `docs/status.md` (regenerate `docs/figures/` with `scripts/figures/`).
- Install command or top-level pitch → `README.md`.

Prefer editing existing sections over appending a "Changelog". The docs describe the *current* state, not history; git log is authoritative for history. If context about a removed feature is worth preserving, put it in the PR description or commit message, not in the docs.

## Interaction Style

Use the `AskUserQuestion` tool extensively before and during any non-trivial task. This includes:

- Clarifying the intent or scope of a request before starting
- Confirming design choices (e.g., module names, function signatures, data formats) before implementing
- Checking assumptions about domain context (e.g., what a model input/output represents biologically)
- Verifying before deleting, refactoring, or restructuring existing code

Prefer asking over assuming, even when the request seems clear.
