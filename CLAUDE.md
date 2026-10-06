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

The canonical library lives under `data/indices/ersilia_reference_library_v0/` (gitignored, via eosvc) or `~/.eosquality/` (`eosquality setup`). Library resolution looks in `./data/indices/` relative to the **current working directory**. When running from elsewhere, set `EOSQUALITY_REFERENCE_LIBRARY_PATH`.

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
- **training**: the model's per-output-column training sets. It covers the `training/` package (`training_sets/` on disk) and the `training_*` components, saved under `<artifacts>/training_mode/`.

**Names.** Components have short internal names (`support`, `training_distance`: attributes, classes, artifacts subfolders). Their public **score names** are prefixed by modality, `ref_` or `trn_` (`ref_support`, `trn_distance`); `_registry.SCORE_NAMES` maps one to the other. Public names are used for the output columns, the run metadata keys (`<score>_<key>`), `exclude=` / `--exclude` and `ALL_SCORES`.

Each modality is fitted only when its data is given (`fit -r`, `fit -t`, or both), and every score of it unless excluded. Training can also be added to existing reference artifacts (`fit -t DIR -a EXISTING` / `ErsiliaQuality.add_training`; the Python argument is `training_sets=`). `run` computes every fitted score.

### Per-score components — `scores/`

Each score subclasses `ScoreComponent` (`scores/_base.py`), which handles fit bookkeeping, `metadata.json` and save/load:
- `save(root)` writes `shared/`, then `knn/` (if `USES_KNN`), then the component's own subfolder.
- `save_component(root)` writes only the component's own subfolder.
- `load(root, shared=None, knn=None)` reads `shared/` and `knn/` from disk unless they are passed in.

Subclasses implement `fit`, `run`, `_save_own`, `_load_own` and `is_fitted_`. Class flags select the upstream tiers: `USES_SHARED` (default True), `USES_KNN`, `USES_TRAINING`.

- **`scores/training_distance.py`** — `TrainingDistance` (training modality, `USES_SHARED=False`). It gives **one value per molecule for the whole model**, built per column without pooling the training sets. Per column (`_per_column`, internal), the raw distance is 1 − mean Tanimoto (Morgan) to the `K_NEIGHBORS = 5` nearest training molecules. A query that is a training molecule drops itself. The calibrated distance is `_cdf_score` (higher = farther) against the column's leave-one-out raw values, from the index's identity-stripped self-kNN, saved as `loo_mean_distances.npz`. There is no cutoff. The scores `trn_distance` / `trn_distance_raw` are `nanquantile(per-column, 0.66)`, plus `trn_in_training` (any column). The details table has one row per query: the whole-model distances and the 5 nearest training molecules over all columns, deduplicated, with `nn_columns`.
- **`scores/training_difficulty.py`** — `TrainingDifficulty` (training modality, needs labels). One value per molecule for the whole model: `nanquantile(per-column, 0.66)` of a calibrated predicted error. It is fitted only for columns with ≥ `MIN_LABELLED = 50` labels; if none qualify, the orchestrator leaves it as `None`. There is no `_raw` column, by design: per-column errors are in different units. The per-column learner is in **`scores/_error_model.py`** (`EndpointErrorModel`, `fit_endpoint`):
  - **Surrogate.** An sklearn RF on Morgan bits with 5 scaffold folds (`training/folds.py`) gives OOF residuals.
  - **Inputs: UNIQUE's feature set (i)** (`feature_names`): 166 MACCS keys (data features); base UQ: kNN distance, 3 KDE log-densities (**`scores/_density.py`**: UNIQUE's 3 kernel/metric variants on MACCS with a grid-searched bandwidth; exact log-sum-exp; leave-one-out for reference molecules; at most 2k molecules for the grid and 5k for the reference), ensemble variance, top-1 probability for binary labels; and the prediction ŷ. Sets (ii)/(iii) and DiffkNN were dropped after the benchmark in `docs/concepts.md`: selecting among sets by OOF Spearman picked the leaky transformed set.
  - **Error model.** An sklearn RF on |residual|. Its OOF predictions are the CDF table, and `spearman` is the honesty check. Folds come from `training/folds.cv_folds` (scaffold, or random when scaffolds cannot split; `cv`). A query that is a training molecule reuses its OOF inputs and OOF predicted error.
  - **Persistence.** Saved with joblib under `training_difficulty/c000/`, with the sklearn version checked on load.
- **`scores/_training_helpers.py`** — `_nearest_training` (top-k training neighbours with the self match dropped) and `_columns_summary` (Q66 across columns), shared by both training scores.

- **`scores/typicality.py`** — `Typicality`. Density-based score: per-column int8 count LUTs, then the Q66 aggregate, then the CDF. Needs only `SharedFitState`.
- **`scores/extremity.py`** — `Extremity`. Position-based score: `min(|scaled|, 1)`, then Q66, then the CDF. Needs only `SharedFitState`.
- **`scores/support.py`** — `Support`. The raw value is the Tanimoto similarity of the **nearest library analogue** (`ref_support_raw`). It is calibrated against the library's own nearest-other-molecule similarities (`vi.self_knn_distances(1)`). Also emits `ref_support_log = −log10(support)`. Needs `SharedFitState` + `KnnFitState`.
- **`scores/_binning.py`** — conditional (binned) CDF calibration used by Consistency: `quantile_bin_edges` (merges ties and small bins), `assign_bins`, `partition_and_sort`, `binned_cdf_score`, and edge/npz save-load helpers.
- **`scores/consistency.py`** — `Consistency`. Output-space L1 distance to the k FP neighbours, calibrated **per FP-distance bin**. The bins start as up to `N_FP_BINS = 10` quantile bins on `mean_fp_distances`. Duplicate edges are merged, and bins smaller than `min(1000, n/20)` are merged into a neighbour. `n_bins_` is the number of bins actually used.
- **`scores/signal.py`** — `Signal` + `SignalLearner`. **Provisional**, fitted by default like the other scores (`--exclude ref_signal` skips it).
  - **Model.** One XGBoost regressor from RDKit physchem descriptors (`DEFAULT_DESCRIPTOR`, fixed) to the scaled, selected outputs.
  - **Training data.** `TRAIN_SAMPLES = 1000` rows of `shared.splits.train_indices` (local subset; the shared split is never modified). Early stopping uses `EARLY_STOP_VAL_MAX = 5000` val rows.
  - **Score.** The per-query Gini of `|SHAP|`, calibrated on the full val slice. `SIGNAL_FORMULA_VERSION = "gini_v2"`.
  - **State.** The descriptor is recorded in `umbrella.json`, with no run-time override. Needs the index folder for the descriptor matrices, but not `KnnFitState`.
- **`scores/_descriptors.py`** — `PhyschemBackend` (`compute_reference_subset`, `query_matrix`, `save_state`, `from_library`, `load_state`). `load_backend` rejects any other recorded descriptor with `ArtifactVersionError`.
  - Reference rows are gathered from the library's memory-mapped `physchem_scaled.npy`.
  - Query rows are computed with the same `library/physchem.py` functions the build used.
  - It checks that RDKit's descriptor list matches the library's.
- **`scores/_helpers.py`** — cross-score helpers:
  - **Calibration.** `_cdf_score` is the single CDF implementation (mid-rank, NaN passes through, `n` comes from the array). `_score_from_aggregates` is its higher-is-higher wrapper. `_sorted_finite` builds a CDF table from finite values. `_nan_aggregate` is the NaN-ignoring Q66 shared by typicality and extremity.
  - **Preprocessing.** `_make_pipeline`, `_make_query_repr`, and `_reference_repr`, which reuses `shared.ref_repr` when it is row-aligned with the reference.
  - **State resolution.** `_resolve_shared`, `_resolve_shared_and_knn`, `_resolve_vector_index` and `_custom_index_path`.
  - **Distances.** `_query_fp_distances` (top-(k+1), dropping the neighbour that is the query molecule itself) and `_query_output_distances` (chunked, NaN-aware L1).

### Shared upstream layers

- **`training/`** — the training-modality tier.
  - `data.py`: `TrainingColumn` and `load_training(folder, output_columns=None, predictions=None)`. One `<column>.csv` per output column (`smiles` or `input`, optional `y` or `value`, optional `key`). Standardisation via `scores/_helpers._standardize` (largest fragment, canonical isomeric). Duplicates are merged: binary by majority, otherwise by median. Columns with fewer than 20 molecules are skipped.
  - `folds.py`: `scaffold_folds` (balanced Murcko-grouped CV folds, acyclic molecules as singletons) and `morgan_bits`.
  - `state.py`: `TrainingFitState` (columns, a per-column `VectorIndex`, `eos_id`, `version`), plus `fit_training`, `save_training_state` and `load_training_state`.
  - Persisted under `<artifacts>/training_mode/training_sets/` with its own `TRAINING_FORMAT_VERSION` (independent of `ARTIFACT_FORMAT_VERSION`).

- **`shared/`** — `SharedFitState` and its `fit_shared` / `save_shared` / `load_shared` functions.
  - **Contents.** schema, eosframes scaler params, binary_class_freq, metadata, reference_ids, splits, selected_columns, and `ref_repr` (the scaled, feature-selected reference matrix, read by Consistency at run time).
  - **`metadata.py`.** Defines `FitMetadata`, which carries `library_id`, `vector_index_path` (custom indices only) and `format_version` (`ARTIFACT_FORMAT_VERSION`, currently 5). `load_shared` rejects other format versions with `ArtifactVersionError`.
  - **`splitter.py`.** Fixed 80/10/10 split with seed 0.
  - **`feature_selection.py`.** Correlation-cluster medoids, capped at `max_features`.
- **`knn/`** — `KnnFitState` holds `k` (`_registry.N_NEIGHBORS = 5`, not configurable), plus the fit-time-only `mean_fp_distances` and `reference_knn_indices`. `fit_knn` slices the precomputed self-kNN from the index. Only `{"k": …}` is persisted.

### Orchestrator + flat infrastructure modules

- **`quality.py`** — `ErsiliaQuality`.
  - **`ErsiliaQuality(verbose=False)`**; there is no other configuration.
  - **`fit(reference=None, training_sets=None, *, eos_id, version="v1", exclude=(), max_features=10, vector_index=None)`.** For the reference: checks unique keys, always loads the index (canonical library unless `vector_index`) and checks that its SMILES match the reference (no separate size minimum), then fits shared + knn once, then each component of `SCORE_ORDER` not in `exclude`. Excluding every reference score is a `ValueError`. `vector_index` is for programmatic use and tests; the CLI has no such flag.
  - **`run`.** Validates and scales the query once, runs the FP kNN once, and returns `RunResult(scores, metadata)`. Score columns come in `ref_<score>, ref_<score>_raw` pairs, plus `ref_support_log`; metadata keys are prefixed with the score name (`ref_support_k`).
  - **`save` / `load` / `add_training`** live in `_artifacts.py`. `save` writes `reference_mode/` (`shared/` and `knn/` once, then `save_component` for each score) and `training_mode/` (`training_sets/` plus training components), each only if fitted, plus `manifest.json`. `load` reads whichever subfolders exist and rejects the old flat layout. The fit and run of the reference modality live in `_reference_modality.py`, and those of the training modality in `_training_modality.py`; score names, orders and constants (`N_NEIGHBORS`, `DEFAULT_MAX_FEATURES`, `SCORE_NAMES`, `split_exclude`) are in `_registry.py` (constants only, so the CLI can read them cheaply), and the name → class map `SCORE_CLASSES` is in `_artifacts.py`.
- **`vectorindex.py`** — `VectorIndex`, the Morgan/FPSim2 kNN index.
  - **API.** `build`, `load` (memory-mapped kNN arrays), `query`, `self_knn_indices` / `self_knn_distances`, and the properties `library_name`, `index_dir`, `smiles`, `n_reference`.
  - **Resume.** `build` resumes only when the SMILES digest and parameters match.
  - **Single-threaded queries.** FPSim2 queries run with `n_workers=1` on purpose, because the order of tied neighbours is unstable with more threads.
- **`basic_descriptors.py`** — `BasicDescriptors.build_physchem` / `build_maccs`, used by `eosquality build`.
- **`preprocess.py`** — `PreprocessPipeline`, a thin wrapper around `eosframes.fit` / `transform`.
- **`schema/`** — `Schema` / `ColumnSpec` and column inference / validation.
- **`library/`** — library identity and the descriptor builders.
  - `identity.py` resolves the canonical library locally: env override → `./data/indices/` → `~/.eosquality/`. It never touches the network.
  - `download.py` is used only by `eosquality setup`.
  - `physchem.py` / `maccs.py` contain the descriptor functions shared by build and query.
- **`cli/`** — the dispatcher is `cli/__init__.py:main(argv=None)`. Subcommands: `setup`, `fit`, `run` and (maintainers) `build`. `_common.run_command(fn, verbose=, command=)` turns the curated console on in the command's colour and prints failures as `✖ error:` lines. Each command opens with a `summary_panel` header and ends with a summary panel. `fit` stages its log into `<artifacts>/eosquality.log` (`staged_log`); `run` writes `<output>.log`.
  - `fit -r CSV -t DIR -a DIR --exclude SCORES -v`. `-a` is a new folder, or an existing one with `reference_mode/` and no `training_mode/` when only `-t` is given (add training).
  - `run -i CSV -a DIR -o CSV -v`; training details always go to `<output stem>.training_details.csv`.
  - **Names carry the model.** There are no `--eos-id` / `--version` flags: the `-r` / `-t` / `-a` names (fit) and `-i` / `-a` / `-o` names (run) must each parse as `[prefix_]<eos_id>_<vN>` (eosframes' rule, `utils/identifiers._STEM_RE`) and agree (`model_from_names`); `run` also checks them against the artifacts' model.
- **`utils/`**
  - Output is in two layers, as in ZairaChem and Olinda; both write through one shared stderr Rich console.
    - `console.py`: the curated, user-facing layer. It provides `section()`, `Steps(n)` (`▪ Step i/N` + a timed `✓` line, `.skip()`), `detail`, `table`, `summary_panel`, `progress` (only on a TTY), `STEP_COLORS` per command, and the path/size/time formatting. It is silent until `console.enable()` is called, by the CLI or by verbose library use.
    - `logging.py`: the diagnostic layer, a loguru logger bound with `extra["eosquality"]` on a `RichHandler`. It prints WARNING by default and DEBUG with `set_verbosity(True)`, which also enables the console. `log_file(path)` adds a DEBUG file sink for one command (`module:function:line`, `diagnose=False`, no rotation or retention) and records any exception escaping the block, with its traceback. The `eosframes` stdlib logger is routed into it (`ROUTED_LOGGERS`).
  - `progress.py`: an alias of `console.progress`.
  - `parallel.py`: `map_rows`, serial below 5,000 items, otherwise a process pool.
  - `identifiers.py`: EOS id / version parsing; `model_from_name` / `model_from_names` for the CLI naming rule.
- **`exceptions.py`** — `SchemaError`, `NotFittedError`, `IncompatibleArtifactsError`, and its subclass `ArtifactVersionError`.

### Save layout

See `docs/diagram.md`: `<artifacts>/manifest.json`, `reference_mode/`, `training_mode/`. Each component's `metadata.json` carries only `component`, `fit_timestamp`, `fit_duration_seconds` and `k`. Dataset information lives once, in `reference_mode/shared/metadata.json`. `manifest.json` is informational, and the loader does not read it.

### When adding new functionality

1. Decide whether it's a **score component**, **shared upstream state**, or **infrastructure**.
2. For a new score, subclass `ScoreComponent` under `scores/<name>.py`, modelled on `Typicality` (no index) or `Support` (index-aware). Then add it to `SCORE_ORDER` and `SCORE_NAMES` (its `ref_` / `trn_` public name) in `_registry.py`, to `SCORE_CLASSES` in `_artifacts.py`, to the fitters in `_reference_modality._fitters` and to the run dispatch in `_reference_modality._run_component`.
3. For new shared upstream state, extend `SharedFitState` (always on, cheap) or `KnnFitState` (kNN tier).
4. Any change to what saved files mean (formula, layout, calibration) must bump `ARTIFACT_FORMAT_VERSION` in `shared/metadata.py`.
5. Add tests under `tests/`; the `library`, `reference` and `query` fixtures in `conftest.py` build a tiny custom library.

## Code Conventions

- **Formatting and linting** use ruff only (`ruff format`, `ruff check`); black is not used.
- **Dependencies** are pinned to exact versions in `pyproject.toml`; bump them deliberately.
- **CLI** is built with Click (`cli/`); commands raise `CliError` for user-facing errors, and `run_command` turns them into `✖ error:` lines and exit status 1. The library fetch command is `setup`, matching the other Ersilia tools.
- **Output:** user-facing status goes through `utils/console.py` (steps, panels), never through `logger.info`. `logger` is for diagnostics, which go to the log file and appear on screen only with `-v`. Library code narrates fit/run with `console.section` + `console.Steps`, which are no-ops while the console is off.
- **Size limits:** modules stay under 600 lines and functions under 80; split them before they grow past that.
- **Start-up imports:** `import eosquality` and the CLI must not import pandas, scikit-learn, SciPy, RDKit, XGBoost, FPSim2 or eosframes, which take about 2 s. The package `__init__` resolves its classes lazily (PEP 562 `__getattr__`), `_registry.py` holds constants only, and CLI modules import heavy modules inside command bodies, using `TYPE_CHECKING` for annotations. `tests/test_startup.py` enforces this.
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
