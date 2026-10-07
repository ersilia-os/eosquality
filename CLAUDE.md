# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`eosquality` scores Ersilia Model Hub predictions for how they compare with the same model's predictions on a fixed reference library (about 1.35M molecules). There are two calibrated scores, typicality and extremity, and two exact-match flags, `ref_match` and `ref_scaffold`, against the reference library. The calibrated scores describe the model's own behaviour, **not** ground truth. See `docs/concepts.md`.

## Setup

```bash
conda create -n eosquality python=3.12
conda activate eosquality
pip install -e ".[dev]"        # add ",viz" for the figure scripts (stylia)
```

The canonical library folder (`smiles.csv`, `metadata.json`, `connectivity_keys.npz`) lives under `data/indices/ersilia_reference_library_v0/` (gitignored, via eosvc) or `~/.eosquality/` (`eosquality setup`). Library resolution looks in `./data/indices/` relative to the **current working directory**. When running from elsewhere, set `EOSQUALITY_REFERENCE_LIBRARY_PATH`.

## Common Commands

```bash
ruff check src tests scripts     # lint (org config + bugbear/pyupgrade)
ruff format src tests scripts    # formatting: ruff format only, no black
pre-commit install               # runs ruff-check and ruff-format on commit
pytest -q                        # ~2 min; builds a tiny custom library from tests/fixtures/
bash scripts/run_all_scores.sh   # refit + score the 5 example models into output/
conda run -n stylia python scripts/figures/<figure>.py   # writes docs/figures/
```

The package is installed in editable mode. A long `run_all_scores.sh` run imports `src/` at each CLI call, so edits made while it runs leak into later models. Pin a snapshot with `PYTHONPATH=<frozen copy>/src`.

## Architecture

The package is organized as **per-score components** (one class per score) on top of two **shared upstream layers**, plus a thin orchestrator and a few flat infrastructure modules.

### Two modalities

- **reference**: the model's predictions on the reference library. It covers typicality, extremity and the match flags, and the `shared/` tier, saved under `<artifacts>/reference_mode/`.
- **training**: the model's per-output-column training sets. It covers the `training/` package (`training_sets/` on disk) and the `training_*` components, saved under `<artifacts>/training_mode/`.

**Names.** Components have short internal names (`match`, `training_distance`: attributes, classes, artifacts subfolders). Their public **score names** are prefixed by modality, `ref_` or `trn_` (`ref_match`, `trn_tanimoto`); `_registry.SCORE_NAMES` maps one to the other. Public names are used for the output columns, the run metadata keys (`<score>_<key>`), `exclude=` / `--exclude` and `ALL_SCORES`.

Each modality is fitted only when its data is given (`fit -r`, `fit -t`, or both), and every score of it unless excluded. With both, the training sets are loaded first and the reference is restricted to the output columns with a usable training set (`quality._restrict_outputs`), so `max_features` selects among them, and the training modality is fitted on `shared.selected_columns` only (`_training_modality.select_training_columns`); there is no adding training later (`-a` must be new). `run` computes every fitted score.

### Per-score components — `scores/`

Each score subclasses `ScoreComponent` (`scores/_base.py`), which handles fit bookkeeping, `metadata.json` and save/load:
- `save(root)` writes `shared/` (if `USES_SHARED`), then the component's own subfolder.
- `save_component(root)` writes only the component's own subfolder.
- `load(root, shared=None, training=None)` reads `shared/` / `training_sets/` from disk unless they are passed in.

Subclasses implement `fit`, `run`, `_save_own`, `_load_own` and `is_fitted_`. Class flags select the upstream tiers: `USES_SHARED` (default True) and `USES_TRAINING`.

- **`scores/training_distance.py`** — `TrainingDistance` (training modality, `USES_SHARED=False`). It gives **one value per molecule for the whole model**, built per column without pooling the training sets. Per column (`_per_column`, internal), the raw distance is 1 − mean Tanimoto (Morgan) to the `K_NEIGHBORS = 5` nearest training molecules. A query that is a training molecule drops itself. The calibrated distance is `_cdf_score` (higher = farther) against the column's leave-one-out raw values, from the index's identity-stripped self-kNN, saved as `loo_mean_distances.npz`. There is no cutoff. Both published columns are **similarities, higher is closer**. `trn_tanimoto_pct` is `1 −` the Q66 across columns (`nanquantile(per-column, 0.66)`) of the per-column calibrated distance percentiles. `trn_tanimoto_raw` is `1 −` the Q66 of the per-column raw distances, i.e. the mean Tanimoto similarity of the query to its 5 nearest training molecules. `trn_in_training` (any column) is in the details file. The details table has one row per query: the whole-model distances and the 5 nearest training molecules over all columns, deduplicated, with `nn_columns`.
- **`scores/_training_helpers.py`** — `TrainingQuery` (the per-run cache of standardised SMILES, physchem descriptors and per-column kNN searches, built once in `_training_modality.run_training` and passed to every score: without it each recomputed them), `_nearest_training` (top-k training neighbours with the self match dropped) and `_columns_summary` (Q66 across columns).

- **`scores/training_match.py`** — `TrainingMatch` (training modality, `USES_SHARED=False`, public name `trn_match`). Two exact set lookups over the union of every column's training molecules, emitted as `trn_match` and `trn_scaffold` (`Int64`, 1 / 0): the InChIKey connectivity layer (first 14 characters, `connectivity_layer`) of the query, and of its Murcko scaffold (`training_match._scaffold`). `trn_scaffold` is NA for a query with no scaffold, both are NA for an unparsable one. The artifact is two sorted string arrays (`connectivity_keys.npz`, `allow_pickle=False`); no calibration.

- **`scores/training_physchem.py`** — `TrainingPhyschem` (training modality, `USES_SHARED=False`). The same method as `TrainingDistance` in a different space: per column, the mean Euclidean distance to the 5 nearest training molecules over RDKit physchem descriptors scaled with the reference library's scaler and clipped to ±10 (`library/physchem_scaler.json`, shipped with the package, loaded by `library/physchem.canonical_scaler`; **`scores/_physchem_domain.py`**, pure numpy persistence so nothing is pickled), calibrated against the training molecules' own leave-one-out distances and Q66'd across columns, then published as the similarity percentile `trn_physchem_pct = 1 − Q66` (higher is closer). `trn_physchem_raw` is the same distance as an unclipped similarity `1 − d / PAIR_MEDIAN` (`PAIR_MEDIAN = 18.70`, the median distance of 1M random library pairs; `scripts/physchem_pair_median.py` recomputes it); the distance itself, `trn_physchem_dist`, is in the details file. The self match is dropped exactly as `_helpers._query_fp_distances` does, and the scaled, clipped training matrix is stored with the artifact (float32, ~34 MB for a 39k column) because a neighbour method needs the reference molecules themselves. No cutoff: `d̄ + Zσ` with Z=0.5 (Golbraikh 2003, *not* an OECD prescription — the guidance endorses no specific method) is the 65th percentile on eos4e40, so the percentile carries it without assuming normality. An ellipsoid form (T², DModX) was built first and removed on evidence — see `docs/concepts.md`. Not an error-model input.
- **`scores/reference_match.py`** — `ReferenceMatch` (reference modality, public name `ref_match`, emits `ref_match` and `ref_scaffold`, `Int64` 1 / 0). The same two exact lookups as `TrainingMatch`, against the **reference library's** molecules: the InChIKey connectivity layer of the (standardised) query, and of its Murcko scaffold. NA rules as in `trn_match`. The keys are **library-level**: `eosquality build` writes `connectivity_keys.npz` (sorted unique string arrays) into the library folder, `library/reference.ReferenceLibrary.match_keys` reads them (and refuses an RDKit different from the `rdkit_version` that `build` recorded; `build` writes the folder atomically, a temporary sibling renamed when complete), and the artifact keeps only the counts (`state.json`) and finds the library again through `shared.metadata` (`library_id`, `library_path` for a custom library). No calibration, no `_raw`. The helpers shared with `TrainingMatch` are in **`scores/_match_keys.py`** (`connectivity_layer`, `_scaffold`, `_layers`, `_flags`).
- **`scores/typicality.py`** — `Typicality`. Density-based score: per-column int8 count LUTs; per-feature `count / max count`, then Q66 (`ref_typicality_raw`). `ref_typicality_pct`: each column's density is placed on that column's own reference distribution (`pct_luts_`, derived from the count LUTs by `percentile_luts`, not saved), the per-column percentiles go through Q66, and that aggregate through the reference's CDF. The per-column raw values and percentiles are `per_feature` / `per_feature_pct`, written to `reference_details`. Needs only `SharedFitState`.
- **`scores/extremity.py`** — `Extremity`. Position-based score: `min(|scaled|, 1)` on the eosframes-scaled values, then Q66 (`ref_extremity_raw`). `ref_extremity_pct`: each column's value is placed on that column's own reference table (`column_tables.npz`, float32, mid-rank), the per-column percentiles go through Q66, and that aggregate through the reference's CDF. The per-column raw values and percentiles are `per_feature` / `per_feature_pct`, written to `reference_details`. Needs only `SharedFitState`.
- **`scores/_helpers.py`** — cross-score helpers:
  - **Calibration.** `_cdf_score` is the single CDF implementation (mid-rank, NaN passes through, `n` comes from the array). `_score_from_aggregates` is its higher-is-higher wrapper. `_sorted_finite` builds a CDF table from finite values. `_nan_aggregate` is the NaN-ignoring Q66 shared by typicality and extremity, and `_aggregate_percentiles` the same for their per-column percentiles.
  - **Preprocessing.** `_make_pipeline`, `_make_query_repr`, and `_reference_repr`, which reuses `shared.ref_repr` when it is row-aligned with the reference.
  - **State resolution.** `_resolve_shared`.
  - **Distances.** `_query_fp_distances_valid` (top-(k+1), dropping the neighbour that is the query molecule itself), used by the training scores.

### Shared upstream layers

- **`training/`** — the training-modality tier.
  - `data.py`: `TrainingColumn` and `load_training(folder, output_columns=None)`. One `<column>.csv` per output column (`smiles` or `input`, optional `key`; other columns are ignored). Standardisation via `scores/_helpers._standardize` (largest fragment, canonical isomeric). Duplicate molecules are merged. Columns with fewer than 20 molecules are skipped.
  - `state.py`: `TrainingFitState` (columns, a per-column `VectorIndex`, `eos_id`, `version`), plus `fit_training`, `save_training_state` and `load_training_state`.
  - Persisted under `<artifacts>/training_mode/training_sets/` with its own `TRAINING_FORMAT_VERSION` (independent of `ARTIFACT_FORMAT_VERSION`).

- **`shared/`** — `SharedFitState` and its `fit_shared` / `save_shared` / `load_shared` functions.
  - **Contents.** schema, eosframes scaler params, binary_class_freq, metadata, reference_ids and selected_columns. `ref_repr` (the scaled, feature-selected reference matrix) exists at fit time only and is not saved.
  - **`metadata.py`.** Defines `FitMetadata`, which carries `library_id`, `library_path` (custom libraries only) and `format_version` (`ARTIFACT_FORMAT_VERSION`, currently 8). `load_shared` rejects other format versions with `ArtifactVersionError`.

### Orchestrator + flat infrastructure modules

- **`quality.py`** — `ErsiliaQuality`.
  - **`ErsiliaQuality(verbose=False)`**; there is no other configuration.
  - **`fit(reference=None, training_sets=None, *, eos_id, version="v1", exclude=(), max_features=10, library=None)`.** For the reference: checks unique keys, loads the library (`ReferenceLibrary`: the canonical one unless `library`) and checks that its SMILES match the reference (no separate size minimum), with training sets drops the outputs without a usable training set, then fits shared once, then each component of `SCORE_ORDER` not in `exclude`. Excluding every reference score is a `ValueError`. `library` is for programmatic use and tests; the CLI has no such flag.
  - **`run`.** Validates and scales the query once and returns `RunResult(scores, metadata)`. Score columns come in `ref_<score>_pct, ref_<score>_raw` pairs, plus the `ref_match` / `ref_scaffold` flags; metadata keys are prefixed with the score name (`ref_match_n_molecules`); typicality and extremity emit `anchor` (their mean over the reference, ~0.5), and `n_reference` is written once, at the top level. `tests/test_roundtrip.py::test_metadata_keys_are_stable` pins this contract.
  - **`save` / `load`** live in `_artifacts.py`. `save` writes `reference_mode/` (`shared/` once, then `save_component` for each score) and `training_mode/` (`training_sets/` plus training components), each only if fitted, plus `manifest.json`. `load` reads whichever subfolders exist and rejects the old flat layout. The fit and run of the reference modality live in `_reference_modality.py`, and those of the training modality in `_training_modality.py`; score names, orders and constants (`DEFAULT_MAX_FEATURES`, `SCORE_NAMES`, `split_exclude`) are in `_registry.py` (constants only, so the CLI can read them cheaply), and the name → class map `SCORE_CLASSES` is in `_artifacts.py`.
- **`vectorindex.py`** — `VectorIndex`, the Morgan/FPSim2 kNN index. Used by the training scores (one per training column), not by the reference modality.
  - **API.** `build`, `load` (memory-mapped kNN arrays), `query`, `self_knn_indices` / `self_knn_distances`, and the properties `library_name`, `index_dir`, `smiles`, `n_reference`.
  - **Resume.** `build` resumes only when the SMILES digest and parameters match.
  - **Single-threaded queries.** FPSim2 queries run with `n_workers=1` on purpose, because the order of tied neighbours is unstable with more threads.
- **`preprocess.py`** — `PreprocessPipeline`, a thin wrapper around `eosframes.fit` / `transform`.
- **`schema/`** — `Schema` / `ColumnSpec` and column inference / validation.
- **`library/`** — library identity, the library folder and the descriptor functions.
  - `identity.py` resolves the canonical library locally: env override → `./data/indices/` → `~/.eosquality/`. It never touches the network.
  - `download.py` is used only by `eosquality setup`.
  - `reference.py`: `ReferenceLibrary` (the folder's SMILES, metadata, `validate_smiles` and `match_keys`).
  - `physchem.py` holds the RDKit descriptor functions and the shipped physchem scaler (`trn_physchem`).
- **`cli/`** — the dispatcher is `cli/__init__.py:main(argv=None)`. Subcommands: `setup`, `fit`, `run` and (maintainers) `build`. `_common.run_command(fn, verbose=, command=)` turns the curated console on in the command's colour and prints failures as `✖ error:` lines. Each command opens with a `summary_panel` header and ends with a summary panel. `fit` stages its log into `<artifacts>/eosquality.log` (`staged_log`); `run` writes `<output>.log`.
  - `fit -r CSV -t DIR -a DIR --exclude SCORES -v`. `-a` must be a new folder.
  - `run -i CSV -a DIR -o CSV -v`; training details always go to `<output stem>.training_details.csv`, and the per-column typicality and extremity (when fitted) to `<output stem>.reference_details.csv`.
  - **Names carry the model.** There are no `--eos-id` / `--version` flags: the `-r` / `-t` / `-a` names (fit) and `-a` / `-o` names (run; the query `-i` can have any name) must each parse as `[prefix_]<eos_id>_<vN>` (eosframes' rule, `utils/identifiers._STEM_RE`) and agree (`model_from_names`); `run` also checks them against the artifacts' model.
- **`utils/`**
  - Output is in two layers, as in ZairaChem and Olinda; both write through one shared stderr Rich console.
    - `console.py`: the curated, user-facing layer. It provides `section()`, `Steps(n)` (`▪ Step i/N` + a timed `✓` line, `.skip()`), `detail`, `table`, `summary_panel`, `progress` (only on a TTY), `STEP_COLORS` per command, and the path/size/time formatting. It is silent until `console.enable()` is called, by the CLI or by verbose library use.
    - `logging.py`: the diagnostic layer, a loguru logger bound with `extra["eosquality"]` on a `RichHandler`. It prints WARNING by default and DEBUG with `set_verbosity(True)`, which also enables the console. `log_file(path)` adds a DEBUG file sink for one command (`module:function:line`, `diagnose=False`, no rotation or retention) and records any exception escaping the block, with its traceback. The `eosframes` stdlib logger is routed into it (`ROUTED_LOGGERS`).
  - `progress.py`: an alias of `console.progress`.
  - `parallel.py`: `map_rows`. In-process by default: a pool is started only when the caller passes `n_jobs` (none of the commands today) and the input reaches `PARALLEL_MIN_ITEMS`. Library code must never start one — with the "spawn" start method a pool started from a user's unguarded script re-executes that script in every worker (`tests/test_robustness.py::test_library_code_never_starts_a_process_pool`).
  - `identifiers.py`: EOS id / version parsing; `model_from_name` / `model_from_names` for the CLI naming rule.
- **`exceptions.py`** — `SchemaError`, `NotFittedError`, `IncompatibleArtifactsError`, and its subclass `ArtifactVersionError`.

### Save layout

See `docs/diagram.md`: `<artifacts>/manifest.json`, `reference_mode/`, `training_mode/`. Each component's `metadata.json` carries only `component`, `fit_timestamp` and `fit_duration_seconds`. Dataset information lives once, in `reference_mode/shared/metadata.json`. `manifest.json` is informational, and the loader does not read it.

### When adding new functionality

1. Decide whether it's a **score component**, **shared upstream state**, or **infrastructure**.
2. For a new score, subclass `ScoreComponent` under `scores/<name>.py`, modelled on `Typicality` (shared state only) or `ReferenceMatch` (reads the library). Then add it to `SCORE_ORDER` and `SCORE_NAMES` (its `ref_` / `trn_` public name) in `_registry.py`, to `SCORE_CLASSES` in `_artifacts.py`, to the fitters in `_reference_modality._fitters` and to the run dispatch in `_reference_modality._run_component`.
3. For new shared upstream state, extend `SharedFitState` (always on, cheap).
4. Any change to what saved files mean (formula, layout, calibration) must bump `ARTIFACT_FORMAT_VERSION` in `shared/metadata.py`.
5. Add tests under `tests/`; the `library`, `reference` and `query` fixtures in `conftest.py` build a tiny custom library.

## Code Conventions

- **Formatting and linting** use ruff only (`ruff format`, `ruff check`); black is not used.
- **Dependencies** are pinned to exact versions in `pyproject.toml`; bump them deliberately.
- **CLI** is built with Click (`cli/`); commands raise `CliError` for user-facing errors, and `run_command` turns them into `✖ error:` lines and exit status 1. The library fetch command is `setup`, matching the other Ersilia tools.
- **Output:** user-facing status goes through `utils/console.py` (steps, panels), never through `logger.info`. `logger` is for diagnostics, which go to the log file and appear on screen only with `-v`. Library code narrates fit/run with `console.section` + `console.Steps`, which are no-ops while the console is off.
- **Size limits:** modules stay under 600 lines and functions under 80; split them before they grow past that. `console.table` prints at most `MAX_TABLE_ROWS` (15) and then `… and N more`, so a 41-column fit does not flood the terminal; the full table is in the log file.
- **Start-up imports:** `import eosquality` and the CLI must not import pandas, scikit-learn, SciPy, RDKit, FPSim2 or eosframes, which take about 2 s. The package `__init__` resolves its classes lazily (PEP 562 `__getattr__`), `_registry.py` holds constants only, and CLI modules import heavy modules inside command bodies, using `TYPE_CHECKING` for annotations. `tests/test_startup.py` enforces this.
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
