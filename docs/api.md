# Python API

```python
from eosquality import ErsiliaQuality, ALL_SCORES
```

## `ErsiliaQuality`

### Constructor

```python
ErsiliaQuality(verbose=False)
```

`verbose=True` turns on the curated step-by-step output (as in the CLI) and DEBUG logging.

### `fit`

```python
eq.fit(
    reference=None,      # DataFrame: key, input (SMILES), numeric outputs
    training_sets=None,  # folder of <output_column>.csv training sets
    *,
    eos_id,              # required, e.g. "eos4e40"
    version="v1",
    exclude=(),          # score names not to fit, e.g. ["ref_signal"]
    max_features=10,     # feature-selection cap; None disables it
    vector_index=None,   # custom index folder (programmatic use and tests)
) -> ErsiliaQuality
```

`fit` fits the **reference modality** when `reference` is given and the **training modality** when `training_sets` is given. At least one of the two is required. Every score of each given modality is fitted unless it is named in `exclude`.
- **Score names.** `exclude` takes the public, modality-prefixed names in `ALL_SCORES`: `ref_typicality`, `ref_extremity`, `ref_support`, `ref_consistency`, `ref_signal`, `trn_distance`, `trn_difficulty`. An unknown name raises `ValueError`, and so does excluding every reference score.
- **The reference library.** `reference["input"]` must match the library's SMILES row for row, which also fixes its size. The canonical library is resolved locally (see [cli.md](cli.md#the-reference-library)). `vector_index` points at a custom index instead; its absolute path is stored in the artifacts.
- **Fixed settings.** Support and consistency use k = 5 fingerprint neighbours. Signal uses RDKit physchem descriptors and 1,000 training rows.
- **Re-fitting.** Calling `fit` again replaces every component, including ones excluded this time.
- **Training sets.** The folder holds one CSV per output column (`smiles`, optional `y` or `value`, optional `key`). With a reference, file names must be among its output columns. See [cli.md](cli.md#eosquality-fit) for the loading rules.

```python
eq.fit_training(training_sets, *, eos_id=None, version=None, exclude=())
ErsiliaQuality.add_training("artifacts/", training_sets, *, eos_id=None, version=None, exclude=())
```

`fit_training` fits (or replaces) only the training modality on an instance. `add_training` adds it to an existing artifacts folder in place:
- the reference files are left untouched;
- it refuses if the folder already has a training modality;
- it refuses if the model id differs.

In both, `exclude` may name `trn_distance` or `trn_difficulty`.

**Score names.** `ALL_SCORES` lists the seven public names above, reference scores first.

### `run`

```python
result = eq.run(query)  # -> RunResult
```

`query` needs the reference's numeric columns when the reference modality was fit. It needs an `input` column when support, consistency, signal or the training modality was fit. For a training-only artifact, `key` and `input` are enough.

### `save` / `load`

```python
eq.save("artifacts/")
eq = ErsiliaQuality.load("artifacts/")
```

`save` writes one subfolder per fitted modality, `reference_mode/` and `training_mode/`, plus `manifest.json` (see [diagram.md](diagram.md#save-layout)). `load` reconstructs whichever modalities are present. It raises the following errors:
- `ArtifactVersionError`: the artifacts were written in an older on-disk format. Refit them.
- `IncompatibleArtifactsError`: the artifacts were fit against a different reference library or package major version.

**Post-fit attributes.**
- `modalities_`: `["reference"]`, `["training"]` or both.
- `reference_typicality_`, `reference_extremity_`, `reference_support_`, `reference_consistency_`, `reference_signal_`.
- `schema_`, `metadata_`, `shared_` (reference modality only).

## `RunResult`

`RunResult` has three fields: `scores`, `metadata` and `training_details`.

### `scores`

`scores` is a DataFrame indexed like the query. It has two columns per fitted score (three for support), in this order:

| column | range | meaning |
|---|---|---|
| `ref_typicality`, `ref_typicality_raw` | (0, 1], [0, 1] | calibrated score, Q66 density aggregate |
| `ref_extremity`, `ref_extremity_raw` | (0, 1], [0, 1] | calibrated score, Q66 position aggregate |
| `ref_support`, `ref_support_raw`, `ref_support_log` | (0, 1], [0, 1], ≥ 0 | calibrated score, Tanimoto similarity of the nearest library analogue, −log10(support) |
| `ref_consistency`, `ref_consistency_raw` | (0, 1], ≥ 0 | calibrated score, mean output L1 distance to the 5 FP neighbours |
| `ref_signal`, `ref_signal_raw` | (0, 1], [0, 1] | calibrated score, Gini of \|SHAP\| |
| `trn_distance`, `trn_distance_raw` | (0, 1], [0, 1] | one value for the whole model: the 66th percentile across output columns of the calibrated distance (percentile among the column's leave-one-out values) and of the raw distance (1 − mean Tanimoto to the 5 nearest training molecules) |
| `trn_difficulty` | (0, 1] | one value for the whole model: the 66th percentile across labelled output columns of the error model's predicted error, as a percentile among the training molecules' out-of-fold predicted errors (higher is harder); no raw column |
| `trn_in_training` | bool | the query is itself a training molecule of some column |

The `ref_` columns come from the reference modality and the `trn_` columns from the training modality. Scores that were not fit are left out; `trn_in_training` is present whenever `trn_distance` is. A row with no usable output feature has NaN typicality, extremity and consistency.

### `metadata`

`metadata` is a dict containing `n_reference` plus each score's run metadata, with keys prefixed by the score name. Examples:
- `ref_typicality_reference_typicality`
- `ref_support_k`
- `ref_consistency_n_fp_bins`
- `ref_signal_descriptor`, `ref_signal_formula_version`
- `trn_distance_n_columns`, `trn_distance_columns`, `trn_distance_k`
- `trn_difficulty_columns`, `trn_difficulty_spearman` (per column: Spearman of out-of-fold predicted vs actual error), `trn_difficulty_cv` (per column: `scaffold` or `random` folds), `trn_difficulty_n_labelled`

### `training_details`

`training_details` is `None` unless `trn_distance` was fit. Otherwise it is a DataFrame with one row per query:

| column | meaning |
|---|---|
| `key` | query key (or index) |
| `distance`, `distance_raw` | the whole-model `trn_distance` and `trn_distance_raw` |
| `difficulty` | the whole-model `trn_difficulty` (only when fitted) |
| `nn1_distance` | 1 − Tanimoto similarity of the nearest training molecule over all columns |
| `in_training` | the query is a training molecule of some column (its own entry is excluded from the neighbours) |
| `nn_keys`, `nn_similarities` | the 5 nearest training molecules over all columns, `\|`-separated, closest first, deduplicated |
| `nn_columns` | the output columns each of those molecules is a training molecule of, `;`-joined within a neighbour, `\|` between neighbours |

## Per-score components

Every reference component can also be used on its own (`TrainingDistance` and `TrainingDifficulty` need a training state; use the orchestrator). Each has `.fit(...)`, `.run(...)`, `.save(root)` and `.load(root)`:

```python
from eosquality import Typicality, Extremity, Support, Consistency, Signal

t = Typicality().fit(reference, eos_id="eos4e40", version="v1")
t.save("art/")             # writes art/shared/ + art/typicality/ (one reference_mode/ worth)
Typicality.load("art/").run(query).score
```

- **Support and Consistency** take `vector_index=` when fitting (and `k=`, 5 by default, as the orchestrator uses).
- **Names.** Components keep their short names (`eq.support`, `Support`, the `support/` artifacts folder); the `ref_` / `trn_` prefixes belong to the orchestrator's output columns, metadata keys and `exclude`.
- **Signal** needs a pre-fit `shared=` state (for example `ErsiliaQuality(...).shared_`) and a `vector_index=`.
- **Run results.** Each component's run result has `score`, `score_raw` and `metadata`. Typicality and extremity also expose `per_feature`. Support also exposes `score_log`, `distance_k_mean` (mean Tanimoto distance to the k neighbours) and `nearest_reference_ids` (closest first).

## Logging

As a library, `eosquality` is silent by default; only warnings are printed.
- `eosquality.set_verbosity(True)`, or `ErsiliaQuality(verbose=True)`, turns on the curated step-by-step output, as the CLI shows it, together with DEBUG messages. `set_verbosity(False)` turns both off again.
- `eosquality.set_log_level("INFO")` changes only the level of the terminal log sink.
- `from eosquality.utils.logging import logger` gives `with logger.log_file("run.log"): ...`, which writes every record, DEBUG included, to a file.

Output goes to stderr, through one shared Rich console. Handlers that the host application adds to loguru are left untouched. The standard-library logger of `eosframes`, which otherwise prints INFO lines on its own, is routed into eosquality's: its messages land in the log file, and reach the screen only as warnings or with verbose output.
