# Python API

```python
from eosquality import ErsiliaQuality, ALL_SCORES, DEFAULT_SCORES
```

## `ErsiliaQuality`

### Constructor

```python
ErsiliaQuality(k=5, verbose=False, config=None)
```

- `k` is the number of fingerprint neighbours used by support and consistency. It must be at most the index's `max_k` (50 for the canonical library).
- `verbose=True` turns on DEBUG logging and the diagnostic tables.

### `fit`

```python
eq.fit(
    reference=None,                 # DataFrame: key, input (SMILES), numeric outputs
    eos_id=None,                    # required, e.g. "eos4e40"
    version="v1",
    vector_index=None,              # path to a custom index; None = canonical library
    ignore_size=False,              # skip the 10,000-row minimum (testing only)
    scores=DEFAULT_SCORES,          # any subset of ALL_SCORES
    max_features=10,                # None disables feature selection
    max_signal_train_samples=1000,  # None/0 = full train slice
    signal_descriptor="physchem",   # or "maccs"
    training=None,                  # folder of <output_column>.csv training sets
    training_predictions=None,      # model predictions on training molecules (CSV/DataFrame)
) -> ErsiliaQuality
```

`fit` fits the **reference modality** when `reference` is given and the **training modality** when `training` is given. At least one of the two is required. The arguments from `vector_index` to `signal_descriptor` apply to the reference modality only.
- **SMILES alignment.** When support, consistency or signal is requested, `reference["input"]` must match the vector index's SMILES row for row.
- **Scores without an index.** Typicality and extremity need no index or `input` column.
- **Re-fitting.** Calling `fit` again replaces every component, including ones not requested this time.
- **Training sets.** The folder holds one CSV per output column (`smiles`, optional `y`, optional `key`). With a reference, file names must be among its output columns. See [cli.md](cli.md#eosquality-fit) for the loading rules.

```python
eq.fit_training(training, training_predictions=None, eos_id=None, version=None)
ErsiliaQuality.add_training("artifacts/", training, training_predictions=None)
```

`fit_training` fits (or replaces) only the training modality on an instance. `add_training` adds it to an existing artifacts folder in place:
- the reference files are left untouched;
- it refuses if the folder already has a training modality;
- it refuses if the model id differs.

**Score sets.**
- `DEFAULT_SCORES = ("typicality", "support", "consistency", "extremity")`
- `ALL_SCORES = DEFAULT_SCORES + ("signal",)`

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
- `training_domain.reference_domain_`: per column, the mean domain of its own training molecules (≈ 0.5).

## `RunResult`

`RunResult` has three fields: `scores`, `metadata` and `training_details`.

### `scores`

`scores` is a DataFrame indexed like the query. It has two columns per fitted score (three for support), in this order:

| column | range | meaning |
|---|---|---|
| `typicality`, `typicality_raw` | (0, 1], [0, 1] | calibrated score, Q66 density aggregate |
| `extremity`, `extremity_raw` | (0, 1], [0, 1] | calibrated score, Q66 position aggregate |
| `support`, `support_raw`, `support_log` | (0, 1], [0, 1], ≥ 0 | calibrated score, Tanimoto similarity of the nearest library analogue, −log10(support) |
| `consistency`, `consistency_raw` | (0, 1], ≥ 0 | calibrated score, mean output L1 distance to k neighbours |
| `signal`, `signal_raw` | (0, 1], [0, 1] | calibrated score, Gini of \|SHAP\| |
| `training_domain`, `training_domain_raw` | (0, 1], [0, 1] | 34th percentile across output columns of the per-column calibrated domain, and of the nearest-training-molecule similarity |
| `training_n_columns` | integer | output columns with a training set that contributed |
| `in_training_any` | bool | the query is itself a training molecule of some column |

Scores that were not fit are left out. A row with no usable output feature has NaN typicality, extremity and consistency.

### `metadata`

`metadata` is a dict containing `n_reference` plus each component's run metadata, with keys prefixed by component name. Examples:
- `typicality_reference_typicality`
- `support_k`
- `consistency_n_fp_bins`
- `signal_descriptor`
- `signal_formula_version`
- `training_domain_n_columns`, `training_domain_columns`

### `training_details`

`training_details` is `None` unless the training modality was fit. Otherwise it is a DataFrame with one row per (query, output column):

| column | meaning |
|---|---|
| `key` | query key (or index) |
| `column` | model output column |
| `domain`, `domain_raw` | calibrated domain for this column; Tanimoto similarity of the nearest training molecule |
| `n_train` | training molecules for this column |
| `in_training` | the query is one of them (its own entry is skipped) |
| `nn_keys`, `nn_similarities`, `nn_y` | the 5 nearest training molecules, `\|`-separated, closest first; `nn_y` is empty without labels |

## Per-score components

Every reference component can also be used on its own (`TrainingDomain` needs a training state; use the orchestrator). Each has `.fit(...)`, `.run(...)`, `.save(root)` and `.load(root)`:

```python
from eosquality import Typicality, Extremity, Support, Consistency, Signal

t = Typicality().fit(reference, eos_id="eos4e40", version="v1")
t.save("art/")             # writes art/shared/ + art/typicality/ (one reference_mode/ worth)
Typicality.load("art/").run(query).score
```

- **Support and Consistency** take `vector_index=` and `k=` when fitting.
- **Signal** needs a pre-fit `shared=` state (for example `ErsiliaQuality(...).shared_`) and a `vector_index=`.
- **Run results.** Each component's run result has `score`, `score_raw` and `metadata`. Typicality and extremity also expose `per_feature`. Support also exposes `score_log`, `distance_k_mean` (mean Tanimoto distance to the k neighbours) and `nearest_reference_ids` (closest first).

## Logging

As a library, `eosquality` only prints warnings by default.
- `eosquality.set_log_level("INFO")` shows progress messages.
- `eosquality.set_verbosity(True)` shows DEBUG messages and the diagnostic tables.

Handlers that the host application adds to loguru are left untouched.
