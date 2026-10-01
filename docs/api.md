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
    reference,                      # DataFrame: key, input (SMILES), numeric outputs
    eos_id,                         # e.g. "eos4e40"
    version="v1",
    vector_index=None,              # path to a custom index; None = canonical library
    ignore_size=False,              # skip the 10,000-row minimum (testing only)
    scores=DEFAULT_SCORES,          # any subset of ALL_SCORES
    max_features=10,                # None disables feature selection
    max_signal_train_samples=1000,  # None/0 = full train slice
    signal_descriptor="physchem",   # or "maccs"
) -> ErsiliaQuality
```

`fit` runs on the reference to be calibrated against.
- **SMILES alignment.** When support, consistency or signal is requested, `reference["input"]` must match the vector index's SMILES row for row.
- **Scores without an index.** Typicality and extremity need no index or `input` column.
- **Re-fitting.** Calling `fit` again replaces every component, including ones not requested this time.

**Score sets.**
- `DEFAULT_SCORES = ("typicality", "support", "consistency", "extremity")`
- `ALL_SCORES = DEFAULT_SCORES + ("signal",)`

### `run`

```python
result = eq.run(query)  # -> RunResult
```

`query` needs the reference's numeric columns. It also needs an `input` column when support, consistency or signal was fit.

### `save` / `load`

```python
eq.save("artifacts/")
eq = ErsiliaQuality.load("artifacts/")
```

`load` reconstructs every component whose subfolder is present. It raises the following errors:
- `ArtifactVersionError`: the artifacts were written in an older on-disk format. Refit them.
- `IncompatibleArtifactsError`: the artifacts were fit against a different reference library or package major version.

**Post-fit attributes.** `reference_typicality_`, `reference_extremity_`, `reference_support_`, `reference_consistency_`, `reference_signal_`, `schema_`, `metadata_`, `shared_`.

## `RunResult`

`RunResult` has two fields, `scores` and `metadata`.

### `scores`

`scores` is a DataFrame indexed like the query. It has two columns per fitted score, in this order:

| column | range | meaning |
|---|---|---|
| `typicality`, `typicality_raw` | (0, 1], [0, 1] | calibrated score, Q66 density aggregate |
| `extremity`, `extremity_raw` | (0, 1], [0, 1] | calibrated score, Q66 position aggregate |
| `support`, `support_raw` | (0, 1], [0, 1] | calibrated score, mean Tanimoto distance to k neighbours |
| `consistency`, `consistency_raw` | (0, 1], ≥ 0 | calibrated score, mean output L1 distance to k neighbours |
| `signal`, `signal_raw` | (0, 1], [0, 1] | calibrated score, Gini of \|SHAP\| |

Scores that were not fit are left out. A row with no usable output feature has NaN typicality, extremity and consistency.

### `metadata`

`metadata` is a dict containing `n_reference` plus each component's run metadata, with keys prefixed by component name. Examples:
- `typicality_reference_typicality`
- `support_k`
- `consistency_n_fp_bins`
- `signal_descriptor`
- `signal_formula_version`

## Per-score components

Every component can also be used on its own. Each has `.fit(...)`, `.run(...)`, `.save(root)` and `.load(root)`:

```python
from eosquality import Typicality, Extremity, Support, Consistency, Signal

t = Typicality().fit(reference, eos_id="eos4e40", version="v1")
t.save("art/")             # writes art/shared/ + art/typicality/
Typicality.load("art/").run(query).score
```

- **Support and Consistency** take `vector_index=` and `k=` when fitting.
- **Signal** needs a pre-fit `shared=` state (for example `ErsiliaQuality(...).shared_`) and a `vector_index=`.
- **Run results.** Each component's run result has `score`, `score_raw` and `metadata`. Typicality and extremity also expose `per_feature`. Support also exposes `distance_k_mean`, `distance_k_max` and `nearest_reference_ids`.

## Logging

As a library, `eosquality` only prints warnings by default.
- `eosquality.set_log_level("INFO")` shows progress messages.
- `eosquality.set_verbosity(True)` shows DEBUG messages and the diagnostic tables.

Handlers that the host application adds to loguru are left untouched.
