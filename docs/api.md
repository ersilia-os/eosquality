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
    include=(),          # scores that are off by default, i.e. ["trn_difficulty"]
    max_features=10,     # cap on output columns, both modalities; None disables it
    vector_index=None,   # custom index folder (programmatic use and tests)
) -> ErsiliaQuality
```

`fit` fits the **reference modality** when `reference` is given and the **training modality** when `training_sets` is given. At least one of the two is required. Every score of each given modality is fitted unless it is named in `exclude`.
- **Score names.** `exclude` takes the public, modality-prefixed names in `ALL_SCORES`: `ref_typicality`, `ref_extremity`, `ref_support`, `ref_consistency`, `ref_signal`, `trn_tanimoto`, `trn_physchem`, `trn_match`, `trn_difficulty`. An unknown name raises `ValueError`, and so does excluding every score of a given input.
- **The reference library.** `reference["input"]` must match the library's SMILES row for row, which also fixes its size. The canonical library is resolved locally (see [cli.md](cli.md#the-reference-library)). `vector_index` points at a custom index instead; its absolute path is stored in the artifacts.
- **Fixed settings.** Support and consistency use k = 5 fingerprint neighbours. Signal uses RDKit physchem descriptors and 1,000 training rows.
- **Re-fitting.** Calling `fit` again replaces every component, including ones excluded this time.
- **Training sets.** The folder holds one CSV per output column (`smiles`, optional `y` or `value`, optional `key`). With a reference, file names must be among its output columns. See [cli.md](cli.md#eosquality-fit) for the loading rules.
- **Both inputs.** The training sets are loaded first. The reference modality is then fitted only on the output columns that have a usable training set (at least 20 valid molecules), and `max_features` selects among those; the training modality is then fitted on the selected columns only, so both modalities cover the same columns. A training-only fit applies `max_features` too, keeping the largest training set of each cluster on `1 − Jaccard` overlap of the training molecules. There is no way to add training sets to a fitted instance or to saved artifacts: fit both together.

**Score names.** `ALL_SCORES` lists the eight public names above, reference scores first.

### `run`

```python
result = eq.run(query)  # -> RunResult
```

`query` needs the reference's numeric columns when the reference modality was fit. It needs an `input` column (or `smiles`) when support, consistency, signal or the training modality was fit. For a training-only artifact, SMILES alone are enough (`key` is optional).

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

`RunResult` has four fields: `scores`, `metadata`, `training_details` and `reference_details`.

### `scores`

`scores` is a DataFrame indexed like the query. It has two columns per fitted score (three for support), in this order:

| column | range | meaning |
|---|---|---|
| `ref_typicality`, `ref_typicality_raw` | (0, 1], [0, 1] | calibrated score, Q66 density aggregate |
| `ref_extremity_pct`, `ref_extremity_raw` | (0, 1], [0, 1] | `_raw`: Q66 over the output columns of `min(\|scaled\|, 1)` (0 = all at the centre, 1 = at least a third at the rails); `_pct`: each column's value is first placed on that column's own reference distribution, the per-column percentiles are combined at Q66, and the result is its percentile among the reference library's own values of that statistic; ~0.5 for a typical reference molecule |
| `ref_support`, `ref_support_raw`, `ref_support_log` | (0, 1], [0, 1], ≥ 0 | calibrated score, Tanimoto similarity of the nearest library analogue, −log10(support) |
| `ref_consistency`, `ref_consistency_raw` | (0, 1], ≥ 0 | calibrated score, mean output L1 distance to the 5 FP neighbours |
| `ref_signal`, `ref_signal_raw` | (0, 1], [0, 1] | calibrated score, Gini of \|SHAP\| |
| `trn_tanimoto_pct`, `trn_tanimoto_raw` | (0, 1], [0, 1] | structural applicability domain, one value for the whole model. `_raw` is the **mean Tanimoto similarity** (Morgan) to the 5 nearest training molecules, higher is closer, taken at the point where at least two-thirds of the output columns are this close or closer (1 − the 66th percentile of the per-column distances). `_pct` is the similarity percentile: 1 − the percentile of the matching distance among the column's training molecules' own leave-one-out distances, so higher is closer, ~0.5 for a query as close as a typical training molecule and near 0 for one farther than almost all of them |
| `trn_physchem_pct` | (0, 1) | physicochemical applicability domain, the similarity percentile: 1 − the percentile of the mean Euclidean distance to the 5 nearest training molecules (over RDKit physchem descriptors scaled with the reference library's scaler, clipped to ±10) among the training molecules' own leave-one-out distances, combined across columns at the 66th percentile of the distance; higher is closer |
| `trn_physchem_raw` | ≤ 1 | physchem similarity, `1 − d / 18.70` with d the mean distance above and 18.70 the median distance between two random reference-library molecules in the same space; 1 is identical, 0 is no closer than a random pair, and it is not clipped, so it can be negative |
| `trn_match` | 1 / 0 | 1 if the query's InChIKey connectivity layer (first 14 characters) equals that of a training molecule of any column, so the same structure ignoring stereochemistry, isotopes and charge; empty if the SMILES does not parse |
| `trn_scaffold` | 1 / 0 | 1 if the connectivity layer of the query's Murcko scaffold equals that of a training molecule's scaffold; empty (`NA`) if the query has no scaffold, e.g. an acyclic molecule, or does not parse |
| `trn_difficulty` | (0, 1] | **off by default** (`include=["trn_difficulty"]`): one value for the whole model, the 66th percentile across labelled output columns of the error model's predicted error (higher is harder); no raw column. Its four inputs, `nn1_tanimoto`, `nn5_tanimoto`, `ensemble_variance`, `surrogate_score`, go to `training_details` as `trn_<name>` |

The `ref_` columns come from the reference modality and the `trn_` columns from the training modality. Scores that were not fit are left out. A row with no usable output feature has NaN typicality, extremity and consistency.

### `metadata`

`metadata` is a dict. It holds `n_reference` (the size of the reference library) plus each score's own run metadata, with keys prefixed by the score name:
- `ref_<score>_anchor`: the score's mean over the reference library (about 0.5 by construction, the calibration check)
- `ref_support_k`, `ref_consistency_k`: fingerprint neighbours
- `ref_consistency_n_fp_bins`
- `ref_signal_descriptor`, `ref_signal_formula_version`, `ref_signal_anchor_raw`
- `trn_tanimoto_n_columns`, `trn_tanimoto_columns`, `trn_tanimoto_k`
- `trn_physchem_columns`, `trn_physchem_k` (neighbours averaged per column)
- `trn_match_n_molecules`, `trn_match_n_scaffolds` (distinct connectivity layers held)
- `trn_difficulty_columns`, `trn_difficulty_spearman` (per column: Spearman of out-of-fold predicted vs actual error), `trn_difficulty_cv` (per column: `scaffold` or `random` folds), `trn_difficulty_n_labelled`

### `reference_details`

`reference_details` is `None` unless extremity is fitted. Otherwise it is a DataFrame with one row per query: `key`, `input` (when the query has them) and, for each selected output column, `<column>_extremity_raw` (`min(|scaled|, 1)`) and `<column>_extremity_pct` (the percentile of that value among the reference library's values of the same column, higher = more extreme). The `ref_extremity_*` columns of `scores` are the whole-model summary of these.

### `training_details`

`training_details` is `None` when no training score is fitted. Otherwise it is a DataFrame with one row per query; columns of a score that was not fit are absent:

| column | meaning |
|---|---|
| `key` | query key (or index) |
| `input` | the query SMILES, as given |
| `trn_tanimoto_pct`, `trn_tanimoto_raw`, `trn_physchem_pct`, `trn_match`, `trn_scaffold` | the same values as in `scores`, repeated so the file stands alone |
| `trn_physchem_dist` | mean Euclidean distance, in library-scaled descriptor units, to the 5 nearest training molecules (Q66 across columns); details file only |
| `trn_physchem_raw` | the similarity of that distance, as in `scores` |
| `trn_in_training` | the query is itself a training molecule of some column (same standardised SMILES; stricter than `trn_match`) |
| `trn_difficulty`, `trn_nn1_tanimoto`, `trn_nn5_tanimoto`, `trn_ensemble_variance`, `trn_surrogate_score` | only when the error model is on |
| `nn1_similarity` | Tanimoto similarity of the nearest training molecule over all columns |
| `nn_smiles`, `nn_keys`, `nn_similarities` | the 5 nearest training molecules over all columns: standardised SMILES, key and similarity, `\|`-separated, closest first, deduplicated |
| `nn_columns` | the output columns each of those molecules is a training molecule of, `;`-joined within a neighbour, `\|` between neighbours |

The query itself is never one of its own neighbours, so a molecule with
`trn_in_training` set still gets the nearest *other* training molecules —
the same leave-one-out view the calibration is built from. Rows whose SMILES
does not parse keep their `key` and `input`, with empty neighbour fields.

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
- **Run results.** Each component's run result has `score`, `score_raw` and `metadata`. The Series carry the public score names (`ref_support`, `trn_tanimoto_pct`), the same as the orchestrator's output columns. Typicality and extremity also expose `per_feature`. Support also exposes `score_log`, `distance_k_mean` (mean Tanimoto distance to the k neighbours) and `nearest_reference_ids` (closest first).

## Logging

As a library, `eosquality` is silent by default; only warnings are printed.
- `eosquality.set_verbosity(True)`, or `ErsiliaQuality(verbose=True)`, turns on the curated step-by-step output, as the CLI shows it, together with DEBUG messages. `set_verbosity(False)` turns both off again.
- `eosquality.set_log_level("INFO")` changes only the level of the terminal log sink.
- `from eosquality.utils.logging import logger` gives `with logger.log_file("run.log"): ...`, which writes every record, DEBUG included, to a file.

Output goes to stderr, through one shared Rich console. Handlers that the host application adds to loguru are left untouched. The standard-library logger of `eosframes`, which otherwise prints INFO lines on its own, is routed into eosquality's: its messages land in the log file, and reach the screen only as warnings or with verbose output.
