# Command-line interface

There are two steps, plus a one-time download:

```bash
eosquality download                                        # once per install
eosquality fit --reference eos4e40_v1.csv -o art_eos4e40/  # once per model
eosquality run -i query_eos4e40_v1.csv -a art_eos4e40/ -o scores.csv
```

**Two modalities.** `fit` fits each modality whose data you give it. The scores add up, and `run` always computes every score in the artifacts:

| `fit` gets | Scores |
|---|---|
| `--reference` (predictions on the reference library) | reference modality: typicality, extremity, support, consistency, signal |
| `--training` (per-output-column training sets) | training modality: training_domain |
| both | both |

```bash
eosquality fit --reference eos4e40_v1.csv --training training_eos4e40_v1/ -o art/  # both
eosquality fit --training training_eos4e40_v1/ -o art/                             # training only
eosquality fit --training training_eos4e40_v1/ --artifacts art/                    # add training later
```

**Depth of the training modality.** What each training file contains decides which training scores its column gets, with no flags involved. SMILES alone give `training_domain` and the nearest training neighbours. Training scores that use labels `y` and the model's predictions on its training molecules are planned.

**Common behaviour:**
- By default, the CLI prints INFO-level progress to stderr. With `-v` it also prints DEBUG messages and the diagnostic tables.
- Each command exits with status 0 on success and 1 on error.
- `fit` and `run` refuse to overwrite an existing output path.

## `eosquality download`

Fetches the canonical reference library from the public S3 bucket into the user cache:
- the index folder → `~/.eosquality/indices/<library>/`
- the source CSV → `~/.eosquality/libraries/<library>.csv`

This is the only command that uses the network. If a valid cached copy already exists, nothing is downloaded.

| flag | default | |
|---|---|---|
| `--force`, `-f` | off | re-download even if cached |
| `--verbose`, `-v` | off | |

## `eosquality fit`

Fits the quality scores of one model and saves the artifacts.

**Reference CSV** (`--reference`). It needs a `key` column, an `input` column (SMILES) and one numeric column per model output. For index-aware scores, the `input` column must equal the library SMILES in library order.

**Training folder** (`--training`). It holds one `<output_column>.csv` per output column:
- a `smiles` column (or `input`);
- optional `y`, numeric, binary or continuous;
- optional `key`, used to name training molecules in the details file.

When `--reference` is also given, every file must name one of its output columns. Columns without a file simply have no training scores.

Training SMILES are standardised: largest fragment, then canonical isomeric SMILES. Unparsable SMILES are dropped. Duplicate molecules are merged, with binary labels by majority vote and continuous labels by median. Columns with fewer than 20 molecules are skipped.

**Model id.** It is read from the `--reference` file name, otherwise from the `--training` folder name (e.g. `eos4e40_v1.csv`, `training_eos4e40_v1/`). When adding training with `--artifacts`, it must match the artifacts' model.

| flag | default | |
|---|---|---|
| `--reference CSV` | — | reference modality: predictions on the reference library |
| `--training DIR` | — | training modality: per-column training sets |
| `--training-predictions CSV` | — | the model's predictions on the training molecules (Ersilia output CSV), stored for planned training scores |
| `--output`, `-o PATH` | — | new artifacts folder (must not exist); required unless `--artifacts` |
| `--artifacts`, `-a PATH` | — | existing artifacts folder to add `--training` to in place; reference files are not touched; refuses if it already has a training modality |
| `--vector-index PATH` | canonical library | fit the reference modality against a custom index built with `build`; its absolute path is stored in the artifacts |
| `--k K` | 5 | fingerprint neighbours (≤ the index's `max_k`) |
| `--version VERSION` | `v1` | used only if the file/folder name has no version |
| `--ignore-size` | off | skip the 10,000-row minimum (testing only) |
| `--max-features N` | 10 | feature-selection cap; `0` disables it |
| `--scores LIST` | `typicality,support,consistency,extremity` | reference scores to fit: comma-separated subset of `typicality,extremity,support,consistency,signal` |
| `--max-signal-samples N` | 1000 | signal training rows; `0` uses the full train slice |
| `--signal-descriptor {physchem,maccs}` | `physchem` | signal feature backend |
| `--verbose`, `-v` | off | |

At least one of `--reference` and `--training` is required.

The canonical library is looked up locally, in this order:
1. `$EOSQUALITY_REFERENCE_LIBRARY_PATH`
2. `./data/indices/<library>/`
3. `~/.eosquality/indices/<library>/`

`fit` never downloads anything.

## `eosquality run`

Scores a query CSV against saved artifacts. Every score that was fit is computed; choose which scores to compute at fit time.

The output CSV contains:
- the query's `key` and `input` columns, if present;
- a calibrated column and a `*_raw` column for each fitted score;
- for the training modality, also `training_n_columns` and `in_training_any` (see [api.md](api.md#runresult)).

If the artifacts hold a training modality, a second CSV is also written. It has one row per (query, output column), with the per-column domain and the 5 nearest training molecules (keys, similarities, labels).

For a training-only artifact, the query only needs `key` and `input`.

| flag | default | |
|---|---|---|
| `--input`, `-i PATH` | required | query CSV |
| `--artifacts`, `-a PATH` | required | folder written by `fit` |
| `--output`, `-o PATH` | required | scores CSV (must not exist) |
| `--training-details PATH` | `<output stem>.training_details.csv` | per-column training details (written only with a training modality; must not exist) |
| `--verbose`, `-v` | off | |

Artifacts written by an older eosquality format fail with a "refit" message.

## `eosquality build` (maintainers)

Builds a vector index and the descriptor matrices from a SMILES CSV that has a `smiles` column. It writes:
- `vector_index.h5`
- `knn_indices.npy` and `knn_distances.npy`
- `smiles.csv`
- `metadata.json`
- `physchem_scaled.npy` and `physchem_scaler.json`
- `maccs.npy`

An interrupted build resumes, but only if the SMILES list and the parameters are unchanged. See [reference-library.md](reference-library.md).

| flag | default | |
|---|---|---|
| `--input`, `-i PATH` | required | library CSV |
| `--output`, `-o PATH` | required | index folder |
| `--max-k K` | 50 | neighbours precomputed per molecule |
| `--radius R` | 2 | Morgan radius |
| `--n-bits N` | 2048 | Morgan bits |
| `--max-samples N` | all | truncate the input (testing) |
| `--verbose`, `-v` | off | |

## Reproducing the example results

`scripts/run_all_scores.sh` fits all five scores for every `data/fit_examples/emh_paper_<eos>_v1.csv`. It then scores every matching `data/run_examples/*_1000_<eos>_v1.csv` query set, writing to `output/` (override with `OUT_DIR=`). The figure scripts in `scripts/figures/` read those CSVs (see [status.md](status.md)).
