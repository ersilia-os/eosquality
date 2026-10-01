# Command-line interface

There are two user steps, plus a one-time download:

```bash
eosquality download                                   # once per install
eosquality fit -i eos4e40_v1.csv -o artifacts_eos4e40/ # once per model
eosquality run -i query_eos4e40_v1.csv -a artifacts_eos4e40/ -o scores.csv
```

By default, the CLI prints INFO-level progress to stderr. With `-v` it also prints DEBUG messages and the diagnostic tables. Each command exits with status 0 on success and 1 on error. `fit` and `run` refuse to overwrite an existing output path.

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

Fits the quality scores on a model's predictions over the reference library and saves the artifacts.

The input CSV needs a `key` column, an `input` column (SMILES) and one numeric column per model output. For index-aware scores, the `input` column must equal the library SMILES in library order. The model id and version are read from the filename, e.g. `eos4e40_v1.csv` or `prefix_eos4e40_v1.csv`.

| flag | default | |
|---|---|---|
| `--input`, `-i PATH` | required | reference CSV |
| `--output`, `-o PATH` | required | artifacts folder (must not exist) |
| `--vector-index PATH` | canonical library | fit against a custom index built with `build`; its absolute path is stored in the artifacts |
| `--k K` | 5 | fingerprint neighbours (≤ the index's `max_k`) |
| `--version VERSION` | `v1` | used only if the filename has no version |
| `--ignore-size` | off | skip the 10,000-row minimum (testing only) |
| `--max-features N` | 10 | feature-selection cap; `0` disables it |
| `--scores LIST` | `typicality,support,consistency,extremity` | comma-separated subset of `typicality,extremity,support,consistency,signal` |
| `--max-signal-samples N` | 1000 | signal training rows; `0` uses the full train slice |
| `--signal-descriptor {physchem,maccs}` | `physchem` | signal feature backend |
| `--verbose`, `-v` | off | |

The canonical library is looked up locally, in this order:
1. `$EOSQUALITY_REFERENCE_LIBRARY_PATH`
2. `./data/indices/<library>/`
3. `~/.eosquality/indices/<library>/`

`fit` never downloads anything.

## `eosquality run`

Scores a query CSV against saved artifacts. The output CSV contains:
- the query's `key` and `input` columns, if present;
- a calibrated column and a `*_raw` column for each fitted score (see [api.md](api.md#runresult)).

Every score that was fit is computed; choose which scores to compute at fit time.

| flag | default | |
|---|---|---|
| `--input`, `-i PATH` | required | query CSV |
| `--artifacts`, `-a PATH` | required | folder written by `fit` |
| `--output`, `-o PATH` | required | scores CSV (must not exist) |
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
