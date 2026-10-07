# Command-line interface

There are two steps, plus a one-time setup:

```bash
eosquality setup                                                    # once per install
eosquality fit -r reference_eos4e40_v1.csv -a artifacts_eos4e40_v1/ # once per model
eosquality run -i query_eos4e40_v1.csv -a artifacts_eos4e40_v1/ -o quality_eos4e40_v1.csv
```

**Names carry the model.** There is no `--eos-id` or `--version` flag. The model id and version are read from the names of the files and folders, which must follow the eosframes rule `[prefix_]<eos_id>_<version>`, e.g. `reference_eos4e40_v1.csv`, `training_eos4e40_v1/`, `artifacts_eos4e40_v1/`. The prefix is free; `<eos_id>` is `eos` + a digit + three characters, `<version>` is `v` + a number. Within a command every such name must give the same model, and `run` also checks it against the model the artifacts were fitted for. The query of `run` (`-i`) is exempt: it can have any name. Anything else stops with an error, for example:

```
✖ error: the names disagree on the model: --reference reference_eos4e40_v1.csv is eos4e40 v1, --artifacts artifacts_eos7m30_v1 is eos7m30 v1.
```

**Two modalities.** `fit` fits each modality whose data you give it, and every score of it unless excluded. Score names are prefixed by modality: `ref_` for the reference modality, `trn_` for the training modality. `run` always computes every score in the artifacts.

| `fit` gets | Scores |
|---|---|
| `-r/--reference` (predictions on the reference library) | `ref_typicality`, `ref_extremity`, `ref_support`, `ref_consistency`, `ref_signal` |
| `-t/--training-sets` (per-output-column training sets) | `trn_tanimoto`, `trn_physchem`, `trn_match` and `trn_scaffold` (the error model `trn_difficulty` is parked and off by default) |
| both | both, on the same columns (below) |

```bash
eosquality fit -r reference_eos4e40_v1.csv -t training_eos4e40_v1/ -a artifacts_eos4e40_v1/  # both
eosquality fit -t training_eos4e40_v1/ -a artifacts_eos4e40_v1/                              # training only
eosquality fit -r reference_eos4e40_v1.csv -a artifacts_eos4e40_v1/ --exclude ref_signal     # skip a score
```

**Depth of the training modality.** What each training file contains decides which training scores its column gets, with no flags involved. SMILES alone give four things: `trn_tanimoto` (distance to the 5 nearest training molecules over Morgan fingerprints, raw and as a percentile of the training set's own leave-one-out distances), `trn_physchem` (the same in standardised physicochemical descriptor space), `trn_match` (1 if the same structure, by InChIKey connectivity layer, is in a training set) and `trn_scaffold` (the same for the Murcko scaffold; empty for acyclic molecules), plus the nearest training neighbours in the details file. The error model `trn_difficulty` (needs labels `y`, at least 50 in some column) is parked: it is off by default.

**Common behaviour:**
- **Terminal output.** Every command prints curated progress to stderr, in the style of the other Ersilia tools (ZairaChem, Olinda). Each command has its own accent colour:
  - a header panel with the inputs;
  - a section per modality, made of numbered steps (`▪ Step i/N · …`) that close with a timed `✓` line and a short result;
  - small tables where useful: training columns, error models, the score summary of `run`;
  - a final summary panel with the outputs and the log file.

  Warnings appear in between. With `-v`, DEBUG messages and full tracebacks are shown too. Progress bars appear only on an interactive terminal.
- **Log files.** Every record, DEBUG included, goes to a log file with `module:function:line` context:
  - `fit`: `<artifacts>/eosquality.log`. If a fit fails, its log is kept in a temporary file whose path is printed.
  - `run`: `<output stem>.log` next to the scores CSV, e.g. `quality_eos4e40_v1.log`.

  When a command fails, the error and its traceback are recorded in its log.

  Variable values are never written into tracebacks, so SMILES don't leak into logs.
- **Exit status.** Errors print as `✖ error: …` and exit with status 1; success exits with 0.
- **Invalid molecules.** Query rows whose `input` SMILES is missing or does not parse are not an error. Their structure-based scores (`ref_support`, `ref_consistency`, `ref_signal` and the `trn_` scores) are NaN, and a warning names the rows. The output-based scores, `ref_typicality` and `ref_extremity`, are still computed.
- `fit` and `run` refuse to overwrite an existing output path.

## `eosquality setup`

Sets eosquality up: fetches the canonical reference library from the public S3 bucket into the user cache:
- the index folder → `~/.eosquality/indices/<library>/`
- the source CSV → `~/.eosquality/libraries/<library>.csv`

This is the only command that uses the network. If a valid cached copy already exists, nothing is fetched.

| flag | default | |
|---|---|---|
| `--force`, `-f` | off | fetch again even if cached |
| `--verbose`, `-v` | off | |

## `eosquality fit`

Fits the quality scores of one model and saves the artifacts.

**Reference CSV** (`-r`). It needs a `key` column, an `input` column (SMILES) and one numeric column per model output. The `input` column must equal the reference library's SMILES, in library order; this also fixes its size.

**Training folder** (`-t`). It holds one `<output_column>.csv` per output column:
- a `smiles` column (or `input`);
- optional `y` (or `value`), numeric, binary or continuous;
- optional `key`, used to name training molecules in the details file.

When `-r` is also given, every file must name one of its output columns. Columns without a file simply have no training scores.

Training SMILES are standardised: largest fragment, then canonical isomeric SMILES. Unparsable SMILES are dropped. Duplicate molecules are merged, with binary labels by majority vote and continuous labels by median. Columns with fewer than 20 molecules are skipped. The fit prints a table of the loaded columns (molecules, label kind, rows dropped as unparsable, molecules merged from conflicting labels) and warns once with the totals.

**Both inputs.** With both `-r` and `-t`, the reference modality uses only the output columns that have a usable training set (at least 20 valid molecules); the other columns are left out of every score. Feature selection then keeps at most 10 of the remaining columns, and both modalities use those same columns: the training scores, including the error models, are fitted on the selected columns only. So `trn_in_training` means "a training molecule of one of the selected columns". Training sets cannot be added to existing artifacts later: fit both together.

**Artifacts folder** (`-a`). Always a new folder; it gets `reference_mode/` and/or `training_mode/`. An existing folder is refused.

**Excluding scores** (`--exclude`). Takes score names, comma-separated or repeated: `--exclude ref_signal`, `--exclude ref_signal,trn_difficulty`. An unknown name is an error. Naming a score of a modality you did not give is ignored. Excluding every score of a given input is an error (nothing to fit).

| flag | default | |
|---|---|---|
| `--reference`, `-r CSV` | — | reference modality: the model's predictions on the reference library |
| `--training-sets`, `-t DIR` | — | training modality: per-column training sets |
| `--artifacts`, `-a DIR` | required | new artifacts folder |
| `--exclude SCORES` | none | scores not to fit, from `ref_typicality`, `ref_extremity`, `ref_support`, `ref_consistency`, `ref_signal`, `trn_tanimoto`, `trn_physchem`, `trn_match`, `trn_difficulty` |
| `--verbose`, `-v` | off | |

At least one of `-r` and `-t` is required.

**Fixed settings.** Support and consistency use k = 5 fingerprint neighbours. Signal uses RDKit physchem descriptors and trains on 1,000 rows. Feature selection keeps at most 10 output columns. With a reference, it clusters them on the correlation of their predictions; with `-t` alone, it clusters them on how many training molecules they share (columns from one screening panel group together) and keeps the largest set of each cluster. The Python API exposes `max_features` (see [api.md](api.md#fit)).

### The reference library

`fit -r` looks up the canonical library locally, in this order:
1. `$EOSQUALITY_REFERENCE_LIBRARY_PATH` (an index folder, used as is)
2. `./data/indices/<library>/`
3. `~/.eosquality/indices/<library>/`

`fit` never downloads anything; run `eosquality setup` first.

## `eosquality run`

Scores a query CSV against saved artifacts. Every score that was fit is computed; choose which scores to compute at fit time.

The artifacts folder and the output must be named for the same model, which must be the one the artifacts were fitted for. The query file can have any name.

The output CSV contains:
- the query's `key` and `input` (or `smiles`) columns, if present;
- for each fitted reference score, a calibrated column and a `*_raw` column (plus `ref_support_log`);
- for the training modality, `trn_tanimoto_pct`, `trn_tanimoto_raw`, `trn_physchem_pct`, `trn_physchem_raw` (a similarity, 1 = identical, 0 = no closer than a random library pair), `trn_match` and `trn_scaffold` (see [api.md](api.md#runresult)). `trn_match` and `trn_scaffold` are 1 / 0, and `trn_scaffold` is empty for a query with no scaffold.

If typicality or extremity is fitted, `<output stem>.reference_details.csv` is written next to the scores CSV: one row per query with `<column>_typicality_raw` / `_pct` and `<column>_extremity_raw` / `_pct` for every selected output column (see [api.md](api.md#reference_details)).

If the artifacts hold training scores, a further CSV, `<output stem>.training_details.csv`, is written next to it, e.g. `quality_eos4e40_v1.training_details.csv`. It has one row per query: the query's `key` and `input`, its `trn_*` scores, and the 5 nearest training molecules over all output columns (SMILES, keys, similarities, and the columns each belongs to). See [api.md](api.md#training_details).

For a training-only artifact, the query only needs SMILES, in an `input` or `smiles` column (`key` is optional).

| flag | default | |
|---|---|---|
| `--input`, `-i PATH` | required | query CSV |
| `--artifacts`, `-a PATH` | required | folder written by `fit` |
| `--output`, `-o CSV` | required | scores CSV, ending in `.csv` (must not exist; nor may its details CSVs) |
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

`scripts/run_all_scores.sh` fits all five reference scores for every `data/fit_examples/emh_paper_<eos>_v1.csv`. It then scores every matching `data/run_examples/*_1000_<eos>_v1.csv` query set, writing artifacts and scores to `output/` (override with `OUT_DIR=`). The figure scripts in `scripts/figures/` read those CSVs (see [status.md](status.md)).
