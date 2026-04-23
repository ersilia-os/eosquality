![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

# Quality of Ersilia predictions

Quality assessment for [Ersilia Model Hub](https://ersilia.io) predictions.

`eosquality` estimates how well a reference population _supports_ each query sample. This is useful for flagging predictions that fall outside a model's applicability domain. It does **not** estimate the _probability_ that a prediction is correct.

## Install

You can install `eosquality` from GitHub directly.

```bash
pip install git+https://github.com/ersilia-os/eosquality.git
```

Development install:

```bash
git clone https://github.com/ersilia-os/eosquality.git
cd eosquality
pip install -e ".[dev]"
```

## Workflow

Each release of `eosquality` is pinned to a single canonical reference library, i.e. a pre-built Morgan-fingerprint kNN index over a curated set of molecules. The library is **not** bundled in the wheel; it's hosted on a public S3 bucket and downloaded lazily on first use into `~/.eosquality/indices/ersilia_reference_library_vN/`.

**0. (Optional) Prefetch** — download the library ahead of time, useful for CI or offline work with `eosquality download`.

**1. Fit** — given a reference dataset (one row per molecule, numeric output columns, in the default Ersilia Model Hub format), infer column types, fit per-column normalization, and compute reference quality diagnostics. The corresponding index is fetched automatically if not already cached.

**2. Run** — score query samples against the fitted reference. Returns per-sample quality scores.

## CLI

```bash
# 1. Fit (eos_id and version are extracted from the filename)
eosquality fit --input eos4e40_v1.csv --output artifacts/ --verbose

# 2. Run
eosquality run --input query.csv --artifacts artifacts/ --output scores.csv --verbose
```

## Output scores

`result.scores` (one row per query sample):

| Column | Description |
|---|---|
| `quality_score` | Geometric mean of `support_score`, `typicality_score`, and `consistency_score` (0–1) |
| `support_score` | Proximity to the reference — exp-decay on mean k-distance to fingerprint-selected neighbors (0–1) |
| `typicality_score` | Plausibility of each individual feature value under the reference empirical CDF, aggregated by geometric mean across features (0–1) |
| `consistency_score` | Uniformity of the neighbor distances — tight neighborhoods score higher (0–1) |
| `distance_k_mean` | Mean L1 distance to the k nearest reference neighbors |
| `distance_k_max` | Max L1 distance to the k nearest reference neighbors |
| `nearest_reference_ids` | Row indices of the k nearest reference neighbors |

`result.typicality_per_feature` — `(n_query, n_features)` DataFrame with the per-column typicality contributions, so you can ask *which* feature made a query look atypical (`result.typicality_per_feature.loc[sample_id].sort_values()`).

`result.metadata` also includes `reference_quality` and `reference_typicality` as calibration baselines computed on the reference itself during fit.

## Python API

```python
import pandas as pd
from eosquality import ErsiliaQuality

# 1. Fit
eq = ErsiliaQuality(k=20, verbose=True)
eq.fit(
    reference_df,           # DataFrame with 'key', 'input' (SMILES), and numeric columns
    eos_id="eos4e40",
    version="v1",
    # vector_index=... is optional — omit to use the shipped reference library
)

# 2. Run
result = eq.run(query_df)
print(result.scores[["quality_score", "support_score", "typicality_score", "consistency_score"]])

# Drill into *which* columns made a sample look atypical
print(result.typicality_per_feature.loc[sample_id].sort_values())

# Save and reload fitted artifacts
eq.save("artifacts/")
eq2 = ErsiliaQuality.load("artifacts/")
result2 = eq2.run(query_df)
```

## Concepts

**Support** — how close a query sits to its k nearest reference neighbors in the normalized output space. A low value means the sample sits far from anything similar in the reference.

**Typicality** — how plausible each individual feature value is on its own, under the reference's per-column empirical distribution. For continuous/count/proportion columns, typicality = `2·min(CDF, 1−CDF)`, so a value at the reference median scores 1.0 and a value in the extreme tail scores near 0. For binary columns, typicality = `min(1, 2·class_freq)`, so a 90/10 imbalance gives the majority class 1.0 and the minority class 0.2. Aggregated across features by geometric mean, with an eps floor of `1/(2·n_ref)` so a single off-chart feature can't collapse the score to zero. A sample can have high support yet low typicality (near structural neighbors, but individual values are unusual) or vice versa — the two signals complement each other.

**Consistency** — how uniform the k neighbor distances are. A tight cluster of neighbors gives high consistency; scattered distances give low consistency.

**Quality** — geometric mean of support, typicality, and consistency. One bad factor pulls the headline down.

## Normalization

Normalization is **kind-aware**. Each column's kind is auto-detected from the reference data (`binary`, `count`, `continuous`, `proportion`) and drives two parallel artifacts:

**Distance representation** (what feeds the kNN distance):
- `binary` / `proportion` → pass-through; values are already in `[0, 1]`.
- `count` / `continuous` → clip to the `[1st, 99th]` percentile range and scale to `[0, 1]`. An optional log1p pre-transform kicks in when the column is sparse (>5% exact zeros) *and* heavy-tailed (non-zero range ratio >10×).

**Quantile grid** (what feeds typicality):
- `count` / `continuous` / `proportion` → a 101-point empirical CDF over the raw (pre-log1p) values, linearly interpolated at query time.
- `binary` → class frequencies `{0: p0, 1: p1}`; no grid needed.
- Constant columns are flagged and contribute 1.0 to typicality (no information either way).

The **anchor** (normalized reference median) is stored per column and used as the fallback for NaN query inputs in distance computations. Distances are mean L1 (`mean |v_i − w_i|`) on the `[0, 1]` normalized values.

## Reference library & versioning

Each release of `eosquality` is pinned to exactly one canonical reference library. Library version tracks the package major:

| Package version | Reference library         |
|-----------------|---------------------------|
| `0.y.z`         | `ersilia_reference_library_v0` |
| `1.y.z`         | `ersilia_reference_library_v1` |
| `2.y.z`         | `ersilia_reference_library_v2` |

`0.y.z` is pre-stable: APIs and scores can still move. Any change to the library — adding/removing molecules, rebuilding the index with different Morgan parameters, correcting SMILES in place — changes query scores, and in post-1.0 releases requires a major version bump. Metadata-only edits (description, citation) do not bump.

Artifacts saved with `eq.save(...)` are tagged with the library they were fit against; loading them under a different major raises `IncompatibleArtifactsError`. The fix is to install a compatible release or refit.

### Download model

A release ships **two** artifacts on the public S3 bucket:

| Artifact | S3 location | Local cache |
|---|---|---|
| Library CSV (source SMILES) | `https://eosvc-public.s3.amazonaws.com/eosquality/libraries/ersilia_reference_library_vN.csv` | `~/.eosquality/libraries/ersilia_reference_library_vN.csv` |
| Index folder (5 files) | `https://eosvc-public.s3.amazonaws.com/eosquality/indices/ersilia_reference_library_vN/` | `~/.eosquality/indices/ersilia_reference_library_vN/` |

Both are downloaded lazily on first use; `eosquality download` prefetches both. Override the base URL with `EOSQUALITY_REFERENCE_BASE_URL` for staging — the sibling `libraries/` prefix is derived automatically from the trailing `/indices/` segment.

The canonical name `ersilia_reference_library_vN` is used consistently everywhere: the CSV filename stem, the library-index folder, the `library_name` baked into `metadata.json`, the cache folder names, and the S3 path segments.

**Resolution order** for each artifact when `fit()` or `eosquality download` needs it:

| # | Library CSV | Library index |
|---|---|---|
| 1 | `EOSQUALITY_REFERENCE_LIBRARY_CSV_PATH` env override | `EOSQUALITY_REFERENCE_LIBRARY_PATH` env override |
| 2 | `./data/libraries/ersilia_reference_library_vN.csv` (CWD) | `./data/indices/ersilia_reference_library_vN/` (CWD) |
| 3 | `~/.eosquality/libraries/ersilia_reference_library_vN.csv` (user cache) | `~/.eosquality/indices/ersilia_reference_library_vN/` (user cache) |
| 4 | Download from S3 into the user cache | Download from S3 into the user cache |

Offline or CI setups can skip the network entirely by running from a repo checkout (which has both `data/libraries/` and `data/indices/` populated), or by placing the artifacts manually and setting the two env overrides.

**Why both?** The SMILES alignment check at fit time compares the reference CSV's `input` column against the canonical library CSV — the source of truth. This lets a maintainer build a subsampled index with `--max-samples` for fast testing while still asserting the reference is a true prefix of the canonical library. If you pass `vector_index=...` explicitly (opting out of the canonical ecosystem), fit falls back to checking against the index's own `smiles.csv`.

### For maintainers: cutting a new library version

Releasing a new library version (adding/removing molecules, rebuilding with different Morgan parameters, correcting SMILES in place) is a coordinated bump across the package. All historical library versions stay in the repo under `data/indices/` and on S3 forever — we never delete published libraries; we only add new majors alongside them. Source SMILES CSVs live at `data/libraries/` and are build inputs (not shipped anywhere at runtime).

**Three things must stay aligned for a given release**, all sharing the same canonical name `ersilia_reference_library_vN`. The pre-release test `tests/test_versioning.py` checks (1) and (2); (3) is a simple filesystem check you'll catch immediately when building or running:

| # | Thing                                  | Value for `vN` |
|---|----------------------------------------|----------------|
| 1 | `pyproject.toml` `project.version`     | `N.y.z`        |
| 2 | `LIBRARY_ID` in `src/eosquality/_library.py` | `ersilia_reference_library_vN` |
| 3 | Both the SMILES CSV stem and the output index folder name | `ersilia_reference_library_vN` |

Why the CSV filename matters: `eosquality index` derives `library_name` from the input CSV's basename (stem) and bakes it into the output `metadata.json`. That `library_name` is what the runtime integrity check compares against `LIBRARY_ID` — so the CSV (and, by convention, the output folder) must be named to match.

**Step-by-step for cutting `vN`:**

```bash
# You're in the root of the eosquality checkout.
#
# 1. Place the SMILES CSV (single column named 'smiles', deduped) at:
#    data/libraries/ersilia_reference_library_vN.csv

# 2. Bump the two code-side anchors in the same commit:
#    - LIBRARY_ID in src/eosquality/_library.py -> "ersilia_reference_library_vN"
#    - pyproject.toml version                   -> N.0.0
# Reinstall so the import-time consistency check sees the new metadata:
pip install -e . --no-deps

# 3. Build the canonical index into the repo.
eosquality index \
    --input  data/libraries/ersilia_reference_library_vN.csv \
    --output data/indices/ersilia_reference_library_vN --verbose

# 4. Sanity-check locally — with no env overrides, fit() picks up the newly
#    built index from ./data/indices/ersilia_reference_library_vN/ (the CWD
#    fallback) without touching the network:
pytest
eosquality fit --input some_eos_model_v1.csv --output /tmp/artifacts/

# 5. Push BOTH the library CSV and the index folder to S3 (maintainer side;
#    regular users do not need eosvc). The library CSV is the SMILES source
#    of truth used for fit-time alignment checks; the index folder is the
#    kNN data used at run time.
eosvc push data/libraries/
eosvc push data/indices/

# 6. Commit, tag (e.g. vN.0.0), and publish the wheel. End users then run
#    pip install eosquality==N.0.0 and, on first fit, eosquality fetches
#    ersilia_reference_library_vN from S3 into ~/.eosquality/indices/
#    automatically.
```

**Non-canonical indices for internal testing.** To fit against an index that is *not* the canonical library (e.g. a scratch build or a subsetted library), put it anywhere and point `fit` at it:

```bash
eosquality index --input scratch.csv --output /tmp/idx --verbose
EOSQUALITY_REFERENCE_LIBRARY_PATH=/tmp/idx \
    eosquality fit --input some_eos_model_v1.csv --output artifacts/
```

Artifacts produced this way are still tagged with the non-canonical `library_name` and will refuse to load under a release whose `LIBRARY_ID` doesn't match — which is the point.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) develops AI/ML tools for antimicrobial drug discovery in the Global South.

![Ersilia Logo](assets/Ersilia_Brand.png)
