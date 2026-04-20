> **Work in progress.** This repository is under active development. APIs and output formats may change without notice.

# eosquality

Quality assessment for [Ersilia Model Hub](https://ersilia.io) predictions.

`eosquality` estimates how well a reference population supports each query sample — useful for flagging predictions that fall outside a model's applicability domain. It does not estimate the probability that a prediction is correct.

## Install

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

There are three steps, run once in order:

**1. Build a vector index** — compute Morgan fingerprints and a pre-built kNN index for a reference molecule collection. Run once per collection; share across models that use the same set.

**2. Fit** — given a reference dataset (one row per molecule, numeric output columns), infer column types, fit per-column normalization, and compute reference quality diagnostics.

**3. Run** — score query samples against the fitted reference. Returns per-sample quality scores.

## Python API

```python
import pandas as pd
from eosquality import ErsiliaQuality

# Step 1: build the vector index (CLI, see below)

# Step 2: fit
eq = ErsiliaQuality(k=20, verbose=True)
eq.fit(
    reference_df,           # DataFrame with 'key', 'input' (SMILES), and numeric columns
    eos_id="eos4e40",
    version="v1",
    vector_index="path/to/index/",
)

# Step 3: run
result = eq.run(query_df)
print(result.scores[["quality_score", "support_score", "consistency_score", "intrinsic_richness"]])

# Save and reload fitted artifacts
eq.save("artifacts/")
eq2 = ErsiliaQuality.load("artifacts/")
result2 = eq2.run(query_df)
```

## CLI

```bash
# Step 1: build the vector index
eosquality index --input library.csv --output index_dir/ --verbose

# Step 2: fit (eos_id and version are extracted from the filename)
eosquality fit --input eos4e40_v1.csv --vector-index index_dir/ --output artifacts/ --verbose

# Step 3: run
eosquality run --input query.csv --artifacts artifacts/ --output scores.csv --verbose
```

## Output scores

| Column | Description |
|---|---|
| `quality_score` | Geometric mean of support and consistency (0–1) |
| `support_score` | Proximity to the reference — exp-decay on mean k-distance (0–1) |
| `consistency_score` | Uniformity of the local neighborhood (0–1) |
| `intrinsic_richness` | Mean deviation from the reference baseline across features (0–1) |
| `distance_k_mean` | Mean L1 distance to the k nearest reference neighbors |
| `distance_k_max` | Max L1 distance to the k nearest reference neighbors |
| `effective_feature_fraction` | Fraction of features used (reserved, currently 1.0) |
| `nearest_reference_ids` | Row indices of the k nearest reference neighbors |

## Concepts

**Support** measures how close a query point is to its k nearest reference neighbors. A low value means the query sits far from anything the reference has seen.

**Consistency** measures how uniform the local neighborhood is. A tight cluster of neighbors gives high consistency; a scattered neighborhood gives low consistency.

**Quality** is the geometric mean of support and consistency.

**Intrinsic richness** measures how much non-baseline signal the query carries. Each feature column has a fitted anchor — the normalized position of the reference median in [0, 1]. Richness is the mean absolute deviation from these anchors. A sample with all features at the reference median scores 0 (no signal above baseline); a sample with outputs far from typical scores higher.

## Normalization

All columns are normalized to [0, 1] before distance computation:

- Columns that are sparse (>5% exact zeros) *and* heavy-tailed (non-zero range >10×) receive a log1p pre-transform before scaling.
- All other columns are clipped to the [1st, 99th] percentile range and scaled linearly.

The **anchor** (normalized reference median) is stored per column. For right-skewed or sparse distributions it falls near 0; for bell-shaped distributions near 0.5. Distances are computed as mean L1 (`mean |v_i − w_i|`) on the [0, 1] normalized values.

---

The [Ersilia Open Source Initiative](https://ersilia.io) develops AI/ML tools for antimicrobial drug discovery in the Global South.

![Ersilia Logo](assets/Ersilia_Brand.png)
