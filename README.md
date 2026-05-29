![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)

# Quality of Ersilia predictions

Quality scoring for [Ersilia Model Hub](https://ersilia.io) predictions. `eosquality` tries to quantify, via multiple metrics, whether a given run output from Ersilia is "trustworthy". It does **not** estimate the probability that a prediction is correct.

## Installation

Install the latest version of `eosquality` directly from GitHub:

```bash
pip install git+https://github.com/ersilia-os/eosquality.git
```

The CLI is then available as `eosquality`. You need to start by downloading the reference library and its indices:

```bash
eosquality download
```

This will take a while. The downloaded library will be stored under `~/.eosquality/`.

## Quick start

The typical workflow is two commands: `fit` once per Ersilia model against its reference predictions, then score any query dataset against the saved artifacts.

### Fitting a reference library

The input CSV must hold the model's predictions on the **exact** molecules of the canonical Ersilia reference library shipped with your installed `eosquality` version: a `key` column, an `input` (SMILES) column, and one numeric column per model output. The filename encodes the model and version (e.g., `eos4e40_v1.csv`). The output folder stores the fitted artifacts.

There is one and only one reference library per major version of `eosquality`, so the molecule set is fixed by your install. If the SMILES entries in the input CSV don't match that library, `fit` refuses with an error.

```bash
eosquality fit --input reference_eos4e40_v1.csv --output artifacts_eos4e40_v1/
```

Please check [Isaura](https://github.com/ersilia-os/isaura) for a large store of pre-calculations across Ersilia models.

### Running against new samples

At querying time, `run` loads a fitted artifacts folder and scores any query CSV containing Ersilia results for a given model. The output CSV is written with `key`, `input`, and the score columns described below.

```bash
eosquality run --input query_eos4e40_v1.csv --artifacts artifacts_eos4e40_v1/ --output quality_eos4e40_v1.csv
```

## Scores

For each query, `eosquality` reports component scores measuring how its predicted values sit relative to the reference population. Note that a reference population is the model's own predictions on a fixed reference library, **not** ground truth.

- **Typicality**. For each prediction column, the int8-quantized query value is looked up in a count histogram built on the reference (`count / max_count`). The most common reference value scores 1.0, unseen values score 0. Aggregated across columns by arithmetic mean.

- **Extremity** — position-based. For each column, `min(|scaled|, 1.0)` after the eosframes scaler. Values at the per-column center score 0; values at the rails score 1. Aggregated by arithmetic mean over non-NaN columns. Complementary to typicality.

- **Support** — neighborhood-based, in fingerprint space. The mean Tanimoto distance from the query to its *k* FP-nearest reference molecules is looked up in the reference's own self-distance CDF. Closer to the reference than every reference point → ~1.0; farther than every reference point → ~eps.

- **Consistency** — neighborhood-based, in output space. After picking the query's *k* FP-nearest reference neighbors, the mean output-space L1 distance to those neighbors is looked up in the reference's own self-distance CDF. Quieter than the reference's typical neighborhood → ~1.0; jaggier → ~eps. Mirrors Support's calibration, in output space rather than FP space.

- **Signal** — attribution-based, on a chemical descriptor. An XGBoost regressor is fit once at fit time on a chosen feature backend → eosframes-scaled model outputs. At run time, per-query `|SHAP|` attributions are reduced to the **Gini coefficient** of the attribution distribution: high (→ 1) when one or a few features carry most of the attribution ("focused" chemistry), low (→ 0) when attribution is spread roughly uniformly across many features ("scattered" chemistry). The raw Gini is calibrated through the reference val slice's own distribution. The feature backend is chosen at fit time via `--signal-descriptor`: `physchem` (default, 217 RDKit physicochemical descriptors) or `maccs` (167-bit RDKit MACCS structural fingerprint). The choice is baked into the saved artifact and used unchanged at run time. *Provisional and opt-in: not in the default score set; request via `--scores ...,signal` on the CLI or `scores=DEFAULT_SCORES + ("signal",)` in Python.*

## For maintainers

Repository maintaniers may build a new reference library index. `eosquality build` is a **release-only** tool that produces a new canonical reference library. **Ordinary users never invoke it.** Two invariants make this section maintainer-only:

- There is **one and only one** reference library per major version of `eosquality`. The library version is synchronized with the package's major version: bumping one bumps the other, in lockstep.
- The reference library is **always** the [canonical Ersilia reference library](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs). Third-party or user-curated libraries are out of scope.

Cutting a new library version therefore means bumping `LIBRARY_ID` in `src/eosquality/library/identity.py` alongside the package's major-version bump, building a fresh index from the new Ersilia SMILES CSV, and uploading both the CSV and the index folder to the canonical S3 location so that `eosquality download` picks them up:

```bash
eosquality build --input ersilia_reference_library_v1.csv --output data/indices/ersilia_reference_library_v1/
```

Uploading to S3 happens with [eosvc](https://github.com/ersilia-os/eosvc). The repo's `access.json` already routes `data/` to the public bucket, so place the source CSV at `data/libraries/ersilia_reference_library_v1.csv` (alongside the index folder built above) and run:

```bash
eosvc upload --path data/libraries/ersilia_reference_library_v1.csv
eosvc upload --path data/indices/ersilia_reference_library_v1/
```

# TODO

- [ ] Column subsampling is currently doing only 10 columns. Maybe increase to 30.
- [ ] In the signal score, the validation set is very small. Maybe increase size to 10,000.
- [ ] In the signal score, the training set is very small. Maybe increase size to 100,000.
- [ ] For the training set selection, it may make sense to selecte high quality compounds, based on low typicality, high extremity, consistency, etc.

## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech-nonprofit fueling sustainable research in the Global South. Ersilia's main asset is the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia), an open-source repository of AI/ML models for drug discovery.

`eosquality` is designed to score predictions produced by [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) models. The library is built on [`eosframes`](https://github.com/ersilia-os/eosframes) as a per-column normalization backend.

![Ersilia Logo](https://raw.githubusercontent.com/ersilia-os/eosquality/main/assets/Ersilia_Brand.png)