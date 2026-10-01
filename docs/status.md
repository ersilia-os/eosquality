# Project status

**Status:** package `0.1.0`, library `ersilia_reference_library_v0` (1,355,109 molecules), artifact format 2. The project is a work in progress. The four default scores are functional and calibrated; Signal is provisional.

## Example results

To reproduce:
1. Run `scripts/run_all_scores.sh`. It fits all five scores for five Ersilia models (the `emh_paper` fit sets in `data/fit_examples/`) and scores 1,000 molecules from each of five query sets (`data/run_examples/`).
2. Run `scripts/figures/*.py` (in an environment with stylia) to regenerate the figures below.

| model | endpoint | outputs | kept after selection |
|---|---|---|---|
| eos3b5e | molecular weight | 1 | 1 |
| eos4e40 | *E. coli* activity | 1 | 1 |
| eos3804 | *A. baumannii* activity | 1 | 1 |
| eos42ez | cytotoxicity | 3 | 3 |
| eos7m30 | ADMET panel | 49 | 10 |

The query sets:
- **Library sample:** 1,000 molecules drawn from the reference library itself.
- **Drugs:** 43% of them are in the library.
- **Natural products, synthetic:** none are in the library.
- **Inert:** 0.4% are in the library.

### Calibration

![Reference calibration](figures/reference_calibration.png)

**Library sample.** Library molecules should score roughly Uniform(0, 1) on every score. For every model and score, the KS distance to uniform is at most 0.04, against a 95% band of about 0.043 for n = 1,000.

**Full reference.** The `reference_<score>` anchors are exactly 0.500, thanks to mid-rank calibration. Format 1 had 0.505–0.509 for typicality on the one-output models.

**Typicality steps.** Typicality's ECDF moves in steps for those one-output models because their raw typicality has only about 130 distinct values across 1.35M molecules (int8 density levels). Ties cannot spread out into a uniform distribution. Mid-rank centres them rather than biasing them upward.

### Distributions

![Score distributions](figures/score_distributions.png)

- **Library sample:** flat for every score, as expected.
- **Support:** separates chemistry cleanly. Synthetic and natural-product molecules sit at about 0, drugs are spread out, and the library sample is uniform. Support depends on chemistry only, so it is identical across models.
- **Output-based scores** react per model:
  - The MW model gives natural products and synthetic molecules extreme, atypical predictions and very low consistency.
  - The Cytotox model treats synthetic molecules as typical but noisy.
- **Inert set:** looks almost exactly like the library sample on every score, including support, even though only 0.4% of it is in the library. Open question: what is this set, and is that expected?

### Redundancy

![Score correlations](figures/score_correlations.png)

Spearman correlations between calibrated scores, pooled over all queries of a model:
- **Typicality vs extremity:** ρ = −0.56 to −0.95 (−0.85 to −0.95 for the one-output models). For single-output models the two are nearly redundant: a value far from the centre is almost always a rare value.
- **Typicality vs consistency:** moderately correlated (0.21–0.61).
- **Support** is roughly independent of the output scores except for MW (0.6 with typicality), where molecular weight is itself a chemical-space proxy.
- **Signal** is weakly correlated with everything (|ρ| ≤ 0.54).

### What changed between artifact formats 1 and 2

Comparing re-fitted against previous scores on the 25 example sets (old CSVs kept in `output/format1/`; the old calibration plot is `figures/reference_calibration_format1.png`):
- **Extremity and signal:** unchanged.
- **Typicality:** shifted by at most 0.017 (mid-rank ties).
- **Missing values:** none of the five example models have NaN outputs, so the NaN-policy fixes do not affect these examples. They matter for models with missing outputs.
- **Support and consistency:** unchanged for most queries, but about 25% of drugs gained support (mean +0.22). These are molecules not in the library as written whose stereo-free form is. See the decision below.

## Decisions to review

- **Self-match rule (support/consistency).** A query now drops only the neighbour that is the same molecule (same SMILES or canonical isomeric SMILES). A different stereoisomer with an identical Morgan fingerprint stays as a distance-0 neighbour, exactly as in the library's own self-kNN, so the calibration is consistent.
  - The alternative treats any fingerprint-identical neighbour as "self". That gives lower support to stereo-variants of library molecules, but it is inconsistent with how the reference CDF is built.
- **Typicality/extremity redundancy.** Consider reporting only one for single-output models, or replacing extremity with a score that is less tied to density.

## Known limitations

- **Signal is provisional.**
  - It trains on 1,000 rows by default (`--max-signal-samples`).
  - With physchem descriptors, raw Gini values cluster near their maximum (about 0.99 on the test fixture), so most of the discrimination comes from small differences.
  - The full val-slice |SHAP| matrix is saved to `signal/val_shap_attributions.npy` so other reductions can be prototyped offline.
- **Feature selection** keeps at most 10 outputs (10 of 49 for eos7m30).
- **Typicality resolution** is limited by int8 quantisation for one-output models (about 130 levels).
- **Library lookup** looks in `./data/indices/` relative to the current working directory. From elsewhere, set `EOSQUALITY_REFERENCE_LIBRARY_PATH` or run `eosquality download`.
- **Run time** for 1,000 queries is about 15 s with all five scores, dominated by FPSim2 queries (about 10 ms each). Queries run single-threaded on purpose: multi-threaded FPSim2 returns ties in an unstable order, which made consistency non-reproducible. Fitting one model takes a few minutes.
- `binary_class_freq` is computed and saved, but no score reads it.

## Open items

Carried over from the previous README TODO list:

- [ ] Raise the feature-selection cap from 10 to around 30 columns.
- [ ] Signal: train on more rows (e.g. 100,000) and check how stable calibration is. Early stopping already evaluates 5,000 val rows, and calibration already uses the full val slice (about 135k rows).
- [ ] Signal: choose training compounds for quality (e.g. high consistency, diverse) instead of at random.
- [ ] Signal: handle trivial models (e.g. molecular weight), where a few descriptors explain everything. One option is to bin the reference.

New:

- [ ] Characterise the "inert" query set (see Distributions).
- [ ] Decide on the self-match rule and on typicality/extremity redundancy (see Decisions to review).
- [ ] Before pushing, check that CI passes on GitHub (`.github/workflows/ci.yml`).
