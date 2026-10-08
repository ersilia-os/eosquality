# Project status

**Status:** package `0.1.0`, library `ersilia_reference_library_v0` (1,355,109 molecules), artifact format 12, training format 14. The project is a work in progress. Typicality and extremity are functional and calibrated, and the reference and training match flags are exact lookups.

## Example results

The example models and query sets are those of `scripts/run_all_scores.sh`: it fits every reference score for five Ersilia models (the `emh_paper` fit sets in `data/fit_examples/`) and scores 1,000 molecules from each of five query sets (`data/run_examples/`). The figures in `figures/` are regenerated from its output with `scripts/figures/*.py` (in an environment with stylia).

| model | endpoint | outputs | kept after selection |
|---|---|---|---|
| eos3b5e | molecular weight | 1 | 1 |
| eos4e40 | *E. coli* activity | 1 | 1 |
| eos3804 | *A. baumannii* activity | 1 | 1 |
| eos42ez | cytotoxicity | 3 | 3 |
| eos7m30 | ADMET panel | 49 | 10 |

The query sets, and how many of their molecules the reference library holds (`ref_match` / `ref_scaffold`, the same for every model):

| query set | `ref_match` = 1 | `ref_scaffold` = 1 (of molecules with a scaffold) | no scaffold (NA) |
|---|---|---|---|
| Library sample | 100% | 100% | 0.4% |
| Drugs | 71% | 92% | 10% |
| Inert | 53% | 92% | 0.4% |
| Synthetic | 0% | 21% | 5% |
| Natural products | 0% | 7% | 34% |

`ref_match` ignores stereochemistry, charge and tautomers, so it finds more drugs and inert molecules than an exact SMILES comparison (43% of the drugs, 0.4% of the inert set).

**Calibration.** Library molecules score roughly Uniform(0, 1) on typicality and extremity, and the anchors are 0.500 by construction (mid-rank calibration). On the library sample the KS distance to uniform is at most 0.040 for every model and both scores, against a 95% critical value of 0.043 for n = 1,000.

![Reference calibration](figures/reference_calibration.png)
![Score distributions](figures/score_distributions.png)

**Typicality is smooth.** The density is interpolated between int8 levels, so the percentile varies continuously with the value (about 770–990 distinct values per 1,000 molecules for the one-output models, against about 130 when each value was rounded to a level). Exact ties remain where the outputs themselves tie (a binary or constant column).

**Redundancy.** For the single-output models typicality and extremity are nearly redundant (Spearman ρ −0.90 to −0.96: a value far from the centre is almost always a rare value); the panels of 3 and 49 outputs are less so (−0.65 and −0.45).

![Score correlations](figures/score_correlations.png)

**Cost.** On eos4e40, `fit` with reference and training sets takes about 13 s and `run` on 1,000 drugs about 11 s (about 4 s for molecules that are in the library, whose descriptors come from its cache), with the RDKit descriptors spread over the cores (`-j`). The largest training sets cost most: the cytotoxicity model (3 columns of 39,000 molecules, one shared index) fits in about 1.1 minutes, the ADMET panel (10 selected columns) in about 1.3 minutes. `eosquality build` of the 1.35M-molecule library takes about 30 minutes (the physchem descriptors dominate). The artifacts are 25–160 MB per model, mostly the training sets (their indices and physchem matrices, shared by columns measured on the same molecules) and the reference CDF tables (about 11 MB for typicality and 16 MB for extremity).

## Decisions to review

- **Typicality/extremity redundancy.** Consider reporting only one for single-output models, or replacing extremity with a score that is less tied to density.

## Known limitations

- **Feature selection** keeps at most 10 outputs (10 of 49 for eos7m30). With
  training sets, the candidates are first restricted to the columns that have
  one (41 of eos7m30's 49), and both modalities then use the same 10.
- **Typicality resolution** follows the int8 grid of the reference counts: a density between two levels is interpolated, not measured.
- **The match flags** say a structure or scaffold is in the library (or a training set), not that the model's prediction for it is right. Their keys depend on the RDKit version the library was built with: a different installed RDKit is refused, not silently accepted.
- **Library lookup** looks in `./data/indices/` relative to the current working directory. From elsewhere, set `EOSQUALITY_REFERENCE_LIBRARY_PATH` or run `eosquality setup`.
- **Training-set quality is not assessed.** The loader standardises SMILES,
  merges duplicates and reports how many rows it dropped or merged, but a set
  whose assay differs from the deployed model's, or
  which is not actually the model's training data will be scored against
  anyway. `trn_*` answers "how does this molecule relate to the data in this
  folder", not "was this model trained well".

## Training modality

Each output column can have its own training set. It is fitted with
`-t/--training-sets`, alone or together with `-r/--reference`; with both, the
reference scores use only the columns that have a training set.

![Training-score distributions](figures/training_scores.png)

Read that figure critically. The columns are the Tanimoto similarity percentile (`trn_tanimoto_pct`), the raw mean Tanimoto similarity (`trn_tanimoto_raw`) and the physchem similarity percentile (`trn_physchem_pct`). Against most query sets, including a sample of the reference library itself, the similarity percentile sits far below the training set's own typical value (the dashed 0.5 line) and often saturates near 0. That is honest: the training sets hold a few thousand molecules against a 1.35M-molecule library, so almost any query is farther from them than their molecules are from each other. It leaves the calibrated score with little resolution once everything is "far", which is why `trn_tanimoto_raw` is reported alongside it. Drugs are the closest set for eos4e40 (E. coli), whose training data is a drug-like screen; the synthetic set is the farthest for every model. The natural products fall outside every training set's physchem range, so `trn_physchem_pct` is near 0 for them.

## Open items

Reference modality:

- [ ] Raise the feature-selection cap from 10 to around 30 columns.
- [ ] Decide on typicality/extremity redundancy (see Decisions to review).

Training modality:

- [ ] `trn_tanimoto_pct` saturates near 0 against query sets that are all far
      from the training sets, so its calibrated form loses resolution exactly
      where a user most wants it. Consider a log companion.
