# Concepts

`eosquality` judges a prediction in two modalities. Each is available only when its data was given at fit time.

**Reference modality.** It compares how an Ersilia model behaves on a query molecule with how the same model behaves on a fixed **reference library**: about 1.35M molecules shipped with each major version. The reference population is the model's own predictions on that library. It is **not** ground truth, so these scores describe how similar a query is to the model's behaviour on the reference. They don't estimate whether a prediction is correct.

**Training modality.** It compares a query with the model's **training sets**: one per output column, optionally with labels. Training labels are observations, so this modality can say whether the model has seen chemistry like the query. Its scores are reported as raw distances, not calibrated. Planned label-aware scores will also estimate how reliable a prediction is likely to be. See [Training modality](#training-modality).

## Shared preprocessing

Every score starts from the same fit-time state (`shared/`):

1. **Scaling.** Each numeric output column is scaled with [`eosframes`](https://github.com/ersilia-os/eosframes). eosframes classifies each column (constant, binary, count, skewed or centred continuous) and maps it robustly into `[-1, 1]`.
2. **Feature selection.** If there are more than `max_features` columns (default 10), columns are clustered on `1 − |Pearson r|` with average linkage, and only the medoid of each cluster is kept. Typicality, extremity, consistency and signal see only the selected columns.
3. **Split.** The reference is split once into a fixed, seeded 80/10/10 train/val/test partition (seed 0). Signal uses the train and val slices. The full split is always the one saved to disk.

## Calibration

Every score maps a **raw value** to a **calibrated score** in `(0, 1]` through the reference's own distribution of that raw value. The CDF uses mid-ranks:

```
cdf(v) = (#{ref < v} + #{ref ≤ v}) / (2n)
score  = clip(cdf(v), 1/(2n), 1)        # scores where "higher raw = higher score"
score  = clip(1 − cdf(v), 1/(2n), 1)    # distance-based scores (support, consistency)
```

With mid-ranks, a raw value shared by many reference rows (common for one-output models or quantised typicality) lands in the middle of its tie block. Reference molecules therefore score close to Uniform(0, 1), and their mean is 0.5; this is the `reference_<score>` anchor reported by each component.

`docs/figures/reference_calibration.png` checks this on 1,000 molecules sampled from the library.

## Missing values

The same rule applies to every score:
- A NaN output feature carries no information, so it is left out of the per-row aggregate and out of output-space distances.
- A row with no usable feature gets a NaN score.
- Reference rows with no usable feature are left out of the calibration CDFs.

## The five scores

| Score | Space | Raw value (`*_raw`) | High score means |
|---|---|---|---|
| Typicality | output | Q66 over features of per-feature density | outputs sit where the reference's outputs are dense |
| Extremity | output | Q66 over features of `min(\|scaled\|, 1)` | outputs sit far from the column centres |
| Support | fingerprint | Tanimoto similarity of the nearest library molecule | the library contains a close analogue of the molecule |
| Consistency | output, FP-conditioned | mean output L1 distance to the k FP neighbours | the outputs agree with those of chemically similar molecules |
| Signal *(opt-in)* | descriptor | Gini of per-feature \|SHAP\| | the model's output is driven by a few descriptors |

### Typicality (density)

At fit time, each selected column is quantised to int8 (`round(scaled × 127)`), and a 256-slot count table is built from the reference. Per-feature typicality is `count(level) / max_count`: the most common level scores 1 and unseen levels score 0. The row aggregate is the 66th percentile over features (Q66), which keeps the aggregate from collapsing towards the mean as the number of features grows. That aggregate is then calibrated.

### Extremity (position)

Per-feature extremity is `min(|scaled|, 1)`: 0 at the column centre, 1 at the rails. The row aggregate is Q66, which is then calibrated. Extremity is position-based where typicality is density-based, and the two are complementary. In practice they are strongly anti-correlated for most models (see `docs/figures/score_correlations.png`).

### Support (closest library analogue)

Support asks whether the library contains a close analogue of the query. The raw value (`support_raw`) is the Tanimoto similarity of the query's **nearest library molecule**, using Morgan fingerprints (radius 2, 2048 bits) queried with FPSim2. It is calibrated through the library's own nearest-analogue similarities, i.e. each library molecule against its closest *other* molecule.

The raw similarity reads directly in chemists' terms. Here is the share of each example set with no library analogue at a given threshold:

| | none ≥ 0.4 (no related chemistry) | none ≥ 0.6 (no close analogue) | none ≥ 0.8 (no near-identical) |
|---|---|---|---|
| Library molecules | 0.5% | 16% | 76% |
| Drugs | 5% | 24% | 62% |
| Natural products | 28% | 51% | 97% |
| Synthetic scaffolds | 99% | 100% | 100% |

**Self-matches.** A query that is itself in the library drops its own entry before the nearest analogue is taken. The entry dropped is the neighbour with the same SMILES or the same canonical SMILES; a different molecule with an identical fingerprint, such as a stereoisomer, counts as an analogue.

**Log scale.** `support_log = −log10(support)` expresses the calibrated tail probability on a log scale:
- about 0.3 for a typical library molecule;
- 2 means a more distant nearest analogue than 99% of library molecules have;
- 3 means more distant than 99.9%.

On the 0–1 scale, molecules far outside the library are all squeezed into 0–0.01; `support_log` keeps them apart.

**Size.** Tanimoto similarity is lower for small molecules (few set bits), so small fragments look somewhat more novel. This is not corrected for.

**Alternatives tested.** These definitions were compared on the five example query sets:
- **Mean distance to the 5 nearest neighbours (the previous definition).** It separated natural products from library molecules less well (AUC 0.72 vs 0.79).
- **Analogue counts above a threshold.** They have too many ties to calibrate (KS up to 0.25).
- **Calibration within fingerprint-size bins.** It tracked neighbour output disagreement no better on average.

How much a molecule's neighbourhood can be trusted to describe the model is left to consistency, which uses all k neighbours.

Support depends only on chemistry, so it is identical across models.

### Consistency (output agreement among chemical neighbours)

For the same k FP neighbours, the raw value is the mean output-space L1 distance between the query's outputs and each neighbour's outputs. The average is taken over features that are finite on both sides.

A plain CDF of this distance would mostly re-measure support, because distant neighbours naturally disagree more. To avoid that, the calibration is **conditioned on the fingerprint regime**:
1. The reference is split into up to 10 quantile bins of its own mean FP distance.
2. Duplicate quantile edges are merged, and bins smaller than `min(1000, n / 20)` rows are merged into a neighbour.
3. Each bin gets its own CDF.
4. A query is scored against the bin that contains its own mean FP distance.

The question consistency answers is: are this prediction's neighbours unusually noisy *for how far away they are*?

### Signal (attribution focus) — provisional, opt-in

At fit time, one XGBoost regressor is trained from a chemical descriptor to the scaled, selected model outputs. The descriptor is either RDKit physchem descriptors (`physchem`, the default) or MACCS keys (`maccs`). Training uses at most `max_signal_train_samples` rows of the train slice (default 1000). Early stopping is evaluated on 5,000 val rows.

For each query, the per-feature `|SHAP|` attributions are reduced to a Gini coefficient: about 1 when one descriptor carries all the attribution, about 0 when attribution is spread evenly. The Gini is then calibrated against the full val slice. The `|SHAP|` matrix of the val slice is saved as `signal/val_shap_attributions.npy` so other reductions can be tried offline.

## Training modality

Each output column of a model may have its own training set: SMILES, plus optional labels `y` (binary or continuous). Training SMILES are standardised (largest fragment, then canonical isomeric SMILES) and duplicates are merged. Each column gets its own Morgan fingerprint index (radius 2, 2048 bits).

### Training distance

Training distance asks how far the query is from the molecules the model was trained on. It gives **one value per molecule for the whole model**. It is the classic kNN applicability domain: the mean similarity to the 5 nearest training molecules is among the best structural predictors of prediction error (Sheridan et al., *J. Chem. Inf. Comput. Sci.* 2004). Following the kNN domain of Tropsha and the OECD principle that a domain is judged against the training set itself, the query is compared with how close training molecules are to each other. There is no in/out cutoff: both a raw distance and a calibrated percentile are reported.

The value is built from each output column's own training set; the sets are **not pooled**:
- **Raw, per column:** `1 − mean Tanimoto similarity` (Morgan, radius 2, 2048 bits) between the query and its **5 nearest training molecules**. A query that is itself a training molecule (same standardised SMILES) drops its own entry, so it gets its leave-one-out value.
- **Calibrated, per column:** the mid-rank percentile of the raw value among the column's **leave-one-out** raw values, where each training molecule is compared with its 5 nearest *other* training molecules. About 0.5 means as close as a typical training molecule; near 1 means farther than almost all of them.
- **Whole model:** `training_distance` (calibrated) and `training_distance_raw` are the **66th percentile** of the per-column values: at least two-thirds of the columns are this close or closer. A single distant column doesn't dominate, but several do.

Why not pool the training sets into one? Pooled, a large training set could hide that the query is far from a small one. Each calibrated per-column value is a percentile of that column's own training set, so columns of very different sizes and densities combine fairly.

How to read the two values: higher is farther for both. The raw value means the same thing across models: as a rough guide, 0.6 or more (mean similarity ≤ 0.4) means no related training chemistry. The calibrated value is relative to how dense the training sets are, so a diverse training set makes the same raw distance look more typical. Read them together.

`in_training` flags a query that is a training molecule of any column. `training_details` lists, per query, the 5 nearest training molecules over all columns (keys, similarities, and the columns each belongs to).

### Training difficulty

Training difficulty asks how hard the query is to predict, judging by the training data. It needs labels `y` and gives **one value per molecule for the whole model**. Distance measures novelty; difficulty also catches regions that are close to the training set but hard to learn: noisy assays, activity cliffs, chemotypes the labels disagree on.

It is an **error model**, following the error models of Novartis's UNIQUE (adapted from DEUP, Lahlou et al. 2021). Each output column with at least 50 labels gets its own:
1. **Surrogate.** A random forest on Morgan bits (scikit-learn) is fitted with 5-fold scaffold-grouped cross-validation. A congeneric training set, with fewer usable Murcko scaffolds than folds or one scaffold holding over 40% of the labelled molecules, falls back to random folds; the fit warns that its out-of-fold errors are then optimistic, and the metadata records `cv`. Every training molecule gets an out-of-fold prediction `ŷ` (P(y = 1) for binary labels) and the variance across the forest's trees. Labelled ones also get an out-of-fold residual `|y − ŷ|`, UNIQUE's L1 error. The surrogate stands in for the Ersilia model, whose own out-of-fold predictions are not available.
2. **Inputs**, grouped as in UNIQUE:
   - **Base UQ methods:**
     - kNN distance: the mean Tanimoto distance to the 5 nearest *other* training molecules.
     - Three kernel density estimates on the MACCS keys: Gaussian/Euclidean, Gaussian/Manhattan and exponential/Manhattan. Each bandwidth is chosen by 5-fold grid search over {0.1, 0.5, 1}.
     - Ensemble variance: across the surrogate's trees.
     - For binary labels, the top-1 class probability `max(p, 1 − p)`.
   - **Transformed UQ methods:**
     - **DiffkNN** on the prediction and on the variance: `|v − mean(v over the 5 nearest training molecules)|`, as UNIQUE defines it.
     - Two eosquality additions that use labels, which UNIQUE does not have: the similarity-weighted out-of-fold error of those neighbours, and the spread of their labels.
   - **Data features:** the 166 MACCS keys.
   - **The prediction** `ŷ`.
3. **Error model.** As in UNIQUE, a second random forest learning inputs → residual is fitted on three feature sets:
   - data features + base UQ + prediction;
   - base UQ + prediction;
   - transformed UQ + prediction.

   Each gets out-of-fold predictions on the same folds, and the one whose predictions correlate best (Spearman) with the true residuals is kept. Its out-of-fold predictions give the calibration table, and its Spearman value is the **honesty check**: 0 means no better than random.

For a query, the error model predicts its error, calibrated as the percentile among the training molecules' out-of-fold predicted errors: about 0.5 is as hard as a typical training molecule, near 1 is among the hardest. `training_difficulty` is the **66th percentile** across labelled columns, as for distance.

How to read it:
- It is a **rank, not an error estimate**. Predicted errors are in each column's own units (log-units, probabilities…), so there is no raw column: only percentiles can be combined across columns.
- It measures how hard the **endpoint** is around the query, for a random forest. It is not the deployed model's error. That part of the error comes mostly from the data (noise, cliffs, sparsity), which is why it transfers, but not entirely.
- Check the per-column Spearman values in the run metadata (`training_difficulty_spearman`, and `training_difficulty_variant_spearman` for all three feature sets) and in the fit log before trusting it. `scripts/evaluate_training.py` measures how well each training score ranks held-out errors on a scaffold split.

Columns without labels, or with fewer than 50, get no error model. A model with no such column has no `training_difficulty`.

A query that is itself a training molecule gets its own out-of-fold predicted error, the value the calibration table was built from. Re-using the final surrogate and error model for it would be in-sample, since both were trained on its label, and would rate it optimistically easy.

The surrogate, densities and error model are saved with joblib (pickle), so only load artifacts from a trusted source. The scikit-learn version is recorded, and loading with another version is refused (refit).

**Differences from UNIQUE.** Some are forced by the black-box setting, the rest are choices:
- **Errors come from a surrogate.** UNIQUE uses the real model's predictions. Ersilia models are black boxes, so we use a surrogate.
- **Training errors are all out-of-fold.** UNIQUE trains its error model on in-sample TRAIN errors plus out-of-sample CALIBRATION errors. Here every error is out-of-fold.
- **Training molecules are left out of their own neighbours and kernel.** UNIQUE counts a training molecule as its own nearest neighbour.
- **Densities are summed exactly.** They are computed in log space; scikit-learn's `score_samples` approximates densities far in the tails. For large training sets, the bandwidth grid search uses at most 2,000 molecules and the densities are built on at most 5,000.
- **Variant choice.** The feature set is chosen per column by out-of-fold Spearman. UNIQUE picks its best method on a held-out test split, with bootstrap and Wilcoxon tests.
- **Output.** We report a percentile combined across columns. UNIQUE reports raw predicted errors for one endpoint.
- **Not included:**
  - distances converted to variances and summed (UNIQUE's SumOfVariances), which needs a separate calibration set;
  - LASSO error models;
  - L2 and signed errors;
  - input standardisation, which doesn't matter for random forests.
- **Random-forest settings differ:** 200 trees and `min_samples_leaf=5`, against 50 trees and `max_depth=10` in UNIQUE's examples.

### Planned

- **Conformal intervals** (needs labelled molecules the model did not train on): coverage-guaranteed error intervals.
