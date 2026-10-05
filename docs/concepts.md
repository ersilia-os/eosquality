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

Training distance asks how far the query is from the molecules each output column was trained on. It is a plain distance and is **not calibrated**.
- **Per column:** `1 − Tanimoto similarity` (Morgan, radius 2, 2048 bits) between the query and its **nearest training molecule**.
  - 0 means the query is itself a training molecule (same standardised SMILES); it is then flagged `in_training`.
  - Values near 1 mean the training set holds nothing similar.
  - As a rough guide, a distance of 0.6 or more (similarity ≤ 0.4) means no related training chemistry.
- **Summary across columns:** `training_distance` in the scores is the **66th percentile** of the per-column distances: at least two-thirds of the columns have a training molecule this close or closer. A single distant column doesn't dominate, but several do.
- **Column count:** `training_n_columns` counts the columns that contributed.

Per-column distances and the 5 nearest training molecules (keys, similarities, labels) are reported in `training_details` for inspection.

### Planned

These planned training scores add information beyond the domain:
- **Training reliability** (needs `y`): how smooth the training labels are around the query.
- **Training fidelity** (needs `y` plus the model's predictions on its training molecules): the model's local error against those labels.
