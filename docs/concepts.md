# Concepts

`eosquality` judges a prediction in two modalities. Each is available only when its data was given at fit time.

**Reference modality.** It compares how an Ersilia model behaves on a query molecule with how the same model behaves on a fixed **reference library**: about 1.35M molecules shipped with each major version. The reference population is the model's own predictions on that library. It is **not** ground truth, so these scores describe how similar a query is to the model's behaviour on the reference. They don't estimate whether a prediction is correct.

**Training modality.** It compares a query with the model's **training sets**: one per output column. This modality can say whether the model has seen chemistry like the query. See [Training modality](#training-modality).

## Shared preprocessing

Every score starts from the same fit-time state (`shared/`):

1. **Scaling.** Each numeric output column is scaled with [`eosframes`](https://github.com/ersilia-os/eosframes). eosframes classifies each column (constant, binary, count, skewed or centred continuous) and maps it robustly into `[-1, 1]`.
2. **Feature selection.** If there are more than `max_features` columns (default 10), columns are clustered on `1 − |Pearson r|` with average linkage, and only the medoid of each cluster is kept. Typicality and extremity see only the selected columns. When training sets are given too, only the columns with a usable training set are candidates, and the training scores use the selected columns. A training-only fit has no predictions to correlate, so its columns are clustered on `1 − Jaccard` overlap of their training molecules instead, keeping the largest set of each cluster.

## Calibration

Typicality and extremity map a **raw value** to a **calibrated score** in `(0, 1]` through the reference's own distribution of that raw value. The CDF uses mid-ranks:

```
cdf(v) = (#{ref < v} + #{ref ≤ v}) / (2n)
score  = clip(cdf(v), 1/(2n), 1)
```

With mid-ranks, a raw value shared by many reference rows (common for one-output models or quantised typicality) lands in the middle of its tie block. Reference molecules therefore score close to Uniform(0, 1), and their mean is 0.5; this is the anchor (`reference_typicality_`, `reference_extremity_`, and the `ref_<score>_anchor` run metadata).

`docs/figures/reference_calibration.png` checks this on 1,000 molecules sampled from the library.

## Missing values

The same rule applies to the output-based scores:
- A NaN output feature carries no information, so it is left out of the per-row aggregate.
- A row with no usable feature gets a NaN score.
- Reference rows with no usable feature are left out of the calibration CDFs.

## The reference scores

| Score | Space | Raw value (`*_raw`) | High score means |
|---|---|---|---|
| Typicality | output | Q66 over features of per-feature density | outputs sit where the reference's outputs are dense |
| Extremity | output | Q66 over features of `min(\|scaled\|, 1)` | outputs sit far from the column centres |
| Match | structure | none: 1 / 0 flags | the molecule, or its scaffold, is in the reference library |

### Typicality (density)

Typicality asks how common each output value is among the reference library's values of that column.

- **Per feature.** At fit time each selected column is quantised to int8 (`round(scaled × 127)`) and a 256-slot count table is built from the reference. The density of a value is `count / max_count`, read from that table by linear interpolation between the two levels around the value, so it varies continuously: the most common level scores 1 and unseen levels score 0. For a binary column the majority class scores 1 and the minority class scores minority/majority; a constant column always scores 1.
- **Whole model, raw.** `ref_typicality_raw` is the 66th percentile (Q66) of the per-feature densities across the output columns, in [0, 1]. The Q66 keeps the value from collapsing towards the mean as columns are added. The densities of different columns are not comparable (a smooth column with many levels has low densities everywhere, a binary one does not), so the raw value is not comparable across models.
- **Per column, percentile.** Each column's density is placed on that column's own reference distribution (mid-rank percentile of the density among the reference library's values of that column; higher = more typical), so every column counts equally whatever its shape. The reference's own densities are kept as a 65536-bin histogram per column (saved), and the percentile table is derived from it. These are `<column>_typicality_pct` in the reference details file, next to `<column>_typicality_raw`.
- **Whole model, percentile.** `ref_typicality_pct` takes the Q66 of the per-column percentiles and maps it through the reference library's own distribution of that same statistic: ~0.5 for a typical reference molecule, near 1 for a molecule more typical than almost all of the library. The last step keeps the value uniform under the reference whatever the number of columns, so it is comparable across models.
- **Resolution.** The counts live on the int8 grid, 1/127 of the scaled range per level; between levels the density is interpolated. A column whose values tie (binary, constant) still gives ties, handled by mid-ranks.

### Extremity (position)

Extremity asks how far from the centre of its range each output sits.

- **Scaled values.** Each selected output column is first transformed by the eosframes scaler, a type-aware robust scaler fitted on the reference: the column's centre goes to 0, the bulk of the data lands well inside ±1, and tails approach ±1 smoothly instead of being cut, so distinct outliers keep distinct values (the exact body width depends on the column type). Binary columns become {0, 1}, and constant columns 0. Columns are therefore on a common scale, whatever their units.
- **Per feature.** `min(|scaled|, 1)`: 0 at the column centre, 1 at (or beyond) the rails.
- **Whole model, raw.** `ref_extremity_raw` is the 66th percentile (Q66) of the per-feature values across the output columns, in [0, 1]. A value of 0.8 means at least a third of the columns sit at 0.8 or beyond; 0 means every column is at its centre. The Q66 keeps the value from collapsing towards the mean as columns are added. Binary columns contribute 0 or 1 by construction, so for a model with several of them the raw value is coarse.
- **Per column, percentile.** Each column's value is placed on that column's own reference distribution (mid-rank percentile of `min(|scaled|, 1)` among the reference library's values of that column), so a column that is rarely extreme counts the same as one that often is. These are `<column>_extremity_pct` in the reference details file, next to `<column>_extremity_raw`.
- **Whole model, percentile.** `ref_extremity_pct` takes the Q66 of the per-column percentiles and maps it through the reference library's own distribution of that same statistic: ~0.5 for a typical reference molecule, 0.97 for one more extreme than 97% of the library. The last step keeps the value uniform under the reference whatever the number of columns, so it is comparable across models, which the raw value is not.

Extremity is position-based where typicality is density-based, and the two are complementary. In practice they are strongly anti-correlated for most models (see `docs/figures/score_correlations.png`).

### Match (is the molecule in the reference library?)

`ref_match` and `ref_scaffold` are exact lookups against the reference library's molecules, with no calibration and no dependence on the model. As in the training modality, both compare the **connectivity layer** of an InChIKey (its first 14 characters, the hash of the heavy-atom skeleton and hydrogens, blind to stereochemistry, isotopes, charge and mobile-hydrogen tautomerism). `ref_match` is 1 when the layer of the query (after standardisation: largest fragment, canonical isomeric SMILES) is among the library's, so an ionised form, enantiomer or salt of a library molecule counts. `ref_scaffold` does the same for the Murcko scaffold, and is empty, not 0, for a query with no scaffold; both are empty for an unparsable SMILES.

A match says the compound is in the reference library, not that the model's prediction for it is right.

The keys are computed once per library by `eosquality build` and shipped with it, not per model. The artifacts keep only the counts and find the library again when they run.

## Training modality

Each output column of a model may have its own training set: SMILES. Training SMILES are standardised (largest fragment, then canonical isomeric SMILES) and duplicates are merged. Each distinct training set gets a Morgan fingerprint index (radius 2, 2048 bits); columns measured on the same molecules, such as one screening panel, share it.

### Training distance

Training distance asks how far the query is from the molecules the model was trained on. It gives **one value per molecule for the whole model**. It is the classic kNN applicability domain: the mean similarity to the 5 nearest training molecules is among the best structural predictors of prediction error (Sheridan et al., *J. Chem. Inf. Comput. Sci.* 2004). Following the kNN domain of Tropsha and the OECD principle that a domain is judged against the training set itself, the query is compared with how close training molecules are to each other. There is no in/out cutoff: both a raw distance and a calibrated percentile are reported.

The value is built from each output column's own training set; the sets are **not pooled**:
- **Raw, per column:** the mean Tanimoto similarity (Morgan, radius 2, 2048 bits) between the query and its **5 nearest training molecules**; internally the distance `1 −` that similarity is calibrated. A query that is itself a training molecule (same standardised SMILES) drops its own entry, so it gets its leave-one-out value.
- **Calibrated, per column:** the mid-rank percentile of the raw value among the column's **leave-one-out** raw values, where each training molecule is compared with its 5 nearest *other* training molecules. This is a distance percentile, higher is farther; the published column is its similarity (1 minus it).
- **Whole model:** `trn_tanimoto_pct` is 1 − the **66th percentile** of the per-column calibrated distances, so higher is closer: about 0.5 means as close as a typical training molecule, and near 0 means farther than almost all of them. `trn_tanimoto_raw` is the similarity at the same point (1 − the 66th percentile of the per-column distances): at least two-thirds of the columns are this close or closer. A single distant column doesn't dominate, but several do.

Why not pool the training sets into one? Pooled, a large training set could hide that the query is far from a small one. Each calibrated per-column value is a percentile of that column's own training set, so columns of very different sizes and densities combine fairly.

How to read the two values: both are similarities, higher is closer. The raw value means the same thing across models: as a rough guide, 0.4 or less (a mean Tanimoto similarity that low) means no related training chemistry. The calibrated value is relative to how dense the training sets are, so a diverse training set makes the same raw distance look more typical. Read them together.

`trn_in_training` (details file) flags a query that is a training molecule of any column. `training_details` lists, per query, the 5 nearest training molecules over all columns (keys, similarities, and the columns each belongs to).

### Exact-structure and scaffold match

`trn_match` and `trn_scaffold` are exact lookups, like `ref_match`, over every training molecule of every column, with no calibration. Both compare the **connectivity layer** of an InChIKey, its first 14 characters: the hash of the heavy-atom skeleton and hydrogens, blind to stereochemistry, isotopes, charge and (standard InChI) mobile-hydrogen tautomerism. `trn_match` is 1 when the query's layer is in the training sets, so the same compound counts whether it is an ionised form, an enantiomer or a tautomer; on eos4e40 the lowest-similarity matches were a thiol-sulfonate against its anion and keto/enol pairs of allopurinol and theophylline. Because a Murcko scaffold keeps exocyclic double bonds, a tautomer pair can match on `trn_match` while its scaffolds differ, giving `trn_match` 1 with `trn_scaffold` 0. `trn_scaffold` does the same for the Murcko scaffold, and is empty, not 0, for a query with no scaffold (an acyclic molecule), because the question has no answer there. Both are 1 / 0 and empty for an unparsable SMILES. A match says the compound was seen, not that the prediction for it is right.

### Physchem applicability domain

**Precedent.** The recipe is a standard one. Sahigara et al. (*J. Cheminform.* 2013, 5, 27) build a k-nearest-neighbour domain on "autoscaled Euclidean distances", and autoscale for every method but the bounding box and convex hull. Klingspohn et al. (*J. Cheminform.* 2017, 9, 44) include a DA-index, γ, defined as the mean distance to the k = 5 nearest training neighbours, with Euclidean distance on auto-scaled MOE (181) and E-State (192) descriptors, which is our measure at the same scale. In published practice the raw distance is never read alone: it is compared with the training set's own distances (a `mean + Z·σ` cutoff, or a per-sample threshold), which is what `trn_physchem_pct` does. So a raw distance is anchored on the library's own random-pair median to give the similarity.

**Dimensionality.** Distances in many dimensions concentrate (Beyer et al., *ICDT* 1999, who show the effect at as few as 10-15 dimensions), and Aggarwal et al. (*ICDT* 2001) find L1 and fractional norms more meaningful than L2. We checked it on eos4e40's 2,304 training molecules: the 217 descriptors have an effective dimension of about 16 (participation ratio; 64 components for 90% of the variance), and the farthest training molecule is a median 18 times as far as the nearest, with the 5th neighbour 1.4 times as far as the 1st. L1 (25, 1.65) and PCA to 10 or 30 components (38 and 27; 1.6 and 1.5) do not give a materially better separation, so we kept L2 on the full set. One dataset only.

`trn_tanimoto` asks whether the query resembles a *specific* training
molecule. This asks a different question with the same method, in a different
space: do its bulk properties fall where the training set's do?

Per output column, the 217 RDKit physchem descriptors are scaled with the
**reference library's scaler** (median imputation, then mean and standard
deviation over its 1.35M molecules; a copy ships with the package as
`library/physchem_scaler.json`) and clipped to ±10, so every model shares one
descriptor space and distances are in library standard deviations.
Clipping is needed because the scaler is not robust: `Ipc` grows exponentially
with molecule size and reached z = 2.7×10²⁶ on eos4e40, and a single
occurrence of a rare fragment count such as `fr_isothiocyan` gives z ≈ 100.
The **mean Euclidean distance to the 5 nearest training molecules** is
`trn_physchem_dist` (details file). Two scores come from it. `trn_physchem_pct`
is the similarity percentile: 1 − the mid-rank percentile of the distance among
the training molecules' own leave-one-out distances, so ~0.5 means "as ordinary
as a typical training molecule" and near 0 means "further out than almost all
of them". `trn_physchem_raw` is a
**similarity**, `1 − d / 18.70`, where 18.70 is the median distance between two
random reference-library molecules in this space (1,000,000 random pairs; recomputed
by `scripts/physchem_pair_median.py`). It is 1 for an identical molecule, 0 for one
no closer than a random pair, and deliberately not clipped, so it goes negative
for a molecule further away than a random pair and keeps its ordering. On
eos4e40 a training molecule's median distance of 7.2 gives 0.61, and the 1,000
drug-like queries have a median of 0.49. The percentile is model-relative
("unusual for this training set"); the similarity is the same in every model
("how close in absolute terms"). The whole-model value is the 66th percentile of
the distance across columns, as for training distance, then converted.

Why the same method twice, rather than something cleverer: distance to the
training set is the oldest applicability-domain signal, and Sheridan et al.
found the similarity to the nearest training molecule, and the number of
neighbours within a cutoff, to be the best discriminators of prediction
accuracy across 20 activity sets — a result they report as holding whichever
QSAR method and descriptor is used ("Similarity to Molecules in the Training
Set Is a Good Discriminator for Prediction Accuracy in QSAR", *J. Chem. Inf.
Comput. Sci.* 2004, 44(6), 1912–1928). k-NN Euclidean distance over autoscaled
descriptors is the standard form of it (Sahigara et al., *J. Cheminform.* 2013,
5, 27; Aniceto et al., *J. Cheminform.* 2016, 8, 69).

OECD guidance asks for the structural and physicochemical domains to be
described separately rather than chosen between: "the different AD methods
should not be seen as in competition with one another, since the combined use
of multiple AD methods should give a higher assurance that query chemicals are
predicted accurately" (ENV/JM/MONO(2007)2 ¶128).

Note what is *not* claimed. No primary source we could find demonstrates that
descriptor-space and fingerprint AD are empirically complementary; the one
head-to-head we could read found Euclidean novelty measures ranked slightly
better than Tanimoto ones but that "most of the rankings according to novelty
measures were not significantly different from chance" (Klingspohn et al.,
*J. Cheminform.* 2017, 9, 44). What we can say is measured here: on eos4e40
against the drugs query set the two correlate at ρ 0.54, so a molecule can be
structurally novel while physicochemically ordinary, or the reverse.

Three implementation details, each with a reason:

- **Standardise first.** Raw Euclidean distance is dominated by whichever
  descriptor has the largest scale — molecular weight swamps a 0–1 fraction.
  Non-finite cells are replaced by the training median first, since RDKit
  returns NaN for a descriptor that overflows or fails.
- **Exclude the self match.** The calibration table is each training
  molecule's distance to its 5 nearest *other* training molecules, and a query
  that is itself a training molecule drops its own zero-distance neighbour —
  the same rule the fingerprint side applies. Without it every training
  molecule's apparent distance would be halved and queries would look far more
  novel than they are.
- **No cutoff.** A widely used threshold is `D = d̄ + Z·σ` over the training
  distances with `Z = 0.5`, traceable to Golbraikh et al. (*J. Comput. Aided
  Mol. Des.* 2003, 17, 241–253) and still quoted as "commonly taken" by
  Rakhimbekova et al. (*Int. J. Mol. Sci.* 2020, 21, 5542). It is *not*,
  despite common belief, prescribed by the OECD guidance, which reviews
  descriptor ranges, convex hull, centroid distance, leverage and density
  methods but contains no k-NN cutoff and endorses no specific method
  (ENV/JM/MONO(2007)2 ch. 4). We report the percentile instead, which carries
  the same information without assuming the distances are normal. The mapping
  is direct: on eos4e40 `Z = 0.5` falls at the 65th percentile of the training
  distribution, `Z = 1.0` at the 81st, `Z = 2.0` at the 100th — so
  `trn_tanimoto_pct <= 0.65` reproduces the conventional flag.

**An ellipsoid-style measure was implemented first and removed.** Hotelling's
T² and DModX on physchem principal components were tried, and were more
complementary to `trn_tanimoto` (ρ 0.09 against 0.54). They were dropped
because the evidence does not support them: leverage, to which T² is
proportional, is an ordinary-least-squares construct — "The original leverage
can only be applied in the case of a linear regression, but not for non-linear
regression" (Dutschmann, Schlenker & Baumann, *Mol. Inf.* 2024, 43,
e202400018) — and T² is a latent-variable construct — "Hotelling's T2 is usually based on
calculation in score space of latent variable projection methods", and DModX
is the PLS X-residual, which a random forest does not have at all (Eriksson et
al., *Environ. Health Perspect.* 2003, 111(10), 1361–1375; the same paper notes
"the leverage h and Hotelling's T2 are, apart from a proportionality constant,
identical"). Empirically, a thresholded Mahalanobis-to-centroid domain made
external Q² *worse* than applying no domain at all, 0.797 → 0.791, while
discarding 6 of 95 test compounds — and the model there was a radial-basis
neural network, so this is a non-linear case rather than a linear-model
artefact (Sahigara et al., *J. Cheminform.* 2013, 5, 27). The trade is
deliberate: less orthogonality, better evidential support.

Honesty about how strong that support is: in the same table the *classical*
k-NN domain scored 0.797, identical to using no domain, and the best method
reached only 0.803. This evidence is a clean knockout for the ellipsoid
family, not a demonstration that k-NN domains earn their keep.

**Cost.** The standardised training matrix is kept with the artifact, because
a nearest-neighbour domain needs the reference molecules themselves, not a
summary of them: about 34 MB for a 39,000-molecule column. Computing the
descriptors runs at 3–5 ms per molecule.
