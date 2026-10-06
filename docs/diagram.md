# Architecture

## Fit

```mermaid
flowchart LR
    subgraph inputs[Inputs]
        REF["Model predictions on the<br/>reference library<br/><i>key, input, outputs…</i>"]
        LIB[("Reference library<br/>Morgan FPSim2 index · self-kNN<br/>physchem · MACCS")]
    end

    REF --> SH["<b>shared/</b><br/>schema · eosframes scaler<br/>feature selection (≤10 medoids)<br/>80/10/10 split · scaled ref matrix"]
    LIB -- "self-kNN, k = 5" --> KNN["<b>knn/</b><br/>5 neighbours per ref row<br/>mean FP distance"]
    SH --> KNN

    SH --> TYP["<b>Typicality</b><br/>int8 density LUTs<br/>CDF of Q66"]
    SH --> EXT["<b>Extremity</b><br/>|scaled| position<br/>CDF of Q66"]
    LIB -- "nearest analogue" --> SUP["<b>Support</b><br/>Tanimoto similarity of<br/>nearest library analogue · CDF"]
    KNN --> CON["<b>Consistency</b><br/>output L1 to FP neighbours<br/>CDF per FP-distance bin"]
    SH --> CON
    SH --> SIG["<b>Signal</b> (provisional)<br/>XGBoost physchem → outputs<br/>CDF of |SHAP| Gini on val"]
    LIB -- "physchem" --> SIG
```

## Fit — training modality

```mermaid
flowchart LR
    TR["training_eosXXXX_vN/<br/><i>&lt;column&gt;.csv: smiles, y?/value?, key?</i>"] --> LD["load + standardise<br/>largest fragment · canonical<br/>merge duplicates"]
    LD --> SEL["select ≤ max_features columns<br/>(the reference's selection, or<br/>least-overlapping training sets)"]
    SEL --> IDX["<b>training/</b><br/>one Morgan index per column<br/>(self-kNN = leave-one-out)"]
    IDX --> TD["<b>Training distance</b><br/>mean distance to the 5 nearest<br/>training molecules per column,<br/>Q66 → one value (raw + calibrated)"]
    IDX --> TDF["<b>Training difficulty</b> (columns with y, ≤ 10k molecules)<br/>surrogate RF, scaffold CV → OOF errors<br/>error model on MACCS + kNN, KDE, variance + ŷ<br/>(UNIQUE feature set i) → calibrated rank,<br/>Q66 → one value"]
```

## Run

```mermaid
flowchart LR
    Q["Query predictions<br/><i>key, input, outputs…</i>"] --> SCALE["validate schema<br/>scale + select features<br/>(once)"]
    Q --> FPQ["FPSim2 top-(k+1)<br/>drop self match<br/>(once)"]
    LIB[("Reference library")] --> FPQ
    SCALE --> TYP[Typicality] & EXT[Extremity] & CON[Consistency]
    FPQ --> SUP[Support] & CON
    Q -- SMILES --> SIG["Signal<br/>physchem → SHAP → Gini"]
    Q -- SMILES --> TQ["TrainingQuery (once)<br/>standardise · MACCS · Morgan<br/>per-column kNN"]
    TQ --> TDR["Training distance<br/>per column → 66th percentile"]
    TQ --> TDF["Training difficulty<br/>error model per column → 66th percentile"]
    TDR & TDF --> DET["&lt;output&gt;.training_details.csv<br/>one row per query · 5 nearest training molecules"]
    TYP & EXT & SUP & CON & SIG & TDR & TDF --> OUT["&lt;output&gt;.csv<br/>ref_* and trn_* columns: score + score_raw<br/>(+ ref_support_log, trn_in_training)"]
```

## Save layout

One subfolder per modality; either or both may be present.

```
<artifacts>/
  manifest.json                           # informational: eos_id, version, modalities, scores
  reference_mode/                         # iff fitted with -r/--reference
    shared/
      schema.json  scaler.json  binary_class_freq.json
      metadata.json                       # n_samples, library_id, vector_index_path, format_version, …
      reference_ids.json  splits.json  selected_columns.json
      reference_repr.npy                  # (n_ref, n_selected) scaled reference
    knn/state.json                        # {"k": 5}; iff support or consistency
    typicality/   state.json  reference_self_aggregates.npy  metadata.json
    extremity/    state.json  reference_self_aggregates.npy  metadata.json
    support/      state.json  reference_nearest_similarities.npy  metadata.json
    consistency/  state.json  reference_self_distances_per_bin.npz  metadata.json
    signal/       learner.json  learner.ubj  umbrella.json  reference_self_aggregates.npy
                  physchem_scaler.json  val_shap_attributions.npy  metadata.json
  training_mode/                          # iff fitted with -t/--training-sets
    training_sets/
      metadata.json                       # training_format_version, eos_id, version, columns
      columns.json  arrays.npz            # per column: n, y_kind, ids, y, predictions
      indices/c000/ …                     # one VectorIndex per output column
    training_distance/  state.json  loo_mean_distances.npz  metadata.json  # trn_distance
    training_difficulty/  state.json  metadata.json  # trn_difficulty; iff some column has ≥ 50 labels
      c000/ …           surrogate.joblib  density.joblib  error_model.joblib
                        arrays.npz  state.json      # one folder per labelled column
```

Each component's `metadata.json` records only `component`, `fit_timestamp`, `fit_duration_seconds` and `k`.

A standalone score class (e.g. `Support().fit(...).save(folder)`) writes its own subfolder plus the `shared/` (and `knn/`) folders it needs directly into `folder`. That is the same structure as one `reference_mode/`.
