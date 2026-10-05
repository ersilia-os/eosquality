# Architecture

## Fit

```mermaid
flowchart LR
    subgraph inputs[Inputs]
        REF["Model predictions on the<br/>reference library<br/><i>key, input, outputs…</i>"]
        LIB[("Reference library<br/>Morgan FPSim2 index · self-kNN<br/>physchem · MACCS")]
    end

    REF --> SH["<b>shared/</b><br/>schema · eosframes scaler<br/>feature selection (≤10 medoids)<br/>80/10/10 split · scaled ref matrix"]
    LIB -- "self-kNN, k" --> KNN["<b>knn/</b><br/>k neighbours per ref row<br/>mean FP distance"]
    SH --> KNN

    SH --> TYP["<b>Typicality</b><br/>int8 density LUTs<br/>CDF of Q66"]
    SH --> EXT["<b>Extremity</b><br/>|scaled| position<br/>CDF of Q66"]
    LIB -- "nearest analogue" --> SUP["<b>Support</b><br/>Tanimoto similarity of<br/>nearest library analogue · CDF"]
    KNN --> CON["<b>Consistency</b><br/>output L1 to FP neighbours<br/>CDF per FP-distance bin"]
    SH --> CON
    SH --> SIG["<b>Signal</b> (opt-in)<br/>XGBoost descriptor → outputs<br/>CDF of |SHAP| Gini on val"]
    LIB -- "physchem / MACCS" --> SIG
```

## Fit — training modality

```mermaid
flowchart LR
    TR["training_eosXXXX_vN/<br/><i>&lt;column&gt;.csv: smiles, y?, key?</i>"] --> LD["load + standardise<br/>largest fragment · canonical<br/>merge duplicates"]
    LD --> IDX["<b>training/</b><br/>one Morgan index per column<br/>(self-kNN = leave-one-out)"]
    IDX --> TD["<b>Training distance</b><br/>1 − Tanimoto to the nearest<br/>training molecule (uncalibrated)"]
```

## Run

```mermaid
flowchart LR
    Q["Query predictions<br/><i>key, input, outputs…</i>"] --> SCALE["validate schema<br/>scale + select features<br/>(once)"]
    Q --> FPQ["FPSim2 top-(k+1)<br/>drop self match<br/>(once)"]
    LIB[("Reference library")] --> FPQ
    SCALE --> TYP[Typicality] & EXT[Extremity] & CON[Consistency]
    FPQ --> SUP[Support] & CON
    Q -- SMILES --> SIG["Signal<br/>descriptor → SHAP → Gini"]
    Q -- SMILES --> TDR["Training distance<br/>per column → 66th percentile"]
    TDR --> DET["training_details.csv<br/>query × column · 5 nearest training molecules"]
    TYP & EXT & SUP & CON & SIG & TDR --> OUT["scores.csv<br/>score + score_raw per component<br/>(+ support_log)"]
```

## Save layout

One subfolder per modality; either or both may be present.

```
<artifacts>/
  manifest.json                           # informational: eos_id, version, modalities, scores
  reference_mode/                         # iff fitted with --reference
    shared/
      schema.json  scaler.json  binary_class_freq.json
      metadata.json                       # n_samples, library_id, vector_index_path, format_version, …
      reference_ids.json  splits.json  selected_columns.json
      reference_repr.npy                  # (n_ref, n_selected) scaled reference
    knn/state.json                        # {"k": …}; iff support or consistency
    typicality/   state.json  reference_self_aggregates.npy  metadata.json
    extremity/    state.json  reference_self_aggregates.npy  metadata.json
    support/      state.json  reference_nearest_similarities.npy  metadata.json
    consistency/  state.json  reference_self_distances_per_bin.npz  metadata.json
    signal/       learner.json  learner.ubj  umbrella.json  reference_self_aggregates.npy
                  physchem_scaler.json (physchem only)  val_shap_attributions.npy  metadata.json
  training_mode/                          # iff fitted with --training
    training_sets/
      metadata.json                       # training_format_version, eos_id, version, columns
      columns.json  arrays.npz            # per column: n, y_kind, ids, y, predictions
      indices/c000/ …                     # one VectorIndex per output column
    training_distance/  state.json  metadata.json
```

Each component's `metadata.json` records only `component`, `fit_timestamp`, `fit_duration_seconds` and `k`.

A standalone score class (e.g. `Support().fit(...).save(folder)`) writes its own subfolder plus the `shared/` (and `knn/`) folders it needs directly into `folder`. That is the same structure as one `reference_mode/`.
