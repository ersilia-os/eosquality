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

## Run

```mermaid
flowchart LR
    Q["Query predictions<br/><i>key, input, outputs…</i>"] --> SCALE["validate schema<br/>scale + select features<br/>(once)"]
    Q --> FPQ["FPSim2 top-(k+1)<br/>drop self match<br/>(once)"]
    LIB[("Reference library")] --> FPQ
    SCALE --> TYP[Typicality] & EXT[Extremity] & CON[Consistency]
    FPQ --> SUP[Support] & CON
    Q -- SMILES --> SIG["Signal<br/>descriptor → SHAP → Gini"]
    TYP & EXT & SUP & CON & SIG --> OUT["scores.csv<br/>score + score_raw per component<br/>(+ support_log)"]
```

## Save layout

```
<root>/
  manifest.json                         # informational summary (format_version, scores, k, library)
  shared/                               # always
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
```

Each component's `metadata.json` records only `component`, `fit_timestamp`, `fit_duration_seconds` and `k`.
