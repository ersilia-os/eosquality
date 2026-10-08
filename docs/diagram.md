# Architecture

## Fit

```mermaid
flowchart LR
    subgraph inputs[Inputs]
        REF["Model predictions on the<br/>reference library<br/><i>key, input, outputs…</i>"]
        LIB[("Reference library folder<br/>smiles.csv · metadata.json<br/>connectivity_keys.npz")]
    end

    REF --> SH["<b>shared/</b><br/>schema · eosframes scaler<br/>feature selection (≤10 medoids)"]
    LIB -- "check: same molecules, same order" --> SH

    SH --> TYP["<b>Typicality</b><br/>int8 density LUTs<br/>per-column pct, Q66, CDF"]
    SH --> EXT["<b>Extremity</b><br/>|scaled| position<br/>per-column pct, Q66, CDF"]
    LIB -- "connectivity keys" --> MAT["<b>Match</b> (ref_match, ref_scaffold)<br/>keys read from the library;<br/>the artifact keeps only the counts"]
    SH --> MAT
```

## Fit — training modality

```mermaid
flowchart LR
    TR["training_eosXXXX_vN/<br/><i>&lt;column&gt;.csv: smiles, key?</i>"] --> LD["load + standardise<br/>largest fragment · canonical<br/>merge duplicates<br/>one point per Morgan fingerprint"]
    LD --> SEL["select ≤ max_features columns<br/>(the reference's selection, or<br/>least-overlapping training sets)"]
    SEL --> IDX["<b>training/</b><br/>a Morgan index per distinct training set<br/>(self-kNN = leave-one-out)"]
    IDX --> TD["<b>Training distance</b> (trn_tanimoto)<br/>mean Morgan distance to the 5 nearest<br/>training molecules per column,<br/>Q66 → one value (pct + raw)"]
    LD --> TP["<b>Training physchem</b> (trn_physchem)<br/>217 descriptors, library scaler, clip ±10<br/>mean distance to the 5 nearest per column,<br/>Q66 → one value (pct + raw)"]
    LD --> TM["<b>Training match</b> (trn_match, trn_scaffold)<br/>connectivity layers of the molecules<br/>and of their Murcko scaffolds"]
```

## Run

```mermaid
flowchart LR
    Q["Query predictions<br/><i>key, input, outputs…</i>"] --> SCALE["validate schema<br/>scale + select features<br/>(once)"]
    LIB[("Reference library<br/>connectivity keys")] --> MAT
    SCALE --> TYP[Typicality] & EXT[Extremity]
    Q -- SMILES --> MAT["Match<br/>standardise · connectivity layers<br/>set lookup"]
    Q -- SMILES --> TQ["TrainingQuery (once)<br/>standardise · physchem<br/>per-column kNN"]
    TQ --> TDR["Training distance<br/>per column → 66th percentile"]
    TQ --> TPH["Training physchem<br/>per column → 66th percentile"]
    TQ --> TMA["Training match<br/>connectivity-layer lookup"]
    TDR & TPH & TMA --> DET["&lt;output&gt;.training_details.csv<br/>one row per query · 5 nearest training molecules"]
    TYP & EXT --> RDET["&lt;output&gt;.reference_details.csv<br/>per-column typicality and extremity"]
    TYP & EXT & MAT & TDR & TPH & TMA --> OUT["&lt;output&gt;.csv<br/>ref_* and trn_* columns: pct + raw<br/>(+ match and scaffold flags)"]
```

## Save layout

One subfolder per modality; either or both may be present.

```
<artifacts>/
  manifest.json                           # informational: eos_id, version, modalities, scores
  reference_mode/                         # iff fitted with -r/--reference
    shared/
      schema.json  scaler.json
      metadata.json                       # n_samples, library_id, library_path, format_version, …
      selected_columns.json
    typicality/   state.json  count_luts.npy  reference_self_aggregates.npy  metadata.json
    extremity/    state.json  column_tables.npz  reference_self_aggregates.npy  metadata.json
    match/        state.json  metadata.json        # counts only; the keys live in the library folder
  training_mode/                          # iff fitted with -t/--training-sets
    training_sets/
      metadata.json                       # training_format_version, eos_id, version, columns
      columns.json  arrays.npz            # per column: n, ids
      indices/c000/ …                     # one VectorIndex per distinct training set
    training_distance/  state.json  loo_mean_distances.npz  metadata.json  # trn_tanimoto
    training_physchem/  state.json  c000/ …  # trn_physchem: one domain per distinct training set
    training_match/  connectivity_keys.npz  metadata.json  # trn_match, trn_scaffold
```

Each component's `metadata.json` records only `component`, `fit_timestamp`, and `fit_duration_seconds`.

A standalone score class (e.g. `Typicality().fit(...).save(folder)`) writes its own subfolder plus the `shared/` folder it needs directly into `folder`. That is the same structure as one `reference_mode/`.
