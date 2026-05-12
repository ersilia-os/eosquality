# `eosquality` at a glance

High-level view of how `eosquality` turns a reference dataset and a query into per-sample quality scores.

## Workflow

```mermaid
flowchart LR
    Ref[Reference dataset]
    Lib[Reference library]
    Fit([Fit])
    Art[Artifacts]
    Query[Query dataset]
    Run([Run])
    Scores[Quality scores]

    Ref --> Fit
    Lib --> Fit
    Fit --> Art
    Art --> Run
    Query --> Run
    Run --> Scores

    classDef input fill:#8CC8FA,stroke:#50285A,color:#2C3E50
    classDef process fill:#50285A,stroke:#50285A,color:#FFFFFF
    classDef artifact fill:#BEE6B4,stroke:#50285A,color:#2C3E50
    classDef output fill:#AA96FA,stroke:#50285A,color:#2C3E50

    class Ref,Lib,Query input
    class Fit,Run process
    class Art artifact
    class Scores output
```

**Fit** learns per-column normalization and reference diagnostics from a reference dataset, anchored to a canonical reference library. **Run** scores each query sample against the fitted reference.

## Score composition

```mermaid
flowchart TB
    Support[Support]
    Typicality[Typicality]
    Consistency[Consistency]
    Quality[Quality score]

    Support --> Quality
    Typicality --> Quality
    Consistency --> Quality

    classDef component fill:#BEE6B4,stroke:#50285A,color:#2C3E50
    classDef headline fill:#50285A,stroke:#50285A,color:#FFFFFF

    class Support,Typicality,Consistency component
    class Quality headline
```

**Quality** is the geometric mean of three complementary signals: *support* (closeness to neighbors), *typicality* (plausibility of feature values), and *consistency* (uniformity of neighborhoods).
