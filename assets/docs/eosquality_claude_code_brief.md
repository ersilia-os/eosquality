# eosquality: Claude Code implementation brief

This document is written to be used directly with Claude Code to scaffold the `eosquality` repository.

Its purpose is not to restate the research background in full, but to translate the design into an implementable repository plan with concrete interfaces, sequencing, file responsibilities, and coding conventions.

---

## Project goal

Build a Python package called `eosquality` that assesses the quality of query data against a fitted reference population.

The package should support a **fit/run** workflow:

- `fit(reference_data)` builds all reference-dependent artifacts
- `run(query_data)` scores new samples against those artifacts

The system must support:

- one or many columns
- arbitrary numeric distributions
- mixed data types
- optional embedding/vector blocks
- scalable nearest-neighbor search
- discovery of high-reliability subsets inside the reference population

The package should be designed for extensibility and scientific transparency, not only convenience.

---

## Core product semantics

Externally, the package is about **quality**.

Internally, keep the decomposition explicit:

- `reference_quality`: quality of the reference population
- `core_membership`: whether a reference point belongs to a high-reliability subset
- `support_score`: how well a query point is supported by the reference
- `consistency_score`: how coherent the local neighborhood is
- `stability_score`: how robust the score is under reference resampling
- `quality_score`: final pointwise aggregate returned to the user

Do not represent the main score as a probability of correctness.

---

## Top-level design

The package should have four main layers:

1. **Schema and preprocessing**
2. **Distance and neighbors**
3. **Reference fitting**
4. **Inference / scoring**

### Fit phase

`fit()` should compute and persist everything needed for efficient inference:

- schema inference or validation
- fitted transforms
- block definitions and feature weights
- distance engine configuration
- reference feature representation
- nearest-neighbor index
- reference kNN graph
- local support statistics on the reference
- optional density estimators
- reference-quality diagnostics
- clustering / core detection
- soft core membership weights
- optional bootstrap-based stability summaries

### Run phase

`run()` should:

- validate incoming schema
- apply fitted transforms
- query neighbors from the fitted reference
- compute pointwise support / consistency / stability
- use the core subset if configured
- return scores and explanations

---

## Recommended naming and public API

Use `fit` and `run` publicly. Internally, `score` can be an alias or helper.

Public API should remain small.

### Main class

```python
from eosquality import EOSQuality

eq = EOSQuality(
    k=20,
    ann_backend="auto",
    core_method="hdbscan",
    bootstrap_B=50,
)

eq.fit(reference_df)
result = eq.run(query_df)
```

### Optional advanced API

```python
eq.reference_report_
eq.core_report_
eq.config_
eq.save("artifacts/eosquality_ref")
eq = EOSQuality.load("artifacts/eosquality_ref")
```

### Public methods

```python
class EOSQuality:
    def __init__(self, ...): ...
    def fit(self, reference): ...
    def run(self, query): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

### Public attributes after fit

- `is_fitted_`
- `schema_`
- `config_`
- `reference_quality_`
- `reference_report_`
- `core_report_`

### Public result object from run

Prefer a lightweight dataclass rather than a raw dict.

```python
@dataclass
class RunResult:
    scores: pd.DataFrame
    metadata: dict[str, Any]
```

`scores` should contain one row per query sample, with columns such as:

- `quality_score`
- `support_score`
- `consistency_score`
- `stability_score`
- `core_support_score`
- `distance_k_mean`
- `distance_k_max`
- `effective_feature_fraction`
- `nearest_reference_ids`

---

## Concrete repository structure

```text
eosquality/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── docs/
│   ├── concepts.md
│   ├── architecture.md
│   ├── api.md
│   ├── fit_run_workflow.md
│   └── tutorials/
│       ├── quickstart.md
│       ├── mixed_types.md
│       ├── scalable_neighbors.md
│       └── interpreting_scores.md
├── examples/
│   ├── quickstart.py
│   ├── mixed_tabular.py
│   ├── embeddings_only.py
│   ├── mixed_blocks.py
│   └── save_load.py
├── tests/
│   ├── conftest.py
│   ├── test_smoke_fit_run.py
│   ├── test_schema.py
│   ├── test_preprocessing.py
│   ├── test_distance_engine.py
│   ├── test_neighbors.py
│   ├── test_reference_quality.py
│   ├── test_core_detection.py
│   ├── test_stability.py
│   ├── test_serialization.py
│   └── test_mixed_types_end_to_end.py
├── eosquality/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── exceptions.py
│   ├── typing.py
│   ├── version.py
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── infer.py
│   │   └── validate.py
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── numeric.py
│   │   ├── categorical.py
│   │   ├── counts.py
│   │   ├── vectors.py
│   │   └── pipeline.py
│   ├── distance/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── numeric.py
│   │   ├── categorical.py
│   │   ├── binary.py
│   │   ├── counts.py
│   │   ├── vectors.py
│   │   ├── mixed.py
│   │   └── engine.py
│   ├── neighbors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sklearn_backend.py
│   │   ├── faiss_backend.py
│   │   ├── graph.py
│   │   └── shared_neighbors.py
│   ├── reference/
│   │   ├── __init__.py
│   │   ├── fit_state.py
│   │   ├── diagnostics.py
│   │   ├── cohesion.py
│   │   ├── null_models.py
│   │   ├── shift.py
│   │   └── reports.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── detect.py
│   │   ├── score.py
│   │   └── bootstrap.py
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── support.py
│   │   ├── consistency.py
│   │   ├── stability.py
│   │   ├── aggregate.py
│   │   └── run.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── save.py
│   │   └── load.py
│   └── utils/
│       ├── __init__.py
│       ├── arrays.py
│       ├── stats.py
│       ├── logging.py
│       └── random.py
└── .github/
    └── workflows/
        ├── ci.yml
        └── publish.yml
```

---

## File-by-file intent

### `api.py`
Contains the main `EOSQuality` class and nothing too low-level.

Responsibilities:
- own the public fit/run interface
- orchestrate the pipeline
- expose fitted attributes
- delegate to internal modules

### `config.py`
Use structured config dataclasses.

Examples:
- `NeighborConfig`
- `DistanceConfig`
- `CoreConfig`
- `BootstrapConfig`
- `AggregationConfig`
- `EOSQualityConfig`

Avoid a single giant dict.

### `schema/models.py`
Define schema dataclasses:

```python
@dataclass
class ColumnSpec:
    name: str
    kind: str  # numeric, binary, categorical, count, vector
    weight: float = 1.0
    missing_policy: str = "ignore"
    block: str | None = None
```

### `schema/infer.py`
Infer schema from pandas DataFrame where possible, but allow users to override explicitly.

### `preprocess/pipeline.py`
Fit and apply transforms blockwise.

Should return:
- transformed representation(s)
- fitted scalers
- feature metadata

### `distance/engine.py`
Central object that composes all per-type distances.

Should support:
- pairwise distances
- query-to-reference distances
- per-feature contribution reporting

### `neighbors/*`
Encapsulate neighbor search backend details.

Need a unified interface so the rest of the code does not care whether the backend is sklearn or FAISS.

### `reference/fit_state.py`
Define the persisted fitted state.

This should be the heart of `save()` / `load()`.

Suggested dataclass:

```python
@dataclass
class FitState:
    config: EOSQualityConfig
    schema: Any
    preprocess_artifacts: Any
    reference_ids: list[Any]
    reference_repr: Any
    neighbor_index: Any
    neighbor_graph: Any
    reference_knn_stats: Any
    reference_quality: float
    reference_report: dict[str, Any]
    core_report: dict[str, Any]
    core_membership: Any
```

### `core/detect.py`
Implements discovery of high-reliability subsets.

First version should support:
- HDBSCAN if installed
- fallback mode based on density thresholding and connected components

### `scoring/run.py`
Runs inference over query samples.

Should produce raw components and final aggregates.

---

## Implementation priorities

Build in phases.

### Phase 1: minimal viable fit/run
Goal: an end-to-end usable package on numeric or mixed tabular data.

Implement:
- schema
- preprocessing
- mixed distance engine
- sklearn neighbor backend
- reference kNN stats
- support score
- simple reference quality
- simple run output
- serialization

### Phase 2: core detection and stability
Implement:
- HDBSCAN integration
- bootstrap reference diagnostics
- core subset discovery
- core-weighted support
- stability score

### Phase 3: scaling and advanced diagnostics
Implement:
- FAISS backend
- shared-neighbor graph
- null-model tests
- shift metrics
- density-ratio proxy

### Phase 4: polish
Implement:
- docs
- benchmarks
- examples
- better result reporting
- CLI if needed

---

## First-pass algorithmic defaults

These defaults should exist in code and docs.

### Distance defaults
- numeric: robust-scaled absolute distance
- count: `log1p` then robust-scaled absolute distance
- binary: Hamming
- categorical: mismatch indicator
- vector blocks: cosine distance if normalized, otherwise Euclidean
- mixed: weighted average of normalized per-feature distances

### Missingness defaults
- default policy: ignore missing pairwise for that feature
- track `effective_feature_fraction`
- downweight scores when effective feature fraction is too small

### Neighbor defaults
- `k = 20`
- backend = sklearn for small data
- backend = faiss for vector-heavy large data when available

### Core defaults
- primary method: HDBSCAN
- fallback: top density quantile + connected component filtering

### Aggregation defaults
Use geometric mean of bounded component scores.

Example:
```python
quality_score = (
    support_score
    * consistency_score
    * stability_score
) ** (1 / 3)
```

Then gate by `reference_quality`.

---

## Fit workflow in detail

Claude Code should implement `EOSQuality.fit()` roughly as follows:

```python
def fit(self, reference):
    # 1. validate input
    # 2. infer or validate schema
    # 3. fit preprocessing transforms
    # 4. transform reference data
    # 5. build distance engine
    # 6. build neighbor index
    # 7. build kNN graph
    # 8. compute reference local statistics
    # 9. compute reference quality diagnostics
    # 10. detect core / reliable subset
    # 11. optionally run bootstrap stability
    # 12. persist fit state
    # 13. expose fitted attributes
    return self
```

### Artifacts computed at fit time

Mandatory:
- transformed reference representation
- neighbor index
- kNN distances and indices for reference self-neighborhoods
- reference local support statistics
- reference quality metrics
- core membership

Optional:
- bootstrap results
- null model statistics
- precomputed density ratios or classifiers

---

## Run workflow in detail

Claude Code should implement `EOSQuality.run()` roughly as follows:

```python
def run(self, query):
    # 1. validate fit state
    # 2. validate query schema
    # 3. transform query using fitted preprocessing
    # 4. query nearest neighbors
    # 5. compute support score
    # 6. compute local consistency
    # 7. compute stability if enabled
    # 8. compute core-aware support if enabled
    # 9. aggregate into quality_score
    # 10. build result object
    return RunResult(...)
```

### Minimum outputs from run

Per sample:
- `quality_score`
- `support_score`
- `consistency_score`
- `distance_k_mean`
- `distance_k_max`
- `effective_feature_fraction`

If core detection is fitted:
- `core_support_score`
- `nearest_core_fraction`

If bootstrap is fitted:
- `stability_score`

---

## Suggested internal data structures

Prefer dataclasses over loose dicts.

### Config dataclasses

```python
@dataclass
class DistanceConfig:
    mixed_strategy: str = "weighted_mean"
    vector_metric: str = "cosine"
    numeric_scaler: str = "robust"

@dataclass
class NeighborConfig:
    k: int = 20
    backend: str = "auto"

@dataclass
class CoreConfig:
    method: str = "hdbscan"
    min_cluster_size: int = 30
    enabled: bool = True

@dataclass
class BootstrapConfig:
    enabled: bool = False
    n_resamples: int = 50
    subsample_fraction: float = 0.8
```

### Result dataclasses

```python
@dataclass
class ReferenceReport:
    reference_quality: float
    cohesion_score: float
    fragmentation_score: float
    stability_score: float | None
    notes: list[str]

@dataclass
class CoreReport:
    enabled: bool
    n_core_points: int
    core_fraction: float
    n_clusters: int
    notes: list[str]
```

---

## What Claude Code should scaffold first

Create these files with working stubs and docstrings:

- `eosquality/__init__.py`
- `eosquality/api.py`
- `eosquality/config.py`
- `eosquality/schema/models.py`
- `eosquality/schema/infer.py`
- `eosquality/preprocess/pipeline.py`
- `eosquality/distance/engine.py`
- `eosquality/neighbors/sklearn_backend.py`
- `eosquality/reference/fit_state.py`
- `eosquality/reference/diagnostics.py`
- `eosquality/core/detect.py`
- `eosquality/scoring/run.py`
- `eosquality/io/save.py`
- `eosquality/io/load.py`
- `tests/test_smoke_fit_run.py`
- `tests/test_distance_engine.py`

The first scaffold should already run a simple numeric example end to end.

---

## Acceptance criteria for v0.1

Claude Code should aim for these concrete outcomes.

### Functional
- `EOSQuality.fit(df_ref)` works on a pandas DataFrame with numeric columns
- `EOSQuality.run(df_query)` returns a `RunResult`
- `save()` and `load()` work
- reference quality is computed
- support score is computed
- tests pass

### Engineering
- typed code
- docstrings on all public classes and methods
- black/ruff-compatible formatting
- no notebook-only logic inside the package
- deterministic tests with fixed random seed

### Documentation
- README contains installation and quickstart
- one tutorial for numeric data
- one tutorial for mixed-type data

---

## README skeleton

Claude Code should generate a concise README with these sections:

1. What eosquality is
2. What it is not
3. Installation
4. Quickstart
5. Fit/run workflow
6. Concepts
7. Roadmap

Important sentence to include:

> eosquality estimates support, coherence, and stability of query data relative to a fitted reference population; it does not estimate probability of correctness.

---

## Recommended coding style

- Python 3.11+
- pandas and numpy first
- scipy and scikit-learn for baseline implementation
- optional extras for:
  - `faiss-cpu`
  - `hdbscan`
- use dataclasses for config and reports
- avoid overengineering the first version
- write narrow, testable functions
- keep public API stable and internal API flexible

---

## Suggested dependencies

### Core
- numpy
- pandas
- scipy
- scikit-learn

### Optional
- hdbscan
- faiss-cpu
- networkx
- joblib
- pydantic or msgspec only if really needed

### Dev
- pytest
- pytest-cov
- ruff
- black
- mypy

---

## pyproject outline

Claude Code can scaffold a `pyproject.toml` with:

- package metadata
- core dependencies
- optional extras:
  - `faiss`
  - `hdbscan`
  - `dev`
- ruff config
- pytest config

---

## Concrete TODO list for Claude Code

Use this as the immediate build order.

### TODO 1
Create the package skeleton and `pyproject.toml`.

### TODO 2
Implement config dataclasses and schema inference.

### TODO 3
Implement preprocessing pipeline with robust scaling for numeric features.

### TODO 4
Implement a first mixed distance engine:
- numeric
- binary
- categorical
- count
- vector block placeholder

### TODO 5
Implement sklearn nearest-neighbor backend.

### TODO 6
Implement `EOSQuality.fit()`:
- schema validation
- preprocessing
- neighbor build
- reference kNN summary stats
- reference quality summary

### TODO 7
Implement `EOSQuality.run()`:
- transform query
- fetch neighbors
- compute support
- compute consistency
- aggregate to quality

### TODO 8
Implement serialization.

### TODO 9
Add smoke tests and one end-to-end example.

### TODO 10
Add optional HDBSCAN-based core detection.

---

## Minimal pseudocode for the first implementation

```python
class EOSQuality:
    def __init__(self, config: EOSQualityConfig | None = None):
        self.config = config or EOSQualityConfig.default()
        self.is_fitted_ = False

    def fit(self, reference: pd.DataFrame) -> "EOSQuality":
        schema = infer_schema(reference)
        preprocess = PreprocessPipeline(schema, self.config)
        ref_repr = preprocess.fit_transform(reference)

        distance_engine = DistanceEngine(schema=schema, config=self.config.distance)
        neighbor_index = SklearnNeighborIndex(
            k=self.config.neighbors.k,
            metric="euclidean",
        ).fit(ref_repr)

        knn = neighbor_index.query(ref_repr)
        reference_report = compute_reference_report(ref_repr, knn, self.config)

        core_report, core_membership = detect_core(
            ref_repr, knn, self.config.core
        )

        self.fit_state_ = FitState(
            config=self.config,
            schema=schema,
            preprocess_artifacts=preprocess.get_state(),
            reference_ids=list(reference.index),
            reference_repr=ref_repr,
            neighbor_index=neighbor_index.get_state(),
            neighbor_graph=knn,
            reference_knn_stats=knn,
            reference_quality=reference_report.reference_quality,
            reference_report=reference_report,
            core_report=core_report,
            core_membership=core_membership,
        )
        self.is_fitted_ = True
        return self

    def run(self, query: pd.DataFrame) -> RunResult:
        check_is_fitted(self)

        preprocess = PreprocessPipeline.from_state(
            self.fit_state_.preprocess_artifacts
        )
        query_repr = preprocess.transform(query)

        neighbor_index = load_neighbor_index(self.fit_state_)
        q_knn = neighbor_index.query(query_repr)

        scores = score_queries(
            query_repr=query_repr,
            query_knn=q_knn,
            fit_state=self.fit_state_,
            config=self.config,
        )
        return RunResult(scores=scores, metadata={})
```

---

## Explicit non-goals for v0.1

Do not try to do all of these immediately:

- full density estimation across all mixed types
- perfect support for all missingness regimes
- learned feature weights
- advanced graph clustering alternatives
- domain-specific chemistry modules
- distributed inference

Get the fit/run core working first.

---

## Final instruction to Claude Code

Use this document to scaffold a clean, testable Python package for `eosquality`.

Prioritize:

1. a working end-to-end fit/run flow
2. transparent internal state
3. simple but defensible defaults
4. modularity for later upgrades

The first successful milestone is a repository where:

- `pip install -e .`
- `pytest`
- a minimal example script

all work cleanly.
