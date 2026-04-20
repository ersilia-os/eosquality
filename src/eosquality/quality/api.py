"""ErsiliaQuality: public fit/run interface."""

MIN_REFERENCE_SAMPLES = 10_000

import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd

from eosquality.config import ErsiliaQualityConfig, NeighborConfig
from eosquality.exceptions import NotFittedError, SchemaError
from eosquality.io.load import load
from eosquality.io.save import save
from eosquality.preprocess.pipeline import PreprocessPipeline
from eosquality.reference.diagnostics import compute_reference_report
from eosquality.reference.fit_state import FitState
from eosquality.reference.metadata import compute_metadata
from eosquality.schema.infer import infer_schema, validate_against_schema
from eosquality.scoring.run import RunResult, score_queries
from eosquality.utils.identifiers import validate_eos_id, validate_version
from eosquality.utils.logging import logger
from eosquality.vectorindex.backend import VectorIndex


class ErsiliaQuality:
    """Assess the quality of query data against a fitted reference population.

    Parameters
    ----------
    k:
        Number of nearest neighbors used for scoring.
    verbose:
        If True, print informative progress and diagnostic output.
    config:
        Full configuration object. If provided, ``k`` is ignored.
    """

    def __init__(
        self,
        k: int = 20,
        verbose: bool = False,
        config: ErsiliaQualityConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = ErsiliaQualityConfig(
                neighbors=NeighborConfig(k=k),
            )
        self.verbose = verbose
        if verbose:
            logger.set_verbosity(True)
        self.is_fitted_: bool = False
        self._fit_state: FitState | None = None
        self._vector_index_cache: Any | None = None

    # ------------------------------------------------------------------
    # Public fit/run interface
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        eos_id: str,
        version: str = "v1",
        vector_index: str | pathlib.Path = "",
        ignore_size: bool = False,
        allow_duplicates: bool = False,
    ) -> "ErsiliaQuality":
        """Build all reference-dependent artifacts from a numeric DataFrame.

        Parameters
        ----------
        reference:
            A pandas DataFrame containing ``key``, ``input`` (SMILES), and
            numeric feature columns.
        eos_id:
            EOS model identifier (e.g. ``"eos4e40"``). Must match the
            pattern ``eos<digit><3 alphanumeric>`` (7 characters).
        version:
            Dataset version string (e.g. ``"v1"``). Must match ``v<digits>``.
        vector_index:
            Path to a pre-built :class:`VectorIndex` folder produced by
            ``eosquality index``. The SMILES in the index must match the
            ``input`` column of ``reference`` row-for-row.
        ignore_size:
            If ``True``, skip the minimum-row check. Intended for tests and
            development only — fitting on small datasets produces unreliable
            scores. Default: ``False``.
        allow_duplicates:
            If ``True``, skip duplicate SMILES and duplicate key checks.
            Default: ``False`` — duplicate rows are almost always a data
            preparation error and should be fixed before fitting.

        Returns
        -------
        self
        """
        validate_eos_id(eos_id)
        validate_version(version)

        if not vector_index:
            raise ValueError(
                "vector_index is required. Build one with: "
                "eosquality index <library.csv> --output <index_dir/>"
            )

        if not ignore_size and len(reference) < MIN_REFERENCE_SAMPLES:
            raise ValueError(
                f"Reference dataset has {len(reference):,} rows. "
                f"Fitting requires at least {MIN_REFERENCE_SAMPLES:,} rows for reliable results. "
                "Pass ignore_size=True to bypass this check (not recommended for production use)."
            )

        # Duplicate checks — fail fast before any heavy work
        if not allow_duplicates:
            if "input" in reference.columns:
                seen: set[str] = set()
                dupes: list[tuple[int, str]] = []
                for idx, smi in enumerate(reference["input"]):
                    if smi in seen:
                        dupes.append((idx, str(smi)))
                    else:
                        seen.add(str(smi))
                if dupes:
                    n_shown = min(5, len(dupes))
                    examples = ", ".join(
                        f"row {i}: {s!r}" for i, s in dupes[:n_shown]
                    )
                    raise ValueError(
                        f"Duplicate SMILES in reference: {len(dupes)} duplicate(s). "
                        f"First {n_shown}: [{examples}]. "
                        "Pass allow_duplicates=True to skip this check."
                    )

            if "key" in reference.columns:
                seen_keys: set[str] = set()
                dupe_keys: list[tuple[int, str]] = []
                for idx, key in enumerate(reference["key"]):
                    k = str(key)
                    if k in seen_keys:
                        dupe_keys.append((idx, k))
                    else:
                        seen_keys.add(k)
                if dupe_keys:
                    n_shown = min(5, len(dupe_keys))
                    examples = ", ".join(
                        f"row {i}: {k!r}" for i, k in dupe_keys[:n_shown]
                    )
                    raise ValueError(
                        f"Duplicate keys in reference: {len(dupe_keys)} duplicate(s). "
                        f"First {n_shown}: [{examples}]. "
                        "Pass allow_duplicates=True to skip this check."
                    )

        t_start = time.perf_counter()
        logger.rule(f"ErsiliaQuality · fit · {eos_id} {version}")

        # 1. Schema + metadata
        t0 = time.perf_counter()
        schema = infer_schema(reference)
        metadata = compute_metadata(reference, eos_id=eos_id, version=version)
        t_schema = time.perf_counter() - t0

        logger.reference_table(
            n_samples=len(reference),
            n_features=len(schema.columns),
            column_names=schema.column_names,
        )
        logger.info(
            f"Reference: {len(reference):,} samples · {len(schema.columns)} features"
        )

        # 2. Load vector index + validate SMILES alignment
        t0 = time.perf_counter()
        if "input" not in reference.columns:
            raise SchemaError(
                "Reference DataFrame must contain an 'input' column with SMILES "
                "strings for vector-index alignment."
            )
        null_smiles = reference["input"].isna()
        if null_smiles.any():
            raise SchemaError(
                f"Reference 'input' column has {null_smiles.sum()} NaN value(s). "
                "All SMILES must be valid strings."
            )
        empty_smiles = reference["input"] == ""
        if empty_smiles.any():
            raise SchemaError(
                f"Reference 'input' column has {empty_smiles.sum()} empty string(s). "
                "All SMILES must be non-empty."
            )
        vi = VectorIndex.load(vector_index)
        vi.validate_smiles(list(reference["input"]))
        vector_index_path = str(pathlib.Path(vector_index).resolve())
        t_vi_load = time.perf_counter() - t0
        logger.debug(f"Vector index loaded | {len(vi._smiles):,} molecules")

        # 3. Preprocessing
        t0 = time.perf_counter()
        column_kinds = {
            col: chars.kind
            for col, chars in metadata.column_characteristics.items()
        }
        pipeline = PreprocessPipeline(schema=schema, column_kinds=column_kinds)
        ref_repr = pipeline.fit_transform(reference)
        t_preprocess = time.perf_counter() - t0
        logger.debug(f"Preprocessing done | repr shape={ref_repr.shape}")

        # 4. Reference kNN statistics using FP-selected neighbors
        k = self.config.neighbors.k
        n_features = ref_repr.shape[1]
        t0 = time.perf_counter()
        vi_knn_idx = vi.self_knn_indices(k)      # (n_ref, k) — FP-selected
        # Compute output-space L1 distances to those FP neighbors (mean |diff| per feature)
        neighbor_reprs = ref_repr[vi_knn_idx]    # (n_ref, k, n_features)
        diffs = ref_repr[:, None, :] - neighbor_reprs
        output_dists = (np.abs(diffs).sum(axis=2) / n_features).astype(np.float64)

        reference_report, knn_distances, knn_indices = compute_reference_report(
            ref_repr=ref_repr,
            knn_distances_with_self=output_dists,
            knn_indices_with_self=vi_knn_idx,
            exclude_self=False,
        )
        t_diagnostics = time.perf_counter() - t0

        logger.reference_report_table(
            reference_quality=reference_report.reference_quality,
            cohesion_score=reference_report.cohesion_score,
            fragmentation_score=reference_report.fragmentation_score,
            median_k_distance=reference_report.median_k_distance,
            notes=reference_report.notes,
        )

        # 5. Persist fit state
        self._fit_state = FitState(
            config=self.config,
            schema=schema,
            preprocess_state=pipeline.get_state(),
            reference_ids=list(reference.index),
            reference_repr=ref_repr,
            reference_knn_distances=knn_distances,
            reference_knn_indices=knn_indices,
            reference_report=reference_report,
            metadata=metadata,
            vector_index_path=vector_index_path,
        )
        self._vector_index_cache = vi
        self.is_fitted_ = True

        t_total = time.perf_counter() - t_start
        logger.timing_table([
            ("Schema inference", t_schema, False),
            ("Vector index load", t_vi_load, False),
            ("Preprocessing (type-aware L1)", t_preprocess, False),
            ("Reference diagnostics (FP kNN)", t_diagnostics, False),
        ])
        logger.success(
            f"Fit complete | reference_quality={reference_report.reference_quality:.4f} "
            f"| {t_total:.2f}s"
        )
        logger.rule()
        return self

    def run(self, query: pd.DataFrame) -> RunResult:
        """Score query samples against the fitted reference population.

        Parameters
        ----------
        query:
            A pandas DataFrame with the same numeric columns as the reference.

        Returns
        -------
        RunResult
            A dataclass with a ``scores`` DataFrame (one row per query sample)
            and a ``metadata`` dict.
        """
        self._check_fitted()
        assert self._fit_state is not None

        t_start = time.perf_counter()
        logger.info(f"Running quality scoring | {len(query):,} query samples")

        validate_against_schema(query, self._fit_state.schema)

        if "input" not in query.columns:
            raise SchemaError(
                "Query DataFrame must contain an 'input' column with SMILES strings."
            )

        pipeline = PreprocessPipeline.from_state(self._fit_state.preprocess_state)
        query_repr = pipeline.transform(query)
        logger.debug(f"Query repr shape={query_repr.shape}")

        k = self.config.neighbors.k

        # FP kNN → output-space distances
        if self._vector_index_cache is None:
            self._vector_index_cache = VectorIndex.load(
                self._fit_state.vector_index_path
            )
        query_smiles = list(query["input"])
        vi_distances, vi_indices = self._vector_index_cache.query(query_smiles, k=k)
        logger.debug(f"FP kNN done | k={k}")

        # Compute output-space L1 distances to FP-selected neighbors
        neighbor_reprs = self._fit_state.reference_repr[vi_indices]  # (n_q, k, n_feat)
        diffs = query_repr[:, None, :] - neighbor_reprs
        knn_distances = np.abs(diffs).sum(axis=2) / query_repr.shape[1]  # (n_q, k)

        result = score_queries(
            query_repr=query_repr,
            query_knn_distances_raw=knn_distances,
            query_knn_indices_raw=vi_indices,
            fit_state=self._fit_state,
            query_index=list(query.index),
        )

        logger.scores_summary_table(result.scores)
        t_total = time.perf_counter() - t_start
        logger.success(
            f"Run complete | mean quality_score={result.scores['quality_score'].mean():.4f} "
            f"| {t_total:.2f}s"
        )
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Persist fitted artifacts to a folder.

        Parameters
        ----------
        path:
            Directory to write. Created if it does not exist.

        Returns
        -------
        pathlib.Path
            The resolved folder path.
        """
        self._check_fitted()
        assert self._fit_state is not None
        written = save(self._fit_state, path)
        logger.info(f"Artifacts saved → {written}")
        return written

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ErsiliaQuality":
        """Load a previously saved ErsiliaQuality instance.

        Parameters
        ----------
        path:
            File path produced by :meth:`save`.

        Returns
        -------
        ErsiliaQuality
            A fitted instance ready to call :meth:`run`.
        """
        logger.info(f"Loading artifacts from {path}")
        fit_state = load(path)
        instance = cls(config=fit_state.config)
        instance._fit_state = fit_state
        instance._vector_index_cache = None  # lazy-loaded on first run()
        instance.is_fitted_ = True
        logger.success(f"Artifacts loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # Post-fit attributes
    # ------------------------------------------------------------------

    @property
    def schema_(self):
        """Inferred schema from the reference DataFrame."""
        self._check_fitted()
        return self._fit_state.schema  # type: ignore[union-attr]

    @property
    def reference_quality_(self) -> float:
        """Scalar quality measure of the reference population (0–1)."""
        self._check_fitted()
        return self._fit_state.reference_report.reference_quality  # type: ignore[union-attr]

    @property
    def reference_report_(self):
        """Full ReferenceReport with cohesion and fragmentation diagnostics."""
        self._check_fitted()
        return self._fit_state.reference_report  # type: ignore[union-attr]

    @property
    def metadata_(self):
        """FitMetadata: provenance and dataset statistics from fit()."""
        self._check_fitted()
        return self._fit_state.metadata  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise NotFittedError(
                "This ErsiliaQuality instance is not fitted yet. Call fit() first."
            )
