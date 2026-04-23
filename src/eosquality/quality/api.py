"""ErsiliaQuality: public fit/run interface."""

MIN_REFERENCE_SAMPLES = 10_000

import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd

from eosquality._library import reference_library_csv_path, reference_library_path
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
from eosquality.scoring.typicality import compute_typicality
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
        vector_index: str | pathlib.Path | None = None,
        ignore_size: bool = False,
    ) -> "ErsiliaQuality":
        """Build all reference-dependent artifacts from a numeric DataFrame.

        Parameters
        ----------
        reference:
            A pandas DataFrame containing ``key``, ``input`` (SMILES), and
            numeric feature columns. Duplicate ``key`` values are always
            rejected — fix the source CSV if you hit this.
        eos_id:
            EOS model identifier (e.g. ``"eos4e40"``). Must match the
            pattern ``eos<digit><3 alphanumeric>`` (7 characters).
        version:
            Dataset version string (e.g. ``"v1"``). Must match ``v<digits>``.
        vector_index:
            Optional path to a pre-built :class:`VectorIndex` folder produced
            by ``eosquality index``. If ``None`` (default), the canonical
            reference library shipped with this release is used. Provide a
            path only to fit against a non-canonical index for internal
            testing; the SMILES in that index must match the ``input`` column
            of ``reference`` row-for-row.
        ignore_size:
            If ``True``, skip the minimum-row check. Intended for tests and
            development only — fitting on small datasets produces unreliable
            scores. Default: ``False``.

        Returns
        -------
        self
        """
        validate_eos_id(eos_id)
        validate_version(version)

        using_canonical_library = not vector_index

        # Earliest possible "wrong CSV" detection: the reference CSV's SMILES
        # must match the library, row-for-row. Everything else (size check,
        # duplicates, schema, preprocessing, index download) is pointless work
        # if the wrong molecules were provided, so we fail before any of it.
        if reference.empty:
            raise SchemaError("Reference DataFrame is empty.")
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

        t_start = time.perf_counter()
        logger.rule(f"ErsiliaQuality · fit · {eos_id} {version}")

        # 1. Validate SMILES alignment.
        #    Canonical path: check against data/libraries/<LIBRARY_ID>.csv
        #    (source of truth) BEFORE fetching the index — so a "wrong CSV"
        #    reference fails fast without paying the index download cost.
        #    Supports subsampled indexes: a --max-samples index is a prefix of
        #    the canonical library, so kNN lookups still line up row-for-row
        #    with any reference that's also a prefix of the same library.
        #    Custom path (user passed vector_index=...): fall back to the
        #    index's own smiles.csv, since they've opted out of the canonical
        #    ecosystem.
        t0 = time.perf_counter()
        if using_canonical_library:
            _validate_reference_against_library_csv(reference)
            vector_index = reference_library_path()
        vi = VectorIndex.load(vector_index)
        if using_canonical_library:
            if len(reference) != len(vi._smiles):
                raise ValueError(
                    f"Reference has {len(reference):,} rows but the vector "
                    f"index has {len(vi._smiles):,} molecules. Fit requires "
                    "matching row counts — rebuild the index with the same "
                    "--max-samples, or use a reference of matching size."
                )
        else:
            vi.validate_smiles(list(reference["input"]))
        vector_index_path = str(pathlib.Path(vector_index).resolve())
        t_vi_load = time.perf_counter() - t0
        logger.debug(f"Vector index loaded | {len(vi._smiles):,} molecules")

        # Reference-local checks, now known to be for the right library
        if not ignore_size and len(reference) < MIN_REFERENCE_SAMPLES:
            raise ValueError(
                f"Reference dataset has {len(reference):,} rows. "
                f"Fitting requires at least {MIN_REFERENCE_SAMPLES:,} rows for reliable results. "
                "Pass ignore_size=True to bypass this check (not recommended for production use)."
            )

        # Reference SMILES dedup is handled upstream: the library itself is
        # built with no duplicates, and validate_smiles() above enforces
        # row-for-row alignment, so reference["input"] is dupe-free by
        # construction. Only key uniqueness needs an explicit check here.
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
                    "The 'key' column must be unique across rows."
                )

        # 2. Schema + metadata
        t0 = time.perf_counter()
        schema = infer_schema(reference)
        metadata = compute_metadata(reference, eos_id=eos_id, version=version)
        # Tag the fit with the library identity declared by the vector index itself.
        # For the canonical shipped index this equals LIBRARY_ID; for a --vector-index
        # override it's whatever that index was built with. The load-time check
        # uses this to reject cross-library artifacts.
        metadata.library_id = str(vi._config.get("library_name", "") or "")
        t_schema = time.perf_counter() - t0

        logger.reference_table(
            n_samples=len(reference),
            n_features=len(schema.columns),
            column_names=schema.column_names,
        )
        logger.info(
            f"Reference: {len(reference):,} samples · {len(schema.columns)} features"
        )

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

        # Reference typicality: the mean aggregate typicality of the reference
        # under its own CDF. Calibration baseline rather than a pass/fail
        # metric — a random reference sample has expected per-feature
        # typicality = E[2·min(U, 1−U)] = 0.5 (U uniform on [0,1]), and the
        # geomean across features lands below that. Compare query
        # typicality_score values against this baseline to judge magnitude.
        pipeline_state = pipeline.get_state()
        reference_raw = pipeline.raw_numeric_values(reference)
        _, ref_typ = compute_typicality(
            raw_values=reference_raw,
            scalers=pipeline_state["scalers"],
            column_names=list(schema.column_names),
            n_reference=len(reference),
            quantile_levels=pipeline_state.get("quantile_levels"),
        )
        reference_report.reference_typicality = float(np.mean(ref_typ))
        t_diagnostics = time.perf_counter() - t0

        logger.reference_report_table(
            reference_quality=reference_report.reference_quality,
            cohesion_score=reference_report.cohesion_score,
            fragmentation_score=reference_report.fragmentation_score,
            median_k_distance=reference_report.median_k_distance,
            reference_typicality=reference_report.reference_typicality,
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
            ("Vector index load + SMILES alignment", t_vi_load, False),
            ("Schema inference", t_schema, False),
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
        query_raw = pipeline.raw_numeric_values(query)
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
            query_raw=query_raw,
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


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _validate_reference_against_library_csv(reference: pd.DataFrame) -> None:
    """Ensure reference.input is a prefix of the canonical library SMILES.

    Loads the library CSV (from CWD, cache, or S3 — resolved lazily) and
    checks that ``reference["input"][i] == library_smiles[i]`` for every row
    of the reference, with ``len(reference) <= len(library_smiles)``.

    Failing early here makes "wrong CSV" the most visible error class for
    users: we distinguish between "shape mismatch with library", "content
    mismatch against library", and (downstream) "content mismatch against
    the actual index".
    """
    csv_path = reference_library_csv_path()
    lib_df = pd.read_csv(csv_path, usecols=["smiles"])
    library_smiles = list(lib_df["smiles"])

    ref_smiles = list(reference["input"])
    if len(ref_smiles) > len(library_smiles):
        raise ValueError(
            f"Reference has {len(ref_smiles):,} rows but the canonical "
            f"library {csv_path.name} has only {len(library_smiles):,}. "
            "Reference must be a prefix of the canonical library."
        )

    mismatches = [
        i for i, (a, b) in enumerate(zip(ref_smiles, library_smiles)) if a != b
    ]
    if mismatches:
        n_shown = min(3, len(mismatches))
        examples = "; ".join(
            f"row {i}: reference={ref_smiles[i]!r} vs library={library_smiles[i]!r}"
            for i in mismatches[:n_shown]
        )
        raise ValueError(
            f"SMILES mismatch against canonical library {csv_path.name}: "
            f"{len(mismatches)} row(s) differ. First {n_shown}: [{examples}]. "
            "The reference CSV must be a prefix of the canonical library "
            "(same molecules, same order, starting from row 0)."
        )
