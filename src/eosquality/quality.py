"""ErsiliaQuality: thin orchestrator composing the per-score components.

For users who want a one-stop fit/run interface, this class fits the
shared state once, the kNN state once, and then each requested score on
top. Each score remains independently saveable / loadable.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from eosquality.config import ErsiliaQualityConfig, NeighborConfig
from eosquality.exceptions import NotFittedError, SchemaError
from eosquality.knn.fit import fit_knn
from eosquality.library.identity import (
    LIBRARY_ID,
    reference_library_csv_path,
    reference_library_path,
)
from eosquality.preprocess import PreprocessPipeline
from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import (
    _query_fp_distances,
    _query_output_distances,
    _resolve_vector_index,
)
from eosquality.scores.consistency import Consistency
from eosquality.scores.extremity import Extremity
from eosquality.scores.support import Support
from eosquality.scores.typicality import Typicality
from eosquality.shared.fit import fit_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.identifiers import validate_eos_id, validate_version
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex


MIN_REFERENCE_SAMPLES = 10_000

DEFAULT_SCORES: tuple[str, ...] = ("typicality", "support", "consistency", "extremity")
_INDEX_AWARE = frozenset({"support", "consistency"})
_KNN_USERS = frozenset({"support", "consistency"})


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    ``scores`` is a per-query DataFrame whose columns are exactly the
    fitted components, in canonical order: ``typicality``, ``extremity``,
    ``support``, ``consistency``. A fit that skips some components yields
    a DataFrame missing those columns; the order of the remaining columns
    is unchanged.
    """

    scores: pd.DataFrame
    metadata: dict[str, Any]


class ErsiliaQuality:
    """Orchestrate the per-score components for a single reference dataset."""

    def __init__(
        self,
        k: int = 5,
        verbose: bool = False,
        config: ErsiliaQualityConfig | None = None,
    ) -> None:
        """Build an unfitted orchestrator.

        Parameters
        ----------
        k:
            Number of nearest neighbors for the FP self-kNN. Ignored
            if ``config`` is provided.
        verbose:
            If ``True``, route the package's loguru output to stderr at
            DEBUG level.
        config:
            Full :class:`ErsiliaQualityConfig`. If omitted, a default
            config is built from ``k``.
        """
        if config is not None:
            self.config = config
        else:
            self.config = ErsiliaQualityConfig(neighbors=NeighborConfig(k=k))
        self.verbose = verbose
        if verbose:
            logger.set_verbosity(True)

        self.typicality: Typicality | None = None
        self.support: Support | None = None
        self.consistency: Consistency | None = None
        self.extremity: Extremity | None = None
        self._shared: SharedFitState | None = None
        self._vector_index_cache: VectorIndex | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Fit / run
    # ------------------------------------------------------------------

    def fit(
        self,
        reference: pd.DataFrame,
        eos_id: str,
        version: str = "v1",
        vector_index: str | pathlib.Path | None = None,
        ignore_size: bool = False,
        scores: Iterable[str] = DEFAULT_SCORES,
    ) -> "ErsiliaQuality":
        """Fit the selected scores on a reference DataFrame.

        Parameters
        ----------
        scores:
            Which components to fit. Defaults to all four
            (``typicality``, ``support``, ``consistency``, ``extremity``).
            Pass e.g. ``scores=["typicality"]`` to skip the vector index
            entirely.
        """
        validate_eos_id(eos_id)
        validate_version(version)
        scores_set = set(scores)
        unknown = scores_set - set(DEFAULT_SCORES)
        if unknown:
            raise ValueError(
                f"Unknown score(s): {sorted(unknown)}. Valid choices: {DEFAULT_SCORES}."
            )

        needs_index = bool(scores_set & _INDEX_AWARE)
        needs_knn = bool(scores_set & _KNN_USERS)
        using_canonical_library = needs_index and not vector_index

        if reference.empty:
            raise SchemaError("Reference DataFrame is empty.")
        if needs_index:
            self._validate_input_column(reference)

        t_start = time.perf_counter()
        logger.rule(f"ErsiliaQuality · fit · {eos_id} {version}")

        vi: VectorIndex | None = None
        library_id = ""

        if needs_index:
            if using_canonical_library:
                _validate_reference_against_library_csv(reference)
                vector_index = reference_library_path()
            vi = VectorIndex.load(vector_index)
            if using_canonical_library:
                if len(reference) != len(vi._smiles):
                    raise ValueError(
                        f"Reference has {len(reference):,} rows but the vector "
                        f"index has {len(vi._smiles):,} molecules. Fit requires "
                        "matching row counts."
                    )
            else:
                vi.validate_smiles(list(reference["input"]))
            library_id = str(vi._config.get("library_name", "") or "")

        if not ignore_size and len(reference) < MIN_REFERENCE_SAMPLES:
            raise ValueError(
                f"Reference dataset has {len(reference):,} rows. "
                f"Fitting requires at least {MIN_REFERENCE_SAMPLES:,} rows for reliable results. "
                "Pass ignore_size=True to bypass this check (not recommended for production use)."
            )

        self._check_unique_keys(reference)

        shared, ref_repr = fit_shared(
            reference, eos_id=eos_id, version=version, library_id=library_id
        )
        self._shared = shared

        knn = None
        if needs_knn:
            assert vi is not None
            knn = fit_knn(
                shared=shared,
                ref_repr=ref_repr,
                vector_index=vi,
                k=self.config.neighbors.k,
            )

        if "typicality" in scores_set:
            self.typicality = Typicality().fit(reference, shared=shared)
        if "support" in scores_set:
            self.support = Support().fit(
                reference,
                vector_index=vi,
                k=self.config.neighbors.k,
                shared=shared,
                knn=knn,
            )
        if "consistency" in scores_set:
            self.consistency = Consistency().fit(
                reference,
                vector_index=vi,
                k=self.config.neighbors.k,
                shared=shared,
                knn=knn,
            )
        if "extremity" in scores_set:
            self.extremity = Extremity().fit(reference, shared=shared)

        self._vector_index_cache = vi
        self.is_fitted_ = True

        self._emit_reference_report(reference, shared)

        t_total = time.perf_counter() - t_start
        ref_s = self.support.reference_support_ if self.support else float("nan")
        logger.success(f"Fit complete | reference_support={ref_s:.4f} | {t_total:.2f}s")
        logger.rule()
        return self

    def run(self, query: pd.DataFrame) -> RunResult:
        """Score query samples against the fitted reference population.

        Computes the FP-selected kNN once for index-aware scores (Support
        + Consistency) and reuses the result across them; per-score
        :meth:`run` methods are called with the precomputed arrays.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference,
            plus an ``'input'`` SMILES column if any index-aware score
            (Support or Consistency) was fit.

        Returns
        -------
        RunResult
            ``scores`` is a per-query DataFrame whose columns are exactly
            the fitted components, in canonical order (``typicality``,
            ``extremity``, ``support``, ``consistency``). Components
            that were not fit are simply absent from the DataFrame.
            ``metadata`` aggregates per-score metadata dicts (Consistency
            keys are prefixed with ``consistency_`` to avoid colliding
            with Support's ``n_reference`` / ``k``).
        """
        self._check_fitted()
        assert self._shared is not None

        t_start = time.perf_counter()
        logger.info(f"Running quality scoring | {len(query):,} query samples")

        validate_against_schema(query, self._shared.schema)

        needs_input_col = self.support is not None or self.consistency is not None
        if needs_input_col and "input" not in query.columns:
            raise SchemaError(
                "Query DataFrame must contain an 'input' column with SMILES strings."
            )

        pipeline = PreprocessPipeline.from_state(
            {
                "schema": self._shared.schema,
                "scaler_params": self._shared.scaler_params,
                "binary_class_freq": self._shared.binary_class_freq,
            }
        )
        query_repr = pipeline.transform(query)

        query_fp_indices: np.ndarray | None = None
        query_fp_distances: np.ndarray | None = None
        query_output_distances: np.ndarray | None = None
        if self.support is not None or self.consistency is not None:
            vi = self._get_vector_index()
            knn = (self.support or self.consistency)._knn  # type: ignore[union-attr]
            assert knn is not None
            query_fp_distances, query_fp_indices = _query_fp_distances(query, vi, knn.k)
            if self.consistency is not None:
                query_output_distances = _query_output_distances(
                    query_repr, knn, query_fp_indices
                )

        columns: dict[str, pd.Series] = {}
        metadata: dict[str, Any] = {"n_reference": len(self._shared.reference_ids)}

        if self.typicality is not None:
            typicality_result = self.typicality.run(query, query_repr=query_repr)
            columns["typicality"] = typicality_result.score
            metadata.update(typicality_result.metadata)

        if self.extremity is not None:
            extremity_result = self.extremity.run(query, query_repr=query_repr)
            columns["extremity"] = extremity_result.score
            metadata.update(extremity_result.metadata)

        if self.support is not None:
            support_result = self.support.run(
                query,
                query_fp_indices=query_fp_indices,
                query_fp_distances=query_fp_distances,
            )
            columns["support"] = support_result.score
            metadata.update(support_result.metadata)

        if self.consistency is not None:
            consistency_result = self.consistency.run(
                query,
                query_repr=query_repr,
                query_fp_indices=query_fp_indices,
                query_output_distances=query_output_distances,
            )
            columns["consistency"] = consistency_result.score
            # Prefix Consistency's metadata keys so they don't collide with
            # Support's ``n_reference`` / ``k`` (both index-aware scores
            # report the same field names).
            metadata.update(
                {f"consistency_{k}": v for k, v in consistency_result.metadata.items()}
            )

        ordered = [
            c
            for c in ("typicality", "extremity", "support", "consistency")
            if c in columns
        ]
        scores_df = pd.DataFrame(
            {c: columns[c] for c in ordered}, index=list(query.index)
        )

        logger.scores_summary_table(scores_df)
        t_total = time.perf_counter() - t_start
        head_col = next(iter(scores_df.columns), None)
        head_val = scores_df[head_col].mean() if head_col else float("nan")
        logger.success(
            f"Run complete | mean {head_col or 'score'}={head_val:.4f} | {t_total:.2f}s"
        )
        return RunResult(
            scores=scores_df,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write fitted artifacts to a folder.

        Each fitted score writes its own subfolder under ``path``:
        ``shared/`` (always), ``knn/`` (iff support or consistency was
        fit), ``typicality/``, ``support/``, ``consistency/``,
        ``extremity/`` (each iff that score was fit). The save calls
        are idempotent on overlapping ``shared/`` / ``knn/`` writes.

        Also writes a top-level ``manifest.json`` summarizing the fit
        for easy inspection (this file is informational only — the
        loader does not consult it).
        """
        self._check_fitted()
        assert self._shared is not None
        folder = pathlib.Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving artifacts → {folder}")
        if self.typicality is not None:
            self.typicality.save(folder)
        if self.support is not None:
            self.support.save(folder)
        if self.consistency is not None:
            self.consistency.save(folder)
        if self.extremity is not None:
            self.extremity.save(folder)
        self._write_manifest(folder)
        logger.info(f"Artifacts saved → {folder}")
        return folder

    def _write_manifest(self, folder: pathlib.Path) -> None:
        """Write the top-level ``manifest.json`` summary."""
        import json

        assert self._shared is not None
        scores: list[str] = []
        k_used: int | None = None
        if self.typicality is not None:
            scores.append("typicality")
        if self.extremity is not None:
            scores.append("extremity")
        if self.support is not None:
            scores.append("support")
            k_used = self.support.knn_.k
        if self.consistency is not None:
            scores.append("consistency")
            k_used = self.consistency.knn_.k
        manifest = {
            "scores": scores,
            "n_samples": self._shared.metadata.n_samples,
            "n_features": self._shared.metadata.n_features,
            "k": k_used,
            "library_id": self._shared.metadata.library_id,
            "fit_timestamp": self._shared.metadata.fit_timestamp,
            "eosquality_version": self._shared.metadata.eosquality_version,
        }
        with open(folder / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "ErsiliaQuality":
        """Reconstruct an orchestrator from a saved folder.

        Walks the subfolders under ``path`` and loads each per-score
        component that's present. At least one score subfolder must
        exist. Library / package compatibility is enforced when any
        index-aware score is found.
        """
        folder = pathlib.Path(path)
        if not folder.exists():
            raise FileNotFoundError(f"No artifacts folder found at: {folder}")
        if not folder.is_dir():
            raise ValueError(
                f"Expected a directory, got a file: {folder}. "
                "Artifacts are stored as a folder — pass the folder path."
            )
        instance = cls()
        if (folder / "typicality").is_dir():
            instance.typicality = Typicality.load(folder)
        if (folder / "support").is_dir():
            instance.support = Support.load(folder)
        if (folder / "consistency").is_dir():
            instance.consistency = Consistency.load(folder)
        if (folder / "extremity").is_dir():
            instance.extremity = Extremity.load(folder)

        first_shared = next(
            (
                c._shared
                for c in (
                    instance.typicality,
                    instance.support,
                    instance.consistency,
                    instance.extremity,
                )
                if c is not None
            ),
            None,
        )
        if first_shared is None:
            raise FileNotFoundError(
                f"No score subfolders found under {folder} — nothing to load."
            )
        instance._shared = first_shared
        _check_artifacts_compatibility(
            instance._shared,
            has_index_scores=any(
                s is not None for s in (instance.support, instance.consistency)
            ),
        )
        instance.is_fitted_ = True
        logger.success(f"Artifacts loaded from {folder}")
        return instance

    # ------------------------------------------------------------------
    # Post-fit attributes
    # ------------------------------------------------------------------

    @property
    def schema_(self):
        """Inferred reference schema."""
        self._check_fitted()
        assert self._shared is not None
        return self._shared.schema

    @property
    def reference_support_(self) -> float:
        """Mean reference-as-query support; requires the support score to be fit."""
        self._check_fitted()
        if self.support is None:
            raise RuntimeError(
                "reference_support is only defined when the support score has been fit."
            )
        return self.support.reference_support_

    @property
    def reference_typicality_(self) -> float:
        """Mean reference-as-query typicality; requires typicality to be fit."""
        self._check_fitted()
        if self.typicality is None:
            raise RuntimeError(
                "reference_typicality is only defined when typicality has been fit."
            )
        return self.typicality.reference_typicality_

    @property
    def reference_extremity_(self) -> float:
        """Mean reference-as-query extremity; requires extremity to be fit."""
        self._check_fitted()
        if self.extremity is None:
            raise RuntimeError(
                "reference_extremity is only defined when extremity has been fit."
            )
        return self.extremity.reference_extremity_

    @property
    def metadata_(self):
        """Shared :class:`FitMetadata` (eos_id, version, sizes, timestamps, ...)."""
        self._check_fitted()
        assert self._shared is not None
        return self._shared.metadata

    @property
    def shared_(self) -> SharedFitState:
        """The shared fit state (schema, scaler, binary_class_freq, metadata)."""
        self._check_fitted()
        assert self._shared is not None
        return self._shared

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise NotFittedError(
                "This ErsiliaQuality instance is not fitted yet. Call fit() first."
            )

    def _get_vector_index(self) -> VectorIndex:
        """Load (and cache) the VectorIndex backing the index-aware scores.

        Resolution is by ``shared.metadata.library_id`` via the canonical
        library resolver — the artifact does not store an index path.
        """
        if self._vector_index_cache is not None:
            return self._vector_index_cache
        score = self.support or self.consistency
        if score is None:
            raise RuntimeError(
                "No vector-index-aware score is loaded; cannot resolve a VectorIndex."
            )
        assert self._shared is not None
        self._vector_index_cache = _resolve_vector_index(self._shared)
        return self._vector_index_cache

    @staticmethod
    def _validate_input_column(reference: pd.DataFrame) -> None:
        if "input" not in reference.columns:
            raise SchemaError(
                "Reference DataFrame must contain an 'input' column with SMILES "
                "strings for vector-index alignment."
            )
        null_smiles = reference["input"].isna()
        if null_smiles.any():
            raise SchemaError(
                f"Reference 'input' column has {int(null_smiles.sum())} NaN value(s). "
                "All SMILES must be valid strings."
            )
        empty_smiles = reference["input"] == ""
        if empty_smiles.any():
            raise SchemaError(
                f"Reference 'input' column has {int(empty_smiles.sum())} empty string(s). "
                "All SMILES must be non-empty."
            )

    @staticmethod
    def _check_unique_keys(reference: pd.DataFrame) -> None:
        if "key" not in reference.columns:
            return
        seen: set[str] = set()
        dupes: list[tuple[int, str]] = []
        for idx, key in enumerate(reference["key"]):
            k = str(key)
            if k in seen:
                dupes.append((idx, k))
            else:
                seen.add(k)
        if dupes:
            n_shown = min(5, len(dupes))
            examples = ", ".join(f"row {i}: {k!r}" for i, k in dupes[:n_shown])
            raise ValueError(
                f"Duplicate keys in reference: {len(dupes)} duplicate(s). "
                f"First {n_shown}: [{examples}]. "
                "The 'key' column must be unique across rows."
            )

    def _emit_reference_report(
        self, reference: pd.DataFrame, shared: SharedFitState
    ) -> None:
        logger.info(
            f"Reference: {len(reference):,} samples · {len(shared.schema.columns)} features"
        )
        if (
            self.support is None
            and self.consistency is None
            and self.typicality is None
            and self.extremity is None
        ):
            return
        logger.reference_report_table(
            reference_support=self.support.reference_support_ if self.support else None,
            reference_typicality=self.typicality.reference_typicality_
            if self.typicality
            else None,
            reference_extremity=self.extremity.reference_extremity_
            if self.extremity
            else None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_artifacts_compatibility(
    shared: SharedFitState, has_index_scores: bool
) -> None:
    """Reject artifacts fit against a different reference library.

    The compatibility check only applies to index-aware artifacts (the only
    ones where library identity actually matters). A typicality-only fit
    has ``library_id == ""`` by construction and is portable.
    """
    if not has_index_scores:
        return
    import importlib.metadata as _md

    from packaging.version import Version

    from eosquality.exceptions import IncompatibleArtifactsError

    if shared.metadata.library_id != LIBRARY_ID:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit against reference library "
            f"{shared.metadata.library_id!r} but this install ships {LIBRARY_ID!r}. "
            "Install a compatible eosquality release or refit against the current library."
        )
    try:
        current = _md.version("eosquality")
    except _md.PackageNotFoundError:
        return
    try:
        saved_major = Version(shared.metadata.eosquality_version).major
        current_major = Version(current).major
    except Exception:
        return
    if saved_major != current_major:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit with eosquality {shared.metadata.eosquality_version} "
            f"(major={saved_major}) but this install is {current} "
            f"(major={current_major}). Install a matching eosquality release or refit."
        )


def _validate_reference_against_library_csv(reference: pd.DataFrame) -> None:
    """Ensure reference.input is a prefix of the canonical library SMILES."""
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
            f"{len(mismatches)} row(s) differ. First {n_shown}: [{examples}]."
        )
