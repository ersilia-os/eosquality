"""ErsiliaQuality: thin orchestrator composing the per-score components.

For users who want a one-stop fit/run interface, this class fits the
shared state once, the kNN state once, and then each requested score on
top. Each score remains independently saveable / loadable.
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eosquality.config import ErsiliaQualityConfig, NeighborConfig
from eosquality.exceptions import (
    IncompatibleArtifactsError,
    NotFittedError,
    SchemaError,
)
from eosquality.knn.fit import fit_knn
from eosquality.knn.load import load_knn
from eosquality.knn.save import save_knn
from eosquality.library.identity import LIBRARY_ID, reference_library_path
from eosquality.schema.infer import validate_against_schema
from eosquality.scores._helpers import (
    _custom_index_path,
    _make_query_repr,
    _query_fp_distances,
    _query_output_distances,
    _resolve_vector_index,
)
from eosquality.scores.consistency import Consistency
from eosquality.scores.extremity import Extremity
from eosquality.scores.signal import SIGNAL_FORMULA_VERSION, Signal
from eosquality.scores.support import Support
from eosquality.scores.typicality import Typicality
from eosquality.shared.fit import DEFAULT_MAX_FEATURES, fit_shared
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.utils.identifiers import validate_eos_id, validate_version
from eosquality.utils.logging import logger
from eosquality.vectorindex import VectorIndex

MIN_REFERENCE_SAMPLES = 10_000

DEFAULT_SCORES: tuple[str, ...] = (
    "typicality",
    "support",
    "consistency",
    "extremity",
)
# All valid score names, including the opt-in ones that are not in DEFAULT_SCORES.
# Signal is opt-in (provisional): users must pass ``scores=DEFAULT_SCORES + ("signal",)``
# or similar to enable it. Validation uses this set, not DEFAULT_SCORES.
ALL_SCORES: tuple[str, ...] = DEFAULT_SCORES + ("signal",)
# Canonical component order: fit, run, save, output columns and metadata.
_SCORE_ORDER: tuple[str, ...] = (
    "typicality",
    "extremity",
    "support",
    "consistency",
    "signal",
)
_SCORE_CLASSES = {
    "typicality": Typicality,
    "extremity": Extremity,
    "support": Support,
    "consistency": Consistency,
    "signal": Signal,
}
# Scores that need the vector index at fit time (Signal reads the library's
# descriptor matrices from the index folder; it never queries the index).
_INDEX_AWARE = frozenset({"support", "consistency", "signal"})
_KNN_USERS = frozenset({"support", "consistency"})


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    ``scores`` is a per-query DataFrame with the columns of each fitted
    component, in canonical order: ``typicality``, ``typicality_raw``,
    ``extremity``, ``extremity_raw``, ``support``, ``support_raw``,
    ``support_log``, ``consistency``, ``consistency_raw``, ``signal``,
    ``signal_raw``. The calibrated column is in ``(0, 1]``; ``*_raw`` is the
    pre-calibration value; ``support_log = −log10(support)``. Components
    that were not fit are absent.

    ``metadata`` has ``n_reference`` plus each component's run metadata
    with keys prefixed by the component name (e.g. ``support_k``,
    ``consistency_n_fp_bins``, ``signal_descriptor``).
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
        self.signal: Signal | None = None
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
        max_features: int | None = DEFAULT_MAX_FEATURES,
        max_signal_train_samples: int | None = 1000,
        signal_descriptor: str = "physchem",
    ) -> ErsiliaQuality:
        """Fit the selected scores on a reference DataFrame.

        Parameters
        ----------
        scores:
            Which components to fit. Defaults to ``DEFAULT_SCORES``
            (``typicality``, ``support``, ``consistency``, ``extremity``).
            Pass e.g. ``scores=["typicality"]`` to skip the vector index
            entirely. ``"signal"`` is provisionally opt-in — request it
            explicitly (e.g. ``scores=DEFAULT_SCORES + ("signal",)``)
            since it trains an XGBoost model and computes SHAP values.
            Valid names are listed in ``ALL_SCORES``.
        max_features:
            Cap on the number of features retained after fit-time
            correlation-cluster medoid selection. Defaults to
            ``DEFAULT_MAX_FEATURES`` (= 10). Pass ``None`` to disable
            reduction. Support is unaffected (it uses fingerprints only);
            typicality, extremity, consistency, and signal see the
            reduced set.
        vector_index:
            Optional path to a non-canonical vector index folder (built with
            ``eosquality build``). Default ``None`` uses the canonical
            reference library. A custom index's absolute path is recorded
            in the artifact and must still exist at run time.
        ignore_size:
            Skip the ``MIN_REFERENCE_SAMPLES`` row-count check (testing only).
        max_signal_train_samples:
            Cap on the number of training rows the ``signal`` XGBoost model
            is fit on. Defaults to ``1000`` for fast iteration. Pass
            ``None`` or ``0`` to use the full training slice (~80% of the
            reference). Calibration and the reported ``r2_val`` always use
            the full validation slice; only the per-round early-stopping
            eval set is sampled to 5,000 rows. Ignored when ``signal`` is
            not in ``scores``.
        signal_descriptor:
            Feature backend the ``signal`` score uses: ``"physchem"``
            (default; RDKit physicochemical descriptors) or ``"maccs"``
            (MACCS structural keys). Both are read precomputed from the
            library for the reference and computed on the fly for queries.
            Recorded in the saved artifact and used unchanged at run time.
            Ignored when ``signal`` is not in ``scores``.
        """
        validate_eos_id(eos_id)
        validate_version(version)
        scores_set = set(scores)
        unknown = scores_set - set(ALL_SCORES)
        if unknown:
            raise ValueError(
                f"Unknown score(s): {sorted(unknown)}. Valid choices: {ALL_SCORES}."
            )

        needs_index = bool(scores_set & _INDEX_AWARE)
        needs_knn = bool(scores_set & _KNN_USERS)

        if reference.empty:
            raise SchemaError("Reference DataFrame is empty.")
        if needs_index:
            self._validate_input_column(reference)

        # A re-fit replaces every component, including ones not requested now.
        for name in _SCORE_ORDER:
            setattr(self, name, None)
        self.is_fitted_ = False

        t_start = time.perf_counter()
        logger.rule(f"ErsiliaQuality · fit · {eos_id} {version}")
        logger.info(
            f"fit | eos_id={eos_id} version={version} "
            f"scores=[{', '.join(sorted(scores_set))}]"
        )

        if not ignore_size and len(reference) < MIN_REFERENCE_SAMPLES:
            raise ValueError(
                f"Reference dataset has {len(reference):,} rows. Fitting requires "
                f"at least {MIN_REFERENCE_SAMPLES:,} rows for reliable results. "
                "Pass ignore_size=True to bypass this check (not recommended "
                "for production use)."
            )
        self._check_unique_keys(reference)

        vi: VectorIndex | None = None
        if needs_index:
            t = time.perf_counter()
            vi = VectorIndex.load(vector_index or reference_library_path())
            # The reference must be the index's molecules, in index order.
            vi.validate_smiles(list(reference["input"]))
            logger.info(
                f"vector index | loaded {vi.library_name or '(unnamed)'} | "
                f"n_ref={vi.n_reference:,} | {time.perf_counter() - t:.1f}s"
            )

        shared = fit_shared(
            reference,
            eos_id=eos_id,
            version=version,
            library_id=vi.library_name if vi is not None else "",
            vector_index_path=_custom_index_path(vi) if vi is not None else "",
            max_features=max_features,
        )
        self._shared = shared

        knn = None
        if needs_knn:
            assert vi is not None
            knn = fit_knn(
                shared=shared,
                vector_index=vi,
                k=self.config.neighbors.k,
            )

        if "typicality" in scores_set:
            t = time.perf_counter()
            logger.info("score 'typicality' | fitting…")
            self.typicality = Typicality().fit(reference, shared=shared)
            logger.info(f"score 'typicality' | done | {time.perf_counter() - t:.1f}s")
        if "support" in scores_set:
            t = time.perf_counter()
            logger.info("score 'support' | fitting…")
            self.support = Support().fit(
                reference,
                vector_index=vi,
                k=self.config.neighbors.k,
                shared=shared,
                knn=knn,
            )
            logger.info(f"score 'support' | done | {time.perf_counter() - t:.1f}s")
        if "consistency" in scores_set:
            t = time.perf_counter()
            logger.info("score 'consistency' | fitting…")
            self.consistency = Consistency().fit(
                reference,
                vector_index=vi,
                k=self.config.neighbors.k,
                shared=shared,
                knn=knn,
            )
            logger.info(f"score 'consistency' | done | {time.perf_counter() - t:.1f}s")
        if "extremity" in scores_set:
            t = time.perf_counter()
            logger.info("score 'extremity' | fitting…")
            self.extremity = Extremity().fit(reference, shared=shared)
            logger.info(f"score 'extremity' | done | {time.perf_counter() - t:.1f}s")
        if "signal" in scores_set:
            assert vi is not None
            t = time.perf_counter()
            logger.info(
                f"score 'signal' | fitting "
                f"(descriptor={signal_descriptor} "
                f"max_train_samples={max_signal_train_samples})…"
            )
            self.signal = Signal().fit(
                reference,
                vector_index=vi,
                shared=shared,
                descriptor=signal_descriptor,
                max_train_samples=max_signal_train_samples,
            )
            logger.info(f"score 'signal' | done | {time.perf_counter() - t:.1f}s")

        self._vector_index_cache = vi
        self.is_fitted_ = True

        self._emit_reference_report(reference, shared)

        t_total = time.perf_counter() - t_start
        fitted = list(self._components())
        logger.success(
            f"Fit complete | {len(fitted)} score(s) [{', '.join(fitted)}] | {t_total:.2f}s"
        )
        logger.rule()
        return self

    def run(self, query: pd.DataFrame) -> RunResult:
        """Score query samples against the fitted reference population.

        Validates and scales the query once, computes the FP-selected kNN
        once for Support + Consistency, and passes the precomputed arrays
        to each component's :meth:`run`.

        Parameters
        ----------
        query:
            DataFrame with the same numeric columns as the reference,
            plus an ``'input'`` SMILES column if Support, Consistency or
            Signal was fit.

        Returns
        -------
        RunResult
            See :class:`RunResult` for the column layout and metadata keys.
        """
        self._check_fitted()
        assert self._shared is not None

        t_start = time.perf_counter()
        components = self._components()
        logger.rule(f"ErsiliaQuality · run · {len(query):,} queries")
        logger.info(f"run | n_query={len(query):,} | scores=[{', '.join(components)}]")

        validate_against_schema(query, self._shared.schema)
        needs_input_col = bool(set(components) & _INDEX_AWARE)
        if needs_input_col and "input" not in query.columns:
            raise SchemaError(
                "Query DataFrame must contain an 'input' column with SMILES strings."
            )

        t = time.perf_counter()
        query_repr = _make_query_repr(self._shared, query)
        logger.info(
            f"run | query scaled | shape={query_repr.shape} | "
            f"{time.perf_counter() - t:.2f}s"
        )

        # FP kNN once, shared by Support and Consistency.
        query_fp_indices: np.ndarray | None = None
        query_fp_distances: np.ndarray | None = None
        query_output_distances: np.ndarray | None = None
        if self.support is not None or self.consistency is not None:
            vi = self._get_vector_index()
            knn = (self.support or self.consistency).knn_  # type: ignore[union-attr]
            t = time.perf_counter()
            query_fp_distances, query_fp_indices = _query_fp_distances(query, vi, knn.k)
            logger.info(
                f"run | FP kNN | k={knn.k} | median dist="
                f"{float(np.median(query_fp_distances)) if len(query) else float('nan'):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )
            if self.consistency is not None:
                assert self._shared.ref_repr is not None
                query_output_distances = _query_output_distances(
                    query_repr, self._shared.ref_repr, query_fp_indices
                )

        columns: dict[str, pd.Series] = {}
        metadata: dict[str, Any] = {"n_reference": len(self._shared.reference_ids)}
        for name, component in components.items():
            t = time.perf_counter()
            if name in ("typicality", "extremity"):
                result = component.run(query, query_repr=query_repr)
            elif name == "support":
                result = component.run(
                    query,
                    query_fp_indices=query_fp_indices,
                    query_fp_distances=query_fp_distances,
                )
            elif name == "consistency":
                result = component.run(
                    query,
                    query_repr=query_repr,
                    query_fp_indices=query_fp_indices,
                    query_fp_distances=query_fp_distances,
                    query_output_distances=query_output_distances,
                )
            else:
                result = component.run(query)
            columns[name] = result.score
            columns[f"{name}_raw"] = result.score_raw
            if hasattr(result, "score_log"):
                columns[f"{name}_log"] = result.score_log
            metadata.update({f"{name}_{k}": v for k, v in result.metadata.items()})
            logger.info(
                f"score {name!r} | mean={float(result.score.mean()):.4f} "
                f"raw mean={float(result.score_raw.mean()):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        scores_df = pd.DataFrame(columns, index=list(query.index))
        logger.scores_summary_table(scores_df)
        means = " · ".join(f"{c}={scores_df[c].mean():.3f}" for c in scores_df.columns)
        logger.success(
            f"Run complete | {len(scores_df):,} queries | {means or 'no scores'} | "
            f"{time.perf_counter() - t_start:.2f}s"
        )
        logger.rule()
        return RunResult(scores=scores_df, metadata=metadata)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write fitted artifacts to a folder.

        Writes ``shared/`` once, ``knn/`` once (iff support or consistency
        was fit), then each fitted component's own subfolder, plus a
        top-level ``manifest.json`` summary (informational only — the
        loader does not consult it).
        """
        self._check_fitted()
        assert self._shared is not None
        folder = pathlib.Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        components = self._components()
        save_shared(self._shared, folder)
        knn_owner = self.support or self.consistency
        if knn_owner is not None:
            save_knn(knn_owner.knn_, folder)
        for component in components.values():
            component.save_component(folder)
        self._write_manifest(folder)
        logger.info(f"Artifacts saved → {folder}")
        return folder

    def _write_manifest(self, folder: pathlib.Path) -> None:
        """Write the top-level ``manifest.json`` summary."""
        assert self._shared is not None
        knn_owner = self.support or self.consistency
        signal_meta = None
        if self.signal is not None:
            signal_meta = {
                "formula_version": SIGNAL_FORMULA_VERSION,
                "descriptor": self.signal.descriptor_,
                "n_features": int(self.signal.backend_.n_features),
            }
        manifest = {
            "format_version": self._shared.metadata.format_version,
            "scores": list(self._components()),
            "n_samples": self._shared.metadata.n_samples,
            "n_features": self._shared.metadata.n_features,
            "n_features_selected": len(self._shared.selected_columns),
            "k": knn_owner.knn_.k if knn_owner is not None else None,
            "signal": signal_meta,
            "library_id": self._shared.metadata.library_id,
            "fit_timestamp": self._shared.metadata.fit_timestamp,
            "eosquality_version": self._shared.metadata.eosquality_version,
        }
        with open(folder / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> ErsiliaQuality:
        """Reconstruct an orchestrator from a saved folder.

        Reads ``shared/`` (and ``knn/`` if present) once, then loads every
        component whose subfolder exists. At least one score subfolder
        must exist. Library / package compatibility is enforced when any
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
        present = [n for n in _SCORE_ORDER if (folder / n).is_dir()]
        if not present:
            raise FileNotFoundError(
                f"No score subfolders found under {folder} — nothing to load."
            )
        logger.info(f"loading artifacts from {folder} | scores=[{', '.join(present)}]")
        shared = load_shared(folder)
        knn = load_knn(folder) if set(present) & _KNN_USERS else None
        _check_artifacts_compatibility(
            shared, has_index_scores=bool(set(present) & _INDEX_AWARE)
        )
        instance = cls(k=knn.k if knn is not None else 5)
        instance._shared = shared
        for name in present:
            setattr(
                instance,
                name,
                _SCORE_CLASSES[name].load(folder, shared=shared, knn=knn),
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
    def reference_consistency_(self) -> float:
        """Mean reference-as-query consistency; requires consistency to be fit."""
        self._check_fitted()
        if self.consistency is None:
            raise RuntimeError(
                "reference_consistency is only defined when consistency has been fit."
            )
        return self.consistency.reference_consistency_

    @property
    def reference_signal_(self) -> float:
        """Mean reference-as-query signal; requires signal to be fit."""
        self._check_fitted()
        if self.signal is None:
            raise RuntimeError(
                "reference_signal is only defined when signal has been fit."
            )
        return self.signal.reference_signal_

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

    def _components(self) -> dict[str, Any]:
        """Fitted components keyed by name, in canonical order."""
        return {
            name: getattr(self, name)
            for name in _SCORE_ORDER
            if getattr(self, name) is not None
        }

    def _get_vector_index(self) -> VectorIndex:
        """Load (and cache) the VectorIndex backing the index-aware scores.

        See :func:`eosquality.scores._helpers._resolve_vector_index`.
        """
        if self._vector_index_cache is not None:
            return self._vector_index_cache
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
        keys = reference["key"].astype(str)
        dupes = keys[keys.duplicated()]
        if len(dupes):
            n_shown = min(5, len(dupes))
            examples = ", ".join(
                f"row {i}: {k!r}" for i, k in list(dupes.items())[:n_shown]
            )
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
            and self.signal is None
        ):
            return
        logger.reference_report_table(
            reference_support=self.support.reference_support_ if self.support else None,
            reference_typicality=(
                self.typicality.reference_typicality_ if self.typicality else None
            ),
            reference_extremity=(
                self.extremity.reference_extremity_ if self.extremity else None
            ),
            reference_consistency=(
                self.consistency.reference_consistency_ if self.consistency else None
            ),
            reference_signal=self.signal.reference_signal_ if self.signal else None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_artifacts_compatibility(
    shared: SharedFitState, has_index_scores: bool
) -> None:
    """Reject artifacts fit against a different reference library or major.

    Only index-aware artifacts are checked (a typicality/extremity-only fit
    has ``library_id == ""`` and is portable). Artifacts fit on the
    canonical library must match this install's :data:`LIBRARY_ID`; those
    fit on a custom index must record its path. The package major version
    must also match.
    """
    if not has_index_scores:
        return
    import importlib.metadata as _md

    from packaging.version import InvalidVersion, Version

    library_id = shared.metadata.library_id
    if library_id != LIBRARY_ID and not shared.metadata.vector_index_path:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit against reference library {library_id!r} but "
            f"this install ships {LIBRARY_ID!r}. Install a compatible "
            "eosquality release or refit against the current library."
        )
    try:
        current = _md.version("eosquality")
        saved_major = Version(shared.metadata.eosquality_version).major
        current_major = Version(current).major
    except (_md.PackageNotFoundError, InvalidVersion):
        return
    if saved_major != current_major:
        raise IncompatibleArtifactsError(
            f"Artifacts were fit with eosquality {shared.metadata.eosquality_version} "
            f"(major={saved_major}) but this install is {current} "
            f"(major={current_major}). Install a matching eosquality release or refit."
        )
