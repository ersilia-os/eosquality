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
from eosquality.scores.signal import Signal
from eosquality.scores.support import Support
from eosquality.scores.typicality import Typicality
from eosquality.shared.fit import DEFAULT_MAX_FEATURES, fit_shared
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
_INDEX_AWARE = frozenset({"support", "consistency", "signal"})
_KNN_USERS = frozenset({"support", "consistency"})


@dataclass
class RunResult:
    """Combined result returned by :meth:`ErsiliaQuality.run`.

    ``scores`` is a per-query DataFrame whose columns are exactly the
    fitted components, in canonical order: ``typicality``, ``extremity``,
    ``support``, ``consistency``, ``signal``. A fit that skips some
    components yields a DataFrame missing those columns; the order of
    the remaining columns is unchanged.
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
    ) -> "ErsiliaQuality":
        """Fit the selected scores on a reference DataFrame.

        Parameters
        ----------
        scores:
            Which components to fit. Defaults to ``DEFAULT_SCORES``
            (``typicality``, ``support``, ``consistency``, ``extremity``).
            Pass e.g. ``scores=["typicality"]`` to skip the vector index
            entirely. ``"signal"`` is provisionally opt-in — request it
            explicitly (e.g. ``scores=DEFAULT_SCORES + ("signal",)``)
            since it pays an ``N+1``-XGBoost training cost. Valid names
            are listed in ``ALL_SCORES``.
        max_features:
            Cap on the number of features retained after fit-time
            correlation-cluster medoid selection. Defaults to
            ``DEFAULT_MAX_FEATURES`` (= 10). Pass ``None`` to disable
            reduction. Support is unaffected (it uses fingerprints only);
            typicality, extremity, consistency, and signal see the
            reduced set.
        max_signal_train_samples:
            Cap on the number of training rows the ``signal`` score
            actually fits its XGBoost models on. Defaults to ``1000``
            for fast iteration during development. Pass ``None`` or
            ``0`` to use the full training slice (~80% of the
            reference). The val slice is never subsampled — early
            stopping and calibration always see the full validation
            set. Ignored when ``signal`` is not in ``scores``.
        signal_descriptor:
            Feature backend the ``signal`` score uses. ``"physchem"``
            (default) — 217 RDKit physicochemical descriptors (loaded
            precomputed from the library). ``"maccs"`` — 167-bit RDKit
            MACCS structural fingerprint (computed on demand at fit +
            run time). The chosen descriptor is recorded in the saved
            artifact and used unchanged at run time. Ignored when
            ``signal`` is not in ``scores``.
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
        using_canonical_library = needs_index and not vector_index

        if reference.empty:
            raise SchemaError("Reference DataFrame is empty.")
        if needs_index:
            self._validate_input_column(reference)

        t_start = time.perf_counter()
        logger.rule(f"ErsiliaQuality · fit · {eos_id} {version}")
        logger.info(
            f"fit | eos_id={eos_id} version={version} "
            f"scores=[{', '.join(sorted(scores_set))}]"
        )

        vi: VectorIndex | None = None
        library_id = ""

        if needs_index:
            t = time.perf_counter()
            logger.info("vector index | loading…")
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
            logger.info(
                f"vector index | loaded | n_ref={len(vi._smiles):,} | "
                f"{time.perf_counter() - t:.1f}s"
            )

        if not ignore_size and len(reference) < MIN_REFERENCE_SAMPLES:
            raise ValueError(
                f"Reference dataset has {len(reference):,} rows. "
                f"Fitting requires at least {MIN_REFERENCE_SAMPLES:,} rows for reliable results. "
                "Pass ignore_size=True to bypass this check (not recommended for production use)."
            )

        self._check_unique_keys(reference)

        shared, _ = fit_shared(
            reference,
            eos_id=eos_id,
            version=version,
            library_id=library_id,
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
        fitted = [
            n
            for n, v in (
                ("typicality", self.typicality),
                ("extremity", self.extremity),
                ("support", self.support),
                ("consistency", self.consistency),
                ("signal", self.signal),
            )
            if v is not None
        ]
        logger.success(
            f"Fit complete | {len(fitted)} score(s) [{', '.join(fitted)}] | {t_total:.2f}s"
        )
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
        logger.rule(f"ErsiliaQuality · run · {len(query):,} queries")
        loaded_scores = [
            n
            for n, v in (
                ("typicality", self.typicality),
                ("extremity", self.extremity),
                ("support", self.support),
                ("consistency", self.consistency),
                ("signal", self.signal),
            )
            if v is not None
        ]
        logger.info(
            f"run | n_query={len(query):,} | scores=[{', '.join(loaded_scores)}]"
        )

        t = time.perf_counter()
        validate_against_schema(query, self._shared.schema)
        logger.info(f"run | schema validated | {time.perf_counter() - t:.2f}s")

        needs_input_col = (
            self.support is not None
            or self.consistency is not None
            or self.signal is not None
        )
        if needs_input_col and "input" not in query.columns:
            raise SchemaError(
                "Query DataFrame must contain an 'input' column with SMILES strings."
            )

        t = time.perf_counter()
        logger.info("run | preprocessing query (eosframes transform + feature filter)…")
        pipeline = PreprocessPipeline.from_state(
            {
                "schema": self._shared.schema,
                "scaler_params": self._shared.scaler_params,
                "binary_class_freq": self._shared.binary_class_freq,
            }
        )
        query_repr = self._shared.filter_features(pipeline.transform(query))
        logger.info(
            f"run | query_repr ready | shape={query_repr.shape} | "
            f"{time.perf_counter() - t:.2f}s"
        )

        query_fp_indices: np.ndarray | None = None
        query_fp_distances: np.ndarray | None = None
        query_output_distances: np.ndarray | None = None
        vi: VectorIndex | None = None
        if self.support is not None or self.consistency is not None:
            t = time.perf_counter()
            logger.info("run | resolving vector index…")
            vi = self._get_vector_index()
            logger.info(
                f"run | vector index ready | n_ref={len(vi._smiles):,} | "
                f"{time.perf_counter() - t:.2f}s"
            )
            knn = (self.support or self.consistency)._knn  # type: ignore[union-attr]
            assert knn is not None
            t = time.perf_counter()
            logger.info(f"run | FP kNN query | k={knn.k} | {len(query):,} queries…")
            query_fp_distances, query_fp_indices = _query_fp_distances(query, vi, knn.k)
            logger.info(
                f"run | FP kNN done | median dist="
                f"{float(np.median(query_fp_distances)):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )
            if self.consistency is not None:
                t = time.perf_counter()
                logger.info(
                    f"run | output-space neighbor distances | "
                    f"{len(query):,} × k={knn.k}…"
                )
                query_output_distances = _query_output_distances(
                    query_repr, self._shared.ref_repr, query_fp_indices
                )
                logger.info(
                    f"run | output-space distances done | "
                    f"{time.perf_counter() - t:.2f}s"
                )
        # Signal is now self-contained: it computes its own physchem
        # descriptors from query SMILES and uses the scaler params it
        # persisted in its own subfolder, so no vector-index lookup is
        # needed here.

        columns: dict[str, pd.Series] = {}
        metadata: dict[str, Any] = {"n_reference": len(self._shared.reference_ids)}

        if self.typicality is not None:
            t = time.perf_counter()
            logger.info("score 'typicality' | running…")
            typicality_result = self.typicality.run(query, query_repr=query_repr)
            columns["typicality"] = typicality_result.score
            columns["typicality_raw"] = typicality_result.score_raw
            metadata.update(typicality_result.metadata)
            logger.info(
                f"score 'typicality' | done | calibrated mean="
                f"{float(typicality_result.score.mean()):.4f} "
                f"raw mean={float(typicality_result.score_raw.mean()):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        if self.extremity is not None:
            t = time.perf_counter()
            logger.info("score 'extremity' | running…")
            extremity_result = self.extremity.run(query, query_repr=query_repr)
            columns["extremity"] = extremity_result.score
            columns["extremity_raw"] = extremity_result.score_raw
            metadata.update(extremity_result.metadata)
            logger.info(
                f"score 'extremity' | done | calibrated mean="
                f"{float(extremity_result.score.mean()):.4f} "
                f"raw mean={float(extremity_result.score_raw.mean()):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        if self.support is not None:
            t = time.perf_counter()
            logger.info("score 'support' | running…")
            support_result = self.support.run(
                query,
                query_fp_indices=query_fp_indices,
                query_fp_distances=query_fp_distances,
            )
            columns["support"] = support_result.score
            columns["support_raw"] = support_result.score_raw
            metadata.update(support_result.metadata)
            logger.info(
                f"score 'support' | done | calibrated mean="
                f"{float(support_result.score.mean()):.4f} "
                f"raw mean dist={float(support_result.score_raw.mean()):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        if self.consistency is not None:
            t = time.perf_counter()
            logger.info("score 'consistency' | running…")
            consistency_result = self.consistency.run(
                query,
                query_repr=query_repr,
                query_fp_indices=query_fp_indices,
                query_fp_distances=query_fp_distances,
                query_output_distances=query_output_distances,
            )
            columns["consistency"] = consistency_result.score
            columns["consistency_raw"] = consistency_result.score_raw
            # Prefix Consistency's metadata keys so they don't collide with
            # Support's ``n_reference`` / ``k`` (both index-aware scores
            # report the same field names).
            metadata.update(
                {f"consistency_{k}": v for k, v in consistency_result.metadata.items()}
            )
            logger.info(
                f"score 'consistency' | done | calibrated mean="
                f"{float(consistency_result.score.mean()):.4f} "
                f"raw mean dist={float(consistency_result.score_raw.mean()):.4f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        if self.signal is not None:
            t = time.perf_counter()
            logger.info("score 'signal' | running…")
            signal_result = self.signal.run(query)
            columns["signal"] = signal_result.score
            columns["signal_raw"] = signal_result.score_raw
            metadata.update(
                {f"signal_{k}": v for k, v in signal_result.metadata.items()}
            )
            logger.info(
                f"score 'signal' | done | calibrated mean="
                f"{float(signal_result.score.mean()):.4f} "
                f"features mean={float(signal_result.score_raw.mean()):.2f} | "
                f"{time.perf_counter() - t:.2f}s"
            )

        ordered = [
            c
            for c in (
                "typicality",
                "typicality_raw",
                "extremity",
                "extremity_raw",
                "support",
                "support_raw",
                "consistency",
                "consistency_raw",
                "signal",
                "signal_raw",
            )
            if c in columns
        ]
        scores_df = pd.DataFrame(
            {c: columns[c] for c in ordered}, index=list(query.index)
        )

        logger.scores_summary_table(scores_df)
        t_total = time.perf_counter() - t_start
        means = " · ".join(f"{c}={scores_df[c].mean():.3f}" for c in scores_df.columns)
        logger.success(
            f"Run complete | {len(scores_df):,} queries | {means or 'no scores'} | {t_total:.2f}s"
        )
        logger.rule()
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
        if self.signal is not None:
            self.signal.save(folder)
        self._write_manifest(folder)
        logger.info(f"Artifacts saved → {folder}")
        return folder

    def _write_manifest(self, folder: pathlib.Path) -> None:
        """Write the top-level ``manifest.json`` summary."""
        import json

        assert self._shared is not None
        scores: list[str] = []
        k_used: int | None = None
        signal_meta: dict[str, Any] | None = None
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
        if self.signal is not None:
            scores.append("signal")
            from eosquality.scores.signal import SIGNAL_FORMULA_VERSION

            signal_meta = {
                "formula_version": SIGNAL_FORMULA_VERSION,
                "descriptor": self.signal.descriptor_,
                "n_features": int(self.signal.backend_.n_features),
            }
        manifest = {
            "scores": scores,
            "n_samples": self._shared.metadata.n_samples,
            "n_features": self._shared.metadata.n_features,
            "n_features_selected": len(self._shared.selected_columns),
            "k": k_used,
            "signal": signal_meta,
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
        logger.info(f"loading artifacts from {folder}")
        instance = cls()
        if (folder / "typicality").is_dir():
            logger.info("  loading typicality…")
            instance.typicality = Typicality.load(folder)
        if (folder / "support").is_dir():
            logger.info("  loading support…")
            instance.support = Support.load(folder)
        if (folder / "consistency").is_dir():
            logger.info("  loading consistency…")
            instance.consistency = Consistency.load(folder)
        if (folder / "extremity").is_dir():
            logger.info("  loading extremity…")
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
        if first_shared is None and (folder / "signal").is_dir():
            # Signal can be the only score on disk; load shared directly.
            from eosquality.shared.load import load_shared

            first_shared = load_shared(folder)
        if first_shared is None:
            raise FileNotFoundError(
                f"No score subfolders found under {folder} — nothing to load."
            )
        if (folder / "signal").is_dir():
            instance.signal = Signal.load(folder, shared=first_shared)
        instance._shared = first_shared
        _check_artifacts_compatibility(
            instance._shared,
            has_index_scores=any(
                s is not None
                for s in (instance.support, instance.consistency, instance.signal)
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

    def _get_vector_index(self) -> VectorIndex:
        """Load (and cache) the VectorIndex backing the index-aware scores.

        Resolution is by ``shared.metadata.library_id`` via the canonical
        library resolver — the artifact does not store an index path.
        """
        if self._vector_index_cache is not None:
            return self._vector_index_cache
        if self.support is None and self.consistency is None and self.signal is None:
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
            and self.signal is None
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
            reference_consistency=self.consistency.reference_consistency_
            if self.consistency
            else None,
            reference_signal=self.signal.reference_signal_ if self.signal else None,
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
