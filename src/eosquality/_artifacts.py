"""Saving, loading and extending :class:`~eosquality.quality.ErsiliaQuality` artifacts."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

import pandas as pd

from eosquality._registry import (
    INDEX_AWARE,
    KNN_USERS,
    SCORE_CLASSES,
    SCORE_ORDER,
    TRAINING_ORDER,
)
from eosquality.config import ErsiliaQualityConfig, NeighborConfig
from eosquality.exceptions import IncompatibleArtifactsError
from eosquality.knn.load import load_knn
from eosquality.knn.save import save_knn
from eosquality.library.identity import LIBRARY_ID
from eosquality.scores.signal import SIGNAL_FORMULA_VERSION
from eosquality.scores.training_domain import TrainingDomain
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.training import load_training_state, save_training_state
from eosquality.utils.logging import logger

if TYPE_CHECKING:
    from eosquality.quality import ErsiliaQuality


def save(eq, path: str | pathlib.Path) -> pathlib.Path:
    """Write fitted artifacts to a folder.

    Reference modality: ``shared/`` once, ``knn/`` once (iff support or
    consistency was fit), then each component's subfolder. Training
    modality: ``training/`` (sets + per-column indices) and each
    training component's subfolder. Plus a top-level ``manifest.json``
    summary (informational only — the loader does not consult it).
    """
    eq._check_fitted()
    folder = pathlib.Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    if eq._shared is not None:
        save_shared(eq._shared, folder)
        knn_owner = eq.support or eq.consistency
        if knn_owner is not None:
            save_knn(knn_owner.knn_, folder)
        for component in eq._components().values():
            component.save_component(folder)
    save_training(eq, folder)
    write_manifest(eq, folder)
    logger.info(f"Artifacts saved → {folder}")
    return folder


def save_training(eq, folder: pathlib.Path) -> None:
    if eq._training is None:
        return
    save_training_state(eq._training, folder)
    for component in eq._training_components().values():
        component.save_component(folder)


def add_training(
    eq_cls,
    path: str | pathlib.Path,
    training: str | pathlib.Path,
    training_predictions: str | pathlib.Path | pd.DataFrame | None = None,
    *,
    eos_id: str | None = None,
    version: str | None = None,
) -> ErsiliaQuality:
    """Add the training modality to an existing artifacts folder in place.

    The reference-modality files are left untouched; ``training/``,
    ``training_domain/`` and ``manifest.json`` are written. Refuses if
    the artifacts already hold a training modality or belong to another
    model.
    """
    folder = pathlib.Path(path)
    if (folder / "training").exists():
        raise FileExistsError(
            f"{folder} already has a training modality; fit into a new folder "
            "to replace it."
        )
    instance = load(eq_cls, folder)
    instance.fit_training(
        training, training_predictions, eos_id=eos_id, version=version
    )
    save_training(instance, folder)
    write_manifest(instance, folder)
    logger.info(f"Training modality added → {folder}")
    return instance


def write_manifest(eq, folder: pathlib.Path) -> None:
    """Write the top-level ``manifest.json`` summary."""
    eos_id, version = eq._model_id()
    manifest: dict[str, Any] = {
        "eos_id": eos_id,
        "version": version,
        "modalities": [
            m
            for m, present in (
                ("reference", eq._shared is not None),
                ("training", eq._training is not None),
            )
            if present
        ],
        "scores": list(eq._components()) + list(eq._training_components()),
    }
    if eq._shared is not None:
        knn_owner = eq.support or eq.consistency
        signal_meta = None
        if eq.signal is not None:
            signal_meta = {
                "formula_version": SIGNAL_FORMULA_VERSION,
                "descriptor": eq.signal.descriptor_,
                "n_features": int(eq.signal.backend_.n_features),
            }
        meta = eq._shared.metadata
        manifest["reference"] = {
            "format_version": meta.format_version,
            "n_samples": meta.n_samples,
            "n_features": meta.n_features,
            "n_features_selected": len(eq._shared.selected_columns),
            "k": knn_owner.knn_.k if knn_owner is not None else None,
            "signal": signal_meta,
            "library_id": meta.library_id,
            "fit_timestamp": meta.fit_timestamp,
            "eosquality_version": meta.eosquality_version,
        }
    if eq._training is not None:
        from eosquality.training.state import TRAINING_FORMAT_VERSION

        manifest["training"] = {
            "format_version": TRAINING_FORMAT_VERSION,
            "columns": {
                name: {
                    "n": col.n,
                    "has_y": col.has_y,
                    "y_kind": col.y_kind,
                    "has_pred": col.has_pred,
                }
                for name, col in eq._training.columns.items()
            },
        }
    with open(folder / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load(eq_cls, path: str | pathlib.Path) -> ErsiliaQuality:
    """Reconstruct an orchestrator from a saved folder.

    Loads the reference modality if ``shared/`` exists (reading it and
    ``knn/`` once) and the training modality if ``training/`` exists,
    then every component whose subfolder is present. Library / package
    compatibility is enforced when an index-aware reference score is
    found.
    """
    folder = pathlib.Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"No artifacts folder found at: {folder}")
    if not folder.is_dir():
        raise ValueError(
            f"Expected a directory, got a file: {folder}. "
            "Artifacts are stored as a folder — pass the folder path."
        )
    present = [n for n in SCORE_ORDER if (folder / n).is_dir()]
    present_training = [n for n in TRAINING_ORDER if (folder / n).is_dir()]
    if not present and not present_training:
        raise FileNotFoundError(
            f"No score subfolders found under {folder} — nothing to load."
        )
    logger.info(
        f"loading artifacts from {folder} | scores="
        f"[{', '.join(present + present_training)}]"
    )
    instance = eq_cls()
    knn = None
    if present:
        shared = load_shared(folder)
        knn = load_knn(folder) if set(present) & KNN_USERS else None
        check_artifacts_compatibility(
            shared, has_index_scores=bool(set(present) & INDEX_AWARE)
        )
        instance._shared = shared
        for name in present:
            setattr(
                instance,
                name,
                SCORE_CLASSES[name].load(folder, shared=shared, knn=knn),
            )
    if present_training:
        training = load_training_state(folder)
        instance._training = training
        instance.training_domain = TrainingDomain.load(
            folder, shared=instance._shared, training=training
        )
    if knn is not None:
        instance.config = ErsiliaQualityConfig(neighbors=NeighborConfig(k=knn.k))
    instance.is_fitted_ = True
    logger.success(f"Artifacts loaded from {folder}")
    return instance


def check_artifacts_compatibility(
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
