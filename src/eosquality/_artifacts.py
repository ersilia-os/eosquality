"""Saving and loading :class:`~eosquality.quality.ErsiliaQuality` artifacts."""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

from eosquality._registry import LIBRARY_USERS, SCORE_ORDER
from eosquality.exceptions import ArtifactVersionError, IncompatibleArtifactsError
from eosquality.library.identity import LIBRARY_ID
from eosquality.scores.extremity import Extremity
from eosquality.scores.reference_match import ReferenceMatch
from eosquality.scores.training_distance import TrainingDistance
from eosquality.scores.training_match import TrainingMatch
from eosquality.scores.training_physchem import TrainingPhyschem
from eosquality.scores.typicality import Typicality
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState
from eosquality.training import load_training_state, save_training_state
from eosquality.utils.logging import logger

if TYPE_CHECKING:
    from eosquality.quality import ErsiliaQuality


SCORE_CLASSES = {
    "typicality": Typicality,
    "extremity": Extremity,
    "match": ReferenceMatch,
}
# Artifacts layout: one subfolder per modality, either or both present.
REFERENCE_DIR = "reference_mode"
TRAINING_DIR = "training_mode"


def save(eq, path: str | pathlib.Path) -> pathlib.Path:
    """Write fitted artifacts to a folder, one subfolder per modality.

    ``reference_mode/`` holds ``shared/`` and each reference score's subfolder;
    ``training_mode/`` holds ``training_sets/`` and each training score's
    subfolder. ``manifest.json`` at the top is informational only.

    Parameters
    ----------
    eq : ErsiliaQuality
        A fitted orchestrator.
    path : str or pathlib.Path
        Artifacts folder (created if needed). It must not already hold
        artifacts: saving over them would mix the old and the new fit.

    Returns
    -------
    pathlib.Path
        The artifacts folder.

    Raises
    ------
    FileExistsError
        If ``path`` already holds a ``reference_mode/`` or ``training_mode/``.
    """
    eq._check_fitted()
    folder = pathlib.Path(path)
    for mode in (REFERENCE_DIR, TRAINING_DIR):
        if (folder / mode).exists():
            raise FileExistsError(
                f"{folder} already holds artifacts ({mode}/); delete it or save "
                "to a new folder."
            )
    folder.mkdir(parents=True, exist_ok=True)
    if eq._shared is not None:
        root = folder / REFERENCE_DIR
        save_shared(eq._shared, root)
        for component in eq._components().values():
            component.save_component(root)
    save_training(eq, folder)
    write_manifest(eq, folder)
    logger.info(f"Artifacts saved → {folder}")
    return folder


def save_training(eq, folder: pathlib.Path) -> None:
    """Write ``<folder>/training_mode/`` if the training modality is fitted.

    Parameters
    ----------
    eq : ErsiliaQuality
        The orchestrator.
    folder : pathlib.Path
        Artifacts folder.
    """
    if eq._training is None:
        return
    root = folder / TRAINING_DIR
    save_training_state(eq._training, root)
    for component in eq._training_components().values():
        component.save_component(root)


def write_manifest(eq, folder: pathlib.Path) -> None:
    """Write the top-level ``manifest.json`` summary.

    Parameters
    ----------
    eq : ErsiliaQuality
        The fitted orchestrator.
    folder : pathlib.Path
        Artifacts folder.
    """
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
        meta = eq._shared.metadata
        manifest["reference"] = {
            "format_version": meta.format_version,
            "n_samples": meta.n_samples,
            "n_features": meta.n_features,
            "n_features_selected": len(eq._shared.selected_columns),
            "library_id": meta.library_id,
            "fit_timestamp": meta.fit_timestamp,
            "eosquality_version": meta.eosquality_version,
        }
    if eq._training is not None:
        from eosquality.training.state import TRAINING_FORMAT_VERSION

        manifest["training"] = {
            "format_version": TRAINING_FORMAT_VERSION,
            "columns": {
                name: {"n": col.n} for name, col in eq._training.columns.items()
            },
        }
    with open(folder / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def load(eq_cls, path: str | pathlib.Path) -> ErsiliaQuality:
    """Reconstruct an orchestrator from a saved artifacts folder.

    Loads ``reference_mode/`` and/or ``training_mode/``, whichever exist.

    Parameters
    ----------
    eq_cls : type
        :class:`ErsiliaQuality` (or a subclass).
    path : str or pathlib.Path
        Artifacts folder.

    Returns
    -------
    ErsiliaQuality
        A fitted instance with every component found.
    """
    folder = pathlib.Path(path)
    if not folder.exists():
        raise FileNotFoundError(f"No artifacts folder found at: {folder}")
    if not folder.is_dir():
        raise ValueError(
            f"Expected a directory, got a file: {folder}. "
            "Artifacts are stored as a folder — pass the folder path."
        )
    if (folder / "shared").is_dir() or (folder / "training").is_dir():
        raise ArtifactVersionError(
            f"Artifacts at {folder} use the old flat layout; this eosquality "
            f"install expects {REFERENCE_DIR}/ and {TRAINING_DIR}/ subfolders. Refit."
        )
    has_reference = (folder / REFERENCE_DIR).is_dir()
    has_training = (folder / TRAINING_DIR).is_dir()
    if not has_reference and not has_training:
        raise FileNotFoundError(
            f"No {REFERENCE_DIR}/ or {TRAINING_DIR}/ under {folder} — nothing to load."
        )
    instance = eq_cls()
    if has_reference:
        _load_reference(instance, folder / REFERENCE_DIR)
    if has_training:
        _load_training(instance, folder / TRAINING_DIR)
    instance.is_fitted_ = True
    logger.success(
        f"Artifacts loaded from {folder} | modalities={instance.modalities_}"
    )
    return instance


def _load_training(instance, root: pathlib.Path) -> None:
    """Fill ``instance`` with the training modality stored under ``root``."""
    instance._training = load_training_state(root)
    for cls in (TrainingDistance, TrainingPhyschem, TrainingMatch):
        if (root / cls.NAME).is_dir():
            component = cls.load(
                root, shared=instance._shared, training=instance._training
            )
            setattr(instance, cls.NAME, component)


def _load_reference(instance, root: pathlib.Path) -> None:
    """Fill ``instance`` with the reference modality stored under ``root``."""
    present = [n for n in SCORE_ORDER if (root / n).is_dir()]
    if not present:
        raise FileNotFoundError(f"No reference score subfolders under {root}.")
    shared = load_shared(root)
    check_artifacts_compatibility(
        shared, has_library_scores=bool(set(present) & LIBRARY_USERS)
    )
    instance._shared = shared
    for name in present:
        setattr(instance, name, SCORE_CLASSES[name].load(root, shared=shared))


def check_artifacts_compatibility(
    shared: SharedFitState, has_library_scores: bool
) -> None:
    """Reject artifacts fit against a different reference library or major.

    Only artifacts with a library-reading score (``ref_match``) are checked.
    Artifacts fit on the canonical library must match this install's
    :data:`LIBRARY_ID`; those fit on a custom library must record its path.
    The package major version must also match.

    Parameters
    ----------
    shared : SharedFitState
        Loaded shared state of the reference modality.
    has_library_scores : bool
        Whether any loaded reference score reads the reference library.
    """
    if not has_library_scores:
        return
    import importlib.metadata as _md

    from packaging.version import InvalidVersion, Version

    library_id = shared.metadata.library_id
    if library_id != LIBRARY_ID and not shared.metadata.library_path:
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
