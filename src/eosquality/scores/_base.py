"""Common scaffolding for the per-score components.

Every score class (Typicality, Extremity, Support, Consistency, Signal)
persists the same three things around its own state: the shared fit
state under ``<root>/shared/``, the kNN state under ``<root>/knn/`` (only
for kNN users), and a ``metadata.json`` with fit bookkeeping in its own
subfolder. :class:`ScoreComponent` owns that boilerplate so each score
only implements its own ``_save_own`` / ``_load_own`` pair.

Standalone use (``Support().fit(...).save(root)`` then
``Support.load(root)``) writes and reads ``shared/`` itself. The
:class:`~eosquality.quality.ErsiliaQuality` orchestrator instead writes
``shared/`` and ``knn/`` once and passes the loaded states into each
component's ``load(root, shared=..., knn=...)``, so the (large) shared
state is serialised and deserialised only once per artifact.
"""

from __future__ import annotations

import json
import pathlib
import time
from datetime import datetime, timezone
from typing import Any, ClassVar

from eosquality.knn.load import load_knn
from eosquality.knn.save import save_knn
from eosquality.knn.state import KnnFitState
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState

METADATA_FILE = "metadata.json"


class ScoreComponent:
    """Base class for score components.

    Subclasses set :attr:`NAME` (also the subfolder name), set
    :attr:`USES_KNN` when they depend on :class:`KnnFitState`, implement
    :attr:`is_fitted_`, :meth:`_save_own` and :meth:`_load_own`, and call
    :meth:`_finish_fit` at the end of ``fit``.
    """

    NAME: ClassVar[str] = ""
    USES_SHARED: ClassVar[bool] = True
    USES_KNN: ClassVar[bool] = False
    USES_TRAINING: ClassVar[bool] = False

    def __init__(self) -> None:
        self._shared: SharedFitState | None = None
        self._knn: KnnFitState | None = None
        self._training = None  # TrainingFitState, for training-modality scores
        self._fit_duration_seconds: float | None = None
        self._fit_timestamp: str | None = None

    # ------------------------------------------------------------------
    # Fit bookkeeping
    # ------------------------------------------------------------------

    def _finish_fit(self, t0: float) -> None:
        """Record wall-clock duration since ``t0`` and a UTC timestamp."""
        self._fit_duration_seconds = float(time.perf_counter() - t0)
        self._fit_timestamp = datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, root: str | pathlib.Path) -> pathlib.Path:
        """Persist ``shared/`` (+ ``knn/``) and this component's subfolder.

        Parameters
        ----------
        root : str or pathlib.Path
            Folder to write into.

        Returns
        -------
        pathlib.Path
            ``root``.
        """
        self._check_fitted()
        if self.USES_SHARED:
            assert self._shared is not None
            save_shared(self._shared, root)
        if self.USES_KNN:
            assert self._knn is not None
            save_knn(self._knn, root)
        if self.USES_TRAINING:
            from eosquality.training.state import save_training_state

            save_training_state(self._training, root)
        return self.save_component(root)

    def save_component(self, root: str | pathlib.Path) -> pathlib.Path:
        """Persist only this component's own subfolder (no ``shared/``/``knn/``).

        Used by :class:`~eosquality.quality.ErsiliaQuality`, which writes
        the shared upstream state once for all components.

        Parameters
        ----------
        root : str or pathlib.Path
            Folder to write the component subfolder into.

        Returns
        -------
        pathlib.Path
            ``root``.
        """
        self._check_fitted()
        folder = pathlib.Path(root) / self.NAME
        folder.mkdir(parents=True, exist_ok=True)
        self._save_own(folder)
        meta = {
            "component": self.NAME,
            "fit_timestamp": self._fit_timestamp,
            "fit_duration_seconds": float(self._fit_duration_seconds or 0.0),
            "k": int(self._knn.k) if self.USES_KNN and self._knn is not None else None,
        }
        with open(folder / METADATA_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        return pathlib.Path(root)

    @classmethod
    def load(
        cls,
        root: str | pathlib.Path,
        *,
        shared: SharedFitState | None = None,
        knn: KnnFitState | None = None,
        training=None,
    ):
        """Reconstruct from ``<root>/``.

        ``shared`` / ``knn`` / ``training`` may be passed in when already
        loaded (the orchestrator does this); otherwise they are read from disk.

        Parameters
        ----------
        root : str or pathlib.Path
            Folder holding the component subfolder (and ``shared/`` / ``knn/``).
        shared : SharedFitState, optional
            Already-loaded shared state.
        knn : KnnFitState, optional
            Already-loaded kNN state.
        training : TrainingFitState, optional
            Already-loaded training state (training-mode components).

        Returns
        -------
        ScoreComponent
            The loaded component.
        """
        folder = pathlib.Path(root) / cls.NAME
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected {cls.NAME} artifacts at {folder}, but the folder "
                "does not exist."
            )
        if cls.USES_SHARED and shared is None:
            shared = load_shared(root)
        if cls.USES_KNN and knn is None:
            knn = load_knn(root)
        if cls.USES_TRAINING and training is None:
            from eosquality.training.state import load_training_state

            training = load_training_state(root)
        instance = cls()
        instance._shared = shared
        instance._knn = knn if cls.USES_KNN else None
        instance._training = training if cls.USES_TRAINING else None
        instance._load_own(folder)
        meta_path = folder / METADATA_FILE
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            instance._fit_duration_seconds = float(
                meta.get("fit_duration_seconds", 0.0)
            )
            instance._fit_timestamp = meta.get("fit_timestamp")
        instance._check_fitted()
        return instance

    def _save_own(self, folder: pathlib.Path) -> None:
        raise NotImplementedError

    def _load_own(self, folder: pathlib.Path) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted_(self) -> bool:
        """Whether the component is fitted (or loaded).

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @property
    def shared_(self) -> SharedFitState:
        """The shared fit state.

        Returns
        -------
        SharedFitState
        """
        self._check_fitted()
        assert self._shared is not None
        return self._shared

    @property
    def knn_(self) -> KnnFitState:
        """The kNN state (kNN-using components only).

        Returns
        -------
        KnnFitState
        """
        self._check_fitted()
        if self._knn is None:
            raise AttributeError(f"{type(self).__name__} does not use a kNN state.")
        return self._knn

    @property
    def fit_duration_seconds_(self) -> float | None:
        """Wall-clock fit duration.

        Returns
        -------
        float or None
        """
        return self._fit_duration_seconds

    @property
    def fit_timestamp_(self) -> str | None:
        """UTC fit timestamp (ISO 8601).

        Returns
        -------
        str or None
        """
        return self._fit_timestamp

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted (or loaded) before use."
            )


def require_file(path: pathlib.Path, component: str) -> pathlib.Path:
    """Raise a uniform "incomplete artifact" error when ``path`` is missing.

    Parameters
    ----------
    path : pathlib.Path
        Required file.
    component : str
        Component name for the error message.

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. The {component} artifact is incomplete (or was "
            "written by an incompatible eosquality version) and must be refit."
        )
    return path


def read_json(path: pathlib.Path, component: str) -> dict[str, Any]:
    """Read a required JSON file from a component folder.

    Parameters
    ----------
    path : pathlib.Path
        Required JSON file.
    component : str
        Component name for the error message.

    Returns
    -------
    dict
        The parsed JSON.
    """
    with open(require_file(path, component)) as f:
        return json.load(f)
