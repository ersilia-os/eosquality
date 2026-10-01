"""Feature backends for the :class:`Signal` score.

Two interchangeable descriptor backends drive the same SHAP-Gini score:

- :class:`PhyschemBackend` — RDKit physicochemical descriptors, scaled
  with the scaler that ``eosquality build`` fits on the library.
  Reference rows come from the library's ``physchem_scaled.npy``; query
  rows are computed on demand with the saved scaler params.
- :class:`MaccsBackend` — 166-bit RDKit MACCS keys. Reference rows come
  from the library's ``maccs.npy``; query rows are computed on demand with
  the same function the library build used.

Both backends expose the same interface so :class:`signal.Signal` can
plug in either one. The choice is decided at fit time and baked into
the saved artifact (``umbrella.json``'s ``descriptor`` field) — runs
load the recorded descriptor; there is no run-time override.
"""

from __future__ import annotations

import json
import pathlib
from typing import Union

import numpy as np
import pandas as pd
from eosquality.exceptions import ArtifactVersionError
from eosquality.library.maccs import MACCS_FILE, N_MACCS, compute_maccs
from eosquality.library.physchem import (
    apply_scaler,
    check_descriptor_names,
    compute_physchem_raw,
)
from eosquality.vectorindex import VectorIndex

PHYSCHEM_NAME = "physchem"
MACCS_NAME = "maccs"
DESCRIPTOR_NAMES: tuple[str, ...] = (PHYSCHEM_NAME, MACCS_NAME)
DEFAULT_DESCRIPTOR: str = PHYSCHEM_NAME

PHYSCHEM_SCALER_FILE = "physchem_scaler.json"
PHYSCHEM_REF_MATRIX_FILE = "physchem_scaled.npy"


def _gather_rows(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """``matrix[indices]`` read in ascending row order (fast on a memmap)."""
    order = np.argsort(indices, kind="stable")
    out = np.empty((len(indices), matrix.shape[1]), dtype=matrix.dtype)
    out[order] = matrix[indices[order]]
    return out


class PhyschemBackend:
    """RDKit physchem descriptors with the library-fitted scaler.

    Reference rows are gathered from the library's memory-mapped
    ``physchem_scaled.npy``; query rows are computed on the fly with the
    saved scaler params. The installed RDKit's descriptor list must match
    the one the library was built with (checked on construction).
    """

    name: str = PHYSCHEM_NAME

    def __init__(
        self,
        scaler_params: dict,
        *,
        reference_matrix: np.ndarray | None = None,
    ) -> None:
        check_descriptor_names(scaler_params)
        self._scaler_params = scaler_params
        self._ref_matrix = reference_matrix

    @property
    def n_features(self) -> int:
        return len(self._scaler_params["descriptor_names"])

    def compute_reference_subset(
        self, reference: pd.DataFrame, indices: np.ndarray
    ) -> np.ndarray:
        if self._ref_matrix is None:
            raise RuntimeError(
                "PhyschemBackend has no cached reference matrix; "
                "construct via PhyschemBackend.from_library(vi)."
            )
        return _gather_rows(self._ref_matrix, indices)

    def query_matrix(self, smiles_list: list[str]) -> np.ndarray:
        raw = compute_physchem_raw(smiles_list)
        return apply_scaler(raw, self._scaler_params)

    def save_state(self, folder: pathlib.Path) -> None:
        with open(folder / PHYSCHEM_SCALER_FILE, "w") as f:
            json.dump(self._scaler_params, f)

    @classmethod
    def from_library(cls, vi: VectorIndex) -> "PhyschemBackend":
        library_dir = vi.index_dir
        scaler_path = library_dir / PHYSCHEM_SCALER_FILE
        matrix_path = library_dir / PHYSCHEM_REF_MATRIX_FILE
        if not scaler_path.is_file():
            raise FileNotFoundError(
                f"Physchem scaler params not found at {scaler_path}. "
                "Re-build the vector index (eosquality build)."
            )
        if not matrix_path.is_file():
            raise FileNotFoundError(
                f"Reference physchem matrix not found at {matrix_path}. "
                "Re-build the vector index (eosquality build)."
            )
        with open(scaler_path) as f:
            scaler = json.load(f)
        # Memory-mapped: fit only gathers the train + val rows.
        return cls(
            scaler_params=scaler,
            reference_matrix=np.load(matrix_path, mmap_mode="r"),
        )

    @classmethod
    def load_state(cls, folder: pathlib.Path) -> "PhyschemBackend":
        path = folder / PHYSCHEM_SCALER_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"physchem_scaler.json not found at {path}; refit signal with "
                "descriptor=physchem."
            )
        with open(path) as f:
            return cls(scaler_params=json.load(f))


class MaccsBackend:
    """166-bit RDKit MACCS keys.

    Reference rows are gathered from the library's memory-mapped
    ``maccs.npy``; query rows are computed with
    :func:`eosquality.library.maccs.compute_maccs`, the same function the
    library build used, so both sides share one bit layout. No per-fit
    state is persisted.
    """

    name: str = MACCS_NAME

    def __init__(self, *, reference_matrix: np.ndarray | None = None) -> None:
        self._ref_matrix = reference_matrix

    @property
    def n_features(self) -> int:
        return N_MACCS

    def compute_reference_subset(
        self, reference: pd.DataFrame, indices: np.ndarray
    ) -> np.ndarray:
        if self._ref_matrix is None:
            raise RuntimeError(
                "MaccsBackend has no reference matrix; construct via "
                "MaccsBackend.from_library(vi)."
            )
        return _gather_rows(self._ref_matrix, indices)

    def query_matrix(self, smiles_list: list[str]) -> np.ndarray:
        return compute_maccs(smiles_list)

    def save_state(self, folder: pathlib.Path) -> None:
        return None

    @classmethod
    def from_library(cls, vi: VectorIndex) -> "MaccsBackend":
        path = vi.index_dir / MACCS_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference MACCS matrix not found at {path}. "
                "Re-build the library (eosquality build)."
            )
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape != (vi.n_reference, N_MACCS):
            raise ValueError(
                f"{path} has shape {matrix.shape}; expected "
                f"({vi.n_reference}, {N_MACCS}). Re-build the library."
            )
        return cls(reference_matrix=matrix)

    @classmethod
    def load_state(cls, folder: pathlib.Path) -> "MaccsBackend":
        del folder
        return cls()


DescriptorBackend = Union[PhyschemBackend, MaccsBackend]


def make_backend(name: str, vi: VectorIndex) -> DescriptorBackend:
    """Construct a fit-time descriptor backend from its name."""
    if name == PHYSCHEM_NAME:
        return PhyschemBackend.from_library(vi)
    if name == MACCS_NAME:
        return MaccsBackend.from_library(vi)
    raise ValueError(
        f"Unknown signal descriptor {name!r}; expected one of {DESCRIPTOR_NAMES}."
    )


def load_backend(name: str, folder: pathlib.Path) -> DescriptorBackend:
    """Reconstruct a backend from a saved ``signal/`` folder."""
    if name == PHYSCHEM_NAME:
        return PhyschemBackend.load_state(folder)
    if name == MACCS_NAME:
        return MaccsBackend.load_state(folder)
    raise ArtifactVersionError(
        f"signal artifact at {folder} declares descriptor={name!r}, which is "
        f"not a recognized descriptor in this eosquality install "
        f"(known: {DESCRIPTOR_NAMES}). Refit with a supported descriptor."
    )
