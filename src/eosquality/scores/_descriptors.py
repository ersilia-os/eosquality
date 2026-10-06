"""Feature backend for the :class:`Signal` score.

:class:`PhyschemBackend`: RDKit physicochemical descriptors, scaled with the
scaler that ``eosquality build`` fits on the library. Reference rows come
from the library's ``physchem_scaled.npy``; query rows are computed on
demand with the saved scaler params. The descriptor is recorded in the
saved artifact (``umbrella.json``'s ``descriptor`` field); artifacts with
any other descriptor are rejected on load.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from eosquality.exceptions import ArtifactVersionError
from eosquality.library.physchem import (
    apply_scaler,
    check_descriptor_names,
    compute_physchem_raw,
)
from eosquality.vectorindex import VectorIndex

PHYSCHEM_NAME = "physchem"
DESCRIPTOR_NAMES: tuple[str, ...] = (PHYSCHEM_NAME,)
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
        """Number of descriptor features.

        Returns
        -------
        int
        """
        return len(self._scaler_params["descriptor_names"])

    def compute_reference_subset(
        self, reference: pd.DataFrame, indices: np.ndarray
    ) -> np.ndarray:
        """Descriptor rows of the given reference molecules.

        Parameters
        ----------
        reference : pandas.DataFrame
            Reference predictions (unused by library-backed backends).
        indices : numpy.ndarray
            Reference row indices.

        Returns
        -------
        numpy.ndarray
            ``(len(indices), n_features)``.
        """
        if self._ref_matrix is None:
            raise RuntimeError(
                "PhyschemBackend has no cached reference matrix; "
                "construct via PhyschemBackend.from_library(vi)."
            )
        return _gather_rows(self._ref_matrix, indices)

    def query_matrix(self, smiles_list: list[str]) -> np.ndarray:
        """Descriptor matrix of query molecules.

        Parameters
        ----------
        smiles_list : list of str
            Query SMILES.

        Returns
        -------
        numpy.ndarray
            ``(n_query, n_features)``.
        """
        raw = compute_physchem_raw(smiles_list)
        return apply_scaler(raw, self._scaler_params)

    def save_state(self, folder: pathlib.Path) -> None:
        """Persist the per-fit backend state into the ``signal/`` folder.

        Parameters
        ----------
        folder : pathlib.Path
            The component folder.
        """
        with open(folder / PHYSCHEM_SCALER_FILE, "w") as f:
            json.dump(self._scaler_params, f)

    @classmethod
    def from_library(cls, vi: VectorIndex) -> PhyschemBackend:
        """Fit-time backend reading the library's precomputed matrix.

        Parameters
        ----------
        vi : VectorIndex
            The reference library's index (its folder holds the matrix).

        Returns
        -------
        PhyschemBackend
        """
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
    def load_state(cls, folder: pathlib.Path) -> PhyschemBackend:
        """Run-time backend from a saved ``signal/`` folder.

        Parameters
        ----------
        folder : pathlib.Path
            The component folder.

        Returns
        -------
        PhyschemBackend
        """
        path = folder / PHYSCHEM_SCALER_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"physchem_scaler.json not found at {path}; refit signal with "
                "descriptor=physchem."
            )
        with open(path) as f:
            return cls(scaler_params=json.load(f))


DescriptorBackend = PhyschemBackend


def make_backend(name: str, vi: VectorIndex) -> DescriptorBackend:
    """Construct a fit-time descriptor backend from its name.

    Parameters
    ----------
    name : {"physchem"}
        Descriptor identifier.
    vi : VectorIndex
        The reference library's index.

    Returns
    -------
    PhyschemBackend
    """
    if name == PHYSCHEM_NAME:
        return PhyschemBackend.from_library(vi)
    raise ValueError(
        f"Unknown signal descriptor {name!r}; expected one of {DESCRIPTOR_NAMES}."
    )


def load_backend(name: str, folder: pathlib.Path) -> DescriptorBackend:
    """Reconstruct a backend from a saved ``signal/`` folder.

    Parameters
    ----------
    name : str
        Descriptor identifier from ``umbrella.json``.
    folder : pathlib.Path
        The ``signal/`` folder.

    Returns
    -------
    PhyschemBackend
    """
    if name == PHYSCHEM_NAME:
        return PhyschemBackend.load_state(folder)
    raise ArtifactVersionError(
        f"signal artifact at {folder} declares descriptor={name!r}; this "
        f"eosquality install only supports {DESCRIPTOR_NAMES}. Refit."
    )
