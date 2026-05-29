"""Feature backends for the :class:`Signal` score.

Two interchangeable descriptor backends drive the same SHAP-Gini score:

- :class:`PhyschemBackend` — 217 RDKit physicochemical descriptors,
  scaled via the same scaler that ``eosquality build`` fits on the
  library. Reference values are loaded from the precomputed
  ``physchem_scaled.npy`` that ships with the library; query values
  are computed on demand with the saved scaler params.
- :class:`MaccsBackend` — 167-bit RDKit MACCS structural fingerprint.
  Nothing is precomputed at library time; reference and query bits
  are both computed on demand via RDKit (fast enough that on-the-fly
  works for the subsampled training rows + the val slice).

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
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from eosquality.library.physchem import apply_scaler, compute_physchem_raw
from eosquality.vectorindex import VectorIndex


PHYSCHEM_NAME = "physchem"
MACCS_NAME = "maccs"
DESCRIPTOR_NAMES: tuple[str, ...] = (PHYSCHEM_NAME, MACCS_NAME)
DEFAULT_DESCRIPTOR: str = PHYSCHEM_NAME

PHYSCHEM_SCALER_FILE = "physchem_scaler.json"
PHYSCHEM_REF_MATRIX_FILE = "physchem_scaled.npy"

MACCS_N_FEATURES: int = 167
MACCS_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"maccs_{i:03d}" for i in range(MACCS_N_FEATURES)
)


class PhyschemBackend:
    """217-dim RDKit physchem descriptors with library-fitted scaler.

    Reference values are loaded once from the library's
    ``physchem_scaled.npy`` and cached as ``_ref_matrix``; query values
    are computed on the fly with the saved scaler params so they match
    the reference scaling bit-for-bit.
    """

    name: str = PHYSCHEM_NAME

    def __init__(
        self,
        scaler_params: dict,
        *,
        reference_matrix: np.ndarray | None = None,
    ) -> None:
        self._scaler_params = scaler_params
        self._ref_matrix = reference_matrix

    @property
    def n_features(self) -> int:
        return len(self._scaler_params["descriptor_names"])

    @property
    def feature_names(self) -> list[str]:
        return list(self._scaler_params["descriptor_names"])

    @property
    def scaler_params(self) -> dict:
        return self._scaler_params

    def compute_reference_subset(
        self, reference: pd.DataFrame, indices: np.ndarray
    ) -> np.ndarray:
        if self._ref_matrix is None:
            raise RuntimeError(
                "PhyschemBackend has no cached reference matrix; "
                "construct via PhyschemBackend.from_library(vi)."
            )
        return self._ref_matrix[indices]

    def query_matrix(self, smiles_list: list[str]) -> np.ndarray:
        raw = compute_physchem_raw(smiles_list)
        return apply_scaler(raw, self._scaler_params)

    def save_state(self, folder: pathlib.Path) -> None:
        with open(folder / PHYSCHEM_SCALER_FILE, "w") as f:
            json.dump(self._scaler_params, f)

    @classmethod
    def from_library(cls, vi: VectorIndex) -> "PhyschemBackend":
        library_dir = pathlib.Path(vi._h5_path).parent
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
        return cls(scaler_params=scaler, reference_matrix=np.load(matrix_path))

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
    """167-bit RDKit MACCS structural fingerprint, computed on demand.

    No library-level precomputation: reference + query bits are
    computed fresh via :func:`rdkit.Chem.MACCSkeys.GenMACCSKeys`. Fast
    enough that on-the-fly works at fit time (for the subsampled train
    rows + the val slice) and at run time. Bit 0 is always 0 (RDKit
    placeholder); kept in the matrix since XGBoost + SHAP handle the
    constant column without issue.
    """

    name: str = MACCS_NAME

    @property
    def n_features(self) -> int:
        return MACCS_N_FEATURES

    @property
    def feature_names(self) -> list[str]:
        return list(MACCS_FEATURE_NAMES)

    def compute_reference_subset(
        self, reference: pd.DataFrame, indices: np.ndarray
    ) -> np.ndarray:
        return self.query_matrix(list(reference["input"].iloc[indices]))

    def query_matrix(self, smiles_list: list[str]) -> np.ndarray:
        out = np.zeros((len(smiles_list), MACCS_N_FEATURES), dtype=np.uint8)
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            bits = MACCSkeys.GenMACCSKeys(mol)
            out[i] = np.array(list(bits), dtype=np.uint8)
        return out

    def save_state(self, folder: pathlib.Path) -> None:
        # No per-fit parameters; the descriptor identifier in umbrella.json
        # is enough to reconstruct the backend at load time.
        return None

    @classmethod
    def from_library(cls, vi: VectorIndex) -> "MaccsBackend":
        # MACCS is library-independent; the VectorIndex argument is
        # accepted for symmetry with PhyschemBackend.from_library.
        del vi
        return cls()

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
    raise FileNotFoundError(
        f"signal artifact at {folder} declares descriptor={name!r}, which is "
        f"not a recognized descriptor in this eosquality install "
        f"(known: {DESCRIPTOR_NAMES}). Refit with a supported descriptor."
    )
