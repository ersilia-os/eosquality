"""The reference library as the package reads it: its SMILES and its match keys.

A library folder (``data/indices/<LIBRARY_ID>/``) holds:

- ``smiles.csv`` — one ``smiles`` column, the library molecules in order;
- ``metadata.json`` — ``library_name`` (its identity), ``n_samples``, the RDKit
  version the keys were built with, build info;
- ``connectivity_keys.npz`` — the sorted unique InChIKey connectivity layers of
  the library molecules and of their Murcko scaffolds, written once by
  ``eosquality build`` and used by ``ref_match`` / ``ref_scaffold``.

The reference predictions of a model must be for exactly these molecules, in
this order (:meth:`ReferenceLibrary.validate_smiles`).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

SMILES_FILE = "smiles.csv"
METADATA_FILE = "metadata.json"
KEYS_FILE = "connectivity_keys.npz"


class ReferenceLibrary:
    """A reference-library folder.

    Parameters
    ----------
    path : str or pathlib.Path
        The library folder.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"Reference library folder not found: {self.path}")
        self._smiles: list[str] | None = None
        self._keys: tuple[np.ndarray, np.ndarray] | None = None
        self._metadata: dict | None = None

    @classmethod
    def load(cls, path: str | pathlib.Path | None = None) -> ReferenceLibrary:
        """Open a library folder, by default the resolved canonical library.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            A custom library folder (tests, other libraries).

        Returns
        -------
        ReferenceLibrary
        """
        if path is None:
            from eosquality.library.identity import reference_library_path

            path = reference_library_path()
        return cls(path)

    @property
    def metadata(self) -> dict:
        """The folder's ``metadata.json``.

        Returns
        -------
        dict
        """
        if self._metadata is None:
            file = self.path / METADATA_FILE
            if not file.is_file():
                raise FileNotFoundError(f"Missing {file}: not a reference library.")
            with open(file) as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def library_name(self) -> str:
        """Library identity (``metadata.json`` ``library_name``).

        Returns
        -------
        str
        """
        return str(self.metadata.get("library_name", ""))

    @property
    def smiles(self) -> list[str]:
        """The library SMILES, in order.

        Returns
        -------
        list of str
        """
        if self._smiles is None:
            import pandas as pd

            file = self.path / SMILES_FILE
            if not file.is_file():
                raise FileNotFoundError(f"Missing {file}: not a reference library.")
            self._smiles = pd.read_csv(file)["smiles"].astype(str).tolist()
        return self._smiles

    @property
    def n_reference(self) -> int:
        """Number of library molecules.

        Returns
        -------
        int
        """
        return len(self.smiles)

    def validate_smiles(self, smiles: list[str]) -> None:
        """Check that ``smiles`` equals the library SMILES, row by row.

        Parameters
        ----------
        smiles : list of str
            The ``input`` column of the model's reference predictions.

        Raises
        ------
        ValueError
            If the lists differ in length or in any element.
        """
        library = self.smiles
        if len(smiles) != len(library):
            raise ValueError(
                f"SMILES count mismatch: the reference predictions have "
                f"{len(smiles)} rows, the reference library has {len(library)} "
                "molecules."
            )
        mismatches = np.flatnonzero(
            np.asarray(smiles, dtype=object) != np.asarray(library, dtype=object)
        ).tolist()
        if mismatches:
            n_shown = min(3, len(mismatches))
            raise ValueError(
                f"SMILES mismatch at {len(mismatches)} row(s). "
                f"First {n_shown}: rows {mismatches[:n_shown]}. "
                "The 'input' column of the reference predictions must match the "
                "reference library SMILES, in the same order."
            )

    def _check_rdkit_version(self) -> None:
        """Raise if RDKit differs from the one the match keys were built with.

        Libraries built before the version was recorded are not checked.
        """
        stored = self.metadata.get("rdkit_version")
        if not stored:
            return
        from rdkit import __version__ as current

        if stored != current:
            from eosquality.exceptions import IncompatibleArtifactsError

            raise IncompatibleArtifactsError(
                f"RDKit version mismatch: the reference library's match keys were "
                f"built with RDKit {stored}, but this environment has RDKit "
                f"{current}. InChIKeys and scaffolds can differ between RDKit "
                f"versions, which would silently corrupt ref_match and "
                f"ref_scaffold. Install RDKit {stored}, or rebuild the library "
                "with 'eosquality build' (or exclude ref_match)."
            )

    def match_keys(self) -> tuple[np.ndarray, np.ndarray]:
        """Sorted unique connectivity layers of the molecules and of their scaffolds.

        Returns
        -------
        tuple of numpy.ndarray
            ``(molecules, scaffolds)``.
        """
        if self._keys is None:
            self._check_rdkit_version()
            file = self.path / KEYS_FILE
            if not file.is_file():
                raise FileNotFoundError(
                    f"Missing {file}. The reference library has no match keys: "
                    "run 'eosquality setup' to fetch a current library, or "
                    "'eosquality build' to compute them."
                )
            with np.load(file, allow_pickle=False) as arrays:
                self._keys = (arrays["molecules"], arrays["scaffolds"])
        return self._keys
