"""The library's physchem descriptors as a cache for the training scores.

Describing a molecule takes about 6 ms, and a good part of the molecules a
user brings (training sets, queries) is already in the reference library. So
``eosquality build`` also stores the raw descriptors of the library's
(standardised) molecules, and :func:`describe` reads them from there instead
of recomputing them.

Two files in the library folder, sharing their row order:

- ``physchem_hashes.npy`` — sorted unique ``uint64`` hashes of the standardised
  SMILES (:func:`smiles_hashes`);
- ``physchem_raw.npy`` — the ``(n, N_DESCRIPTORS)`` float32 descriptors, as
  :func:`~eosquality.library.physchem.compute_physchem_raw` returns them.

The cache is optional and changes no value: a molecule found there gets exactly
the row it would have been computed to. It is ignored (with a debug message)
when it is missing, was built with another RDKit, or has another descriptor set.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Iterable

import numpy as np

from eosquality.library.physchem import N_DESCRIPTORS, compute_physchem_raw
from eosquality.utils.logging import logger

HASHES_FILE = "physchem_hashes.npy"
MATRIX_FILE = "physchem_raw.npy"
CACHE_FILES = (HASHES_FILE, MATRIX_FILE)

_loaded: dict[str, PhyschemCache | None] = {}  # library folder → its cache


def smiles_hashes(smiles: Iterable[str]) -> np.ndarray:
    """Stable 64-bit hash of each SMILES (the same in every process).

    Parameters
    ----------
    smiles : iterable of str
        Standardised SMILES.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` uint64.
    """
    digests = b"".join(
        hashlib.blake2b(s.encode(), digest_size=8).digest() for s in smiles
    )
    return np.frombuffer(digests, dtype="<u8").astype(np.uint64)


def write_cache(folder: str | pathlib.Path, standardised: list[str]) -> int:
    """Describe the library molecules and write the cache files into ``folder``.

    Parameters
    ----------
    folder : str or pathlib.Path
        Library folder.
    standardised : list of str
        The library's standardised SMILES (repeats allowed).

    Returns
    -------
    int
        Number of cached molecules (distinct hashes).
    """
    folder = pathlib.Path(folder)
    hashes, first = np.unique(smiles_hashes(standardised), return_index=True)
    ordered = [standardised[i] for i in first]  # in hash order: no copy afterwards
    matrix = compute_physchem_raw(
        ordered, show_progress=True, label="library physchem descriptors"
    )
    np.save(folder / HASHES_FILE, hashes)
    np.save(folder / MATRIX_FILE, matrix)
    return len(hashes)


class PhyschemCache:
    """A library's cached descriptors, memory-mapped.

    Parameters
    ----------
    hashes : numpy.ndarray
        Sorted unique molecule hashes.
    matrix : numpy.ndarray
        Descriptors, one row per hash.
    """

    def __init__(self, hashes: np.ndarray, matrix: np.ndarray) -> None:
        self.hashes = hashes
        self.matrix = matrix

    @classmethod
    def load(cls, folder: str | pathlib.Path) -> PhyschemCache | None:
        """Open the cache of a library folder, or ``None`` when it cannot be used.

        Parameters
        ----------
        folder : str or pathlib.Path
            Library folder.

        Returns
        -------
        PhyschemCache or None
        """
        folder = pathlib.Path(folder)
        if not all((folder / name).is_file() for name in CACHE_FILES):
            return None
        from eosquality.library.reference import ReferenceLibrary

        stored = ReferenceLibrary(folder).metadata.get("rdkit_version")
        from rdkit import __version__ as current

        if stored and stored != current:
            logger.debug(f"physchem cache ignored: built with RDKit {stored}")
            return None
        hashes = np.load(folder / HASHES_FILE)
        matrix = np.load(folder / MATRIX_FILE, mmap_mode="r")
        if matrix.shape != (len(hashes), N_DESCRIPTORS):
            logger.debug(f"physchem cache ignored: shape {matrix.shape}")
            return None
        return cls(hashes, matrix)

    def rows_of(self, smiles: list[str]) -> np.ndarray:
        """Row of each standardised SMILES in the cache.

        Parameters
        ----------
        smiles : list of str
            Standardised SMILES.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` int, ``-1`` for a molecule that is not cached.
        """
        wanted = smiles_hashes(smiles)
        pos = np.minimum(np.searchsorted(self.hashes, wanted), len(self.hashes) - 1)
        return np.where(self.hashes[pos] == wanted, pos, -1)


def default_cache() -> PhyschemCache | None:
    """The cache of the resolved canonical library, if there is one.

    Never raises and never downloads: no library or no cache gives ``None``.

    Returns
    -------
    PhyschemCache or None
    """
    from eosquality.library.reference import ReferenceLibrary

    try:
        folder = ReferenceLibrary.load().path
    except FileNotFoundError:
        return None
    key = str(folder)
    if key not in _loaded:
        _loaded[key] = PhyschemCache.load(folder)
    return _loaded[key]


def describe(
    smiles: list[str],
    *,
    label: str = "physchem descriptors",
    show_progress: bool | None = None,
    cache: PhyschemCache | None | bool = True,
) -> np.ndarray:
    """Raw physchem descriptors of standardised SMILES, from the cache where possible.

    Parameters
    ----------
    smiles : list of str
        Standardised SMILES.
    label : str, optional
        Progress-bar title.
    show_progress : bool, optional
        Show a progress bar for the molecules that are computed.
    cache : PhyschemCache or bool, optional
        The cache to read: ``True`` (default) the canonical library's, if present;
        ``False`` or ``None`` none.

    Returns
    -------
    numpy.ndarray
        ``(n, N_DESCRIPTORS)`` float32, identical to
        :func:`~eosquality.library.physchem.compute_physchem_raw` of ``smiles``.
    """
    if cache is True:
        cache = default_cache()
    if not cache or not len(smiles):
        return compute_physchem_raw(smiles, show_progress=show_progress, label=label)
    rows = cache.rows_of(smiles)
    hit = rows >= 0
    out = np.empty((len(smiles), N_DESCRIPTORS), dtype=np.float32)
    out[hit] = cache.matrix[rows[hit]]
    missing = np.flatnonzero(~hit)
    if len(missing):
        out[missing] = compute_physchem_raw(
            [smiles[i] for i in missing], show_progress=show_progress, label=label
        )
    logger.debug(f"physchem cache | {int(hit.sum()):,} of {len(smiles):,} molecules")
    return out
