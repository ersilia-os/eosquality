"""VectorIndex: pre-computed Morgan vector index for molecular kNN.

Built per training column (``training/state.py``) for the training scores.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file
from rdkit import __version__ as _RDKIT_VERSION

from eosquality.utils import console
from eosquality.utils.logging import logger

_PROGRESS_THRESHOLD = 25  # show the FP kNN bar once n_query is at least this
# FPSim2 top-k searches run single-threaded on purpose: with n_workers > 1
# the order of equally-similar neighbors is not stable, so which molecule
# fills the k-th slot (and therefore the training distances) would vary between runs.
_QUERY_WORKERS = 1

MAX_K_DEFAULT = 50
RADIUS_DEFAULT = 2
N_BITS_DEFAULT = 2048


class VectorIndex:
    """Pre-computed Morgan vector index of a set of unique molecules.

    Parameters
    ----------
    smiles : list of str
        The molecules, in index row order.
    knn_distances : numpy.ndarray
        ``(n, max_k)`` Tanimoto distances to each molecule's nearest *other*
        molecules, closest first.
    h5_path : pathlib.Path
        The FPSim2 database inside the index folder.
    config : dict
        The folder's ``metadata.json``.
    """

    def __init__(
        self,
        smiles: list[str],
        knn_distances: np.ndarray,
        h5_path: pathlib.Path,
        config: dict,
    ) -> None:
        self._smiles = smiles
        self._knn_distances = knn_distances  # (n, max_k)
        self._h5_path = h5_path
        self._config = config
        self._engine: FPSim2Engine | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        smiles: list[str],
        output_dir: str | pathlib.Path,
        max_k: int = MAX_K_DEFAULT,
        radius: int = RADIUS_DEFAULT,
        n_bits: int = N_BITS_DEFAULT,
    ) -> VectorIndex:
        """Build a VectorIndex from unique SMILES and persist it to ``output_dir``.

        Writes ``vector_index.h5`` (FPSim2 database), ``knn_distances.npy`` (the
        self-kNN distances, identity stripped), ``smiles.csv`` and ``metadata.json``.

        Parameters
        ----------
        smiles : list of str
            Unique SMILES, in index order.
        output_dir : str or pathlib.Path
            Folder for the index files (created if needed).
        max_k : int, optional
            Neighbours precomputed per molecule; at most ``len(smiles) - 2``.
        radius, n_bits : int, optional
            Morgan fingerprint radius and size.

        Returns
        -------
        VectorIndex
            The built index, with its FPSim2 engine already loaded.
        """
        _check_build_inputs(smiles, max_k)
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        h5_path = output_dir / "vector_index.h5"
        create_db_file(
            [(smi, i) for i, smi in enumerate(smiles)],
            str(h5_path),
            "smiles",
            "Morgan",
            {"radius": radius, "fpSize": n_bits},
        )
        engine = FPSim2Engine(str(h5_path), in_memory_fps=True)
        knn_distances = _self_knn_distances(engine, smiles, max_k)
        np.save(output_dir / "knn_distances.npy", knn_distances)
        (output_dir / "smiles.csv").write_text(
            "smiles\n" + "".join(f"{s}\n" for s in smiles)
        )
        config = {
            "n_samples": len(smiles),
            "radius": radius,
            "n_bits": n_bits,
            "max_k": max_k,
            "rdkit_version": _RDKIT_VERSION,
        }
        (output_dir / "metadata.json").write_text(json.dumps(config, indent=2))
        logger.debug(
            f"vector index | {len(smiles):,} molecules | max_k={max_k} | "
            f"{time.perf_counter() - t0:.2f}s"
        )
        index = cls(smiles, knn_distances, h5_path, config)
        index._engine = engine
        return index

    @classmethod
    def load(cls, index_dir: str | pathlib.Path) -> VectorIndex:
        """Load a VectorIndex from a folder produced by :meth:`build`.

        Parameters
        ----------
        index_dir : str or pathlib.Path
            Folder written by :meth:`build`.

        Returns
        -------
        VectorIndex
        """
        index_dir = pathlib.Path(index_dir)
        if not index_dir.is_dir():
            raise FileNotFoundError(f"No vector index folder at: {index_dir}")
        smiles = (index_dir / "smiles.csv").read_text().splitlines()[1:]
        config = json.loads((index_dir / "metadata.json").read_text())
        _check_rdkit_version(config)
        # Memory-mapped: the full (n, max_k) arrays never need to be resident.
        return cls(
            smiles=smiles,
            knn_distances=np.load(index_dir / "knn_distances.npy", mmap_mode="r"),
            h5_path=index_dir / "vector_index.h5",
            config=config,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def index_dir(self) -> pathlib.Path:
        """Folder holding this index.

        Returns
        -------
        pathlib.Path
        """
        return pathlib.Path(self._h5_path).parent

    @property
    def smiles(self) -> list[str]:
        """The indexed SMILES, in row order.

        Returns
        -------
        list of str
        """
        return self._smiles

    def self_knn_distances(self, k: int) -> np.ndarray:
        """Precomputed self-kNN Tanimoto distances ``(n, k)``, closest first (float32).

        Parameters
        ----------
        k : int
            Neighbours per molecule; at most ``max_k``.

        Returns
        -------
        numpy.ndarray
        """
        return np.ascontiguousarray(self._knn_distances[:, : self._check_k(k)])

    def _check_k(self, k: int) -> int:
        if k > self._config["max_k"]:
            raise ValueError(
                f"Requested k={k} exceeds the pre-computed max_k={self._config['max_k']}."
            )
        return k

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self, smiles_list: list[str], k: int, *, show_progress: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k most similar indexed molecules of each SMILES.

        Parameters
        ----------
        smiles_list : list of str
            Query SMILES.
        k : int
            Neighbours to return.
        show_progress : bool, optional
            Draw a progress bar (default: from 25 queries on).

        Returns
        -------
        distances : numpy.ndarray
            Tanimoto distances (1 − similarity), ``(n_query, k)``.
        indices : numpy.ndarray
            Row indices into the index, ``(n_query, k)``.
        """
        _check_rdkit_version(self._config)
        if self._engine is None:
            self._engine = FPSim2Engine(str(self._h5_path), in_memory_fps=True)
        n_query = len(smiles_list)
        distances = np.zeros((n_query, k), dtype=np.float32)
        indices = np.zeros((n_query, k), dtype=np.int32)
        if show_progress is None:
            show_progress = n_query >= _PROGRESS_THRESHOLD
        progress = console.progress("FP kNN query") if show_progress else None
        task = None
        if progress is not None:
            progress.start()
            task = progress.add_task("FP kNN query", total=n_query)
        try:
            for i, smi in enumerate(smiles_list):
                hits = self._engine.top_k(
                    smi, k=k, threshold=0.0, n_workers=_QUERY_WORKERS
                )
                if len(hits) < k:
                    raise ValueError(
                        f"Query SMILES at row {i} returned only {len(hits)} "
                        f"neighbors (k={k} requested); the index has "
                        f"{len(self._smiles)} molecules."
                    )
                indices[i] = hits["mol_id"][:k]
                distances[i] = 1.0 - hits["coeff"].astype(np.float32)[:k]
                if progress is not None:
                    progress.advance(task)
        finally:
            if progress is not None:
                progress.stop()
        return distances, indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_rdkit_version(config: dict) -> None:
    """Raise RuntimeError if the current RDKit differs from the index's."""
    stored = config.get("rdkit_version")
    if stored is not None and stored != _RDKIT_VERSION:
        raise RuntimeError(
            f"RDKit version mismatch: the vector index was built with RDKit "
            f"{stored}, but this environment has RDKit {_RDKIT_VERSION}. Morgan "
            "vectors may differ between RDKit versions, which can silently "
            "corrupt query results. Rebuild the index (refit) with this RDKit, or "
            "install the version it was built with."
        )


def _check_build_inputs(smiles: list[str], max_k: int) -> None:
    """Require at least ``max_k + 2`` molecules and no duplicate SMILES."""
    if len(smiles) < max_k + 2:
        raise ValueError(
            f"Need at least max_k + 2 = {max_k + 2} molecules to build "
            f"self-kNN with max_k={max_k}, got {len(smiles)}."
        )
    seen: set[str] = set()
    dupes = [i for i, s in enumerate(smiles) if s in seen or seen.add(s)]
    if dupes:
        shown = ", ".join(f"row {i}: {smiles[i]!r}" for i in dupes[:5])
        raise ValueError(
            f"Duplicate SMILES detected: {len(dupes)} duplicate rows. First "
            f"{min(5, len(dupes))}: [{shown}]. The vector index requires a unique "
            "molecule per row."
        )


def _self_knn_distances(
    engine: FPSim2Engine, smiles: list[str], max_k: int
) -> np.ndarray:
    """Distances from each molecule to its ``max_k`` nearest *other* molecules."""
    distances = np.zeros((len(smiles), max_k), dtype=np.float32)
    for i, smi in enumerate(smiles):
        hits = engine.top_k(smi, k=max_k + 1, threshold=0.0, n_workers=_QUERY_WORKERS)
        not_self = hits["mol_id"] != i
        distances[i] = 1.0 - hits["coeff"].astype(np.float32)[not_self][:max_k]
    return distances
