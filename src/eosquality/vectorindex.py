"""VectorIndex: pre-computed Morgan vector index for molecular kNN.

Build once per reference molecule collection; share across many models.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file
from rdkit import __version__ as _RDKIT_VERSION

from eosquality.utils.logging import logger
from eosquality.utils.progress import make_progress

_PROGRESS_THRESHOLD = 25  # show the FP kNN bar once n_query is at least this
# FPSim2 top-k searches run single-threaded on purpose: with n_workers > 1
# the order of equally-similar neighbors is not stable, so which molecule
# fills the k-th slot (and therefore Consistency) would vary between runs.
_QUERY_WORKERS = 1

MAX_K_DEFAULT = 50
RADIUS_DEFAULT = 2
N_BITS_DEFAULT = 2048


def _build_fpsim2_db(smiles: list[str], h5_path: str, radius: int, n_bits: int) -> None:
    """Build a FPSim2 .h5 database from a list of SMILES."""
    mols = [(smi, i) for i, smi in enumerate(smiles)]
    create_db_file(
        mols,
        h5_path,
        "smiles",
        "Morgan",
        {"radius": radius, "fpSize": n_bits},
    )


def _smiles_digest(smiles: list[str]) -> str:
    """SHA-256 over the ordered SMILES list (identity of the index contents)."""
    h = hashlib.sha256()
    for smi in smiles:
        h.update(smi.encode())
        h.update(b"\n")
    return h.hexdigest()


class VectorIndex:
    """Pre-computed Morgan vector index for a reference molecule collection.

    Build once per reference library; share across multiple Ersilia models that
    use the same set of molecules.

    Parameters
    ----------
    index_dir:
        Folder produced by :meth:`build`. Contains the Morgan-FP files
        consumed by the loader — ``vector_index.h5`` (FPSim2 database),
        ``knn_indices.npy`` and ``knn_distances.npy`` (precomputed
        ``(n_ref, max_k)`` self-kNN, identity neighbor already
        stripped), ``smiles.csv`` (one SMILES per reference row), and
        ``metadata.json`` (library name, radius, n_bits, max_k, RDKit
        version). The same folder also holds the per-molecule descriptor
        matrices written by :class:`eosquality.basic_descriptors.BasicDescriptors`
        (``physchem_scaled.npy``, ``physchem_scaler.json``, ``maccs.npy``).
        Those are not loaded by :meth:`load`; the Signal score's descriptor
        backends read them directly from :attr:`index_dir`.
    """

    def __init__(
        self,
        smiles: list[str],
        knn_indices: np.ndarray,
        knn_distances: np.ndarray,
        h5_path: pathlib.Path,
        config: dict,
    ) -> None:
        self._smiles = smiles
        self._knn_indices = knn_indices  # (n_ref, max_k)
        self._knn_distances = knn_distances  # (n_ref, max_k)
        self._h5_path = h5_path
        self._config = config
        self._engine = None  # lazy-loaded FPSim2Engine

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        smiles: list[str],
        output_dir: str | pathlib.Path,
        max_k: int = MAX_K_DEFAULT,
        radius: int = RADIUS_DEFAULT,
        n_bits: int = N_BITS_DEFAULT,
        verbose: bool = False,
        library_name: str = "",
        max_samples: int | None = None,
    ) -> VectorIndex:
        """Build a VectorIndex from a list of SMILES and persist it to ``output_dir``.

        An interrupted build resumes, reusing finished steps, but only if the
        SMILES list and parameters are unchanged.

        Parameters
        ----------
        smiles : list of str
            Unique SMILES, one per reference molecule, in index order.
        output_dir : str or pathlib.Path
            Folder for the index files (created if needed).
        max_k : int, optional
            Neighbours precomputed per molecule; must be below ``len(smiles) - 1``.
        radius, n_bits : int, optional
            Morgan fingerprint radius and size.
        verbose : bool, optional
            Show DEBUG logs and the diagnostic tables.
        library_name : str, optional
            Library identifier stored in ``metadata.json``.
        max_samples : int, optional
            Truncate the input to its first ``max_samples`` molecules (testing).

        Returns
        -------
        VectorIndex
            The built index, with its FPSim2 engine already loaded.
        """
        if verbose:
            logger.set_verbosity(True)
        if max_samples is not None:
            smiles = smiles[:max_samples]
        _check_build_inputs(smiles, max_k)
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = {
            "n_samples": len(smiles),
            "smiles_sha256": _smiles_digest(smiles),
            "radius": radius,
            "n_bits": n_bits,
            "max_k": max_k,
        }
        _check_resume(output_dir, fingerprint)

        t0 = time.perf_counter()
        logger.rule("VectorIndex · build")
        logger.info(
            f"Building index | {len(smiles):,} molecules | max_k={max_k} "
            f"| radius={radius} | n_bits={n_bits}"
        )
        h5_path = output_dir / "vector_index.h5"
        engine = _open_or_build_db(smiles, h5_path, radius, n_bits)
        knn_indices, knn_distances = _load_or_compute_self_knn(
            engine, smiles, max_k, output_dir
        )
        config = _write_index_files(output_dir, smiles, fingerprint, library_name)
        logger.success(
            f"Vector index built | {len(smiles):,} molecules | max_k={max_k} | "
            f"{time.perf_counter() - t0:.2f}s"
        )
        logger.rule()

        instance = cls(
            smiles=smiles,
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            h5_path=h5_path,
            config=config,
        )
        instance._engine = engine  # reuse the already-loaded engine
        return instance

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_dir: str | pathlib.Path) -> VectorIndex:
        """Load a VectorIndex from a folder produced by :meth:`build`.

        Parameters
        ----------
        index_dir:
            Folder written by :meth:`build`.

        Returns
        -------
        VectorIndex
        """
        index_dir = pathlib.Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(f"No vector index folder at: {index_dir}")
        if not index_dir.is_dir():
            raise ValueError(f"Expected a directory, got a file: {index_dir}")

        with open(index_dir / "smiles.csv") as f:
            next(f)  # skip header
            smiles = [line.rstrip("\n") for line in f]

        with open(index_dir / "metadata.json") as f:
            config = json.load(f)

        cls._check_rdkit_version(config)

        # Memory-mapped: run time only queries the .h5, and fit time slices
        # the first k columns, so the full (n_ref, max_k) arrays never need
        # to be resident.
        knn_indices = np.load(index_dir / "knn_indices.npy", mmap_mode="r")
        knn_distances = np.load(index_dir / "knn_distances.npy", mmap_mode="r")
        h5_path = index_dir / "vector_index.h5"

        return cls(
            smiles=smiles,
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            h5_path=h5_path,
            config=config,
        )

    @classmethod
    def _check_rdkit_version(cls, config: dict) -> None:
        """Raise RuntimeError if current RDKit version differs from the index."""
        stored = config.get("rdkit_version")
        if stored is None:
            return  # old index without version info — skip check
        current = _RDKIT_VERSION
        if current != stored:
            raise RuntimeError(
                f"RDKit version mismatch: the vector index was built with "
                f"RDKit {stored}, but the current environment has RDKit {current}. "
                "Morgan vectors may differ between RDKit versions, which can silently "
                "corrupt query results. Rebuild the index with the current RDKit version "
                "or downgrade/upgrade RDKit to match."
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def library_name(self) -> str:
        """Library identifier recorded in ``metadata.json`` (``""`` if unset).

        Returns
        -------
        str
        """
        return str(self._config.get("library_name", "") or "")

    @property
    def index_dir(self) -> pathlib.Path:
        """Folder holding this index (and the library's descriptor files).

        Returns
        -------
        pathlib.Path
        """
        return pathlib.Path(self._h5_path).parent

    @property
    def n_reference(self) -> int:
        """Number of molecules in the index.

        Returns
        -------
        int
        """
        return len(self._smiles)

    @property
    def smiles(self) -> list[str]:
        """Reference SMILES, in index row order.

        Returns
        -------
        list of str
        """
        return self._smiles

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_smiles(self, smiles: list[str]) -> None:
        """Check that ``smiles`` matches the index SMILES (ordered, row-by-row).

        Parameters
        ----------
        smiles:
            SMILES from the model output DataFrame's ``input`` column.

        Raises
        ------
        ValueError
            If the lists differ in length or in any element.
        """
        if len(smiles) != len(self._smiles):
            raise ValueError(
                f"SMILES count mismatch: model CSV has {len(smiles)} rows, "
                f"vector index has {len(self._smiles)} molecules."
            )
        mismatches = np.flatnonzero(
            np.asarray(smiles, dtype=object) != np.asarray(self._smiles, dtype=object)
        ).tolist()
        if mismatches:
            n_shown = min(3, len(mismatches))
            raise ValueError(
                f"SMILES mismatch at {len(mismatches)} row(s). "
                f"First {n_shown}: rows {mismatches[:n_shown]}. "
                "The model CSV 'input' column must match the reference library "
                "SMILES used to build the vector index, in the same order."
            )

    # ------------------------------------------------------------------
    # Accessors for self-kNN (used at fit time)
    # ------------------------------------------------------------------

    def self_knn_indices(self, k: int) -> np.ndarray:
        """Return precomputed self-kNN indices, shape (n_ref, k).

        Parameters
        ----------
        k:
            Number of neighbors to return. Must be ≤ max_k.

        Returns
        -------
        numpy.ndarray
            int32 neighbour row indices, closest first.
        """
        max_k = self._config["max_k"]
        if k > max_k:
            raise ValueError(
                f"Requested k={k} exceeds the pre-computed max_k={max_k}. "
                "Rebuild the vector index with a larger max_k."
            )
        return np.ascontiguousarray(self._knn_indices[:, :k])

    def self_knn_distances(self, k: int) -> np.ndarray:
        """Return precomputed self-kNN Tanimoto distances, shape (n_ref, k).

        Parameters
        ----------
        k : int
            Number of neighbours (at most ``max_k``).

        Returns
        -------
        numpy.ndarray
            float32 Tanimoto distances, closest first.
        """
        max_k = self._config["max_k"]
        if k > max_k:
            raise ValueError(f"Requested k={k} exceeds the pre-computed max_k={max_k}.")
        return np.ascontiguousarray(self._knn_distances[:, :k])

    # ------------------------------------------------------------------
    # Query (used at run time)
    # ------------------------------------------------------------------

    def _get_engine(self) -> FPSim2Engine:
        if self._engine is None:
            self._engine = FPSim2Engine(str(self._h5_path), in_memory_fps=True)
        return self._engine

    def query(
        self,
        smiles_list: list[str],
        k: int,
        *,
        show_progress: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the FPSim2 index for each SMILES; return top-k neighbors.

        Parameters
        ----------
        smiles_list:
            Query SMILES strings.
        k:
            Number of neighbors to return.
        show_progress:
            Whether to render a rich progress bar on stderr while the
            per-SMILES loop runs. ``None`` (default) shows the bar when
            ``len(smiles_list) >= 25``; pass ``True`` / ``False`` to
            force it on or off.

        Returns
        -------
        distances:
            Tanimoto distances (1 − similarity), shape (n_query, k).
        indices:
            Row indices into the reference library, shape (n_query, k).
        """
        self._check_rdkit_version(self._config)
        engine = self._get_engine()

        n_ref = len(self._smiles)
        n_query = len(smiles_list)
        all_dists = np.zeros((n_query, k), dtype=np.float32)
        all_idx = np.zeros((n_query, k), dtype=np.int32)

        if show_progress is None:
            show_progress = n_query >= _PROGRESS_THRESHOLD

        progress = make_progress("FP kNN query") if show_progress else None
        task_id = None
        if progress is not None:
            progress.start()
            task_id = progress.add_task("FP kNN query", total=n_query)

        try:
            for i, smi in enumerate(smiles_list):
                result = engine.top_k(smi, k=k, threshold=0.0, n_workers=_QUERY_WORKERS)
                got = len(result)
                if got < k:
                    raise ValueError(
                        f"Query SMILES at row {i} returned only {got} neighbors "
                        f"(k={k} requested). The reference index has {n_ref} molecules."
                    )
                mol_ids = result["mol_id"].astype(np.int32)
                sims = result["coeff"].astype(np.float32)
                all_idx[i] = mol_ids[:k]
                all_dists[i] = 1.0 - sims[:k]
                if progress is not None:
                    progress.advance(task_id)
        finally:
            if progress is not None:
                progress.stop()

        return all_dists, all_idx


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------


def _check_build_inputs(smiles: list[str], max_k: int) -> None:
    """Require at least ``max_k + 2`` molecules and no duplicate SMILES."""
    if len(smiles) < max_k + 2:
        raise ValueError(
            f"Need at least max_k + 2 = {max_k + 2} molecules to build "
            f"self-kNN with max_k={max_k}, got {len(smiles)}."
        )
    dupes = pd.Series(smiles)[pd.Series(smiles).duplicated()]
    if len(dupes):
        shown = ", ".join(f"row {i}: {s!r}" for i, s in list(dupes.items())[:5])
        raise ValueError(
            f"Duplicate SMILES detected: {len(dupes)} duplicate rows. First "
            f"{min(5, len(dupes))}: [{shown}]. The vector index requires a unique "
            "molecule per row — dedupe the source CSV and rebuild."
        )


def _check_resume(output_dir: pathlib.Path, fingerprint: dict) -> None:
    """Refuse to reuse partial outputs built from different inputs.

    An interrupted build leaves ``build_state.json``; a finished one leaves
    ``metadata.json``. Either must match ``fingerprint`` exactly.
    """
    for name in ("metadata.json", "build_state.json"):
        prior_path = output_dir / name
        if not prior_path.exists():
            continue
        with open(prior_path) as f:
            prior = json.load(f)
        mismatched = {
            key: (prior.get(key), value)
            for key, value in fingerprint.items()
            if prior.get(key) != value
        }
        if mismatched:
            details = ", ".join(
                f"{key}: on disk={old!r}, requested={new!r}"
                for key, (old, new) in mismatched.items()
            )
            raise ValueError(
                f"Output directory '{output_dir}' contains a (partial) index built "
                f"from different inputs ({details}). Delete the folder and "
                "rebuild, or match those inputs."
            )
    with open(output_dir / "build_state.json", "w") as f:
        json.dump(fingerprint, f, indent=2)


def _open_or_build_db(
    smiles: list[str], h5_path: pathlib.Path, radius: int, n_bits: int
) -> FPSim2Engine:
    """Load ``vector_index.h5`` if present (resume), else build it from ``smiles``."""
    if not h5_path.exists():
        logger.debug("Building FPSim2 database (Morgan vectors)…")
        _build_fpsim2_db(smiles, str(h5_path), radius=radius, n_bits=n_bits)
    return FPSim2Engine(str(h5_path), in_memory_fps=True)


def _load_or_compute_self_knn(
    engine: FPSim2Engine, smiles: list[str], max_k: int, output_dir: pathlib.Path
) -> tuple[np.ndarray, np.ndarray]:
    """Self-kNN (identity stripped), reusing ``knn_*.npy`` from a resumed build."""
    idx_path = output_dir / "knn_indices.npy"
    dist_path = output_dir / "knn_distances.npy"
    if idx_path.exists() and dist_path.exists():
        logger.debug("Self-kNN files found; reusing them")
        return np.load(idx_path), np.load(dist_path)
    knn_indices, knn_distances = _self_knn(engine, smiles, max_k)
    np.save(idx_path, knn_indices)
    np.save(dist_path, knn_distances)
    return knn_indices, knn_distances


def _self_knn(
    engine: FPSim2Engine, smiles: list[str], max_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each molecule's ``max_k`` nearest *other* molecules and their distances."""
    n = len(smiles)
    knn_indices = np.zeros((n, max_k), dtype=np.int32)
    knn_distances = np.zeros((n, max_k), dtype=np.float32)
    t0 = t_last = time.perf_counter()
    for i, smi in enumerate(smiles):
        result = engine.top_k(smi, k=max_k + 1, threshold=0.0, n_workers=_QUERY_WORKERS)
        mol_ids = result["mol_id"].astype(np.int32)
        not_self = mol_ids != i
        knn_indices[i] = mol_ids[not_self][:max_k]
        knn_distances[i] = 1.0 - result["coeff"].astype(np.float32)[not_self][:max_k]
        if time.perf_counter() - t_last >= 30.0:
            t_last = time.perf_counter()
            eta = (n - i - 1) * (t_last - t0) / (i + 1)
            logger.info(f"  kNN {i + 1:,}/{n:,} | ETA ~{eta:.0f}s")
    return knn_indices, knn_distances


def _write_index_files(
    output_dir: pathlib.Path, smiles: list[str], fingerprint: dict, library_name: str
) -> dict:
    """Write ``smiles.csv`` and ``metadata.json``; clear the resume marker."""
    with open(output_dir / "smiles.csv", "w") as f:
        f.write("smiles\n")
        f.writelines(f"{smi}\n" for smi in smiles)
    config = _index_config(fingerprint, library_name)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(config, f, indent=2)
    (output_dir / "build_state.json").unlink(missing_ok=True)
    return config


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _index_config(fingerprint: dict, library_name: str) -> dict:
    """Contents of the index's ``metadata.json``."""
    return {
        **fingerprint,
        "method": "morgan_fpsim2",
        "rdkit_version": _RDKIT_VERSION,
        "fpsim2_version": _package_version("FPSim2"),
        "eosquality_version": _package_version("eosquality"),
        "build_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "library_name": library_name,
    }
