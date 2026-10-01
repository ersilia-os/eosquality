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
        """Build a VectorIndex from a list of SMILES and persist to ``output_dir``.

        Parameters
        ----------
        smiles:
            Ordered list of SMILES strings (one per reference molecule).
        output_dir:
            Folder to write index artifacts to. Created if it does not exist.
        max_k:
            Maximum k for self-kNN. Must be < len(smiles). Stored nearest
            neighbors can be sliced at query time for any k ≤ max_k.
        radius:
            Morgan radius (default 2).
        n_bits:
            Number of bits in the Morgan vector (default 2048).
        verbose:
            If True, print progress tables and timing to stderr.
        library_name:
            Name of the reference library (e.g. ``"ersilia_reference_library"``).
            Stored in metadata and shown in verbose output.
        max_samples:
            If set, truncate the input to the first ``max_samples`` molecules
            before building. Useful for quick tests. Default: None (use all).

        Returns
        -------
        VectorIndex
        """
        if verbose:
            logger.set_verbosity(True)

        if max_samples is not None:
            smiles = smiles[:max_samples]

        n = len(smiles)
        if n < max_k + 2:
            raise ValueError(
                f"Need at least max_k + 2 = {max_k + 2} molecules to build "
                f"self-kNN with max_k={max_k}, got {n}."
            )

        seen: set[str] = set()
        duplicates: list[tuple[int, str]] = []
        for idx, smi in enumerate(smiles):
            if smi in seen:
                duplicates.append((idx, smi))
            else:
                seen.add(smi)
        if duplicates:
            n_shown = min(5, len(duplicates))
            examples = ", ".join(f"row {i}: {s!r}" for i, s in duplicates[:n_shown])
            raise ValueError(
                f"Duplicate SMILES detected: {len(duplicates)} duplicate(s) found. "
                f"First {n_shown}: [{examples}]. "
                "The vector index requires a unique molecule per row — "
                "dedupe the source CSV and rebuild."
            )

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = output_dir / "vector_index.h5"
        knn_indices_path = output_dir / "knn_indices.npy"
        knn_distances_path = output_dir / "knn_distances.npy"
        config_path = output_dir / "metadata.json"

        # Resume support: an interrupted build leaves ``build_state.json``
        # behind; a finished one leaves ``metadata.json``. Either way, reuse
        # partial outputs only if they were produced from exactly this
        # SMILES list with exactly these parameters.
        build_state_path = output_dir / "build_state.json"
        fingerprint = {
            "n_samples": n,
            "smiles_sha256": _smiles_digest(smiles),
            "radius": radius,
            "n_bits": n_bits,
            "max_k": max_k,
        }
        for prior_path in (config_path, build_state_path):
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
                    f"Output directory '{output_dir}' contains a (partial) index "
                    f"built from different inputs ({details}). Delete the folder "
                    "and rebuild, or match those inputs."
                )
        with open(build_state_path, "w") as f:
            json.dump(fingerprint, f, indent=2)

        rdkit_version = _RDKIT_VERSION
        t_build_start = time.perf_counter()

        logger.rule("VectorIndex · build")
        logger.index_input_table(n_molecules=n, n_unique=len(set(smiles)))
        logger.index_config_table(
            max_k=max_k,
            radius=radius,
            n_bits=n_bits,
            rdkit_version=rdkit_version,
            output_dir=str(output_dir),
            library_name=library_name,
        )
        logger.info(
            f"Building index | {n:,} molecules | max_k={max_k} "
            f"| radius={radius} | n_bits={n_bits}"
        )

        # 1. Build FPSim2 database — Morgan vectors computed once here.
        #    Skipped if vector_index.h5 already exists (resume support).
        if h5_path.exists():
            logger.debug("Step 1/3 — vector_index.h5 found, loading (skipped)…")
            t0 = time.perf_counter()
            engine = FPSim2Engine(str(h5_path), in_memory_fps=True)
            t_fpsim2 = time.perf_counter() - t0
        else:
            logger.debug("Step 1/3 — Building FPSim2 database (Morgan vectors)…")
            t0 = time.perf_counter()
            _build_fpsim2_db(smiles, str(h5_path), radius=radius, n_bits=n_bits)
            engine = FPSim2Engine(str(h5_path), in_memory_fps=True)
            t_fpsim2 = time.perf_counter() - t0
            logger.debug(f"vector_index.h5 written → {h5_path} | {t_fpsim2:.2f}s")

        # 2. Self-kNN: for each molecule, query its own top-(max_k+1) Tanimoto
        #    neighbors, then strip self. Vectors reused from the .h5 — no
        #    redundant vector computation.
        #    Skipped if knn_indices.npy + knn_distances.npy already exist.
        knn_done = knn_indices_path.exists() and knn_distances_path.exists()
        if knn_done:
            logger.debug("Step 2/3 — knn files found, loading (skipped)…")
            t0 = time.perf_counter()
            knn_indices = np.load(knn_indices_path)
            knn_distances = np.load(knn_distances_path)
            t_knn = time.perf_counter() - t0
        else:
            logger.debug(f"Step 2/3 — Self-kNN (FPSim2 Tanimoto, k={max_k})…")
            t0 = time.perf_counter()
            t_last_log = t0
            knn_indices = np.zeros((n, max_k), dtype=np.int32)
            knn_distances = np.zeros((n, max_k), dtype=np.float32)
            for i, smi in enumerate(smiles):
                result = engine.top_k(
                    smi, k=max_k + 1, threshold=0.0, n_workers=_QUERY_WORKERS
                )
                mol_ids = result["mol_id"].astype(np.int32)
                sims = result["coeff"].astype(np.float32)
                not_self = mol_ids != i
                knn_indices[i] = mol_ids[not_self][:max_k]
                knn_distances[i] = 1.0 - sims[not_self][:max_k]
                now = time.perf_counter()
                if now - t_last_log >= 30.0:
                    elapsed = now - t0
                    rate = (i + 1) / elapsed
                    eta = (n - i - 1) / rate
                    logger.info(
                        f"  kNN {i + 1:,}/{n:,} ({100 * (i + 1) / n:.0f}%) "
                        f"| {elapsed:.0f}s elapsed | ETA ~{eta:.0f}s"
                    )
                    t_last_log = now
            t_knn = time.perf_counter() - t0
            logger.debug(f"Self-kNN done | {t_knn:.2f}s")

        # 3. Persist artifacts (only write files that were (re)computed)
        logger.debug("Step 3/3 — Saving artifacts to disk…")
        t0 = time.perf_counter()
        if not knn_done:
            np.save(knn_indices_path, knn_indices)
            np.save(knn_distances_path, knn_distances)

        with open(output_dir / "smiles.csv", "w") as f:
            f.write("smiles\n")
            f.writelines(f"{smi}\n" for smi in smiles)

        try:
            eq_version = importlib.metadata.version("eosquality")
        except importlib.metadata.PackageNotFoundError:
            eq_version = "unknown"
        try:
            fpsim2_version = importlib.metadata.version("FPSim2")
        except importlib.metadata.PackageNotFoundError:
            fpsim2_version = "unknown"

        config = {
            **fingerprint,
            "method": "morgan_fpsim2",
            "rdkit_version": rdkit_version,
            "fpsim2_version": fpsim2_version,
            "eosquality_version": eq_version,
            "build_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "library_name": library_name,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        build_state_path.unlink(missing_ok=True)
        t_save = time.perf_counter() - t0
        logger.debug(f"Artifacts saved → {output_dir} | {t_save:.2f}s")

        t_total = time.perf_counter() - t_build_start
        logger.timing_table(
            steps=[
                ("FPSim2 database (Morgan vectors)", t_fpsim2, False),
                ("Self-kNN (Tanimoto)", t_knn, False),
                ("Save artifacts", t_save, False),
            ],
            title="Index build timing",
        )
        logger.success(
            f"Vector index built | {n:,} molecules | max_k={max_k} | {t_total:.2f}s"
        )
        logger.rule()

        instance = cls(
            smiles=smiles,
            knn_indices=knn_indices,
            knn_distances=knn_distances,
            h5_path=h5_path,
            config=config,
        )
        instance._engine = engine  # reuse already-loaded engine
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
        """Library identifier recorded in ``metadata.json`` (``""`` if unset)."""
        return str(self._config.get("library_name", "") or "")

    @property
    def index_dir(self) -> pathlib.Path:
        """Folder holding this index (and the library's descriptor files)."""
        return pathlib.Path(self._h5_path).parent

    @property
    def n_reference(self) -> int:
        """Number of molecules in the index."""
        return len(self._smiles)

    @property
    def smiles(self) -> list[str]:
        """Reference SMILES, in index row order."""
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
        """
        max_k = self._config["max_k"]
        if k > max_k:
            raise ValueError(
                f"Requested k={k} exceeds the pre-computed max_k={max_k}. "
                "Rebuild the vector index with a larger max_k."
            )
        return np.ascontiguousarray(self._knn_indices[:, :k])

    def self_knn_distances(self, k: int) -> np.ndarray:
        """Return precomputed self-kNN Tanimoto distances, shape (n_ref, k)."""
        max_k = self._config["max_k"]
        if k > max_k:
            raise ValueError(f"Requested k={k} exceeds the pre-computed max_k={max_k}.")
        return np.ascontiguousarray(self._knn_distances[:, :k])

    # ------------------------------------------------------------------
    # Query (used at run time)
    # ------------------------------------------------------------------

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

        if self._engine is None:
            self._engine = FPSim2Engine(str(self._h5_path), in_memory_fps=True)

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
                result = self._engine.top_k(
                    smi, k=k, threshold=0.0, n_workers=_QUERY_WORKERS
                )
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
