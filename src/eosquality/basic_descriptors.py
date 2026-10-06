"""Build the per-molecule descriptor matrices of a reference library.

A peer of :class:`eosquality.vectorindex.VectorIndex` used by
``eosquality build``. Writes, into the same library folder:

- **Physchem** — RDKit physicochemical descriptors, median-imputed and
  standard-scaled (``physchem_scaled.npy`` as float16 +
  ``physchem_scaler.json``).
- **MACCS** — 166-bit structural keys (``maccs.npy`` as uint8).

Row order follows the input SMILES list, which must be the same list (and
order) the vector index was built from — ``smiles.csv`` in the folder is
the canonical ordering. Both builders are resume-cached: existing files
are kept unless ``force=True``. The Signal score's descriptor backends
(:mod:`eosquality.scores._descriptors`) read these files at fit time.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from eosquality.library.maccs import MACCS_FILE, compute_maccs
from eosquality.library.physchem import (
    PHYSCHEM_SCALED_FILE,
    PHYSCHEM_SCALER_FILE,
    apply_scaler,
    compute_physchem_raw,
    fit_scaler,
)
from eosquality.utils.logging import logger


class BasicDescriptors:
    """Builders for the physchem + MACCS descriptor matrices."""

    @staticmethod
    def build_physchem(
        smiles: list[str],
        output_dir: str | pathlib.Path,
        *,
        n_jobs: int | None = None,
        force: bool = False,
    ) -> pathlib.Path:
        """Compute + persist the scaled physchem matrix and scaler params.

        Parameters
        ----------
        smiles : list of str
            Library SMILES, in index order.
        output_dir : str or pathlib.Path
            Library folder (created if needed).
        n_jobs : int, optional
            Worker processes (default: in-process; ``-1``: every CPU).
        force : bool, optional
            Recompute even if the files already exist.

        Returns
        -------
        pathlib.Path
            ``output_dir``.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scaled_path = output_dir / PHYSCHEM_SCALED_FILE
        scaler_path = output_dir / PHYSCHEM_SCALER_FILE
        if not force and scaled_path.is_file() and scaler_path.is_file():
            logger.info(f"physchem | already present in {output_dir} — skipped")
            return output_dir

        t0 = time.perf_counter()
        raw = compute_physchem_raw(smiles, n_jobs=n_jobs)
        non_finite_rows = int((~np.isfinite(raw)).any(axis=1).sum())
        logger.info(
            f"physchem | raw matrix {raw.shape} | rows with any non-finite "
            f"descriptor: {non_finite_rows:,}"
        )
        scaler_params = fit_scaler(raw)
        np.save(scaled_path, apply_scaler(raw, scaler_params).astype(np.float16))
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)
        logger.success(
            f"physchem | saved → {output_dir} | {time.perf_counter() - t0:.2f}s"
        )
        return output_dir

    @staticmethod
    def build_maccs(
        smiles: list[str],
        output_dir: str | pathlib.Path,
        *,
        n_jobs: int | None = None,
        force: bool = False,
    ) -> pathlib.Path:
        """Compute + persist the 166-bit MACCS matrix.

        Parameters
        ----------
        smiles : list of str
            Library SMILES, in index order.
        output_dir : str or pathlib.Path
            Library folder (created if needed).
        n_jobs : int, optional
            Worker processes (default: in-process; ``-1``: every CPU).
        force : bool, optional
            Recompute even if the files already exist.

        Returns
        -------
        pathlib.Path
            ``output_dir``.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        maccs_path = output_dir / MACCS_FILE
        if not force and maccs_path.is_file():
            logger.info(f"maccs | already present in {output_dir} — skipped")
            return output_dir

        t0 = time.perf_counter()
        matrix = compute_maccs(smiles, n_jobs=n_jobs)
        empty_rows = int((matrix.sum(axis=1) == 0).sum())
        logger.info(
            f"maccs | matrix {matrix.shape} | empty rows (parse failure): {empty_rows:,}"
        )
        np.save(maccs_path, matrix)
        logger.success(
            f"maccs | saved → {output_dir} | {time.perf_counter() - t0:.2f}s"
        )
        return output_dir


__all__ = ["BasicDescriptors"]
