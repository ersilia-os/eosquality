"""Per-molecule basic descriptors for the reference library.

``BasicDescriptors`` is a peer of :class:`eosquality.vectorindex.VectorIndex`
that bundles the two non-FP, non-kNN molecular representations used by
the reference library:

- **Physchem** — RDKit ~200 physicochemical descriptors with a
  per-column median imputer + sklearn ``StandardScaler`` (artifacts:
  ``physchem_scaled.npy`` as float16 + ``physchem_scaler.json``).
- **MACCS** — 166-bit binary keys (artifact: ``maccs.npy`` as uint8).

Both representations are computed purely from a SMILES list — no kNN,
no learned step, no dependency on the vector index — and persist into
the same library folder used by ``VectorIndex.build``. The two
descriptor families are independently constructable: a caller can run
just one of them via :meth:`build_physchem` or :meth:`build_maccs`.
:meth:`load` reads whichever artifacts are present on disk; missing
families are left as ``None`` rather than raising.

The class does **not** read or write ``smiles.csv``. The row order of
its matrices is implicit in the row order of the input SMILES list —
``VectorIndex`` owns ``smiles.csv`` as the canonical ordering file. If
a caller needs to map rows back to SMILES, they read ``smiles.csv``
separately.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from eosquality.library.maccs import (
    MACCS_FILE,
    compute_maccs,
    load_maccs,
)
from eosquality.library.physchem import (
    PHYSCHEM_SCALED_FILE,
    PHYSCHEM_SCALER_FILE,
    apply_scaler,
    compute_physchem_raw,
    fit_scaler,
    load_physchem,
)
from eosquality.utils.logging import logger


class BasicDescriptors:
    """Physchem + MACCS descriptor matrices for the reference library."""

    def __init__(
        self,
        *,
        physchem_scaled: np.ndarray | None = None,
        physchem_scaler_params: dict | None = None,
        maccs: np.ndarray | None = None,
    ) -> None:
        self.physchem_scaled = physchem_scaled
        self.physchem_scaler_params = physchem_scaler_params
        self.maccs = maccs

    # ------------------------------------------------------------------
    # Build — physchem
    # ------------------------------------------------------------------

    @classmethod
    def build_physchem(
        cls,
        smiles: list[str],
        output_dir: str | pathlib.Path,
        *,
        max_samples: int | None = None,
        n_jobs: int | None = None,
        force: bool = False,
    ) -> "BasicDescriptors":
        """Compute + persist the physchem matrices into ``output_dir``.

        Resume-cached: if all three physchem files are already present
        and ``force`` is False, the existing artifacts are loaded into
        the returned instance without recomputation. Returns a
        :class:`BasicDescriptors` populated with the physchem arrays
        plus MACCS if ``maccs.npy`` also happens to be on disk.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        smiles_used = _truncate(smiles, max_samples)

        scaled_path = output_dir / PHYSCHEM_SCALED_FILE
        scaler_path = output_dir / PHYSCHEM_SCALER_FILE

        if not force and scaled_path.is_file() and scaler_path.is_file():
            logger.info(
                f"physchem | artifacts already present in {output_dir} — "
                "skipped (pass force=True to recompute)"
            )
            return cls._populate_from_disk(output_dir)

        t0 = time.perf_counter()
        n = len(smiles_used)
        logger.info(
            f"physchem | computing descriptors × {n:,} molecules"
            + (f" | n_jobs={n_jobs}" if n_jobs is not None else "")
        )
        raw = compute_physchem_raw(smiles_used, n_jobs=n_jobs)
        non_finite_rows = int((~np.isfinite(raw)).any(axis=1).sum())
        logger.info(
            f"physchem | raw matrix ready | shape={raw.shape} "
            f"| rows with any non-finite descriptor: {non_finite_rows:,} "
            f"({100 * non_finite_rows / max(n, 1):.2f}%)"
        )

        scaler_params = fit_scaler(raw)
        scaled = apply_scaler(raw, scaler_params).astype(np.float16)

        np.save(scaled_path, scaled)
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)

        logger.success(
            f"physchem | saved → {output_dir} | "
            f"scaled={scaled_path.name} (float16) scaler={scaler_path.name} "
            f"| {time.perf_counter() - t0:.2f}s"
        )

        return cls._populate_from_disk(output_dir)

    # ------------------------------------------------------------------
    # Build — MACCS
    # ------------------------------------------------------------------

    @classmethod
    def build_maccs(
        cls,
        smiles: list[str],
        output_dir: str | pathlib.Path,
        *,
        max_samples: int | None = None,
        n_jobs: int | None = None,
        force: bool = False,
    ) -> "BasicDescriptors":
        """Compute + persist the MACCS matrix into ``output_dir``.

        Resume-cached: if ``maccs.npy`` is already present and
        ``force`` is False, the existing matrix is loaded into the
        returned instance without recomputation. Returns a
        :class:`BasicDescriptors` populated with MACCS plus physchem
        if those artifacts also happen to be on disk.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        smiles_used = _truncate(smiles, max_samples)

        maccs_path = output_dir / MACCS_FILE

        if not force and maccs_path.is_file():
            logger.info(
                f"maccs | artifact already present in {output_dir} — "
                "skipped (pass force=True to recompute)"
            )
            return cls._populate_from_disk(output_dir)

        t0 = time.perf_counter()
        n = len(smiles_used)
        logger.info(
            f"maccs | computing keys × {n:,} molecules"
            + (f" | n_jobs={n_jobs}" if n_jobs is not None else "")
        )
        matrix = compute_maccs(smiles_used, n_jobs=n_jobs)
        empty_rows = int((matrix.sum(axis=1) == 0).sum())
        mean_on_bits = float(matrix.sum(axis=1).mean()) if n else 0.0
        logger.info(
            f"maccs | matrix ready | shape={matrix.shape} "
            f"| empty rows (parse / compute failure): {empty_rows:,} "
            f"({100 * empty_rows / max(n, 1):.2f}%) "
            f"| mean on-bits per molecule: {mean_on_bits:.1f}"
        )

        np.save(maccs_path, matrix)

        logger.success(
            f"maccs | saved → {output_dir} | maccs={maccs_path.name} "
            f"| {time.perf_counter() - t0:.2f}s"
        )

        return cls._populate_from_disk(output_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, index_dir: str | pathlib.Path) -> "BasicDescriptors":
        """Read whatever descriptor families are present in ``index_dir``.

        Each family loads independently — a missing family leaves the
        corresponding attributes as ``None`` rather than raising. Use
        :attr:`has_physchem` / :attr:`has_maccs` to check what was
        found.
        """
        return cls._populate_from_disk(pathlib.Path(index_dir))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_physchem(self) -> bool:
        return (
            self.physchem_scaled is not None and self.physchem_scaler_params is not None
        )

    @property
    def has_maccs(self) -> bool:
        return self.maccs is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @classmethod
    def _populate_from_disk(cls, output_dir: pathlib.Path) -> "BasicDescriptors":
        """Construct an instance reflecting whatever is currently on disk."""
        physchem_scaled = None
        physchem_scaler_params = None
        if (output_dir / PHYSCHEM_SCALED_FILE).is_file() and (
            output_dir / PHYSCHEM_SCALER_FILE
        ).is_file():
            payload = load_physchem(output_dir)
            physchem_scaled = payload["scaled"]
            physchem_scaler_params = payload["scaler_params"]

        maccs = None
        if (output_dir / MACCS_FILE).is_file():
            maccs = load_maccs(output_dir)

        return cls(
            physchem_scaled=physchem_scaled,
            physchem_scaler_params=physchem_scaler_params,
            maccs=maccs,
        )


def _truncate(smiles: list[str], max_samples: int | None) -> list[str]:
    if max_samples is not None and max_samples > 0:
        return list(smiles)[:max_samples]
    return list(smiles)


__all__ = ["BasicDescriptors"]
