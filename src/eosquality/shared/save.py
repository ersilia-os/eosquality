"""Serialize a SharedFitState into <root>/shared/."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from eosquality.shared.splitter import Splitter
from eosquality.shared.state import SharedFitState
from eosquality.utils.logging import logger


SUBFOLDER = "shared"


def save_shared(state: SharedFitState, root: str | pathlib.Path) -> pathlib.Path:
    """Write the SharedFitState into ``<root>/shared/``.

    Files written:

    - ``schema.json`` — :class:`Schema` column specs.
    - ``scaler.json`` — eosframes scaler params.
    - ``binary_class_freq.json`` — per-binary-column class frequencies.
    - ``metadata.json`` — :class:`FitMetadata` (provenance + stats).
    - ``reference_ids.json`` — JSON list of reference row index labels.
      Strings and ints round-trip cleanly; mixed/exotic Python types
      will fail at this serializer.
    - ``splits.json`` — train / val / test row indices produced by the
      fixed :class:`Splitter`, plus the seed and ratios for provenance.
    - ``selected_columns.json`` — names of the columns kept after the
      fit-time correlation-cluster reduction. Equals the full schema when
      no reduction was applied; older artifacts without this file fall
      back to the full schema at load time.
    - ``reference_repr.npy`` — float64 ``(n_ref, n_selected)`` matrix of
      eosframes-scaled values, projected onto ``selected_columns``. Saved
      whenever ``state.ref_repr`` is populated (always true after
      :func:`fit_shared`). Useful for offline inspection / debugging.
    """
    folder = pathlib.Path(root) / SUBFOLDER
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "schema.json", "w") as f:
        json.dump(dataclasses.asdict(state.schema), f, indent=2)
    with open(folder / "scaler.json", "w") as f:
        json.dump(state.scaler_params, f, indent=2)
    with open(folder / "binary_class_freq.json", "w") as f:
        json.dump(state.binary_class_freq, f, indent=2)
    with open(folder / "metadata.json", "w") as f:
        json.dump(dataclasses.asdict(state.metadata), f, indent=2)
    with open(folder / "reference_ids.json", "w") as f:
        json.dump(list(state.reference_ids), f)
    with open(folder / "splits.json", "w") as f:
        json.dump(
            {
                "seed": Splitter.SEED,
                "train_frac": Splitter.TRAIN_FRAC,
                "val_frac": Splitter.VAL_FRAC,
                "test_frac": Splitter.TEST_FRAC,
                "n_train": int(state.splits.train_indices.size),
                "n_val": int(state.splits.val_indices.size),
                "n_test": int(state.splits.test_indices.size),
                "train_indices": state.splits.train_indices.tolist(),
                "val_indices": state.splits.val_indices.tolist(),
                "test_indices": state.splits.test_indices.tolist(),
            },
            f,
        )
    with open(folder / "selected_columns.json", "w") as f:
        json.dump({"selected_columns": list(state.selected_columns)}, f, indent=2)
    if state.ref_repr is not None:
        np.save(folder / "reference_repr.npy", state.ref_repr)

    logger.debug(
        f"  shared/ | {len(state.schema.columns)} columns"
        f" | selected {len(state.selected_columns)}"
        f" | splits {state.splits.train_indices.size:,}/"
        f"{state.splits.val_indices.size:,}/"
        f"{state.splits.test_indices.size:,}"
    )
    return folder
