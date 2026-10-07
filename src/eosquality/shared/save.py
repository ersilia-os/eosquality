"""Serialize a SharedFitState into <root>/shared/."""

from __future__ import annotations

import dataclasses
import json
import pathlib

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
    - ``selected_columns.json`` — names of the columns kept after the
      fit-time correlation-cluster reduction. Equals the full schema when
      no reduction was applied.

    Parameters
    ----------
    state : SharedFitState
        The shared state.
    root : str or pathlib.Path
        Folder to write ``shared/`` into.

    Returns
    -------
    pathlib.Path
        The ``shared`` folder.
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
    with open(folder / "selected_columns.json", "w") as f:
        json.dump({"selected_columns": list(state.selected_columns)}, f, indent=2)

    logger.debug(
        f"  shared/ | {len(state.schema.columns)} columns"
        f" | selected {len(state.selected_columns)}"
    )
    return folder
