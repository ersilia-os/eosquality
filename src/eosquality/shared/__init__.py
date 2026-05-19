"""Shared fit state: schema + eosframes scaler + binary class freqs + metadata.

Every score depends on these. ``fit_shared`` computes them once from a raw
reference DataFrame; per-score fits can either receive a pre-computed
:class:`SharedFitState` or fit it themselves.
"""

from eosquality.shared.fit import fit_shared
from eosquality.shared.load import load_shared
from eosquality.shared.save import save_shared
from eosquality.shared.state import SharedFitState

__all__ = ["SharedFitState", "fit_shared", "save_shared", "load_shared"]
