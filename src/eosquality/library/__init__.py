"""Reference-library registration.

Maintenance-only and **model-independent**: builds and resolves the canonical
Morgan-fingerprint vector index over the SMILES reference library. End users
never call this submodule directly except through the ``eosquality build``
and ``eosquality download`` CLI commands; the canonical artifacts are
prefetched into ``data/`` and shipped to S3 by maintainers.
"""

from eosquality.library.identity import (
    LIBRARY_ID,
    library_major,
    reference_library_csv_path,
    reference_library_path,
)

__all__ = [
    "LIBRARY_ID",
    "library_major",
    "reference_library_csv_path",
    "reference_library_path",
]
