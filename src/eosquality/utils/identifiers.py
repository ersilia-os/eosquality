"""EOS model identifier and version validation utilities.

Patterns follow the eosframes naming convention
(github.com/ersilia-os/eosframes).
"""

import os
import re

# eos<digit><3 alphanumeric> — exactly 7 characters
EOS_ID_RE = re.compile(r"^eos\d[A-Za-z0-9]{3}$")

# v followed by one or more digits
VERSION_RE = re.compile(r"^v\d+$")

# Matches both eos_id and version anywhere in a filename stem,
# allowing an optional leading prefix (e.g. "project_eos4e40_v1")
_STEM_RE = re.compile(r"(?:^|_)(eos\d[A-Za-z0-9]{3})_(v\d+)$")

# Matches eos_id alone anywhere in a filename stem (no version required)
_EOS_ID_ANYWHERE_RE = re.compile(
    r"(?<![A-Za-z0-9])(eos\d[A-Za-z0-9]{3})(?![A-Za-z0-9])"
)


def validate_eos_id(eos_id: str) -> None:
    """Raise ValueError if eos_id does not match the expected pattern.

    Valid format: ``eos`` + 1 digit + 3 alphanumeric characters (7 chars total).
    Examples: ``eos4e40``, ``eos7m30``, ``eos3804``.
    """
    if not EOS_ID_RE.match(eos_id):
        raise ValueError(
            f"Invalid EOS identifier {eos_id!r}. "
            "Expected format: 'eos' + 1 digit + 3 alphanumeric chars (e.g. 'eos4e40')."
        )


def validate_version(version: str) -> None:
    """Raise ValueError if version does not match the expected pattern.

    Valid format: ``v`` followed by one or more digits.
    Examples: ``v1``, ``v2``, ``v10``.
    """
    if not VERSION_RE.match(version):
        raise ValueError(
            f"Invalid version {version!r}. "
            "Expected format: 'v' followed by digits (e.g. 'v1', 'v2')."
        )


def extract_from_path(path: str | os.PathLike) -> tuple[str, str]:
    """Extract (eos_id, version) from a filename.

    The filename stem must contain both an EOS identifier and a version
    in the pattern ``<eos_id>_<version>`` (optionally preceded by a prefix).

    Examples of valid filenames:
        ``eos4e40_v1.csv``
        ``project_eos4e40_v2.csv``
        ``260313_eos7m30_v1.csv``

    Parameters
    ----------
    path:
        File path. Only the basename is used.

    Returns
    -------
    tuple[str, str]
        ``(eos_id, version)`` e.g. ``("eos4e40", "v1")``.

    Raises
    ------
    ValueError
        If the filename does not contain a valid ``<eos_id>_<version>`` pattern.
    """
    basename = os.path.basename(str(path))
    # Strip extension(s)
    stem = basename.split(".")[0]

    m = _STEM_RE.search(stem)
    if m:
        return m.group(1), m.group(2)

    # Provide a helpful error — check if at least an eos_id is present
    id_match = _EOS_ID_ANYWHERE_RE.search(basename)
    if id_match:
        raise ValueError(
            f"Found EOS identifier {id_match.group(1)!r} in filename {basename!r} "
            "but no version (e.g. '_v1'). "
            "Rename the file to match the pattern '<eos_id>_<version>.csv' "
            "(e.g. 'eos4e40_v1.csv')."
        )

    raise ValueError(
        f"Could not find a valid EOS identifier + version in filename {basename!r}. "
        "Rename the file to match the pattern '<eos_id>_<version>.csv' "
        "(e.g. 'eos4e40_v1.csv')."
    )
