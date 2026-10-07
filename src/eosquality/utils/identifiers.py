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

    Parameters
    ----------
    eos_id : str
        Candidate identifier.
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

    Parameters
    ----------
    version : str
        Candidate version string.
    """
    if not VERSION_RE.match(version):
        raise ValueError(
            f"Invalid version {version!r}. "
            "Expected format: 'v' followed by digits (e.g. 'v1', 'v2')."
        )


def model_from_name(path: str | os.PathLike) -> tuple[str, str]:
    """``(eos_id, version)`` from a file or folder name, eosframes-style.

    The name (without extension) must end in ``<eos_id>_<version>``,
    optionally after a prefix: ``eos4e40_v1.csv``,
    ``reference_eos4e40_v1.csv``, ``training_eos4e40_v1/``,
    ``artifacts_eos4e40_v1``.

    Parameters
    ----------
    path : str or os.PathLike
        File or folder path; only its name is used.

    Returns
    -------
    tuple of (str, str)
        ``(eos_id, version)``.

    Raises
    ------
    ValueError
        If the name does not follow the convention.
    """
    name = os.path.basename(str(path).rstrip("/\\"))
    m = _STEM_RE.search(name.split(".")[0])
    if m:
        return m.group(1), m.group(2)
    found = _EOS_ID_ANYWHERE_RE.search(name)
    hint = (
        f"it has {found.group(1)!r} but no '_<version>' after it"
        if found
        else "it has no EOS identifier"
    )
    raise ValueError(
        f"{name!r} must be named '[prefix_]<eos_id>_<version>' (e.g. "
        f"'eos4e40_v1.csv' or 'training_eos4e40_v1'); {hint}."
    )


def model_from_names(paths: dict[str, str | os.PathLike]) -> tuple[str, str]:
    """The single ``(eos_id, version)`` named by several paths.

    Parameters
    ----------
    paths : dict of str to path
        Label (for messages, e.g. ``"--reference"``) → file or folder path.

    Returns
    -------
    tuple of (str, str)
        The common ``(eos_id, version)``.

    Raises
    ------
    ValueError
        If a name does not follow the convention or the names disagree.
    """
    models = {label: model_from_name(p) for label, p in paths.items()}
    if len(set(models.values())) > 1:
        listed = ", ".join(
            f"{label} {os.path.basename(str(paths[label]).rstrip('/'))} is "
            f"{eos_id} {version}"
            for label, (eos_id, version) in models.items()
        )
        raise ValueError(f"the names disagree on the model: {listed}.")
    return next(iter(models.values()))
