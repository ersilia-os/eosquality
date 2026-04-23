"""Guardrails that pin the reference library to the package major version.

Policy (see src/eosquality/_library.py):
    ``eosquality X.y.z`` ships exactly one canonical reference library,
    ``ersilia_reference_library_vX``. A mismatch means a release-engineering
    bug that must not escape CI.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from packaging.version import Version

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover — project requires Python >= 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from eosquality._library import LIBRARY_ID, library_major
from eosquality.exceptions import IncompatibleArtifactsError
from eosquality.quality.api import ErsiliaQuality


PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"

EOS_ID = "eos4e40"
VERSION = "v1"


def _declared_package_version() -> str:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_pyproject_major_matches_library_major():
    """The declared version in pyproject.toml and LIBRARY_ID must agree on major."""
    pkg_major = Version(_declared_package_version()).major
    assert pkg_major == library_major(), (
        f"pyproject.toml declares version with major={pkg_major} but "
        f"LIBRARY_ID={LIBRARY_ID!r} encodes major={library_major()}. "
        "Bump the package major when bumping the library, and vice versa."
    )


def test_library_id_format():
    """LIBRARY_ID must follow the canonical 'ersilia_reference_library_vN' pattern."""
    assert LIBRARY_ID.startswith("ersilia_reference_library_v")
    assert LIBRARY_ID[len("ersilia_reference_library_v"):].isdigit()


def test_load_rejects_wrong_library_id(
    reference_df_vi, vector_index_dir, tmp_path
):
    """Artifacts tagged with a foreign library_id must not load."""
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi,
        eos_id=EOS_ID,
        version=VERSION,
        vector_index=vector_index_dir,
        ignore_size=True,
    )
    folder = tmp_path / "mismatched"
    eq.save(folder)

    metadata_path = folder / "metadata.json"
    with open(metadata_path) as f:
        data = json.load(f)
    data["library_id"] = "ersilia_reference_library_v999"
    with open(metadata_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(IncompatibleArtifactsError, match="reference library"):
        ErsiliaQuality.load(folder)


def test_load_rejects_wrong_eosquality_major(
    reference_df_vi, vector_index_dir, tmp_path
):
    """Artifacts whose saved eosquality_version major differs from the installed
    package major must not load, even if library_id happens to match."""
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi,
        eos_id=EOS_ID,
        version=VERSION,
        vector_index=vector_index_dir,
        ignore_size=True,
    )
    folder = tmp_path / "wrong_major"
    eq.save(folder)

    metadata_path = folder / "metadata.json"
    with open(metadata_path) as f:
        data = json.load(f)
    # Bump saved eosquality major to something impossible.
    current_major = Version(data["eosquality_version"]).major
    data["eosquality_version"] = f"{current_major + 99}.0.0"
    with open(metadata_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(IncompatibleArtifactsError, match="major"):
        ErsiliaQuality.load(folder)


def test_fit_tags_artifacts_with_library_id(reference_df_vi, vector_index_dir):
    """library_id on the fitted metadata should come from the index's library_name."""
    eq = ErsiliaQuality(k=10).fit(
        reference_df_vi,
        eos_id=EOS_ID,
        version=VERSION,
        vector_index=vector_index_dir,
        ignore_size=True,
    )
    assert eq.metadata_.library_id == LIBRARY_ID
