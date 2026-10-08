import pytest

from eosquality.library import identity
from eosquality.library.download import (
    LibraryDownloadError,
    ensure_library_downloaded,
    is_library_cached_and_valid,
)


@pytest.fixture
def served(library, tmp_path):
    """The test library, served from a ``file://`` URL under its folder name."""
    return library.parent.as_uri() + "/", library.name


def test_a_library_is_valid_with_all_files_and_the_right_identity(library):
    assert is_library_cached_and_valid(library, "test_library")
    assert not is_library_cached_and_valid(library, "another_library")
    assert not is_library_cached_and_valid(library / "nope", "test_library")


def test_a_missing_file_makes_it_invalid(library, tmp_path):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(library, broken)
    (broken / "connectivity_keys.npz").unlink()
    assert not is_library_cached_and_valid(broken, "test_library")


def test_download_fetches_verifies_and_caches(served, tmp_path):
    base_url, name = served
    cache = tmp_path / "cache"
    folder = ensure_library_downloaded(base_url, name, cache, "test_library")
    assert folder == cache / name
    assert is_library_cached_and_valid(folder, "test_library")
    assert not [p for p in cache.iterdir() if p.name.startswith(".")]  # no temp left


def test_download_is_a_no_op_when_cached_and_refetches_with_force(served, tmp_path):
    base_url, name = served
    cache = tmp_path / "cache"
    folder = ensure_library_downloaded(base_url, name, cache, "test_library")
    marker = folder / "marker.txt"
    marker.write_text("x")
    ensure_library_downloaded(base_url, name, cache, "test_library")
    assert marker.exists()
    ensure_library_downloaded(base_url, name, cache, "test_library", force=True)
    assert not marker.exists()


def test_download_rejects_a_library_with_another_identity(served, tmp_path):
    base_url, name = served
    cache = tmp_path / "cache"
    with pytest.raises(LibraryDownloadError, match="expects 'other_id'"):
        ensure_library_downloaded(base_url, name, cache, "other_id")
    assert not (cache / name).exists()  # a failed download leaves no folder


def test_download_reports_a_missing_file(tmp_path):
    with pytest.raises(LibraryDownloadError, match="Network error|HTTP"):
        ensure_library_downloaded(
            (tmp_path / "empty").as_uri() + "/", "nothing", tmp_path / "c", "x"
        )


def test_library_is_found_through_the_environment_variable(library, monkeypatch):
    monkeypatch.setenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", str(library))
    assert identity.reference_library_path() == library.resolve()
    monkeypatch.setenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", str(library / "nope"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        identity.reference_library_path()


def test_library_is_found_in_the_working_directory_then_the_cache(
    library, tmp_path, monkeypatch
):
    monkeypatch.delenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(identity, "LIBRARY_ID", "test_library")
    monkeypatch.setattr(identity, "user_cache_dir", lambda: tmp_path / "cache")
    work = tmp_path / "work"
    (work / "data" / "indices").mkdir(parents=True)
    monkeypatch.chdir(work)
    with pytest.raises(FileNotFoundError, match="eosquality setup"):
        identity.reference_library_path()
    # The cache is searched next ...
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "test_library").symlink_to(library)
    assert identity.reference_library_path() == tmp_path / "cache" / "test_library"
    # ... and the working directory comes first.
    (work / "data" / "indices" / "test_library").symlink_to(library)
    assert (
        identity.reference_library_path() == work / "data" / "indices" / "test_library"
    )


def test_library_ids():
    assert identity.is_library_id("ersilia_reference_library_v12")
    assert not identity.is_library_id("ersilia_reference_library_v")
    assert not identity.is_library_id("mylib")
    assert identity.library_major() == int(identity.LIBRARY_ID.rsplit("_v", 1)[1])


def test_a_folder_that_is_not_a_library_is_a_clear_error(tmp_path):
    from eosquality.library.reference import ReferenceLibrary

    with pytest.raises(FileNotFoundError, match="not found"):
        ReferenceLibrary(tmp_path / "nope")
    empty = ReferenceLibrary(tmp_path)
    with pytest.raises(FileNotFoundError, match="not a reference library"):
        _ = empty.metadata
    with pytest.raises(FileNotFoundError, match="not a reference library"):
        _ = empty.smiles


def test_library_major_matches_the_package_major():
    """``eosquality X.y.z`` ships exactly library ``vX``; a release must keep them together."""
    import importlib.metadata

    from packaging.version import Version

    major = Version(importlib.metadata.version("eosquality")).major
    assert identity.library_major() == major
