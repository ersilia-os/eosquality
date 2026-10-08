"""The maintainer scripts must keep working as the package changes."""

import json
import pathlib
import runpy

import pytest

SCRIPTS = pathlib.Path(__file__).parent.parent / "scripts"


@pytest.mark.parametrize(
    "path", sorted(SCRIPTS.rglob("*.py")), ids=lambda p: str(p.relative_to(SCRIPTS))
)
def test_every_script_compiles(path):
    compile(path.read_text(), str(path), "exec")


@pytest.fixture
def small_library(library, monkeypatch):
    monkeypatch.setenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", str(library))


def test_pair_median_script_runs_on_a_library(small_library, capsys):
    module = runpy.run_path(str(SCRIPTS / "physchem_pair_median.py"), run_name="x")
    module["main"](n_molecules=60, n_pairs=2000)
    out = capsys.readouterr().out
    assert "median" in out and "PAIR_MEDIAN in the package" in out


def test_scaler_script_prints_a_usable_scaler(small_library, capsys):
    from eosquality.library.physchem import DESCRIPTOR_NAMES

    module = runpy.run_path(str(SCRIPTS / "fit_physchem_scaler.py"), run_name="x")
    module["main"](n_molecules=60, n_jobs=1)
    scaler = json.loads(capsys.readouterr().out)
    assert scaler["descriptor_names"] == DESCRIPTOR_NAMES
    assert len(scaler["median"]) == len(scaler["mean"]) == len(scaler["scale"])
