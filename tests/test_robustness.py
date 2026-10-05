"""Regression tests for edge cases found in review (invalid input, CLI state)."""

import pytest

import eosquality
from eosquality import ErsiliaQuality
from eosquality.cli import main
from eosquality.cli.run import log_path_for
from eosquality.config import NeighborConfig
from eosquality.utils import console
from eosquality.utils.logging import logger


def _run(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, ignore_size=True
    )


def test_unparsable_smiles_score_nan_without_failing(fitted, query):
    q = query.head(6).copy()
    q.loc[q.index[1], "input"] = "not_a_smiles"
    q.loc[q.index[3], "input"] = None
    scores = fitted.run(q).scores
    for name in ("support", "consistency"):
        assert scores[name].iloc[[1, 3]].isna().all()
        assert scores[name].drop(index=q.index[[1, 3]]).notna().all()
    assert scores["typicality"].notna().all()  # output-based, needs no SMILES
    support = fitted.support.run(q)
    assert support.nearest_reference_ids[1] == []


def test_log_path_never_equals_output():
    assert str(log_path_for("out/scores.csv")) == "out/scores.log"
    assert str(log_path_for("scores.log")) == "scores.log.log"
    assert str(log_path_for("scores")) == "scores.log"


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        NeighborConfig(k=0)
    assert _run(["fit", "--reference", "x.csv", "-o", "y", "--k", "0"]) == 2


def test_markup_in_user_strings_is_shown_not_parsed(tmp_path, capsys):
    art = tmp_path / "nope[/]"
    code = _run(["run", "-i", "q.csv", "-a", str(art), "-o", str(tmp_path / "o.csv")])
    assert code == 1
    assert "nope[/]" in capsys.readouterr().err


def test_cli_restores_global_state(tmp_path):
    console.enable(False)
    logger.set_verbosity(False)
    _run(["run", "-v", "-i", "q.csv", "-a", str(tmp_path / "missing"), "-o", "o.csv"])
    assert not console.enabled()
    assert not logger.verbose


def test_failure_is_recorded_in_the_run_log(tmp_path, reference, library, query):
    art = tmp_path / "art"
    ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, ignore_size=True
    ).save(art)
    bad = tmp_path / "bad.csv"
    query.drop(columns=["mw"]).to_csv(bad, index=False)  # schema mismatch
    out = tmp_path / "scores.csv"
    assert _run(["run", "-i", str(bad), "-a", str(art), "-o", str(out)]) == 1
    log = (tmp_path / "scores.log").read_text()
    assert "failed:" in log and "Traceback" in log


def test_empty_query_is_a_clear_error(tmp_path, reference, library, capsys):
    art = tmp_path / "art"
    ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, ignore_size=True
    ).save(art)
    empty = tmp_path / "empty.csv"
    empty.write_text("key,input,mw\n")
    out = tmp_path / "s.csv"
    assert _run(["run", "-i", str(empty), "-a", str(art), "-o", str(out)]) == 1
    assert "has no rows" in capsys.readouterr().err


def test_dir_lists_lazy_names():
    assert {"ErsiliaQuality", "Support", "RunResult"} <= set(dir(eosquality))


def test_set_verbosity_false_silences_the_console():
    eosquality.set_verbosity(True)
    assert console.enabled()
    eosquality.set_verbosity(False)
    assert not console.enabled()
