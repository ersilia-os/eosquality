"""Regression tests for edge cases found in review (invalid input, CLI state)."""

import pytest

import eosquality
from eosquality import ErsiliaQuality
from eosquality.cli import main
from eosquality.cli.run import log_path_for
from eosquality.utils import console
from eosquality.utils.logging import logger


def _unwrapped(text: str) -> str:
    """Console output with Rich's terminal-width line wrapping undone."""
    return "".join(text.split())


def _run(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


@pytest.fixture(scope="module")
def fitted(reference, library):
    return ErsiliaQuality().fit(
        reference, eos_id="eos0aaa", vector_index=library, exclude=["ref_signal"]
    )


def test_unparsable_smiles_score_nan_without_failing(fitted, query):
    q = query.head(6).copy()
    q.loc[q.index[1], "input"] = "not_a_smiles"
    q.loc[q.index[3], "input"] = None
    scores = fitted.run(q).scores
    for name in ("ref_support", "ref_consistency"):
        assert scores[name].iloc[[1, 3]].isna().all()
        assert scores[name].drop(index=q.index[[1, 3]]).notna().all()
    assert scores["ref_typicality"].notna().all()  # output-based, needs no SMILES
    support = fitted.support.run(q)
    assert support.nearest_reference_ids[1] == []


def test_log_path_never_equals_output():
    assert str(log_path_for("out/scores.csv")) == "out/scores.log"
    assert str(log_path_for("scores.log")) == "scores.log.log"
    assert str(log_path_for("scores")) == "scores.log"


@pytest.mark.parametrize(
    "flag",
    [["--k", "5"], ["-o", "x"], ["--vector-index", "x"], ["--ignore-size"]],
)
def test_removed_fit_options_are_rejected(flag):
    assert _run(["fit", "-r", "r_eos0aaa_v1.csv", "-a", "a_eos0aaa_v1", *flag]) == 2


def test_markup_in_user_strings_is_shown_not_parsed(tmp_path, capsys):
    art = tmp_path / "nope[/]_eos0aaa_v1"
    out = str(tmp_path / "o_eos0aaa_v1.csv")
    assert _run(["run", "-i", "q_eos0aaa_v1.csv", "-a", str(art), "-o", out]) == 1
    assert "nope[/]" in _unwrapped(capsys.readouterr().err)


def test_cli_restores_global_state(tmp_path):
    console.enable(False)
    logger.set_verbosity(False)
    _run(["run", "-v", "-i", "q.csv", "-a", str(tmp_path / "missing"), "-o", "o.csv"])
    assert not console.enabled()
    assert not logger.verbose


def test_failure_is_recorded_in_the_run_log(tmp_path, fitted, query):
    art = tmp_path / "art_eos0aaa_v1"
    fitted.save(art)
    bad = tmp_path / "bad_eos0aaa_v1.csv"
    query.drop(columns=["mw"]).to_csv(bad, index=False)  # schema mismatch
    out = tmp_path / "scores_eos0aaa_v1.csv"
    assert _run(["run", "-i", str(bad), "-a", str(art), "-o", str(out)]) == 1
    log = (tmp_path / "scores_eos0aaa_v1.log").read_text()
    assert "failed:" in log and "Traceback" in log


def test_empty_query_is_a_clear_error(tmp_path, fitted, capsys):
    art = tmp_path / "art_eos0aaa_v1"
    fitted.save(art)
    empty = tmp_path / "empty_eos0aaa_v1.csv"
    empty.write_text("key,input,mw\n")
    out = tmp_path / "s_eos0aaa_v1.csv"
    assert _run(["run", "-i", str(empty), "-a", str(art), "-o", str(out)]) == 1
    assert "hasnorows" in _unwrapped(capsys.readouterr().err)


def test_dir_lists_lazy_names():
    assert {"ErsiliaQuality", "Support", "RunResult"} <= set(dir(eosquality))


def test_set_verbosity_false_silences_the_console():
    eosquality.set_verbosity(True)
    assert console.enabled()
    eosquality.set_verbosity(False)
    assert not console.enabled()
