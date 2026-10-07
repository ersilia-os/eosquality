import pandas as pd
import pytest

from eosquality.cli import main


def _run(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


def _err(capsys) -> str:
    """Stderr with Rich's line wrapping undone."""
    return "".join(capsys.readouterr().err.split())


@pytest.fixture
def files(tmp_path, reference, query, library, monkeypatch):
    """Inputs named after eos0aaa v1, and the fixture library as the canonical one."""
    monkeypatch.setenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", str(library))
    ref = tmp_path / "reference_eos0aaa_v1.csv"
    reference.to_csv(ref, index=False)
    q = tmp_path / "query_eos0aaa_v1.csv"
    query.to_csv(q, index=False)
    return {
        "reference": str(ref),
        "query": str(q),
        "artifacts": str(tmp_path / "artifacts_eos0aaa_v1"),
        "output": str(tmp_path / "quality_eos0aaa_v1.csv"),
        "tmp": tmp_path,
    }


def test_fit_and_run(files, query):
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", "ref_signal"]) == 0
    run = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", files["output"]]
    assert _run(run) == 0
    scores = pd.read_csv(files["output"])
    assert list(scores.columns[:4]) == [
        "key",
        "input",
        "ref_typicality_pct",
        "ref_typicality_raw",
    ]
    assert "ref_signal" not in scores.columns
    assert len(scores) == len(query)


def test_fit_refuses_existing_artifacts_with_reference(files):
    (files["tmp"] / "artifacts_eos0aaa_v1").mkdir()
    assert _run(["fit", "-r", files["reference"], "-a", files["artifacts"]]) == 1


def test_fit_with_training_and_details(files, query, training_dir):
    fit = ["fit", "-r", files["reference"], "-t", str(training_dir)]
    assert _run([*fit, "-a", files["artifacts"], "--exclude", "ref_signal"]) == 0
    run = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", files["output"]]
    assert _run(run) == 0
    columns = pd.read_csv(files["output"]).columns
    assert [c for c in columns if c.startswith("trn_")] == [
        "trn_tanimoto_pct",
        "trn_tanimoto_raw",
        "trn_physchem_pct",
        "trn_physchem_raw",
        "trn_match",
        "trn_scaffold",
    ]  # the error model is off by default
    details = pd.read_csv(files["tmp"] / "quality_eos0aaa_v1.training_details.csv")
    assert len(details) == len(query)  # one row per query, not per column
    reference_details = pd.read_csv(
        files["tmp"] / "quality_eos0aaa_v1.reference_details.csv"
    )
    assert len(reference_details) == len(query)
    assert any(c.endswith("_extremity_pct") for c in reference_details.columns)


def test_training_only_and_no_adding_later(files, training_dir, capsys):
    only = str(files["tmp"] / "training_only_eos0aaa_v1")
    assert _run(["fit", "-t", str(training_dir), "-a", only]) == 0
    out = str(files["tmp"] / "s_eos0aaa_v1.csv")
    assert _run(["run", "-i", files["query"], "-a", only, "-o", out]) == 0
    # Training sets cannot be added to existing artifacts: -a must be new.
    ref_only = ["ref_extremity", "ref_support", "ref_consistency", "ref_signal"]
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", ",".join(ref_only)]) == 0
    capsys.readouterr()
    assert _run(["fit", "-t", str(training_dir), "-a", files["artifacts"]]) == 1
    assert "alreadyexists" in _err(capsys)


def test_fit_argument_errors(files, training_dir, capsys):
    assert _run(["fit", "-a", files["artifacts"]]) == 1  # no inputs
    assert _run(["fit", "-t", str(training_dir)]) == 2  # no --artifacts
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", "support"]) == 1
    assert "unknownscore" in _err(capsys)
    every = "ref_typicality,ref_extremity,ref_support,ref_consistency,ref_signal"
    assert _run([*fit, "--exclude", every]) == 1
    assert "nothingtofit" in _err(capsys)


def test_names_must_carry_the_same_model(files, training_dir, capsys):
    ref = files["reference"]
    assert _run(["fit", "-r", ref, "-a", str(files["tmp"] / "artifacts")]) == 1
    assert "noEOSidentifier" in _err(capsys)
    other = str(files["tmp"] / "artifacts_eos7m30_v1")
    assert _run(["fit", "-r", ref, "-a", other]) == 1
    assert "disagreeonthemodel" in _err(capsys)
    unversioned = files["tmp"] / "reference_eos0aaa.csv"
    unversioned.write_text("x\n")
    assert _run(["fit", "-r", str(unversioned), "-a", files["artifacts"]]) == 1
    assert "butno'_<version>'" in _err(capsys)


def test_run_refuses_artifacts_of_another_model(files, capsys):
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", "ref_signal"]) == 0
    renamed = files["tmp"] / "artifacts_eos9zzz_v1"
    (files["tmp"] / "artifacts_eos0aaa_v1").rename(renamed)
    q = files["tmp"] / "query_eos9zzz_v1.csv"
    pd.read_csv(files["query"]).to_csv(q, index=False)
    out = str(files["tmp"] / "quality_eos9zzz_v1.csv")
    assert _run(["run", "-i", str(q), "-a", str(renamed), "-o", out]) == 1
    assert "fittedforeos0aaav1" in _err(capsys)


def test_fit_and_run_write_log_files(files):
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", "ref_signal"]) == 0
    fit_log = (files["tmp"] / "artifacts_eos0aaa_v1" / "eosquality.log").read_text()
    assert "| INFO     | eosquality." in fit_log
    assert "[eosframes]" not in fit_log and "eosframes" in fit_log  # routed
    run = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", files["output"]]
    assert _run(run) == 0
    log = (files["tmp"] / "quality_eos0aaa_v1.log").read_text()
    assert "eosquality.cli.run" in log


def test_commands():
    from eosquality.cli import cli

    assert list(cli.commands) == ["setup", "fit", "run", "build"]


def test_query_name_needs_no_model(files, training_dir, query):
    only = str(files["tmp"] / "training_only_eos0aaa_v1")
    assert _run(["fit", "-t", str(training_dir), "-a", only]) == 0
    plain = files["tmp"] / "drugs.csv"  # a plain SMILES file, any name
    query[["input"]].rename(columns={"input": "smiles"}).to_csv(plain, index=False)
    out = str(files["tmp"] / "quality_eos0aaa_v1.csv")
    assert _run(["run", "-i", str(plain), "-a", only, "-o", out]) == 0
    scores = pd.read_csv(out)
    assert scores.columns[0] == "smiles" and "trn_tanimoto_pct" in scores.columns


def test_run_output_must_be_csv(files, capsys):
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", "ref_signal"]) == 0
    capsys.readouterr()
    bad = str(files["tmp"] / "quality_eos0aaa_v1")
    assert _run(["run", "-i", files["query"], "-a", files["artifacts"], "-o", bad]) == 1
    assert "mustbea.csvfile" in _err(capsys)
