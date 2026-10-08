import json
import pathlib

import pandas as pd
import pytest

from eosquality.cli import main


def _run(argv):
    """Run the CLI; ``fit`` / ``run`` stay in-process unless ``-j`` is given."""
    if argv[0] in ("fit", "run") and "-j" not in argv:
        argv = [*argv, "-j", "1"]
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
    assert _run([*fit, "--exclude", "ref_match"]) == 0
    run = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", files["output"]]
    assert _run(run) == 0
    scores = pd.read_csv(files["output"])
    assert list(scores.columns[:4]) == [
        "key",
        "input",
        "ref_typicality_pct",
        "ref_typicality_raw",
    ]
    assert "ref_match" not in scores.columns
    assert len(scores) == len(query)


def test_fit_refuses_existing_artifacts_with_reference(files):
    (files["tmp"] / "artifacts_eos0aaa_v1").mkdir()
    assert _run(["fit", "-r", files["reference"], "-a", files["artifacts"]]) == 1


def test_fit_with_training_and_details(files, query, training_dir):
    fit = ["fit", "-r", files["reference"], "-t", str(training_dir)]
    assert _run([*fit, "-a", files["artifacts"], "--exclude", "ref_match"]) == 0
    run = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", files["output"]]
    assert _run(run) == 0
    assert not list(files["tmp"].glob("*details.csv"))  # opt-in
    columns = pd.read_csv(files["output"]).columns
    assert [c for c in columns if c.startswith("trn_")] == [
        "trn_tanimoto_pct",
        "trn_tanimoto_raw",
        "trn_physchem_pct",
        "trn_physchem_raw",
        "trn_match",
        "trn_scaffold",
    ]
    out = str(files["tmp"] / "withdetails_eos0aaa_v1.csv")
    with_details = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", out]
    assert _run([*with_details, "--details"]) == 0
    details = pd.read_csv(files["tmp"] / "withdetails_eos0aaa_v1.training_details.csv")
    assert len(details) == len(query)  # one row per query, not per column
    reference_details = pd.read_csv(
        files["tmp"] / "withdetails_eos0aaa_v1.reference_details.csv"
    )
    assert len(reference_details) == len(query)
    assert any(c.endswith("_extremity_pct") for c in reference_details.columns)


def test_training_only_and_no_adding_later(files, training_dir, capsys):
    only = str(files["tmp"] / "training_only_eos0aaa_v1")
    assert _run(["fit", "-t", str(training_dir), "-a", only]) == 0
    out = str(files["tmp"] / "s_eos0aaa_v1.csv")
    assert _run(["run", "-i", files["query"], "-a", only, "-o", out]) == 0
    # Training sets cannot be added to existing artifacts: -a must be new.
    ref_only = ["ref_extremity", "ref_match"]
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
    every = "ref_typicality,ref_extremity,ref_match"
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
    assert _run([*fit, "--exclude", "ref_match"]) == 0
    renamed = files["tmp"] / "artifacts_eos9zzz_v1"
    (files["tmp"] / "artifacts_eos0aaa_v1").rename(renamed)
    q = files["tmp"] / "query_eos9zzz_v1.csv"
    pd.read_csv(files["query"]).to_csv(q, index=False)
    out = str(files["tmp"] / "quality_eos9zzz_v1.csv")
    assert _run(["run", "-i", str(q), "-a", str(renamed), "-o", out]) == 1
    assert "fittedforeos0aaav1" in _err(capsys)


def test_fit_and_run_write_log_files(files):
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run([*fit, "--exclude", "ref_match"]) == 0
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
    assert _run([*fit, "--exclude", "ref_match"]) == 0
    capsys.readouterr()
    bad = str(files["tmp"] / "quality_eos0aaa_v1")
    assert _run(["run", "-i", files["query"], "-a", files["artifacts"], "-o", bad]) == 1
    assert "mustbea.csvfile" in _err(capsys)


def _library_csv(tmp_path, smiles, name, n=30):
    path = tmp_path / name
    pd.DataFrame({"smiles": smiles[:n]}).to_csv(path, index=False)
    return str(path)


def test_build_writes_exactly_the_library_files(tmp_path, smiles):
    csv = _library_csv(tmp_path, smiles, "ersilia_reference_library_v7.csv")
    out = tmp_path / "lib"
    assert _run(["build", "-i", csv, "-o", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "connectivity_keys.npz",
        "metadata.json",
        "smiles.csv",
    ]
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["library_name"] == "ersilia_reference_library_v7"


def test_build_refuses_an_existing_folder(tmp_path, smiles, capsys):
    csv = _library_csv(tmp_path, smiles, "ersilia_reference_library_v7.csv")
    out = tmp_path / "lib"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert _run(["build", "-i", csv, "-o", str(out)]) == 1
    assert "alreadyexists" in _err(capsys)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_build_needs_a_library_id_name_or_an_explicit_one(tmp_path, smiles, capsys):
    csv = _library_csv(tmp_path, smiles, "mylib.csv")
    assert _run(["build", "-i", csv, "-o", str(tmp_path / "a")]) == 1
    assert "notalibraryid" in _err(capsys)
    assert not (tmp_path / "a").exists()
    assert _run(["build", "-i", csv, "-o", str(tmp_path / "b"), "--name", "mylib"]) == 0
    meta = json.loads((tmp_path / "b" / "metadata.json").read_text())
    assert meta["library_name"] == "mylib"


def test_help_lists_user_and_maintainer_commands(capsys):
    assert _run(["--help"]) == 0
    out = capsys.readouterr().out
    assert "Commands" in out and "Developer commands" in out
    assert out.index("setup") < out.index("Developer commands") < out.index("build")


@pytest.fixture
def served_library(library, tmp_path, monkeypatch):
    """``setup`` pointed at the test library, served from a ``file://`` URL."""
    import importlib

    setup_cli = importlib.import_module(
        "eosquality.cli.setup"
    )  # the name `setup` is the command

    monkeypatch.setenv("EOSQUALITY_REFERENCE_BASE_URL", library.parent.as_uri() + "/")
    monkeypatch.setattr(setup_cli, "LIBRARY_ID", "test_library")
    monkeypatch.setattr(setup_cli, "library_dirname", lambda: library.name)
    monkeypatch.setattr(setup_cli, "user_cache_dir", lambda: tmp_path / "cache")
    return tmp_path / "cache" / library.name


def test_setup_fetches_the_library_once(served_library, capsys):
    assert _run(["setup"]) == 0
    assert (served_library / "connectivity_keys.npz").is_file()
    marker = served_library / "marker.txt"
    marker.write_text("x")
    assert _run(["setup"]) == 0  # cached: nothing is fetched again
    assert marker.exists()
    assert _run(["setup", "--force"]) == 0
    assert not marker.exists()


def test_setup_reports_a_library_it_cannot_fetch(served_library, monkeypatch, capsys):
    monkeypatch.setenv("EOSQUALITY_REFERENCE_BASE_URL", "file:///nonexistent/")
    assert _run(["setup"]) == 1
    assert "couldnotfetchthereferencelibrary" in _err(capsys)


def test_the_maintainer_workflow_build_then_fit_then_run(
    tmp_path, smiles, reference, query, monkeypatch
):
    """``build`` makes a library that ``fit`` and ``run`` then use."""
    csv = _library_csv(tmp_path, smiles, "ersilia_reference_library_v0.csv", n=600)
    lib = tmp_path / "lib"
    assert _run(["build", "-i", csv, "-o", str(lib), "-j", "1"]) == 0
    monkeypatch.setenv("EOSQUALITY_REFERENCE_LIBRARY_PATH", str(lib))
    reference.to_csv(tmp_path / "reference_eos0aaa_v1.csv", index=False)
    query.to_csv(tmp_path / "query.csv", index=False)
    art = str(tmp_path / "artifacts_eos0aaa_v1")
    assert (
        _run(["fit", "-r", str(tmp_path / "reference_eos0aaa_v1.csv"), "-a", art]) == 0
    )
    out = tmp_path / "quality_eos0aaa_v1.csv"
    assert (
        _run(["run", "-i", str(tmp_path / "query.csv"), "-a", art, "-o", str(out)]) == 0
    )
    scores = pd.read_csv(out)
    meta = pathlib.Path(art) / "reference_mode/shared/metadata.json"
    assert json.loads(meta.read_text())["library_path"] == ""  # canonical: by identity
    assert {"ref_match", "ref_scaffold", "ref_typicality_pct"} <= set(scores.columns)
    assert scores["ref_match"].tail(40).eq(1).all()  # the 40 reference rows


def test_fit_and_run_use_a_pool_for_the_descriptors_when_asked(files, training_dir):
    """The default ``-j -1`` path: a pool for a column of 200 or more molecules."""
    only = str(files["tmp"] / "training_only_eos0aaa_v1")
    assert _run(["fit", "-t", str(training_dir), "-a", only, "-j", "2"]) == 0
    out = str(files["tmp"] / "pooled_eos0aaa_v1.csv")
    assert _run(["run", "-i", files["query"], "-a", only, "-o", out, "-j", "2"]) == 0
    assert "trn_physchem_pct" in pd.read_csv(out).columns


def test_a_custom_library_found_through_the_environment_is_recorded_by_path(files):
    """Not the canonical id: the artifacts must find it again at run time."""
    fit = ["fit", "-r", files["reference"], "-a", files["artifacts"]]
    assert _run(fit) == 0  # every reference score, ref_match included
    shared = pathlib.Path(files["artifacts"]) / "reference_mode/shared/metadata.json"
    meta = json.loads(shared.read_text())
    assert meta["library_id"] == "test_library" and meta["library_path"]
    run = ["run", "-i", files["query"], "-a", files["artifacts"], "-o", files["output"]]
    assert _run(run) == 0


def test_fit_and_build_report_unreadable_inputs(tmp_path, capsys):
    assert (
        _run(
            [
                "fit",
                "-r",
                str(tmp_path / "reference_eos0aaa_v1.csv"),
                "-a",
                "a_eos0aaa_v1",
            ]
        )
        == 1
    )
    assert "couldnotreadreferenceCSV" in _err(capsys)
    assert (
        _run(
            [
                "build",
                "-i",
                str(tmp_path / "ersilia_reference_library_v7.csv"),
                "-o",
                str(tmp_path / "o"),
            ]
        )
        == 1
    )
    assert "couldnotreadlibraryfile" in _err(capsys)
    wrong = tmp_path / "ersilia_reference_library_v8.csv"
    pd.DataFrame({"molecule": ["CCO"]}).to_csv(wrong, index=False)
    assert _run(["build", "-i", str(wrong), "-o", str(tmp_path / "o2")]) == 1
    assert "mustcontaina'smiles'column" in _err(capsys)
    assert not (tmp_path / "o2").exists()


def test_build_can_truncate_the_library(tmp_path, smiles):
    csv = _library_csv(tmp_path, smiles, "ersilia_reference_library_v7.csv", n=50)
    assert (
        _run(
            [
                "build",
                "-i",
                csv,
                "-o",
                str(tmp_path / "lib"),
                "--max-samples",
                "20",
                "-j",
                "1",
            ]
        )
        == 0
    )
    meta = json.loads((tmp_path / "lib" / "metadata.json").read_text())
    assert meta["n_samples"] == 20


def test_the_built_folder_has_the_usual_permissions_not_a_private_temp_folder(
    tmp_path, smiles
):
    import os
    import stat

    csv = _library_csv(tmp_path, smiles, "ersilia_reference_library_v7.csv")
    assert _run(["build", "-i", csv, "-o", str(tmp_path / "lib"), "-j", "1"]) == 0
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE((tmp_path / "lib").stat().st_mode) == 0o777 & ~umask
