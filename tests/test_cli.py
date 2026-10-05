import pandas as pd
import pytest

from eosquality.cli import main


def _run(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code


def test_fit_and_run(tmp_path, reference, query, library):
    ref_csv = tmp_path / "eos0aaa_v1.csv"
    reference.to_csv(ref_csv, index=False)
    query_csv = tmp_path / "query.csv"
    query.to_csv(query_csv, index=False)
    art = tmp_path / "art"
    assert (
        _run(
            [
                "fit",
                "--reference",
                str(ref_csv),
                "-o",
                str(art),
                "--vector-index",
                str(library),
                "--ignore-size",
            ]
        )
        == 0
    )
    out = tmp_path / "scores.csv"
    assert _run(["run", "-i", str(query_csv), "-a", str(art), "-o", str(out)]) == 0
    scores = pd.read_csv(out)
    assert list(scores.columns[:2]) == ["key", "input"]
    assert len(scores) == len(query)


def test_fit_refuses_existing_output(tmp_path, reference):
    ref_csv = tmp_path / "eos0aaa_v1.csv"
    reference.to_csv(ref_csv, index=False)
    (tmp_path / "art").mkdir()
    assert _run(["fit", "--reference", str(ref_csv), "-o", str(tmp_path / "art")]) == 1


def test_fit_with_training_and_details(
    tmp_path, reference, query, library, training_dir
):
    ref_csv = tmp_path / "eos0aaa_v1.csv"
    reference.to_csv(ref_csv, index=False)
    query_csv = tmp_path / "query.csv"
    query.to_csv(query_csv, index=False)
    art = tmp_path / "art"
    assert (
        _run(
            [
                "fit",
                "--reference",
                str(ref_csv),
                "--training",
                str(training_dir),
                "-o",
                str(art),
                "--vector-index",
                str(library),
                "--ignore-size",
            ]
        )
        == 0
    )
    out = tmp_path / "scores.csv"
    assert _run(["run", "-i", str(query_csv), "-a", str(art), "-o", str(out)]) == 0
    assert "training_domain" in pd.read_csv(out).columns
    details = pd.read_csv(tmp_path / "scores.training_details.csv")
    assert len(details) == 3 * len(query)


def test_training_only_and_add_later(tmp_path, reference, query, library, training_dir):
    # training only: model id from the folder name training_eos0aaa_v1/
    art = tmp_path / "train_only"
    assert _run(["fit", "--training", str(training_dir), "-o", str(art)]) == 0
    query[["key", "input"]].to_csv(tmp_path / "q.csv", index=False)
    assert (
        _run(
            [
                "run",
                "-i",
                str(tmp_path / "q.csv"),
                "-a",
                str(art),
                "-o",
                str(tmp_path / "s.csv"),
            ]
        )
        == 0
    )
    # add later: reference-only artifacts, then --artifacts
    ref_csv = tmp_path / "eos0aaa_v1.csv"
    reference.to_csv(ref_csv, index=False)
    art2 = tmp_path / "art2"
    assert (
        _run(
            [
                "fit",
                "--reference",
                str(ref_csv),
                "-o",
                str(art2),
                "--vector-index",
                str(library),
                "--ignore-size",
                "--scores",
                "typicality",
            ]
        )
        == 0
    )
    assert _run(["fit", "--training", str(training_dir), "--artifacts", str(art2)]) == 0
    assert (art2 / "training").is_dir()


def test_fit_argument_errors(tmp_path, training_dir):
    assert _run(["fit", "-o", str(tmp_path / "x")]) == 1  # no inputs
    assert _run(["fit", "--training", str(training_dir)]) == 1  # no output
