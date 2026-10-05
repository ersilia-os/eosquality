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
                "-i",
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
    assert _run(["fit", "-i", str(ref_csv), "-o", str(tmp_path / "art")]) == 1
