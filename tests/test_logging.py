import logging

import pandas as pd

from eosquality import ErsiliaQuality
from eosquality.utils import console
from eosquality.utils.logging import logger


def test_library_use_is_silent(training_dir, capsys):
    console.enable(False)
    logger.set_verbosity(False)
    eq = ErsiliaQuality().fit(eos_id="eos0aaa", training_sets=training_dir)
    eq.run(pd.DataFrame({"input": ["CCO", "not a smiles"]}))
    captured = capsys.readouterr()
    assert captured.out == ""
    # Only genuine warnings surface (here, the unparsable query SMILES).
    assert "Step" not in captured.err and "INFO" not in captured.err
    assert "WARNING" in captured.err


def test_verbose_turns_curated_output_on(training_dir):
    console.enable(False)
    with console.console.capture() as capture:
        ErsiliaQuality(verbose=True).fit(eos_id="eos0aaa", training_sets=training_dir)
    logger.set_verbosity(False)
    console.enable(False)
    text = capture.get()
    assert "Training sets" in text and "Training modality" in text
    assert "Step 1/4" in text


def test_dependency_loggers_are_routed(tmp_path):
    path = tmp_path / "x.log"
    with logger.log_file(path):
        logging.getLogger("eosframes").info("hello from eosframes")
    line = path.read_text().strip()
    assert line.endswith("hello from eosframes")
    assert "| eosframes:" in line


def test_log_file_records_the_caller(tmp_path):
    path = tmp_path / "x.log"
    with logger.log_file(path):
        logger.info("caller check")
    assert "test_logging:test_log_file_records_the_caller" in path.read_text()


def test_long_tables_are_cut_short():
    console.enable(True)
    with console.console.capture() as capture:
        console.table(("a", "b"), [(str(i), "x") for i in range(40)])
    console.enable(False)
    text = capture.get()
    assert "… and 25 more" in text
    assert "14" in text and "15" not in text  # 15 rows printed, 0 … 14
