"""Curated terminal output for eosquality, in the Ersilia (ZairaChem / Olinda) style.

Two layers, as in ZairaChem and Olinda: this module prints the **curated,
user-facing** stream (themed section rules, numbered steps that close with a
timed ``✓`` line, borderless detail blocks, rounded summary panels, progress
bars); :mod:`eosquality.utils.logging` keeps the **diagnostic** stream
(loguru: warnings on screen, everything in the log file). Both write through
the one shared :data:`console`, so log lines and live progress interleave
cleanly.

As a library, eosquality is silent: nothing here prints until
:func:`enable` is called, which the CLI does for every command and
``ErsiliaQuality(verbose=True)`` / ``eosquality.set_verbosity(True)`` do too.

    from eosquality.utils import console

    with console.section("Reference modality"):
        steps = console.Steps(3)
        with steps("Shared state") as st:
            ...
            st.summary = "4 → 3 features"
"""

from eosquality.utils.console._core import (
    STEP_COLORS,
    active_color,
    console,
    echo,
    enable,
    enabled,
    rule,
    set_active_color,
    success,
)
from eosquality.utils.console.bars import progress, track
from eosquality.utils.console.display import (
    MAX_TABLE_ROWS,
    detail,
    median_summary,
    share_summary,
    summary_panel,
    table,
)
from eosquality.utils.console.fmt import (
    elapsed,
    filesize,
    folder_size,
    path,
    plain,
    plural,
    resources,
)
from eosquality.utils.console.steps import Steps, section

__all__ = [
    "MAX_TABLE_ROWS",
    "STEP_COLORS",
    "Steps",
    "active_color",
    "console",
    "detail",
    "echo",
    "elapsed",
    "enable",
    "enabled",
    "filesize",
    "folder_size",
    "median_summary",
    "path",
    "plain",
    "plural",
    "progress",
    "resources",
    "rule",
    "section",
    "set_active_color",
    "share_summary",
    "success",
    "summary_panel",
    "table",
    "track",
]
