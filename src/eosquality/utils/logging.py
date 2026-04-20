"""Pretty, informative logging for eosquality using loguru + rich."""

import sys
from typing import Optional

from loguru import logger as _loguru
from rich import box
from rich.console import Console
from rich.table import Table

_loguru.remove()
_loguru.level("DEBUG", color="<cyan><bold>")
_loguru.level("INFO", color="<blue><bold>")
_loguru.level("WARNING", color="<white><bold><bg yellow>")
_loguru.level("ERROR", color="<white><bold><bg red>")
_loguru.level("CRITICAL", color="<white><bold><bg red>")
_loguru.level("SUCCESS", color="<black><bold><bg green>")

_FORMAT = (
    "<green>{time:HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "{message}"
)


class Logger:
    """Thin wrapper around loguru + rich for informative, pretty output.

    Output is suppressed by default. Call ``set_verbosity(True)`` or pass
    ``verbose=True`` to ``ErsiliaQuality`` to enable it.
    """

    def __init__(self) -> None:
        self.logger = _loguru
        self._console = Console(stderr=True, highlight=False)
        self._sink_id: Optional[int] = None
        self._verbose: bool = False

    @property
    def verbose(self) -> bool:
        return self._verbose

    def set_verbosity(self, verbose: bool) -> None:
        """Enable or disable log output."""
        self._verbose = verbose
        if verbose and self._sink_id is None:
            self._sink_id = self.logger.add(
                sys.stderr,
                format=_FORMAT,
                colorize=True,
                level="DEBUG",
            )
        elif not verbose and self._sink_id is not None:
            try:
                self.logger.remove(self._sink_id)
            except Exception:
                pass
            self._sink_id = None

    # ------------------------------------------------------------------
    # Standard log levels
    # ------------------------------------------------------------------

    def debug(self, text: str) -> None:
        self.logger.debug(text)

    def info(self, text: str) -> None:
        self.logger.info(text)

    def warning(self, text: str) -> None:
        self.logger.warning(text)

    def error(self, text: str) -> None:
        self.logger.error(text)

    def success(self, text: str) -> None:
        self.logger.success(text)

    # ------------------------------------------------------------------
    # Rich display helpers
    # ------------------------------------------------------------------

    def rule(self, title: str = "", style: str = "dim blue") -> None:
        """Print a horizontal rule, optionally with a title."""
        if not self._verbose:
            return
        if title:
            self._console.rule(f"[bold cyan]{title}[/]", style=style)
        else:
            self._console.rule(style=style)

    def reference_table(
        self,
        n_samples: int,
        n_features: int,
        column_names: list[str],
    ) -> None:
        """Display a summary of the reference population being fitted."""
        if not self._verbose:
            return
        table = Table(
            title="[bold]Reference population[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan", no_wrap=True, min_width=16)
        table.add_column("Value", justify="right", min_width=14)

        table.add_row("Samples", f"{n_samples:,}")
        table.add_row("Features", f"{n_features:,}")
        table.add_row("Columns", ", ".join(column_names[:8]) + ("…" if len(column_names) > 8 else ""))

        self._console.print(table)
        self._console.line()

    def reference_report_table(
        self,
        reference_quality: float,
        cohesion_score: float,
        fragmentation_score: float,
        median_k_distance: float,
        notes: list[str],
    ) -> None:
        """Display reference quality diagnostics computed during fit."""
        if not self._verbose:
            return
        table = Table(
            title="[bold]Reference quality diagnostics[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan", no_wrap=True, min_width=22)
        table.add_column("Value", justify="right", min_width=10)

        def _quality_style(v: float) -> str:
            if v >= 0.7:
                return "green"
            if v >= 0.4:
                return "yellow"
            return "red"

        table.add_row(
            "reference_quality",
            f"[{_quality_style(reference_quality)}]{reference_quality:.4f}[/]",
        )
        table.add_row("cohesion_score", f"{cohesion_score:.4f}")
        table.add_row("fragmentation_score", f"{fragmentation_score:.4f}")
        table.add_row("median_k_distance", f"{median_k_distance:.4f}")

        self._console.print(table)
        for note in notes:
            self._console.print(f"  [dim yellow]⚠ {note}[/dim yellow]")
        self._console.line()

    def scores_summary_table(self, scores_df) -> None:
        """Display a summary of score distributions from a run() call."""
        if not self._verbose:
            return
        score_cols = ["quality_score", "support_score", "consistency_score", "intrinsic_richness"]
        present = [c for c in score_cols if c in scores_df.columns]

        table = Table(
            title="[bold]Score summary[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Score", style="cyan", no_wrap=True, min_width=20)
        table.add_column("mean", justify="right", min_width=8)
        table.add_column("median", justify="right", min_width=8)
        table.add_column("min", justify="right", min_width=8)
        table.add_column("max", justify="right", min_width=8)

        for col in present:
            s = scores_df[col]
            table.add_row(
                col,
                f"{s.mean():.4f}",
                f"{s.median():.4f}",
                f"{s.min():.4f}",
                f"{s.max():.4f}",
            )

        self._console.print(table)
        self._console.line()

    def index_input_table(self, n_molecules: int, n_unique: int) -> None:
        """Display a summary of the molecule collection passed to VectorIndex.build()."""
        if not self._verbose:
            return
        table = Table(
            title="[bold]Input data[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Metric", style="cyan", no_wrap=True, min_width=20)
        table.add_column("Value", justify="right", min_width=14)

        table.add_row("Molecules", f"{n_molecules:,}")
        table.add_row("Unique SMILES", f"{n_unique:,}")

        self._console.print(table)
        self._console.line()

    def index_config_table(
        self,
        max_k: int,
        radius: int,
        n_bits: int,
        rdkit_version: str,
        output_dir: str,
        library_name: str = "",
    ) -> None:
        """Display the configuration used to build a VectorIndex."""
        if not self._verbose:
            return
        table = Table(
            title="[bold]Index configuration[/bold]",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Parameter", style="cyan", no_wrap=True, min_width=20)
        table.add_column("Value", justify="right", min_width=16)

        if library_name:
            table.add_row("Library", library_name)
        table.add_row("Output dir", output_dir)
        table.add_row("Method", "Morgan (FPSim2)")
        table.add_row("Radius", str(radius))
        table.add_row("FP bits", f"{n_bits:,}")
        table.add_row("max_k", str(max_k))
        table.add_row("RDKit version", rdkit_version)

        self._console.print(table)
        self._console.line()

    def timing_table(self, steps: list[tuple[str, float, bool]], title: str = "Fit timing breakdown") -> None:
        """Print a per-step timing breakdown.

        Parameters
        ----------
        steps:
            List of (name, seconds, is_subtask) tuples.
            ``is_subtask=True`` → indented row, % column left blank.
        title:
            Table title (default: "Fit timing breakdown").
        """
        if not self._verbose:
            return
        top_level_times = [t for _, t, sub in steps if not sub]
        total = sum(top_level_times) if top_level_times else 0.0

        table = Table(
            title=f"[bold]{title}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_justify="left",
            padding=(0, 1),
        )
        table.add_column("Step", min_width=30)
        table.add_column("Time (s)", justify="right", min_width=9)
        table.add_column("%", justify="right", min_width=5)

        for name, t, is_subtask in steps:
            label = f"  {name}" if is_subtask else name
            pct_str = "" if is_subtask else (f"{100 * t / total:.0f}%" if total > 0 else "—")
            style = "dim" if is_subtask else ""
            table.add_row(label, f"{t:.2f}", pct_str, style=style)

        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{total:.2f}[/bold]",
            "[bold]100%[/bold]",
        )
        self._console.print(table)
        self._console.line()


logger = Logger()
