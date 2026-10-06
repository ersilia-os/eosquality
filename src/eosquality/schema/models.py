"""Schema dataclasses describing the structure of input data."""

from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    """Specification for a single input column."""

    name: str
    kind: str  # "numeric" in v0.1; future: "binary", "categorical", "count", "vector"
    weight: float = 1.0
    missing_policy: str = "ignore"
    block: str | None = None


@dataclass
class Schema:
    """Full schema for a DataFrame, inferred or user-supplied."""

    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        """Names of the schema's columns, in order.

        Returns
        -------
        list of str
        """
        return [c.name for c in self.columns]
