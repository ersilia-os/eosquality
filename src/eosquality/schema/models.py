"""Schema dataclasses describing the structure of input data."""

from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    """Specification for a single input column."""

    name: str
    kind: str  # always "numeric" today


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
