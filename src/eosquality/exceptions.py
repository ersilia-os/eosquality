class NotFittedError(RuntimeError):
    """Raised when ErsiliaQuality methods are called before fit()."""


class SchemaError(ValueError):
    """Raised when input data does not conform to the expected schema."""
