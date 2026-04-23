class NotFittedError(RuntimeError):
    """Raised when ErsiliaQuality methods are called before fit()."""


class SchemaError(ValueError):
    """Raised when input data does not conform to the expected schema."""


class IncompatibleArtifactsError(ValueError):
    """Raised when saved artifacts were produced against a different reference library.

    The reference library is pinned to the package major version: ``eosquality
    X.y.z`` ships with ``ersilia_reference_library_vX``. Artifacts fit under one
    major cannot be loaded under another — scores would silently differ. Refit
    with a compatible release, or install the release that produced the artifacts.
    """
