"""Per-score components: Typicality, Support, Consistency, Extremity.

Each is independently fittable / runnable / saveable / loadable. Typicality
and Extremity need only :class:`~eosquality.shared.state.SharedFitState` —
they never touch the vector index. Support and Consistency add
:class:`~eosquality.knn.state.KnnFitState`.
"""

from eosquality.scores.consistency import Consistency, ConsistencyRunResult
from eosquality.scores.extremity import Extremity, ExtremityRunResult
from eosquality.scores.support import Support, SupportRunResult
from eosquality.scores.typicality import Typicality, TypicalityRunResult

__all__ = [
    "Typicality",
    "TypicalityRunResult",
    "Support",
    "SupportRunResult",
    "Consistency",
    "ConsistencyRunResult",
    "Extremity",
    "ExtremityRunResult",
]
