"""Per-score components of the reference modality: Typicality, Extremity, Match.

Each is independently fittable / runnable / saveable / loadable. Typicality
and Extremity need only :class:`~eosquality.shared.state.SharedFitState`;
:class:`ReferenceMatch` also reads the reference library's match keys.
"""

from eosquality.scores._percentile_score import PercentileRunResult
from eosquality.scores.extremity import Extremity
from eosquality.scores.reference_match import ReferenceMatch, ReferenceMatchRunResult
from eosquality.scores.typicality import Typicality

__all__ = [
    "Extremity",
    "PercentileRunResult",
    "ReferenceMatch",
    "ReferenceMatchRunResult",
    "Typicality",
]
