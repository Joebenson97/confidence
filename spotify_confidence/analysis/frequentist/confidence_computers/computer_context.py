from __future__ import annotations

from typing import List, Optional

from typing_extensions import TypedDict

from spotify_confidence.analysis.constants import (
    ABSOLUTE,
    BOOTSTRAPS,
    CORRECTION_METHOD,
    DENOMINATOR,
    FEATURE,
    FEATURE_CROSS,
    FEATURE_SUMSQ,
    FINAL_EXPECTED_SAMPLE_SIZE,
    INTERVAL_SIZE,
    IS_BINARY,
    MDE,
    METHOD,
    NUMBER_OF_COMPARISONS,
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    ORDINAL_GROUP_COLUMN,
    PREFERENCE_TEST,
    TREATMENT_WEIGHTS,
)


class ComputerContext(TypedDict, total=False):
    """Typed replacement for the untyped **kwargs: Any pattern used throughout
    confidence computer functions.

    Keys correspond to the string constants defined in
    ``spotify_confidence.analysis.constants``.  All fields are optional
    (``total=False``) because different call-sites supply different subsets.
    """

    numerator: Optional[str]
    numerator_sum_of_squares: Optional[str]
    denominator: Optional[str]
    bootstraps: Optional[str]
    interval_size: float
    method_column: Optional[str]
    correction_method: str
    absolute: bool
    number_of_comparisons: int
    final_expected_sample_size: Optional[str]
    ordinal_group_column: Optional[str]
    mde: Optional[str]
    treatment_weights: Optional[List[float]]
    is_binary: bool
    feature: Optional[str]
    feature_sumsq: Optional[str]
    feature_cross: Optional[str]
    preference_used_in_test: Optional[str]


# Mapping from TypedDict field names back to the constant strings so that
# call-sites can be migrated incrementally.
CONTEXT_KEY_MAP = {
    "numerator": NUMERATOR,
    "numerator_sum_of_squares": NUMERATOR_SUM_OF_SQUARES,
    "denominator": DENOMINATOR,
    "bootstraps": BOOTSTRAPS,
    "interval_size": INTERVAL_SIZE,
    "method_column": METHOD,
    "correction_method": CORRECTION_METHOD,
    "absolute": ABSOLUTE,
    "number_of_comparisons": NUMBER_OF_COMPARISONS,
    "final_expected_sample_size": FINAL_EXPECTED_SAMPLE_SIZE,
    "ordinal_group_column": ORDINAL_GROUP_COLUMN,
    "mde": MDE,
    "treatment_weights": TREATMENT_WEIGHTS,
    "is_binary": IS_BINARY,
    "feature": FEATURE,
    "feature_sumsq": FEATURE_SUMSQ,
    "feature_cross": FEATURE_CROSS,
    "preference_used_in_test": PREFERENCE_TEST,
}
