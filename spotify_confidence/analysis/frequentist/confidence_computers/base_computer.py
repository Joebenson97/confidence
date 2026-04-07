from __future__ import annotations

from typing import Any, Union

from pandas import DataFrame, Series

from spotify_confidence.analysis.abstract_base_classes.statistical_method_abc import StatisticalMethodABC
from spotify_confidence.analysis.constants import DENOMINATOR, NUMERATOR


class BaseNumeratorDenominatorComputer(StatisticalMethodABC):
    """Shared base for computers whose ``point_estimate`` is simply
    ``numerator / denominator``.  Both ``ZTestComputer`` and
    ``TTestComputer`` inherit from this.
    """

    def point_estimate(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        numerator = kwargs[NUMERATOR]
        denominator = kwargs[DENOMINATOR]
        if (df[denominator] == 0).any():
            raise ValueError("""Can't compute point estimate: denominator is 0""")
        return df[numerator] / df[denominator]
