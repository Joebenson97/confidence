from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from pandas import DataFrame, Series

from spotify_confidence.analysis.abstract_base_classes.statistical_method_abc import StatisticalMethodABC
from spotify_confidence.analysis.constants import BOOTSTRAPS, CI_LOWER, CI_UPPER, INTERVAL_SIZE, SFX1, SFX2


class BootstrapComputer(StatisticalMethodABC):
    """Bootstrap-based statistical method (stub implementations for p_value etc.)."""

    def point_estimate(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        bootstrap_samples = kwargs[BOOTSTRAPS]
        return df[bootstrap_samples].map(lambda a: a.mean())

    def variance(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        bootstrap_samples = kwargs[BOOTSTRAPS]
        variance = df[bootstrap_samples].map(lambda a: a.var())

        if (variance < 0).any():
            raise ValueError("Computed variance is negative. Please check your inputs.")
        return variance

    def std_err(self, df: DataFrame, **kwargs: Any) -> Optional[Union[float, Series]]:
        return None

    def add_point_estimate_ci(self, df: DataFrame, **kwargs: Any) -> DataFrame:
        bootstrap_samples = kwargs[BOOTSTRAPS]
        interval_size = kwargs[INTERVAL_SIZE]
        df[CI_LOWER] = df[bootstrap_samples].map(lambda a: np.percentile(a, 100 * (1 - interval_size) / 2))
        df[CI_UPPER] = df[bootstrap_samples].map(lambda a: np.percentile(a, 100 * (1 - (1 - interval_size) / 2)))
        return df

    def p_value(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        return -1

    def ci(self, df: DataFrame, alpha_column: str, **kwargs: Any) -> Tuple[Series, Series]:
        bootstrap_samples = kwargs[BOOTSTRAPS]
        lower = df.apply(
            lambda row: np.percentile(
                row[bootstrap_samples + SFX2] - row[bootstrap_samples + SFX1], 100 * row[alpha_column] / 2
            ),
            axis=1,
        )
        upper = df.apply(
            lambda row: np.percentile(
                row[bootstrap_samples + SFX2] - row[bootstrap_samples + SFX1], 100 * (1 - row[alpha_column] / 2)
            ),
            axis=1,
        )
        return lower, upper

    def achieved_power(self, df: DataFrame, mde: float, alpha: float, **kwargs: Any) -> Optional[DataFrame]:
        return None


# ---------------------------------------------------------------------------
# Backward-compatible module-level function aliases
# ---------------------------------------------------------------------------

_singleton = BootstrapComputer()

point_estimate = _singleton.point_estimate
variance = _singleton.variance
std_err = _singleton.std_err
add_point_estimate_ci = _singleton.add_point_estimate_ci
p_value = _singleton.p_value
ci = _singleton.ci
achieved_power = _singleton.achieved_power
