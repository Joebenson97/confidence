from __future__ import annotations

from functools import reduce
from typing import Any, Union

import numpy as np
from pandas import DataFrame, Series

from spotify_confidence.analysis.confidence_utils import dfmatmul, unlist
from spotify_confidence.analysis.constants import (
    DENOMINATOR,
    FEATURE,
    FEATURE_CROSS,
    FEATURE_SUMSQ,
    NUMERATOR,
    REGRESSION_PARAM,
)
from spotify_confidence.analysis.frequentist.confidence_computers.z_test_computer import ZTestComputer


# Keep the module-level z_test singleton for delegation
_z_test = ZTestComputer()


def _estimate_slope(df, **kwargs: Any) -> DataFrame:
    if kwargs[FEATURE] not in df:
        return df

    def col_sum(x):
        return reduce(lambda x, y: x + y, x)

    def dimension(x):
        return x.shape[0] if isinstance(x, np.ndarray) and x.size > 1 else 1

    k = df[kwargs[FEATURE_SUMSQ]].apply(dimension).iloc[0]

    XX0 = np.zeros((k + 1, k + 1))
    XX0[1 : (k + 1), 1 : (k + 1)] = col_sum(df[kwargs[FEATURE_SUMSQ]])
    XX0[0, 0] = col_sum(df[kwargs[DENOMINATOR]])
    XX0[0, 1 : (k + 1)] = col_sum(df[kwargs[FEATURE]])
    XX0[1 : (k + 1), 0] = col_sum(df[kwargs[FEATURE]])

    Xy0 = np.zeros((k + 1, 1))
    Xy0[0,] = col_sum(df[kwargs[NUMERATOR]])
    Xy0[1 : (k + 1),] = np.atleast_2d(col_sum(df[kwargs[FEATURE_CROSS]])).reshape(-1, 1)

    try:
        b = np.matmul(np.linalg.inv(XX0), Xy0)
    except np.linalg.LinAlgError:
        b = np.zeros((k + 1, 1))
    out = b[1 : (k + 1)]
    if out.size == 1:
        out = out.item()

    outseries = Series(index=df.index, dtype=df[kwargs[FEATURE]].dtype)
    df[REGRESSION_PARAM] = outseries.apply(lambda x: out)
    return df


def _lin_reg_variance_delta(row, **kwargs):
    y = row[kwargs[NUMERATOR]]
    n = row[kwargs[DENOMINATOR]]

    XX = unlist(row[kwargs[FEATURE_SUMSQ]])
    X = unlist(row[kwargs[FEATURE]])
    Xy = unlist(row[kwargs[FEATURE_CROSS]])

    sample_var = XX / n - dfmatmul(X / n, X / n)
    sample_cov = Xy / n - dfmatmul(X / n, y / n)
    b = np.atleast_2d(row[REGRESSION_PARAM])
    variance2 = np.matmul(np.transpose(b), np.matmul(sample_var, b)).item()
    variance3 = -2 * np.matmul(np.transpose(b), sample_cov).item()

    return variance2 + variance3


class ZTestLinregComputer(ZTestComputer):
    """Z-test with linear regression adjustment."""

    @property
    def supports_sequential(self) -> bool:
        return True

    @property
    def supports_powered_effect(self) -> bool:
        return True

    @property
    def supports_mde(self) -> bool:
        return True

    def point_estimate(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        df = _estimate_slope(df, **kwargs)
        pe = df[kwargs[NUMERATOR]] / df[kwargs[DENOMINATOR]]

        if REGRESSION_PARAM in df:
            feature_mean = df[kwargs[FEATURE]].sum() / df[kwargs[DENOMINATOR]].sum()

            def lin_reg_point_estimate_delta(row: Series, feature_mean: float, **kwargs: Any) -> Series:
                return dfmatmul(
                    row[REGRESSION_PARAM], row[kwargs[FEATURE]] - feature_mean * row[kwargs[DENOMINATOR]], outer=False
                )

            return (
                pe
                - df.apply(lin_reg_point_estimate_delta, feature_mean=feature_mean, axis=1, **kwargs)
                / df[kwargs[DENOMINATOR]]
            )

        return pe

    def variance(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        variance1 = _z_test.variance(df, **kwargs)
        if kwargs[FEATURE] in df:
            computed_variances = variance1 + df.apply(_lin_reg_variance_delta, axis=1, **kwargs)
            if (computed_variances < 0).any():
                raise ValueError("Computed variance is negative, please check sufficient statistics.")
            return computed_variances
        else:
            return variance1


# ---------------------------------------------------------------------------
# Backward-compatible module-level function aliases
# ---------------------------------------------------------------------------

_singleton = ZTestLinregComputer()

# Keep old names available at module level
estimate_slope = _estimate_slope
lin_reg_variance_delta = _lin_reg_variance_delta

point_estimate = _singleton.point_estimate
variance = _singleton.variance
add_point_estimate_ci = _singleton.add_point_estimate_ci
std_err = _singleton.std_err
p_value = _singleton.p_value
ci = _singleton.ci
powered_effect = _singleton.powered_effect
required_sample_size = _singleton.required_sample_size
