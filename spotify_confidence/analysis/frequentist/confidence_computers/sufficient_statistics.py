"""Encapsulates the logic for computing sufficient statistics from raw data."""

from __future__ import annotations

from typing import Any, Union

from pandas import DataFrame

from spotify_confidence.analysis.confidence_utils import reset_named_indices
from spotify_confidence.analysis.constants import (
    BOOTSTRAPS,
    DENOMINATOR,
    FEATURE,
    FEATURE_CROSS,
    FEATURE_SUMSQ,
    INTERVAL_SIZE,
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    ORIGINAL_POINT_ESTIMATE,
    ORIGINAL_VARIANCE,
    POINT_ESTIMATE,
    VARIANCE,
    ZTEST,
    ZTESTLINREG,
)
from spotify_confidence.analysis.frequentist.confidence_computers import confidence_computers


class SufficientStatisticsBuilder:
    """Builds a sufficient-statistics DataFrame from raw experiment data.

    Extracted from ``ConfidenceComputer._sufficient_statistics`` so the logic
    can be tested and reused independently.
    """

    def __init__(
        self,
        numerator: Union[str, None],
        numerator_sumsq: Union[str, None],
        denominator: Union[str, None],
        bootstrap_samples_column: Union[str, None],
        interval_size: float,
        feature: Union[str, None],
        feature_ssq: Union[str, None],
        feature_cross: Union[str, None],
        method_column: str,
        metric_column: Union[str, None],
    ):
        self._numerator = numerator
        self._numerator_sumsq = numerator_sumsq
        self._denominator = denominator
        self._bootstrap_samples_column = bootstrap_samples_column
        self._interval_size = interval_size
        self._feature = feature
        self._feature_ssq = feature_ssq
        self._feature_cross = feature_cross
        self._method_column = method_column
        self._metric_column = metric_column

    @property
    def _kwargs(self) -> dict[str, Any]:
        return {
            NUMERATOR: self._numerator,
            NUMERATOR_SUM_OF_SQUARES: self._numerator_sumsq,
            DENOMINATOR: self._denominator,
            BOOTSTRAPS: self._bootstrap_samples_column,
            INTERVAL_SIZE: self._interval_size,
            FEATURE: self._feature,
            FEATURE_SUMSQ: self._feature_ssq,
            FEATURE_CROSS: self._feature_cross,
        }

    def build(self, df: DataFrame) -> DataFrame:
        """Compute sufficient statistics (point estimate, variance, CI) for *df*."""
        kwargs = self._kwargs
        method_col = self._method_column
        groupby = [col for col in [method_col, self._metric_column] if col is not None]

        return (
            df.groupby(groupby, sort=False, group_keys=True)
            .apply(
                lambda df: (
                    df.assign(
                        **{
                            POINT_ESTIMATE: lambda df: confidence_computers[df[method_col].values[0]].point_estimate(
                                df, **kwargs
                            )
                        }
                    )
                    .assign(
                        **{
                            ORIGINAL_POINT_ESTIMATE: lambda df: (
                                confidence_computers[ZTEST].point_estimate(df, **kwargs)
                                if df[method_col].values[0] == ZTESTLINREG
                                else confidence_computers[df[method_col].values[0]].point_estimate(df, **kwargs)
                            )
                        }
                    )
                    .assign(
                        **{VARIANCE: lambda df: confidence_computers[df[method_col].values[0]].variance(df, **kwargs)}
                    )
                    .assign(
                        **{
                            ORIGINAL_VARIANCE: lambda df: (
                                confidence_computers[ZTEST].variance(df, **kwargs)
                                if df[method_col].values[0] == ZTESTLINREG
                                else confidence_computers[df[method_col].values[0]].variance(df, **kwargs)
                            )
                        }
                    )
                    .pipe(
                        lambda df: confidence_computers[df[method_col].values[0]].add_point_estimate_ci(df, **kwargs)
                    )
                )
            )
            .pipe(reset_named_indices)
        )
