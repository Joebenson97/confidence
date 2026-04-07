"""Pipeline functions for computing pairwise comparisons.

Extracted from ``confidence_computer.py`` so the comparison logic is
self-contained and testable independently of the orchestrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import isnan
from pandas import DataFrame
from scipy import stats as st

from spotify_confidence.analysis.constants import (
    ABSOLUTE,
    ADJUSTED_ALPHA_POWER_SAMPLE_SIZE,
    ADJUSTED_LOWER,
    ADJUSTED_POWER,
    ADJUSTED_UPPER,
    ALTERNATIVE_HYPOTHESIS,
    CHI2,
    CI_LOWER,
    CI_UPPER,
    DENOMINATOR,
    DIFFERENCE,
    MDE,
    METHOD,
    NIM,
    NULL_HYPOTHESIS,
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    ORIGINAL_POINT_ESTIMATE,
    ORIGINAL_VARIANCE,
    P_VALUE,
    POINT_ESTIMATE,
    POWERED_EFFECT,
    PREFERENCE,
    PREFERENCE_DICT,
    PREFERENCE_TEST,
    REQUIRED_SAMPLE_SIZE,
    REQUIRED_SAMPLE_SIZE_METRIC,
    SFX1,
    SFX2,
    STD_ERR,
    TWO_SIDED,
    VARIANCE,
    VARIANCE_REDUCTION,
    ZTESTLINREG,
)
from spotify_confidence.analysis.frequentist.confidence_computers import confidence_computers
from spotify_confidence.analysis.frequentist.multiple_comparison import (
    add_ci,
    set_alpha_and_adjust_preference,
)


def compute_comparisons(df: DataFrame, **kwargs: Any) -> DataFrame:
    """Compute difference, std_err, p-value, powered effect, and preference for a comparison."""
    return (
        df.assign(**{DIFFERENCE: lambda df: df[POINT_ESTIMATE + SFX2] - df[POINT_ESTIMATE + SFX1]})
        .assign(**{STD_ERR: confidence_computers[df[kwargs[METHOD]].values[0]].std_err(df, **kwargs)})
        .pipe(add_p_value, **kwargs)
        .pipe(powered_effect_and_required_sample_size, **kwargs)
        .assign(**{PREFERENCE: lambda df: df[PREFERENCE].map(PREFERENCE_DICT)})
        .pipe(add_variance_reduction_rate, **kwargs)
    )


def add_variance_reduction_rate(df: DataFrame, **kwargs: Any) -> DataFrame:
    """Add variance reduction rate column for linreg methods."""
    denominator = kwargs[DENOMINATOR]
    method_column = kwargs[METHOD]
    if (df[method_column] == ZTESTLINREG).any():
        variance_no_reduction = (
            df[ORIGINAL_VARIANCE + SFX1] / df[denominator + SFX1]
            + df[ORIGINAL_VARIANCE + SFX2] / df[denominator + SFX2]
        )
        variance_w_reduction = (
            df[VARIANCE + SFX1] / df[denominator + SFX1] + df[VARIANCE + SFX2] / df[denominator + SFX2]
        )
        df = df.assign(**{VARIANCE_REDUCTION: 1 - np.divide(variance_w_reduction, variance_no_reduction)})
    return df


def add_p_value(df: DataFrame, **kwargs: Any) -> DataFrame:
    """Set alpha/preference and compute p-value."""
    return df.pipe(set_alpha_and_adjust_preference, **kwargs).assign(
        **{P_VALUE: lambda df: df.pipe(_p_value, **kwargs)}
    )


def add_ci_and_adjust_if_absolute(df: DataFrame, **kwargs: Any) -> DataFrame:
    """Add confidence intervals and adjust for absolute/relative difference."""
    return df.pipe(add_ci, **kwargs).pipe(_adjust_if_absolute, absolute=kwargs[ABSOLUTE])


def _adjust_if_absolute(df: DataFrame, absolute: bool) -> DataFrame:
    if absolute:
        return df.assign(absolute_difference=absolute)
    else:
        return (
            df.assign(absolute_difference=absolute)
            .assign(**{DIFFERENCE: df[DIFFERENCE] / df[POINT_ESTIMATE + SFX1]})
            .assign(**{CI_LOWER: df[CI_LOWER] / df[POINT_ESTIMATE + SFX1]})
            .assign(**{CI_UPPER: df[CI_UPPER] / df[POINT_ESTIMATE + SFX1]})
            .assign(**{ADJUSTED_LOWER: df[ADJUSTED_LOWER] / df[POINT_ESTIMATE + SFX1]})
            .assign(**{ADJUSTED_UPPER: df[ADJUSTED_UPPER] / df[POINT_ESTIMATE + SFX1]})
            .assign(**{NULL_HYPOTHESIS: df[NULL_HYPOTHESIS] / df[POINT_ESTIMATE + SFX1]})
            .assign(**{POWERED_EFFECT: df[POWERED_EFFECT] / df[POINT_ESTIMATE + SFX1]})
        )


def _p_value(df: DataFrame, **kwargs: Any) -> float:
    if df[kwargs[METHOD]].values[0] == CHI2 and (df[NIM].notna()).any():
        raise ValueError("Non-inferiority margins not supported in ChiSquared. Use StudentsTTest or ZTest instead.")
    return confidence_computers[df[kwargs[METHOD]].values[0]].p_value(df, **kwargs)


def powered_effect_and_required_sample_size(df: DataFrame, **kwargs: Any) -> DataFrame:
    """Compute powered effect size and required sample size."""
    method = df[kwargs[METHOD]].values[0]
    computer = confidence_computers[method]
    if not computer.supports_mde and kwargs[MDE] in df:
        raise ValueError("Minimum detectable effects only supported for ZTest.")
    elif not computer.supports_powered_effect or (df[ADJUSTED_POWER].isna()).any():
        df[POWERED_EFFECT] = None
        df[REQUIRED_SAMPLE_SIZE] = None
        df[REQUIRED_SAMPLE_SIZE_METRIC] = None
        return df
    else:
        n1, n2 = df[kwargs[DENOMINATOR] + SFX1], df[kwargs[DENOMINATOR] + SFX2]
        kappa = n1 / n2
        binary = (df[kwargs[NUMERATOR_SUM_OF_SQUARES] + SFX1] == df[kwargs[NUMERATOR] + SFX1]).all()
        proportion_of_total = (n1 + n2) / df[f"current_total_{kwargs[DENOMINATOR]}"]

        z_alpha = st.norm.ppf(
            1
            - df[ADJUSTED_ALPHA_POWER_SAMPLE_SIZE].values[0] / (2 if df[PREFERENCE_TEST].values[0] == TWO_SIDED else 1)
        )
        z_power = st.norm.ppf(df[ADJUSTED_POWER].values[0])

        nim = df[NIM].values[0]
        if isinstance(nim, float):
            non_inferiority = not isnan(nim)
        elif nim is None:
            non_inferiority = nim is not None

        df[POWERED_EFFECT] = confidence_computers[df[kwargs[METHOD]].values[0]].powered_effect(
            df=df.assign(kappa=kappa)
            .assign(current_number_of_units=df[f"current_total_{kwargs[DENOMINATOR]}"])
            .assign(proportion_of_total=proportion_of_total),
            z_alpha=z_alpha,
            z_power=z_power,
            binary=binary,
            non_inferiority=non_inferiority,
            avg_column=ORIGINAL_POINT_ESTIMATE + SFX1,
            var_column=VARIANCE + SFX1,
        )

        if ALTERNATIVE_HYPOTHESIS in df and NULL_HYPOTHESIS in df and (df[ALTERNATIVE_HYPOTHESIS].notna()).all():
            df[REQUIRED_SAMPLE_SIZE] = confidence_computers[df[kwargs[METHOD]].values[0]].required_sample_size(
                proportion_of_total=1,
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                hypothetical_effect=df[ALTERNATIVE_HYPOTHESIS] - df[NULL_HYPOTHESIS],
                control_avg=df[ORIGINAL_POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                kappa=kappa,
            )
            df[REQUIRED_SAMPLE_SIZE_METRIC] = confidence_computers[df[kwargs[METHOD]].values[0]].required_sample_size(
                proportion_of_total=proportion_of_total,
                z_alpha=z_alpha,
                z_power=z_power,
                binary=binary,
                non_inferiority=non_inferiority,
                hypothetical_effect=df[ALTERNATIVE_HYPOTHESIS] - df[NULL_HYPOTHESIS],
                control_avg=df[ORIGINAL_POINT_ESTIMATE + SFX1],
                control_var=df[VARIANCE + SFX1],
                kappa=kappa,
            )
        else:
            df[REQUIRED_SAMPLE_SIZE] = None
            df[REQUIRED_SAMPLE_SIZE_METRIC] = None

        return df
