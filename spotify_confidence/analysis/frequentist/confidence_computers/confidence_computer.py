# Copyright 2017-2020 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, List, Optional, Tuple, Union

from pandas import DataFrame

from spotify_confidence.analysis.abstract_base_classes.confidence_computer_abc import ConfidenceComputerABC
from spotify_confidence.analysis.confidence_utils import (
    drop_and_rename_columns,
    get_all_categorical_group_columns,
    get_all_group_columns,
    get_remaining_groups,
    groupbyApplyParallel,
    level2str,
    listify,
    remove_group_columns,
    validate_and_rename_columns,
    validate_data,
    validate_levels,
)
from spotify_confidence.analysis.constants import (
    ABSOLUTE,
    ADJUSTED_LOWER,
    ADJUSTED_P,
    ADJUSTED_UPPER,
    ALTERNATIVE_HYPOTHESIS,
    BOOTSTRAP,
    BOOTSTRAPS,
    CHI2,
    CI_LOWER,
    CI_UPPER,
    CORRECTION_METHOD,
    CORRECTION_METHODS,
    DENOMINATOR,
    DIFFERENCE,
    FINAL_EXPECTED_SAMPLE_SIZE,
    INTERVAL_SIZE,
    IS_SIGNIFICANT,
    MDE,
    METHOD,
    NIM,
    NIM_COLUMN_DEFAULT,
    NIM_TYPE,
    NULL_HYPOTHESIS,
    NUMBER_OF_COMPARISONS,
    NUMERATOR,
    NUMERATOR_SUM_OF_SQUARES,
    ORDINAL_GROUP_COLUMN,
    P_VALUE,
    POINT_ESTIMATE,
    POWER,
    POWERED_EFFECT,
    PREFERENCE,
    PREFERENCE_TEST,
    PREFERRED_DIRECTION_COLUMN_DEFAULT,
    REQUIRED_SAMPLE_SIZE,
    SFX1,
    SFX2,
    TTEST,
    ZTEST,
    ZTESTLINREG,
)
from spotify_confidence.analysis.frequentist.confidence_computers import confidence_computers
from spotify_confidence.analysis.frequentist.confidence_computers.comparison_pipeline import (
    add_ci_and_adjust_if_absolute,
    compute_comparisons,
)
from spotify_confidence.analysis.frequentist.confidence_computers.sufficient_statistics import (
    SufficientStatisticsBuilder,
)
from spotify_confidence.analysis.frequentist.multiple_comparison import (
    add_adjusted_p_and_is_significant,
    add_adjusted_power,
    get_num_comparisons,
    get_preference,
)
from spotify_confidence.analysis.frequentist.nims_and_mdes import (
    add_nim_input_columns_from_tuple_or_dict,
    add_nims_and_mdes,
)


class ConfidenceComputer(ConfidenceComputerABC):
    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: Union[str, None],
        numerator_sum_squares_column: Union[str, None],
        denominator_column: Union[str, None],
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: Union[str, None],
        interval_size: float,
        correction_method: str,
        method_column: str,
        bootstrap_samples_column: Union[str, None],
        metric_column: Union[str, None],
        treatment_column: Union[str, None],
        power: float,
        feature_column: Union[str, None],
        feature_sum_squares_column: Union[str, None],
        feature_cross_sum_column: Union[str, None],
    ):
        self._df = data_frame.reset_index(drop=True)
        self._numerator = numerator_column
        self._numerator_sumsq = numerator_sum_squares_column
        if self._numerator is not None and (self._numerator_sumsq is None or self._numerator_sumsq == self._numerator):
            if (data_frame[numerator_column] <= data_frame[denominator_column]).all():
                # Treat as binomial data
                self._numerator_sumsq = self._numerator
            else:
                raise ValueError(
                    f"numerator_sum_squares_column missing or same as "
                    f"numerator_column, but since {numerator_column} is not "
                    f"always smaller than {denominator_column} it can't be "
                    f"binomial data. Please check your data."
                )

        self._denominator = denominator_column
        self._categorical_group_columns = get_all_categorical_group_columns(
            categorical_group_columns, metric_column, treatment_column
        )
        self._segments = remove_group_columns(self._categorical_group_columns, metric_column)
        self._segments = remove_group_columns(self._segments, treatment_column)
        self._ordinal_group_column = ordinal_group_column
        self._metric_column = metric_column
        self._interval_size = interval_size
        self._power = power
        self._treatment_column = treatment_column
        self._feature = feature_column
        self._feature_ssq = feature_sum_squares_column
        self._feature_cross = feature_cross_sum_column

        if correction_method.lower() not in CORRECTION_METHODS:
            raise ValueError(f"Use one of the correction methods in {CORRECTION_METHODS}")
        self._correction_method = correction_method
        self._method_column = method_column

        self._single_metric = False
        if self._metric_column is not None and data_frame.groupby(self._metric_column, sort=False).ngroups == 1:
            self._single_metric = True

        self._all_group_columns = get_all_group_columns(self._categorical_group_columns, self._ordinal_group_column)
        self._bootstrap_samples_column = bootstrap_samples_column

        columns_that_must_exist = []
        if (
            CHI2 in self._df[self._method_column]
            or TTEST in self._df[self._method_column]
            or ZTEST in self._df[self._method_column]
        ):
            columns_that_must_exist += [self._numerator, self._denominator]
            columns_that_must_exist += [] if self._numerator_sumsq is None else [self._numerator_sumsq]
        if BOOTSTRAP in self._df[self._method_column]:
            columns_that_must_exist += [self._bootstrap_samples_column]
        if ZTESTLINREG in self._df[self._method_column]:
            columns_that_must_exist += [self._feature, self._feature_ssq, self._feature_cross]

        validate_data(self._df, columns_that_must_exist, self._all_group_columns, self._ordinal_group_column)

        self._sufficient = None
        self._stats_builder = SufficientStatisticsBuilder(
            numerator=self._numerator,
            numerator_sumsq=self._numerator_sumsq,
            denominator=self._denominator,
            bootstrap_samples_column=self._bootstrap_samples_column,
            interval_size=self._interval_size,
            feature=self._feature,
            feature_ssq=self._feature_ssq,
            feature_cross=self._feature_cross,
            method_column=self._method_column,
            metric_column=self._metric_column,
        )

    def compute_summary(self, verbose: bool) -> DataFrame:
        return (
            self._sufficient_statistics
            if verbose
            else self._sufficient_statistics[
                self._all_group_columns
                + ([self._metric_column] if self._metric_column is not None and self._single_metric else [])
                + [c for c in [self._numerator, self._denominator] if c is not None]
                + [POINT_ESTIMATE, CI_LOWER, CI_UPPER]
            ]
        )

    @property
    def _sufficient_statistics(self) -> DataFrame:
        if self._sufficient is None:
            self._sufficient = self._stats_builder.build(self._df)
        return self._sufficient

    def compute_difference(
        self,
        level_1: Union[str, Iterable],
        level_2: Union[str, Iterable],
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        nims: Optional[NIM_TYPE],
        final_expected_sample_size_column: Optional[str],
        verbose: bool,
        mde_column: Optional[str],
    ) -> DataFrame:
        level_columns = get_remaining_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(
            level_columns=level_columns,
            levels=[(level_1, level_2)],
            absolute=absolute,
            groupby=groupby,
            level_as_reference=True,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column,
            mde_column=mde_column,
        )
        return (
            difference_df
            if verbose
            else difference_df[
                listify(groupby)
                + ["level_1", "level_2", "absolute_difference", DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE]
                + [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT, POWERED_EFFECT, REQUIRED_SAMPLE_SIZE]
                + ([NIM, NULL_HYPOTHESIS, PREFERENCE] if nims is not None else [])
            ]
        )

    def compute_multiple_difference(
        self,
        level: Union[str, Iterable, int],
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        level_as_reference: Optional[bool],
        nims: Optional[NIM_TYPE],
        final_expected_sample_size_column: Optional[str],
        verbose: bool,
        mde_column: Optional[str],
    ) -> DataFrame:
        level_columns = get_remaining_groups(self._all_group_columns, groupby)
        other_levels = [
            other
            for other in self._sufficient_statistics.groupby(level_columns, sort=False).groups.keys()
            if other != level
        ]
        levels = [(level, other) for other in other_levels]
        difference_df = self._compute_differences(
            level_columns=level_columns,
            levels=levels,
            absolute=absolute,
            groupby=groupby,
            level_as_reference=level_as_reference,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column,
            mde_column=mde_column,
        )
        return (
            difference_df
            if verbose
            else difference_df[
                listify(groupby)
                + [
                    "level_1",
                    "level_2",
                    "absolute_difference",
                    DIFFERENCE,
                    CI_LOWER,
                    CI_UPPER,
                    P_VALUE,
                    POWERED_EFFECT,
                    REQUIRED_SAMPLE_SIZE,
                ]
                + [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT]
                + ([NIM, NULL_HYPOTHESIS, PREFERENCE] if nims is not None else [])
            ]
        )

    def compute_differences(
        self,
        levels: Union[Tuple, List[Tuple]],
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        nims: Optional[NIM_TYPE],
        final_expected_sample_size_column: Optional[str],
        verbose: bool,
        mde_column: Optional[str],
    ) -> DataFrame:
        level_columns = get_remaining_groups(self._all_group_columns, groupby)
        difference_df = self._compute_differences(
            level_columns=level_columns,
            levels=[levels] if isinstance(levels, tuple) else levels,
            absolute=absolute,
            groupby=groupby,
            level_as_reference=True,
            nims=nims,
            final_expected_sample_size_column=final_expected_sample_size_column,
            mde_column=mde_column,
        )
        return (
            difference_df
            if verbose
            else difference_df[
                listify(groupby)
                + ["level_1", "level_2", "absolute_difference", DIFFERENCE, CI_LOWER, CI_UPPER, P_VALUE]
                + [ADJUSTED_LOWER, ADJUSTED_UPPER, ADJUSTED_P, IS_SIGNIFICANT, POWERED_EFFECT, REQUIRED_SAMPLE_SIZE]
                + ([NIM, NULL_HYPOTHESIS, PREFERENCE] if nims is not None else [])
            ]
        )

    def _compute_differences(
        self,
        level_columns: Iterable,
        levels: Union[str, Iterable],
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        level_as_reference: Optional[bool],
        nims: Optional[NIM_TYPE],
        final_expected_sample_size_column: Optional[str],
        mde_column: Optional[str],
    ):
        if type(level_as_reference) is not bool:
            raise ValueError(f"level_as_reference must be either True or False, but is {level_as_reference}.")
        groupby = listify(groupby)
        unique_levels = set([level[0] for level in levels] + [level[1] for level in levels])
        validate_levels(self._sufficient_statistics, level_columns, unique_levels)
        str2level = {level2str(lv): lv for lv in unique_levels}
        levels = [
            (level2str(lvl[0]), level2str(lvl[1])) if level_as_reference else (level2str(lvl[1]), level2str(lvl[0]))
            for lvl in levels
        ]

        def assign_total_denominator(df, groupby):
            if self._denominator is None:
                return df.assign(**{f"current_total_{self._denominator}": None})

            if len(groupby) == 0:
                return df.assign(
                    **{f"current_total_{self._denominator}": self._sufficient_statistics[self._denominator].sum()}
                )
            else:
                return df.merge(
                    df.groupby(groupby, sort=False)[self._denominator]
                    .sum()
                    .reset_index()
                    .rename(columns={self._denominator: f"current_total_{self._denominator}"})
                )

        return (
            self._sufficient_statistics.assign(
                level=self._sufficient_statistics[level_columns].agg(level2str, axis="columns")
            )
            .pipe(assign_total_denominator, groupby)
            .query(f"level in {[l1 for l1, l2 in levels] + [l2 for l1, l2 in levels]}")
            .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
            .pipe(
                self._create_comparison_df,
                groups_to_compare=levels,
                absolute=absolute,
                nims=nims,
                mde_column=mde_column,
                final_expected_sample_size_column=final_expected_sample_size_column,
            )
            .assign(level_1=lambda df: df["level_1"].map(lambda s: str2level[s]))
            .assign(level_2=lambda df: df["level_2"].map(lambda s: str2level[s]))
            .pipe(lambda df: df.reset_index([name for name in df.index.names if name is not None]))
            .reset_index(drop=True)
            .sort_values(by=groupby + ["level_1", "level_2"])
            .reset_index(drop=True)
        )

    def _create_comparison_df(
        self,
        df: DataFrame,
        groups_to_compare: List[Tuple[str, str]],
        absolute: bool,
        nims: NIM_TYPE,
        mde_column: bool,
        final_expected_sample_size_column: str,
    ) -> DataFrame:
        def join(df: DataFrame) -> DataFrame:
            has_index = not all(idx is None for idx in df.index.names)
            if has_index:
                # self-join on index (the index will typically model the date,
                # i.e., rows with the same date are joined)
                return df.merge(df, left_index=True, right_index=True, suffixes=(SFX1, SFX2))
            else:
                # join on dummy column, i.e. conduct a cross join
                return (
                    df.assign(dummy_join_column=1)
                    .merge(right=df.assign(dummy_join_column=1), on="dummy_join_column", suffixes=(SFX1, SFX2))
                    .drop(columns="dummy_join_column")
                )

        comparison_df = (
            df.pipe(add_nim_input_columns_from_tuple_or_dict, nims=nims, mde_column=mde_column)
            .pipe(
                add_nims_and_mdes,
                mde_column=mde_column,
                nim_column=NIM_COLUMN_DEFAULT,
                preferred_direction_column=PREFERRED_DIRECTION_COLUMN_DEFAULT,
            )
            .pipe(join)
            .query(
                "("
                + " or ".join([f"(level_1=='{l1}' and level_2=='{l2}')" for l1, l2 in groups_to_compare])
                + ")"
                + "and level_1 != level_2"
            )
            .pipe(
                validate_and_rename_columns,
                [NIM, mde_column, PREFERENCE, final_expected_sample_size_column, self._method_column],
            )
            .pipe(
                drop_and_rename_columns,
                [NULL_HYPOTHESIS, ALTERNATIVE_HYPOTHESIS, f"current_total_{self._denominator}"],
            )
            .assign(**{PREFERENCE_TEST: lambda df: get_preference(df, self._correction_method)})
            .assign(**{POWER: self._power})
            .pipe(
                add_adjusted_power,
                correction_method=self._correction_method,
                metric_column=self._metric_column,
                single_metric=self._single_metric,
            )
        )

        groups_except_ordinal = [
            column
            for column in df.index.names
            if column is not None
            and (column != self._ordinal_group_column or final_expected_sample_size_column is None)
        ]
        n_comparisons = get_num_comparisons(
            comparison_df,
            self._correction_method,
            number_of_level_comparisons=comparison_df.groupby(["level_1", "level_2"], sort=False).ngroups,
            groupby=groups_except_ordinal,
            metric_column=self._metric_column,
            treatment_column=self._treatment_column,
            single_metric=self._single_metric,
            segments=self._segments,
        )

        kwargs = {
            NUMERATOR: self._numerator,
            NUMERATOR_SUM_OF_SQUARES: self._numerator_sumsq,
            DENOMINATOR: self._denominator,
            BOOTSTRAPS: self._bootstrap_samples_column,
            FINAL_EXPECTED_SAMPLE_SIZE: final_expected_sample_size_column,
            ORDINAL_GROUP_COLUMN: self._ordinal_group_column,
            MDE: mde_column,
            METHOD: self._method_column,
            CORRECTION_METHOD: self._correction_method,
            INTERVAL_SIZE: self._interval_size,
            ABSOLUTE: absolute,
            NUMBER_OF_COMPARISONS: n_comparisons,
        }
        comparison_df = groupbyApplyParallel(
            comparison_df.groupby(
                groups_except_ordinal + [self._method_column, "level_1", "level_2"], as_index=False, sort=False
            ),
            lambda df: compute_comparisons(df, **kwargs),
        )
        comparison_df = comparison_df.pipe(add_adjusted_p_and_is_significant, **kwargs)
        comparison_df = groupbyApplyParallel(
            comparison_df.groupby(
                groups_except_ordinal + [self._method_column, "level_1", "level_2"], as_index=False, sort=False
            ),
            lambda df: add_ci_and_adjust_if_absolute(df, **kwargs),
        )

        return comparison_df

    def achieved_power(
        self,
        level_1: Union[str, Iterable],
        level_2: Union[str, Iterable],
        mde: float,
        alpha: float,
        groupby: Optional[Union[str, Iterable]],
    ) -> DataFrame:
        groupby = listify(groupby)
        level_columns = get_remaining_groups(self._all_group_columns, groupby)
        kwargs = {NUMERATOR: self._numerator, DENOMINATOR: self._denominator}
        return (
            self._compute_differences(
                level_columns,
                [(level_1, level_2)],
                True,
                groupby,
                level_as_reference=True,
                nims=None,
                final_expected_sample_size_column=None,
                mde_column=None,
            )  # level_as_reference=True is correct: power is computed relative to the control level
            .pipe(lambda df: df if groupby == [] else df.set_index(groupby))
            .assign(
                achieved_power=lambda df: df.apply(
                    lambda row: confidence_computers[row[self._method_column]].achieved_power(
                        row, mde=mde, alpha=alpha, **kwargs
                    ),
                    axis=1,
                )
            )
        )[["level_1", "level_2", "achieved_power"]]


# ---------------------------------------------------------------------------
# Backward-compatible aliases — logic lives in comparison_pipeline.py
# ---------------------------------------------------------------------------
_compute_comparisons = compute_comparisons
_add_ci_and_adjust_if_absolute = add_ci_and_adjust_if_absolute
