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

from typing import Iterable, Optional, Union

from pandas import DataFrame

from spotify_confidence.chartgrid import ChartGrid

from ..abstract_base_classes.confidence_grapher_abc import ConfidenceGrapherABC
from ..constants import NIM_TYPE

_INSTALL_MSG = "Install spotify-confidence[viz] for plotting support: pip install spotify-confidence[viz]"


class NullGrapher(ConfidenceGrapherABC):
    """A no-op grapher that raises ImportError when any plot method is called.

    Used as a fallback when chartify is not installed, so that the core
    computation functionality remains available without the visualization
    dependency.
    """

    def __init__(
        self,
        data_frame: DataFrame,
        numerator_column: str,
        denominator_column: str,
        categorical_group_columns: Union[str, Iterable],
        ordinal_group_column: Optional[str],
    ):
        pass

    def plot_summary(self, summary_df: DataFrame, groupby: Optional[Union[str, Iterable]]) -> ChartGrid:
        raise ImportError(_INSTALL_MSG)

    def plot_difference(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        nims: Optional[NIM_TYPE],
        use_adjusted_intervals: bool,
        split_plot_by_groups: bool,
    ) -> ChartGrid:
        raise ImportError(_INSTALL_MSG)

    def plot_differences(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        nims: Optional[NIM_TYPE],
        use_adjusted_intervals: bool,
        split_plot_by_groups: bool,
    ) -> ChartGrid:
        raise ImportError(_INSTALL_MSG)

    def plot_multiple_difference(
        self,
        difference_df: DataFrame,
        absolute: bool,
        groupby: Optional[Union[str, Iterable]],
        level_as_reference: Optional[bool],
        nims: Optional[NIM_TYPE],
        use_adjusted_intervals: bool,
        split_plot_by_groups: bool,
    ) -> ChartGrid:
        raise ImportError(_INSTALL_MSG)
