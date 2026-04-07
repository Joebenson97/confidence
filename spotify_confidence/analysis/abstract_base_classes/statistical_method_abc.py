from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

from pandas import DataFrame, Series


class StatisticalMethodABC(ABC):
    """Abstract base class that formalizes the implicit interface currently
    used by all individual confidence-computer modules (z_test_computer,
    t_test_computer, chi_squared_computer, bootstrap_computer,
    z_test_linreg_computer).

    Each concrete subclass wraps the statistical primitives for one test
    type.  Instances are stored in the ``confidence_computers`` registry
    and looked up by method name (e.g. ``"z-test"``, ``"t-test"``).
    """

    # ------------------------------------------------------------------
    # Capability flags — override in subclasses as needed
    # ------------------------------------------------------------------

    @property
    def supports_sequential(self) -> bool:
        """Whether this method supports group-sequential testing."""
        return False

    @property
    def supports_powered_effect(self) -> bool:
        """Whether this method supports powered-effect / sample-size calculations."""
        return False

    @property
    def supports_mde(self) -> bool:
        """Whether this method supports minimum-detectable-effect columns."""
        return False

    # ------------------------------------------------------------------
    # Core abstract methods — every subclass MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def point_estimate(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        """Compute the point estimate from sufficient statistics."""

    @abstractmethod
    def variance(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        """Compute the variance from sufficient statistics."""

    @abstractmethod
    def std_err(self, df: DataFrame, **kwargs: Any) -> Union[float, Series, None]:
        """Compute the standard error of the difference between two groups."""

    @abstractmethod
    def p_value(self, df: DataFrame, **kwargs: Any) -> Union[float, Series]:
        """Compute the p-value for the test."""

    @abstractmethod
    def ci(self, df: DataFrame, alpha_column: str, **kwargs: Any) -> Tuple[Series, Series]:
        """Compute confidence / credible interval bounds.

        Returns:
            A tuple ``(lower, upper)`` of Series.
        """

    @abstractmethod
    def achieved_power(
        self, df: DataFrame, mde: float, alpha: float, **kwargs: Any
    ) -> Optional[Union[int, float, DataFrame]]:
        """Compute the achieved statistical power."""

    @abstractmethod
    def add_point_estimate_ci(self, df: DataFrame, **kwargs: Any) -> DataFrame:
        """Add ``ci_lower`` and ``ci_upper`` columns for the point estimate."""

    # ------------------------------------------------------------------
    # Optional capability methods — default to raising NotImplementedError
    # ------------------------------------------------------------------

    def compute_sequential_adjusted_alpha(self, df: DataFrame, **kwargs: Any) -> Series:
        """Compute sequentially adjusted alpha values.

        Only z-test based methods support this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sequential testing")

    def powered_effect(self, df: DataFrame, **kwargs: Any) -> Series:
        """Compute the minimum detectable effect for current power/sample-size.

        Only z-test and z-test-linreg support this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support powered_effect")

    def required_sample_size(self, **kwargs: Any) -> Union[Series, float]:
        """Compute the required sample size.

        Only z-test and z-test-linreg support this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support required_sample_size")

    def ci_for_multiple_comparison_methods(
        self, df: DataFrame, correction_method: str, alpha: float, w: float = 1.0
    ) -> Tuple[Union[Series, float], Union[Series, float]]:
        """Compute CIs adjusted for non-Bonferroni multiple-comparison methods.

        Only z-test supports this.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support ci_for_multiple_comparison_methods")
