"""Financial Returns Calculation Module.

Calculates logarithmic and simple returns, validates mathematical consistency,
detects price anomalies, and computes benchmark return aggregates.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_log_returns(
    prices: Union[pd.Series, pd.DataFrame],
    drop_first: bool = True,
    extreme_return_threshold: float = 0.50,
) -> Union[pd.Series, pd.DataFrame]:
    """Calculate daily continuously compounded (logarithmic) returns.

    Formula:
        r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Args:
        prices: Series or DataFrame of positive price observations.
        drop_first: If True, drop the first observation (which is always NaN after diff).
        extreme_return_threshold: Warn if any absolute single-day return exceeds this threshold
            (e.g., 0.50 = 50% single-day change, indicating possible data error).

    Returns:
        Series or DataFrame of logarithmic returns.

    Raises:
        TypeError: If prices is not a pandas Series or DataFrame.
        ValueError: If prices contain zero, negative, or non-numeric values.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pandas Series or DataFrame, got {type(prices).__name__}")

    if prices.empty:
        raise ValueError("Input price series/dataframe is empty.")

    # Check for strictly positive prices
    min_price = prices.min().min() if isinstance(prices, pd.DataFrame) else prices.min()
    if min_price <= 0:
        raise ValueError(
            f"Prices must be strictly positive (> 0) to compute logarithmic returns. "
            f"Encountered minimum price: {min_price}"
        )

    # Compute log returns using vectorized numpy
    log_prices = np.log(prices)
    log_returns = log_prices.diff()

    if drop_first:
        log_returns = log_returns.iloc[1:]

    # Check for extreme anomalies (> 50% single-day moves)
    max_abs_ret = log_returns.abs().max().max() if isinstance(log_returns, pd.DataFrame) else log_returns.abs().max()
    if max_abs_ret > extreme_return_threshold:
        logger.warning(
            "Extreme single-day return detected (max absolute log return: %.2f%%). "
            "Inspect data for potential unadjusted splits or abnormal jumps.",
            max_abs_ret * 100,
        )

    return log_returns


def calculate_simple_returns(
    prices: Union[pd.Series, pd.DataFrame],
    drop_first: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """Calculate daily simple percentage returns.

    Formula:
        R_t = (P_t - P_{t-1}) / P_{t-1} = (P_t / P_{t-1}) - 1

    Args:
        prices: Series or DataFrame of price observations.
        drop_first: If True, drop the first observation.

    Returns:
        Series or DataFrame of simple returns.

    Raises:
        TypeError: If input is not a pandas Series or DataFrame.
        ValueError: If prices are empty or non-positive.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError(f"Expected pandas Series or DataFrame, got {type(prices).__name__}")

    if prices.empty:
        raise ValueError("Input price series/dataframe is empty.")

    min_price = prices.min().min() if isinstance(prices, pd.DataFrame) else prices.min()
    if min_price <= 0:
        raise ValueError(f"Prices must be strictly positive. Encountered minimum price: {min_price}")

    simple_returns = prices.pct_change()

    if drop_first:
        simple_returns = simple_returns.iloc[1:]

    return simple_returns


def compute_equal_weighted_index(
    returns_df: pd.DataFrame,
    name: str = "EqualWeighted_Europe",
) -> pd.Series:
    """Compute an equal-weighted synthetic market benchmark return across all available markets.

    Formula:
        r_{bench, t} = (1 / N_t) * sum_{i=1}^{N_t} r_{i, t}

    Args:
        returns_df: DataFrame of daily market returns (columns = markets).
        name: Name for the resulting benchmark series.

    Returns:
        pd.Series of equal-weighted daily benchmark returns.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame of returns, received {type(returns_df).__name__}")

    if returns_df.empty:
        raise ValueError("Returns DataFrame is empty.")

    benchmark_series = returns_df.mean(axis=1)
    benchmark_series.name = name
    return benchmark_series


def compute_annualized_summary(
    returns: Union[pd.Series, pd.DataFrame],
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Compute basic annualized summary statistics (mean, volatility, Sharpe ratio, skewness, kurtosis).

    Args:
        returns: Daily log returns Series or DataFrame.
        trading_days_per_year: Annualization factor (typically 252 for daily equity markets).

    Returns:
        pd.DataFrame of summary metrics.
    """
    if isinstance(returns, pd.Series):
        df = returns.to_frame()
    else:
        df = returns.copy()

    ann_return = df.mean() * trading_days_per_year
    ann_vol = df.std() * np.sqrt(trading_days_per_year)
    sharpe = np.where(ann_vol > 0, ann_return / ann_vol, np.nan)
    skew = df.skew()
    kurt = df.kurtosis()  # Excess kurtosis (normal distribution = 0)

    summary_df = pd.DataFrame(
        {
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Annualized Sharpe (Rf=0)": sharpe,
            "Daily Skewness": skew,
            "Daily Excess Kurtosis": kurt,
            "Observations": df.count(),
        }
    )
    return summary_df
