"""Unit tests for returns calculation module."""

import numpy as np
import pandas as pd
import pytest

from src.returns import (
    calculate_log_returns,
    calculate_simple_returns,
    compute_equal_weighted_index,
    compute_annualized_summary,
)


@pytest.fixture
def sample_prices_series() -> pd.Series:
    """Fixture providing a known price sequence for exact calculation checks."""
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    # Prices: 100 -> 100 (0%), 100 -> 200 (+100%, ln(2)), 200 -> 100 (-50%, -ln(2)), 100 -> 150 (ln(1.5))
    prices = [100.0, 100.0, 200.0, 100.0, 150.0]
    return pd.Series(prices, index=dates, name="Asset_A")


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """Fixture providing a multi-asset price DataFrame."""
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    data = {
        "UK": [100.0, 102.0, 101.0, 105.0],
        "Germany": [200.0, 204.0, 206.0, 210.0],
        "France": [50.0, 51.0, 50.5, 52.0],
    }
    return pd.DataFrame(data, index=dates)


def test_log_returns_synthetic_values(sample_prices_series: pd.Series):
    """Test log returns match analytical formulas on known values."""
    log_ret = calculate_log_returns(sample_prices_series, drop_first=True)

    expected = np.array([
        0.0,                  # ln(100/100)
        np.log(2.0),          # ln(200/100)
        -np.log(2.0),         # ln(100/200)
        np.log(1.5),          # ln(150/100)
    ])

    assert len(log_ret) == 4
    np.testing.assert_allclose(log_ret.values, expected, rtol=1e-7)
    assert log_ret.index[0] == pd.Timestamp("2026-01-02")


def test_log_and_simple_returns_identity(sample_prices_df: pd.DataFrame):
    """Verify fundamental identity: exp(r_log) - 1 == r_simple."""
    log_ret = calculate_log_returns(sample_prices_df, drop_first=True)
    simple_ret = calculate_simple_returns(sample_prices_df, drop_first=True)

    derived_simple = np.exp(log_ret) - 1.0

    np.testing.assert_allclose(derived_simple.values, simple_ret.values, rtol=1e-7)
    assert list(log_ret.columns) == list(sample_prices_df.columns)


def test_drop_first_flag(sample_prices_series: pd.Series):
    """Test behavior when drop_first is False (initial NaN preserved)."""
    ret_with_nan = calculate_log_returns(sample_prices_series, drop_first=False)
    assert len(ret_with_nan) == len(sample_prices_series)
    assert np.isnan(ret_with_nan.iloc[0])
    assert not np.isnan(ret_with_nan.iloc[1])


def test_negative_prices_raise_value_error():
    """Negative prices must raise ValueError for log returns."""
    dates = pd.date_range("2026-01-01", periods=3)
    bad_prices = pd.Series([100.0, -50.0, 80.0], index=dates)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_log_returns(bad_prices)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_simple_returns(bad_prices)


def test_zero_prices_raise_value_error():
    """Zero prices must raise ValueError."""
    dates = pd.date_range("2026-01-01", periods=3)
    zero_prices = pd.Series([100.0, 0.0, 80.0], index=dates)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_log_returns(zero_prices)


def test_invalid_type_raises_type_error():
    """Passing non-pandas objects must raise TypeError."""
    with pytest.raises(TypeError, match="Expected pandas"):
        calculate_log_returns([100.0, 105.0, 110.0])  # type: ignore


def test_empty_dataframe_raises_value_error():
    """Empty inputs must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        calculate_log_returns(pd.DataFrame())


def test_compute_equal_weighted_index(sample_prices_df: pd.DataFrame):
    """Test equal-weighted index calculates exact cross-asset arithmetic mean."""
    returns = calculate_log_returns(sample_prices_df, drop_first=True)
    bench = compute_equal_weighted_index(returns, name="Custom_Benchmark")

    assert bench.name == "Custom_Benchmark"
    assert len(bench) == len(returns)

    expected_day1 = (returns.iloc[0]["UK"] + returns.iloc[0]["Germany"] + returns.iloc[0]["France"]) / 3.0
    assert pytest.approx(bench.iloc[0], rel=1e-7) == expected_day1


def test_compute_annualized_summary(sample_prices_df: pd.DataFrame):
    """Test annualized summary metrics DataFrame formatting and calculations."""
    returns = calculate_log_returns(sample_prices_df, drop_first=True)
    summary = compute_annualized_summary(returns, trading_days_per_year=252)

    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {
        "Annualized Return",
        "Annualized Volatility",
        "Annualized Sharpe (Rf=0)",
        "Daily Skewness",
        "Daily Excess Kurtosis",
        "Observations",
    }
    assert len(summary) == 3
    assert (summary["Observations"] == 3).all()
