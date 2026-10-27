"""Unit tests for preprocessing and calendar alignment module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    validate_price_series,
    inspect_data_quality,
    align_market_calendars,
    generate_data_quality_report,
)


@pytest.fixture
def clean_prices_df() -> pd.DataFrame:
    """Fixture providing a clean, strictly positive price DataFrame."""
    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "^FTSE": np.linspace(7000, 7200, 10),
            "^GDAXI": np.linspace(15000, 15500, 10),
            "^FCHI": np.linspace(6500, 6700, 10),
        },
        index=dates,
    )


@pytest.fixture
def multi_calendar_prices_df() -> pd.DataFrame:
    """Fixture simulating mismatched national market holiday schedules.

    Day 1: All open
    Day 2: UK closed (UK bank holiday) -> NaN
    Day 3: Germany closed (German Unity Day) -> NaN
    Day 4: All open
    Day 5: All closed -> all NaN
    """
    dates = pd.date_range("2026-05-01", periods=5, freq="D")
    data = {
        "UK": [100.0, np.nan, 102.0, 103.0, np.nan],
        "Germany": [200.0, 201.0, np.nan, 205.0, np.nan],
        "France": [50.0, 50.5, 51.0, 51.5, np.nan],
    }
    return pd.DataFrame(data, index=dates)


def test_validate_price_series_valid(clean_prices_df: pd.DataFrame):
    """Clean data should pass validation without raising an exception."""
    validate_price_series(clean_prices_df)


def test_validate_price_series_non_monotonic():
    """Non-monotonic timestamps must raise ValueError."""
    dates = pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"])
    df = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_price_series(df)


def test_validate_price_series_duplicate_dates():
    """Duplicate dates must raise ValueError."""
    dates = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"])
    df = pd.DataFrame({"A": [10.0, 11.0, 12.0]}, index=dates)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_price_series(df)


def test_validate_price_series_negative_prices():
    """Negative prices must raise ValueError."""
    dates = pd.date_range("2026-01-01", periods=3)
    df = pd.DataFrame({"A": [10.0, -1.0, 12.0]}, index=dates)
    with pytest.raises(ValueError, match="non-positive"):
        validate_price_series(df)


def test_validate_price_series_all_nan_column():
    """Column with all NaNs must raise ValueError."""
    dates = pd.date_range("2026-01-01", periods=3)
    df = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [np.nan, np.nan, np.nan]}, index=dates)
    with pytest.raises(ValueError, match="contains only NaN"):
        validate_price_series(df)


def test_inspect_data_quality(multi_calendar_prices_df: pd.DataFrame):
    """Test data quality audit dictionary matches actual missing counts."""
    info = inspect_data_quality(multi_calendar_prices_df)

    assert info["total_calendar_rows"] == 5
    assert info["total_columns"] == 3
    assert info["has_duplicates"] is False

    uk_info = info["instruments"]["UK"]
    assert uk_info["valid_observations"] == 3
    assert uk_info["missing_observations"] == 2
    assert uk_info["missing_percentage"] == 40.0


def test_align_market_calendars_intersection(multi_calendar_prices_df: pd.DataFrame):
    """'intersection' method must keep only days where ALL markets are open."""
    # Filter out Day 5 (all NaN) first to avoid tolerance drop
    df_no_all_nan = multi_calendar_prices_df.dropna(how="all")
    aligned_df, report = align_market_calendars(
        df_no_all_nan,
        method="intersection",
        max_missing_tolerance=0.50,
    )

    # In df_no_all_nan (Days 1-4), only Day 1 and Day 4 have all 3 markets open
    assert len(aligned_df) == 2
    assert aligned_df.isna().sum().sum() == 0
    assert report["alignment_method"] == "intersection"
    assert report["synthetic_zero_returns_induced"] == 0
    assert report["final_rows"] == 2


def test_align_market_calendars_drops_high_nan_column():
    """Columns with missing proportion > max_missing_tolerance should be dropped."""
    dates = pd.date_range("2026-01-01", periods=10)
    df = pd.DataFrame({
        "Good_Market": np.ones(10) * 100.0,
        "Bad_Market": [100.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 100.0],  # 80% NaN
    }, index=dates)

    aligned_df, report = align_market_calendars(
        df,
        method="intersection",
        max_missing_tolerance=0.25,
    )

    assert "Bad_Market" in report["dropped_columns"]
    assert "Good_Market" in report["retained_columns"]
    assert "Bad_Market" not in aligned_df.columns
    assert len(aligned_df) == 10


def test_align_market_calendars_invalid_method(clean_prices_df: pd.DataFrame):
    """Unsupported alignment methods must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown alignment method"):
        align_market_calendars(clean_prices_df, method="invalid_method_name")


def test_generate_data_quality_report(clean_prices_df: pd.DataFrame):
    """Diagnostic report generation produces valid text summary."""
    aligned, rep = align_market_calendars(clean_prices_df, method="intersection")
    report_text = generate_data_quality_report(clean_prices_df, aligned, rep)

    assert "EUROPEAN FINANCIAL MARKETS: DATA QUALITY" in report_text
    assert "^FTSE" in report_text
    assert "intersection" in report_text
