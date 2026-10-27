"""Data Preprocessing and Calendar Alignment Module.

Performs rigorous validation of price time series, calendar synchronization across
European markets, missing data audits, and non-distorting observation filtering.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_price_series(df: pd.DataFrame) -> None:
    """Validate that the input price DataFrame meets quantitative data standards.

    Checks:
        - Input is a pandas DataFrame.
        - Index is a DatetimeIndex and strictly monotonically increasing.
        - No duplicate timestamps exist in the index.
        - All values are numeric (float/int).
        - Prices are strictly positive (> 0); zero or negative prices raise an error.
        - At least one valid observation exists per column.

    Raises:
        TypeError: If input is not a DataFrame or has non-datetime index.
        ValueError: If duplicate dates, negative prices, or empty columns are detected.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(df).__name__}")

    if df.empty:
        raise ValueError("Price DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"DataFrame index must be DatetimeIndex, received {type(df.index).__name__}")

    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be strictly monotonically increasing.")

    if df.index.has_duplicates:
        duplicate_dates = df.index[df.index.duplicated()].unique()
        raise ValueError(f"Duplicate timestamps detected in price index: {list(duplicate_dates[:5])}")

    for col in df.columns:
        valid_prices = df[col].dropna()
        if len(valid_prices) == 0:
            raise ValueError(f"Column '{col}' contains only NaN values.")

        if not np.issubdtype(valid_prices.dtype, np.number):
            raise TypeError(f"Column '{col}' must contain numeric data, found {valid_prices.dtype}")

        if (valid_prices <= 0).any():
            invalid_count = (valid_prices <= 0).sum()
            min_val = valid_prices.min()
            raise ValueError(
                f"Column '{col}' contains {invalid_count} non-positive price values (minimum: {min_val}). "
                f"Financial prices must be strictly positive for logarithmic returns."
            )


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect missing data, gaps, duplicate dates, and date boundaries.

    Args:
        df: Raw or processed prices DataFrame.

    Returns:
        Structured audit dictionary detailing observations, missing counts/percentages,
        and date ranges per instrument.
    """
    total_dates = len(df)
    missing_counts = df.isna().sum()
    missing_pcts = (missing_counts / total_dates * 100) if total_dates > 0 else 0.0

    per_instrument_stats = {}
    for col in df.columns:
        s = df[col].dropna()
        per_instrument_stats[col] = {
            "valid_observations": int(len(s)),
            "missing_observations": int(missing_counts[col]),
            "missing_percentage": float(round(missing_pcts[col], 2)),
            "first_valid_date": str(s.index[0].date()) if len(s) > 0 else None,
            "last_valid_date": str(s.index[-1].date()) if len(s) > 0 else None,
        }

    return {
        "total_calendar_rows": int(total_dates),
        "total_columns": int(len(df.columns)),
        "start_date": str(df.index[0].date()) if total_dates > 0 else None,
        "end_date": str(df.index[-1].date()) if total_dates > 0 else None,
        "instruments": per_instrument_stats,
        "has_duplicates": bool(df.index.has_duplicates),
    }


def align_market_calendars(
    df: pd.DataFrame,
    method: str = "intersection",
    max_missing_tolerance: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Synchronize multiple market time series across distinct European holiday calendars.

    Why this matters:
    European national stock exchanges observe divergent bank holidays (e.g., UK Summer Bank
    Holiday, German Day of Unity, Swiss National Day).
    Forward-filling prices on closed days produces artificial zero returns, which:
      1. Artificially depresses realized volatility.
      2. Artificially lowers pairwise cross-market correlation.
      3. Generates artificial catch-up price jumps upon reopening.

    Alignment Methods:
        - 'intersection' (RECOMMENDED): Keep only trading days where ALL selected instruments
          have active, valid market closing prices. Completely eliminates artificial zero returns.
        - 'union_drop_all_nan': Drop dates where all markets are closed; keeps partial trading
          dates as NaN (used for pairwise rolling calculations).
        - 'ffill_strict': Forward-fill missing prices up to limit=1, with an audit count of
          how many artificial zero returns are introduced.

    Args:
        df: Input prices DataFrame.
        method: Alignment strategy ('intersection', 'union_drop_all_nan', 'ffill_strict').
        max_missing_tolerance: Maximum allowable fraction of missing data for a column before dropping.

    Returns:
        Tuple of:
            - Aligned prices DataFrame
            - Alignment report dictionary
    """
    validate_price_series(df)
    initial_shape = df.shape
    initial_dates = len(df)

    # 1. Filter out columns that exceed missing tolerance
    missing_fractions = df.isna().mean()
    retained_cols = missing_fractions[missing_fractions <= max_missing_tolerance].index.tolist()
    dropped_cols = [c for c in df.columns if c not in retained_cols]

    if not retained_cols:
        raise ValueError(
            f"All columns exceeded the maximum missing tolerance of {max_missing_tolerance:.1%}. "
            f"Missing fractions: {missing_fractions.to_dict()}"
        )

    working_df = df[retained_cols].copy()

    if method == "intersection":
        # Keep only days where every retained market actively traded
        aligned_df = working_df.dropna(how="any").copy()
        zeros_induced = 0
    elif method == "union_drop_all_nan":
        aligned_df = working_df.dropna(how="all").copy()
        zeros_induced = 0
    elif method == "ffill_strict":
        # Forward fill up to 1 day only (e.g. 1-day holiday)
        ffilled = working_df.ffill(limit=1)
        zeros_induced = int((ffilled.isna().diff() == 0).sum().sum())
        aligned_df = ffilled.dropna(how="any").copy()
    else:
        raise ValueError(f"Unknown alignment method: '{method}'. Choose 'intersection', 'union_drop_all_nan', or 'ffill_strict'.")

    aligned_df.sort_index(inplace=True)

    # Re-validate aligned prices
    if not aligned_df.empty:
        validate_price_series(aligned_df)

    report = {
        "alignment_method": method,
        "initial_rows": int(initial_dates),
        "initial_columns": int(initial_shape[1]),
        "retained_columns": retained_cols,
        "dropped_columns": dropped_cols,
        "final_rows": int(len(aligned_df)),
        "rows_removed": int(initial_dates - len(aligned_df)),
        "retention_rate_pct": float(round((len(aligned_df) / initial_dates * 100) if initial_dates > 0 else 0, 2)),
        "synthetic_zero_returns_induced": zeros_induced,
        "aligned_start_date": str(aligned_df.index[0].date()) if len(aligned_df) > 0 else None,
        "aligned_end_date": str(aligned_df.index[-1].date()) if len(aligned_df) > 0 else None,
    }

    logger.info(
        "Calendar alignment (%s): %d -> %d rows (%.1f%% retained)",
        method, initial_dates, len(aligned_df), report["retention_rate_pct"]
    )

    return aligned_df, report


def generate_data_quality_report(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    alignment_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Format a human-readable diagnostic report comparing raw and preprocessed data.

    Args:
        raw_df: Raw downloaded price DataFrame.
        cleaned_df: Cleaned and aligned price DataFrame.
        alignment_report: Output dictionary from align_market_calendars.

    Returns:
        Formatted markdown-style text summary.
    """
    raw_info = inspect_data_quality(raw_df)
    clean_info = inspect_data_quality(cleaned_df)

    lines = [
        "=" * 70,
        "EUROPEAN FINANCIAL MARKETS: DATA QUALITY & PREPROCESSING AUDIT",
        "=" * 70,
        f"Raw observations (rows):       {raw_info['total_calendar_rows']}",
        f"Cleaned observations (rows):   {clean_info['total_calendar_rows']}",
        f"Raw date range:                {raw_info['start_date']} to {raw_info['end_date']}",
        f"Cleaned date range:            {clean_info['start_date']} to {clean_info['end_date']}",
    ]

    if alignment_report:
        lines.extend([
            f"Alignment strategy applied:    {alignment_report['alignment_method']}",
            f"Retention rate:                {alignment_report['retention_rate_pct']}% of trading days",
            f"Artificial zeros introduced:   {alignment_report['synthetic_zero_returns_induced']}",
        ])
        if alignment_report.get("dropped_columns"):
            lines.append(f"Dropped columns (> tolerance): {alignment_report['dropped_columns']}")

    lines.append("-" * 70)
    lines.append(f"{'Ticker':12s} | {'Raw Count':10s} | {'Clean Count':11s} | {'Raw % Missing':14s}")
    lines.append("-" * 70)

    for col in cleaned_df.columns:
        raw_stat = raw_info["instruments"].get(col, {})
        clean_stat = clean_info["instruments"].get(col, {})
        raw_cnt = raw_stat.get("valid_observations", 0)
        clean_cnt = clean_stat.get("valid_observations", 0)
        raw_miss = raw_stat.get("missing_percentage", 0.0)
        lines.append(f"{col:12s} | {raw_cnt:10d} | {clean_cnt:11d} | {raw_miss:13.2f}%")

    lines.append("=" * 70)
    return "\n".join(lines)
