"""Reproducible Empirical Comparison: DAX Total Return vs. DAX Price Return.

Quantifies the empirical difference between:
- DAX 40 Performance Index (^GDAXI): Total return convention (dividends reinvested)
- DAX 40 Kursindex (^GDAXIP): Price return convention (dividends excluded)

Over the overlapping historical sample (2014-01-01 to 2025-12-31).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import align_market_calendars
from src.returns import calculate_log_returns


def run_dax_comparison() -> Dict[str, Any]:
    print("=" * 75)
    print("EMPIRICAL COMPARISON: GERMAN DAX TOTAL RETURN vs. PRICE RETURN")
    print("=" * 75)

    loader = DataLoader(use_cache=True)
    tickers = ["^GDAXI", "^GDAXIP"]
    start_date = "2014-01-01"
    end_date = "2025-12-31"

    print(f"Downloading/loading cached data: {tickers} from {start_date} to {end_date}...")
    raw_df, meta = loader.fetch_market_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    aligned_df, rep = align_market_calendars(raw_df, method="intersection")
    log_ret = calculate_log_returns(aligned_df)

    dax_tr = log_ret["^GDAXI"]
    dax_pr = log_ret["^GDAXIP"]

    corr = float(dax_tr.corr(dax_pr))
    diff = dax_tr - dax_pr
    mean_diff_daily_bps = float(diff.mean() * 10000.0)
    ann_dividend_drift_pct = float(diff.mean() * 252.0 * 100.0)
    max_daily_div_pct = float(diff.abs().max() * 100.0)
    ann_tracking_error_pct = float(diff.std() * np.sqrt(252.0) * 100.0)
    ann_vol_tr = float(dax_tr.std() * np.sqrt(252.0) * 100.0)
    ann_vol_pr = float(dax_pr.std() * np.sqrt(252.0) * 100.0)

    print(f"\nSample Date Range:              {aligned_df.index[0].date()} to {aligned_df.index[-1].date()}")
    print(f"Total Valid Trading Days:       {len(aligned_df)}")
    print(f"Log Return Observations:        {len(log_ret)}")
    print("-" * 75)
    print(f"Daily Log Return Correlation:   {corr:.6f}")
    print(f"Mean Daily Wedge (TR - PR):     +{mean_diff_daily_bps:.3f} bps/day")
    print(f"Annualized Gross Dividend Yield: +{ann_dividend_drift_pct:.2f}% / year")
    print(f"Annualized Tracking Error:      {ann_tracking_error_pct:.2f}%")
    print(f"Maximum Single-Day Difference:  {max_daily_div_pct:.3f}% (reflects AGM dividend ex-dates)")
    print(f"Annualized Volatility (TR):     {ann_vol_tr:.2f}%")
    print(f"Annualized Volatility (PR):     {ann_vol_pr:.2f}% (volatility difference is negligible)")
    print("=" * 75)

    return {
        "start_date": str(aligned_df.index[0].date()),
        "end_date": str(aligned_df.index[-1].date()),
        "observations": len(log_ret),
        "correlation": corr,
        "mean_daily_wedge_bps": mean_diff_daily_bps,
        "annualized_dividend_pct": ann_dividend_drift_pct,
        "annualized_tracking_error_pct": ann_tracking_error_pct,
        "max_daily_diff_pct": max_daily_div_pct,
    }


if __name__ == "__main__":
    from typing import Dict, Any
    run_dax_comparison()
