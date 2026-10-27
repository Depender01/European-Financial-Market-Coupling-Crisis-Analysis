"""Unit tests for crisis event windows and safe-haven candidate analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.crisis_analysis import (
    STRUCTURED_EVENT_MANIFEST,
    detect_overlapping_events,
    partition_event_windows,
    compute_leave_one_out_benchmarks,
    holm_bonferroni_correction,
    block_bootstrap_mean_ci,
    block_bootstrap_corr_ci,
    block_bootstrap_diff_means,
    analyze_safe_haven_candidates,
    compare_crisis_window_metrics,
)


@pytest.fixture
def sample_market_returns() -> pd.DataFrame:
    """Fixture with 300 days of synthetic returns for 3 markets."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    data = {
        "UK": np.random.normal(0.0002, 0.015, 300),
        "Germany": np.random.normal(0.0003, 0.016, 300),
        "France": np.random.normal(0.0001, 0.014, 300),
    }
    # Simulate a severe market crash on days 30-45 (COVID-like period)
    for c in ["UK", "Germany", "France"]:
        data[c][30:45] -= 0.03  # Massive negative shock

    return pd.DataFrame(data, index=dates)


def test_detect_overlapping_events():
    """Verify detection of overlapping date windows across events."""
    custom_manifest = {
        "Event_A": {"start": "2020-01-01", "end": "2020-03-31", "name": "A"},
        "Event_B": {"start": "2020-03-15", "end": "2020-05-31", "name": "B"},  # Overlaps March 15-31
        "Event_C": {"start": "2020-07-01", "end": "2020-09-30", "name": "C"},  # Disjoint
    }

    overlaps = detect_overlapping_events(custom_manifest)
    assert len(overlaps) == 1
    ov = overlaps[0]
    assert ov["event_1"] == "Event_A"
    assert ov["event_2"] == "Event_B"
    assert ov["overlap_start"] == "2020-03-15"
    assert ov["overlap_end"] == "2020-03-31"
    assert ov["overlap_days"] == 17


def test_partition_event_windows():
    """Verify date partitioning into in-event, pre-event, post-event, and calm baseline."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    custom_manifest = {
        "TestEvent": {
            "start": "2020-02-01",
            "end": "2020-02-15",
            "name": "Test Event",
        }
    }

    partition_df = partition_event_windows(
        dates,
        event_manifest=custom_manifest,
        pre_event_days=10,
        post_event_days=10,
        buffer_days=1,
    )

    assert set(partition_df.columns) == {"event_id", "window_type", "is_overlapping"}
    assert len(partition_df) == 100

    # In-event check
    assert partition_df.loc["2020-02-05", "window_type"] == "in_event"
    assert partition_df.loc["2020-02-05", "event_id"] == "TestEvent"

    # Pre-event check (2020-01-22 to 2020-01-31)
    assert partition_df.loc["2020-01-25", "window_type"] == "pre_event"

    # Post-event check (2020-02-17 to 2020-02-25)
    assert partition_df.loc["2020-02-20", "window_type"] == "post_event"

    # Calm baseline check (far outside)
    assert partition_df.loc["2020-01-05", "window_type"] == "calm_baseline"
    assert partition_df.loc["2020-03-25", "window_type"] == "calm_baseline"


def test_compute_leave_one_out_benchmarks():
    """Verify leave-one-out benchmark excludes market i's own return."""
    dates = pd.date_range("2026-01-01", periods=3)
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [10.0, 20.0, 30.0],
            "C": [100.0, 200.0, 300.0],
        },
        index=dates,
    )

    loo = compute_leave_one_out_benchmarks(df)

    # For asset A: benchmark is mean of B and C
    expected_bench_A = (df["B"] + df["C"]) / 2.0
    np.testing.assert_allclose(loo["A"].values, expected_bench_A.values)

    # For asset C: benchmark is mean of A and B
    expected_bench_C = (df["A"] + df["B"]) / 2.0
    np.testing.assert_allclose(loo["C"].values, expected_bench_C.values)


def test_holm_bonferroni_correction():
    """Test step-down Holm-Bonferroni multiple testing adjustment."""
    raw_p = np.array([0.01, 0.04, 0.03])
    # Sorted: p_1 = 0.01 (m=3 -> 0.01 * 3 = 0.03)
    #         p_2 = 0.03 (m=2 -> 0.03 * 2 = 0.06)
    #         p_3 = 0.04 (m=1 -> 0.04 * 1 = 0.04 -> clamped to max(0.06, 0.04) = 0.06)
    adj_p = holm_bonferroni_correction(raw_p)

    expected = np.array([0.03, 0.06, 0.06])
    np.testing.assert_allclose(adj_p, expected, rtol=1e-5)
    # Boundary: adjusted p cannot exceed 1.0
    assert (adj_p <= 1.0).all()


def test_analyze_safe_haven_candidates(sample_market_returns: pd.DataFrame):
    """Verify safe-haven conditional return analysis across multiple stress thresholds."""
    df_sh = analyze_safe_haven_candidates(
        sample_market_returns,
        stress_quantiles=(0.01, 0.05, 0.10),
        use_leave_one_out=True,
        n_bootstraps=200,
    )

    assert isinstance(df_sh, pd.DataFrame)
    assert set(df_sh["quantile_threshold"].unique()) == {"1%", "5%", "10%"}
    assert set(df_sh["market"].unique()) == {"UK", "Germany", "France"}
    assert "mean_return" in df_sh.columns
    assert "ret_ci_95_lower" in df_sh.columns
    assert "ret_ci_95_upper" in df_sh.columns
    assert "holm_adj_p_value_ret" in df_sh.columns
    assert "corr_stress" in df_sh.columns
    assert "corr_stress_ci_lower" in df_sh.columns
    assert "corr_stress_ci_upper" in df_sh.columns
    assert "corr_calm" in df_sh.columns
    assert "delta_corr" in df_sh.columns
    assert "baur_lucey_status" in df_sh.columns
    assert "capital_preservation_status" in df_sh.columns

    # CI lower must be <= CI upper
    valid_ci = df_sh.dropna(subset=["ret_ci_95_lower", "ret_ci_95_upper"])
    assert (valid_ci["ret_ci_95_lower"] <= valid_ci["ret_ci_95_upper"]).all()


def test_compare_crisis_window_metrics(sample_market_returns: pd.DataFrame):
    """Verify in-event vs calm baseline metric comparison and dependence-aware inference."""
    from src.correlation import compute_rolling_correlation_matrices
    from src.metrics import compute_rolling_metrics

    corr_mats, dates, _ = compute_rolling_correlation_matrices(sample_market_returns, window=15)
    metrics_df = compute_rolling_metrics(corr_mats, dates)

    # Use a custom test manifest covering days in the sample
    custom_manifest = {
        "ShockEvent": {
            "name": "Synthetic Shock",
            "start": str(dates[20].date()),
            "end": str(dates[40].date()),
        }
    }

    partition_df = partition_event_windows(dates, event_manifest=custom_manifest, pre_event_days=10, post_event_days=10)
    df_comp = compare_crisis_window_metrics(
        metrics_df, partition_df, event_manifest=custom_manifest, block_size=10, n_bootstraps=200
    )

    assert len(df_comp) == 1
    assert df_comp.iloc[0]["event_id"] == "ShockEvent"
    assert "in_mean_corr" in df_comp.columns
    assert "delta_corr_vs_calm" in df_comp.columns
    assert "delta_corr_ci_95_lower" in df_comp.columns
    assert "delta_corr_ci_95_upper" in df_comp.columns
    assert "boot_p_value_corr" in df_comp.columns
    assert "holm_adj_boot_p_value" in df_comp.columns
    assert "ks_statistic" in df_comp.columns
    assert "ks_p_value_descriptive_only" in df_comp.columns
    assert "pre_event_status" in df_comp.columns


def test_block_bootstrap_mean_ci():
    """Verify moving-block bootstrap confidence intervals preserve serial dependence."""
    np.random.seed(42)
    # Generate AR(1) process with positive autocorrelation
    n = 200
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + np.random.normal(0, 1)

    # Compute block bootstrap CI with block_size=5
    ci_lower, ci_upper = block_bootstrap_mean_ci(
        x, block_size=5, n_bootstraps=500, confidence_level=0.95, random_seed=42
    )

    sample_mean = float(np.mean(x))
    assert ci_lower < ci_upper
    assert ci_lower <= sample_mean <= ci_upper

    # Edge cases
    assert block_bootstrap_mean_ci([]) == (np.nan, np.nan)
    assert block_bootstrap_mean_ci([5.0]) == (np.nan, np.nan)


def test_block_bootstrap_corr_ci():
    """Verify paired moving-block bootstrap correlation confidence intervals."""
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    y = 0.8 * x + np.random.normal(0, 0.5, n)

    corr, ci_low, ci_upp = block_bootstrap_corr_ci(x, y, block_size=5, n_bootstraps=300, random_seed=42)
    assert not np.isnan(corr)
    assert 0.7 < corr < 0.95
    assert ci_low < ci_upp
    assert ci_low <= corr <= ci_upp


def test_block_bootstrap_diff_means():
    """Verify moving-block bootstrap difference in means inference."""
    np.random.seed(42)
    x = np.random.normal(1.0, 0.2, 50)
    baseline = np.random.normal(0.0, 0.2, 100)

    delta, ci_low, ci_upp, p_val = block_bootstrap_diff_means(x, baseline, block_size=10, n_bootstraps=300, random_seed=42)
    assert not np.isnan(delta)
    assert 0.8 < delta < 1.2
    assert ci_low < ci_upp
    assert ci_low <= delta <= ci_upp
    assert p_val <= 0.05


def test_block_bootstrap_corr_series_alignment():
    """Verify that paired observations remain strictly aligned when passing pandas Series with NaNs and differing indices."""
    dates_x = pd.date_range("2020-01-01", periods=60, freq="B")
    dates_y = pd.date_range("2020-01-03", periods=60, freq="B")  # slight offset

    rng = np.random.RandomState(42)
    s_x = pd.Series(rng.normal(0, 1, len(dates_x)), index=dates_x)
    s_y = 0.75 * s_x + pd.Series(rng.normal(0, 0.4, len(dates_y)), index=dates_y)
    s_x.iloc[5] = np.nan
    s_y.iloc[10] = np.nan

    corr, ci_low, ci_upp = block_bootstrap_corr_ci(s_x, s_y, block_size=5, n_bootstraps=200, random_seed=42)
    assert not np.isnan(corr)
    assert 0.65 <= corr <= 0.95
    assert ci_low <= corr <= ci_upp


def test_block_bootstrap_block_length_sensitivity():
    """Verify moving-block bootstrap sensitivity across multiple block lengths."""
    rng = np.random.RandomState(42)
    n = 120
    # Generate autocorrelated AR(1) series
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.5 * x[i - 1] + rng.normal(0, 1)
    y = 0.7 * x + rng.normal(0, 0.5, n)

    for b in [2, 5, 10, 20]:
        corr, ci_low, ci_upp = block_bootstrap_corr_ci(x, y, block_size=b, n_bootstraps=200, random_seed=42)
        assert not np.isnan(corr)
        assert ci_low < ci_upp
        assert ci_low <= corr <= ci_upp



