"""Crisis Event Window and Safe-Haven Candidate Analysis Module.

Implements structured historical crisis windows, overlap detection, pre/in/post window partitioning,
dependence-aware comparative metrics, leave-one-market-out benchmarks, and Baur-Lucey
conditional tail-return safe-haven testing with Holm-Bonferroni multiple testing corrections.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import HISTORICAL_CRISIS_WINDOWS

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Historical Crisis Windows & Overlap Detection
# -----------------------------------------------------------------------------
# Structured crisis manifest with primary historical documentation sources
STRUCTURED_EVENT_MANIFEST: Dict[str, Dict[str, Any]] = {
    "DotCom_Crash": {
        "name": "Dot-Com Crash / Telecom Bust",
        "start": "2000-03-10",
        "end": "2002-10-09",
        "rationale": "Peak of technology valuation bubble followed by multi-year equity bear market.",
        "source": "Shiller (2000); NBER US/Global Business Cycle dating.",
    },
    "GFC_Lehman": {
        "name": "Global Financial Crisis (Lehman Collapse)",
        "start": "2007-08-09",
        "end": "2009-03-09",
        "rationale": "BNP Paribas fund suspension, subprime contagion, Lehman Brothers bankruptcy (2008-09-15), market trough.",
        "source": "BIS Annual Report (2009); Federal Reserve Financial Crisis Timeline.",
    },
    "Euro_Debt_Crisis": {
        "name": "European Sovereign Debt Crisis (PIIGS)",
        "start": "2010-04-23",
        "end": "2012-07-26",
        "rationale": "Greek formal bailout request to Draghi 'Whatever it takes' London speech.",
        "source": "ECB Economic Bulletin; Lane (2012) 'The European Sovereign Debt Crisis'.",
    },
    "Grexit_China_Slowdown": {
        "name": "Grexit Referendum & China Yuan Devaluation",
        "start": "2015-06-01",
        "end": "2016-02-11",
        "rationale": "Greek capital controls, commodity price collapse, and PBOC currency devaluation.",
        "source": "IMF Global Financial Stability Report (Oct 2015).",
    },
    "Brexit_Referendum": {
        "name": "UK Brexit Referendum Shock",
        "start": "2016-06-23",
        "end": "2016-07-15",
        "rationale": "Shock UK vote to exit the European Union; sterling depreciation and European bank selloff.",
        "source": "Bank of England Financial Stability Report (Nov 2016).",
    },
    "COVID_Crash": {
        "name": "COVID-19 Pandemic Market Crash",
        "start": "2020-02-19",
        "end": "2020-03-23",
        "rationale": "Global pandemic lockdowns, extreme volatility spike, swift liquidity collapse ending with Fed/ECB backstops.",
        "source": "WHO Pandemic Declaration; ECB Pandemic Emergency Purchase Programme (PEPP).",
    },
    "Ukraine_War_Inflation": {
        "name": "Russia-Ukraine War & European Energy Shock",
        "start": "2022-01-03",
        "end": "2022-10-14",
        "rationale": "Russian invasion of Ukraine, natural gas supply shock, European inflation surge, and rapid ECB rate hikes.",
        "source": "ECB Financial Stability Review (May 2022); IEA European Gas Security Analysis.",
    },
}


def detect_overlapping_events(
    event_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Detect overlapping date spans across configured historical events.

    Why this matters:
    Silently treating overlapping days as independent evidence across multiple events
    causes double-counting and distorts statistical hypothesis testing.

    Returns:
        List of dictionaries detailing overlapping event pairs and shared days.
    """
    manifest = event_manifest or STRUCTURED_EVENT_MANIFEST
    event_ids = list(manifest.keys())
    overlaps = []

    for i in range(len(event_ids)):
        for j in range(i + 1, len(event_ids)):
            id1, id2 = event_ids[i], event_ids[j]
            s1, e1 = pd.Timestamp(manifest[id1]["start"]), pd.Timestamp(manifest[id1]["end"])
            s2, e2 = pd.Timestamp(manifest[id2]["start"]), pd.Timestamp(manifest[id2]["end"])

            # Check overlap
            latest_start = max(s1, s2)
            earliest_end = min(e1, e2)

            if latest_start <= earliest_end:
                overlap_days = (earliest_end - latest_start).days + 1
                overlaps.append({
                    "event_1": id1,
                    "event_2": id2,
                    "overlap_start": str(latest_start.date()),
                    "overlap_end": str(earliest_end.date()),
                    "overlap_days": overlap_days,
                })

    if overlaps:
        logger.warning("Detected %d overlapping event window pair(s)", len(overlaps))
    return overlaps


def partition_event_windows(
    index_dates: pd.DatetimeIndex,
    event_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    pre_event_days: int = 60,
    post_event_days: int = 60,
    buffer_days: int = 5,
) -> pd.DataFrame:
    """Partition calendar dates into Pre-Event, In-Event, Post-Event, and Calm Baseline windows.

    Windows:
        - In-Event: [start, end]
        - Pre-Event: [start - pre_event_days, start - buffer_days]
        - Post-Event: [end + buffer_days, end + post_event_days]
        - Calm Baseline: All dates not classified as Pre, In, or Post for ANY event.

    Args:
        index_dates: DatetimeIndex of market observations.
        event_manifest: Event definitions. Defaults to STRUCTURED_EVENT_MANIFEST.
        pre_event_days: Calendar days for pre-event window.
        post_event_days: Calendar days for post-event window.
        buffer_days: Exclusion gap between pre/post windows and in-event dates.

    Returns:
        pd.DataFrame indexed by date with columns:
            - 'event_id': Name of active event or 'None'
            - 'window_type': 'in_event', 'pre_event', 'post_event', or 'calm_baseline'
            - 'is_overlapping': Boolean indicating if date falls within multiple events
    """
    manifest = event_manifest or STRUCTURED_EVENT_MANIFEST
    dates_series = pd.Series(index=index_dates, data="calm_baseline", dtype=object)
    event_col = pd.Series(index=index_dates, data="None", dtype=object)
    overlap_counter = pd.Series(index=index_dates, data=0, dtype=int)

    for ev_id, ev in manifest.items():
        s = pd.Timestamp(ev["start"])
        e = pd.Timestamp(ev["end"])

        in_mask = (index_dates >= s) & (index_dates <= e)
        pre_mask = (index_dates >= (s - pd.Timedelta(days=pre_event_days))) & (
            index_dates <= (s - pd.Timedelta(days=buffer_days))
        )
        post_mask = (index_dates >= (e + pd.Timedelta(days=buffer_days))) & (
            index_dates <= (e + pd.Timedelta(days=post_event_days))
        )

        overlap_counter[in_mask] += 1

        # Apply classifications (In-event takes highest precedence)
        for d in index_dates[pre_mask]:
            if dates_series[d] == "calm_baseline":
                dates_series[d] = "pre_event"
                event_col[d] = ev_id

        for d in index_dates[post_mask]:
            if dates_series[d] == "calm_baseline":
                dates_series[d] = "post_event"
                event_col[d] = ev_id

        for d in index_dates[in_mask]:
            dates_series[d] = "in_event"
            event_col[d] = ev_id

    partition_df = pd.DataFrame(
        {
            "event_id": event_col,
            "window_type": dates_series,
            "is_overlapping": overlap_counter > 1,
        },
        index=index_dates,
    )
    return partition_df


# -----------------------------------------------------------------------------
# 2. Leave-One-Market-Out Benchmark
# -----------------------------------------------------------------------------
def compute_leave_one_out_benchmarks(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute benchmark returns excluding market i:

    Formula:
        r_{bench}^{(-i)}(t) = 1 / (N - 1) * sum_{j != i} r_j(t)

    Why this matters:
    Including market i in the benchmark when testing whether market i provides protection
    introduces mechanical self-correlation and endogeneity, especially for large markets
    like Germany or the UK.

    Args:
        returns_df: DataFrame of asset returns (dates x assets).

    Returns:
        DataFrame of same shape where column i is the benchmark excluding asset i.
    """
    N = returns_df.shape[1]
    if N < 2:
        raise ValueError(f"Need at least 2 markets for leave-one-out benchmark, got {N}")

    total_sum = returns_df.sum(axis=1)
    loo_df = pd.DataFrame(index=returns_df.index)

    for col in returns_df.columns:
        # Subtract asset i's return from the total sum and divide by N - 1
        loo_df[col] = (total_sum - returns_df[col]) / float(N - 1)

    return loo_df


# -----------------------------------------------------------------------------
# 3. Safe-Haven Candidate Analysis (Baur-Lucey Conditional Tail Returns)
# -----------------------------------------------------------------------------
def holm_bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """Apply Holm-Bonferroni step-down correction for multiple hypothesis testing.

    Controls the Family-Wise Error Rate (FWER) without the excessive conservatism
    of single-step Bonferroni.

    Args:
        p_values: 1D array of unadjusted p-values.

    Returns:
        1D array of Holm-adjusted p-values bounded in [0.0, 1.0].
    """
    m = len(p_values)
    if m == 0:
        return np.array([])

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted = np.empty(m, dtype=float)

    current_max = 0.0
    for i in range(m):
        rank = i + 1
        adj_p = (m - rank + 1) * sorted_p[i]
        adj_p = min(1.0, max(current_max, adj_p))
        adjusted[order[i]] = adj_p
        current_max = adj_p

    return adjusted


def block_bootstrap_mean_ci(
    data: Union[np.ndarray, List[float], pd.Series],
    block_size: int = 5,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """Compute moving-block bootstrap confidence intervals for the sample mean.

    Preserves serial correlation and short-term autocorrelation in financial time series.

    Args:
        data: 1D array-like of time series observations.
        block_size: Length of contiguous sub-blocks (default 5 ~ 1 business week).
        n_bootstraps: Number of bootstrap resamples.
        confidence_level: Nominal coverage (default 0.95).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    N = len(arr)
    if N < 2:
        return (np.nan, np.nan)

    rng = np.random.RandomState(random_seed)
    b = min(block_size, max(1, N // 2))

    if b <= 1:
        boot_means = [
            np.mean(rng.choice(arr, size=N, replace=True))
            for _ in range(n_bootstraps)
        ]
    else:
        n_blocks = int(np.ceil(N / b))
        num_valid_starts = N - b + 1
        boot_means = []
        for _ in range(n_bootstraps):
            starts = rng.randint(0, num_valid_starts, size=n_blocks)
            resampled = np.concatenate([arr[s : s + b] for s in starts])[:N]
            boot_means.append(float(np.mean(resampled)))

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return ci_lower, ci_upper


def block_bootstrap_corr_ci(
    x: Union[np.ndarray, List[float], pd.Series],
    y: Union[np.ndarray, List[float], pd.Series],
    block_size: int = 5,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute Pearson correlation and moving-block bootstrap confidence intervals.

    Preserves serial correlation and cross-correlation structure in paired financial series.

    Args:
        x: First time series.
        y: Second time series.
        block_size: Length of contiguous blocks (default 5 ~ 1 business week).
        n_bootstraps: Number of bootstrap iterations.
        confidence_level: Coverage level (default 0.95).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        paired_df = pd.concat([x, y], axis=1).dropna()
        arr_x = paired_df.iloc[:, 0].values.astype(float)
        arr_y = paired_df.iloc[:, 1].values.astype(float)
    else:
        arr_x = np.asarray(x, dtype=float)
        arr_y = np.asarray(y, dtype=float)
        valid_mask = (~np.isnan(arr_x)) & (~np.isnan(arr_y))
        arr_x = arr_x[valid_mask]
        arr_y = arr_y[valid_mask]
    N = len(arr_x)

    if N < 4 or np.std(arr_x) < 1e-12 or np.std(arr_y) < 1e-12:
        return (np.nan, np.nan, np.nan)

    point_corr = float(stats.pearsonr(arr_x, arr_y)[0])
    rng = np.random.RandomState(random_seed)
    b = min(block_size, max(1, N // 2))

    boot_corrs = []
    if b <= 1:
        for _ in range(n_bootstraps):
            idx = rng.choice(N, size=N, replace=True)
            bx, by = arr_x[idx], arr_y[idx]
            if np.std(bx) > 1e-12 and np.std(by) > 1e-12:
                r, _ = stats.pearsonr(bx, by)
                boot_corrs.append(float(r))
    else:
        n_blocks = int(np.ceil(N / b))
        num_valid_starts = N - b + 1
        for _ in range(n_bootstraps):
            starts = rng.randint(0, num_valid_starts, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + b) for s in starts])[:N]
            bx, by = arr_x[idx], arr_y[idx]
            if np.std(bx) > 1e-12 and np.std(by) > 1e-12:
                r, _ = stats.pearsonr(bx, by)
                boot_corrs.append(float(r))

    if len(boot_corrs) < 10:
        return (point_corr, np.nan, np.nan)

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_corrs, alpha * 100.0))
    ci_upper = float(np.percentile(boot_corrs, (1.0 - alpha) * 100.0))
    return point_corr, ci_lower, ci_upper


def block_bootstrap_diff_means(
    x: np.ndarray,
    baseline: np.ndarray,
    block_size: int = 20,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Moving-block bootstrap inference for the difference in means between two serially dependent series.

    Why this matters:
    Overlapping rolling-window metrics exhibit strong autocorrelation (MA(M-1) moving average structure).
    Ordinary two-sample tests (e.g. Student's t or Kolmogorov-Smirnov) assume independent draws,
    which severely deflates standard errors and generates spurious statistical significance.
    This function resamples contiguous blocks of length `block_size` to preserve the dependence structure.

    Args:
        x: In-event observations.
        baseline: Baseline (e.g. calm period) observations.
        block_size: Contiguous block length in days (default 20, matching typical epoch window M).
        n_bootstraps: Resampling iterations.
        confidence_level: Nominal coverage.
        random_seed: Random seed.

    Returns:
        Tuple of (delta_mean, ci_lower, ci_upper, p_value_two_sided).
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(baseline, dtype=float)
    arr_x = arr_x[~np.isnan(arr_x)]
    arr_y = arr_y[~np.isnan(arr_y)]

    Nx, Ny = len(arr_x), len(arr_y)
    if Nx < 2 or Ny < 2:
        return (np.nan, np.nan, np.nan, 1.0)

    delta_point = float(np.mean(arr_x) - np.mean(arr_y))
    rng = np.random.RandomState(random_seed)

    bx = min(block_size, max(1, Nx // 2))
    by = min(block_size, max(1, Ny // 2))

    n_blocks_x = int(np.ceil(Nx / bx))
    n_blocks_y = int(np.ceil(Ny / by))

    starts_x_limit = max(1, Nx - bx + 1)
    starts_y_limit = max(1, Ny - by + 1)

    boot_diffs = []
    for _ in range(n_bootstraps):
        # Resample in-event blocks
        sx = rng.randint(0, starts_x_limit, size=n_blocks_x)
        sample_x = np.concatenate([arr_x[s : s + bx] for s in sx])[:Nx]

        # Resample baseline blocks
        sy = rng.randint(0, starts_y_limit, size=n_blocks_y)
        sample_y = np.concatenate([arr_y[s : s + by] for s in sy])[:Ny]

        boot_diffs.append(float(np.mean(sample_x) - np.mean(sample_y)))

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100.0))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100.0))

    # Two-sided empirical p-value testing H_0: delta = 0 vs H_1: delta != 0
    boot_arr = np.array(boot_diffs)
    p_nonpos = float(np.mean(boot_arr <= 0.0))
    p_nonneg = float(np.mean(boot_arr >= 0.0))
    p_val_two_sided = float(min(1.0, 2.0 * min(p_nonpos, p_nonneg)))

    return delta_point, ci_lower, ci_upper, p_val_two_sided



def analyze_safe_haven_candidates(
    returns_df: pd.DataFrame,
    stress_quantiles: Tuple[float, ...] = (0.01, 0.05, 0.10),
    use_leave_one_out: bool = True,
    block_size: int = 5,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Empirically test whether any European equity market behaves as a safe haven under tail stress.

    Verifiable Bibliographic References:
        1. Baur, D. G., & Lucey, B. M. (2010). "Is gold a hedge or a safe haven? An analysis of stocks,
           bonds and gold." Financial Review, 45(2), 217-229.
        2. Baur, D. G., & McDermott, T. K. (2010). "Is gold a safe haven? International evidence."
           Journal of Banking & Finance, 34(8), 1886-1898.

    Rigorous Econometric Definitions:
        - Diversifier: An asset positively correlated with another asset/benchmark on average,
          with correlation strictly less than 1.0.
        - Hedge: An asset uncorrelated or negatively correlated with another asset/benchmark
          on average (rho <= 0).
        - Safe Haven (Baur & Lucey 2010): An asset that is uncorrelated or negatively correlated
          with another asset/benchmark in times of market stress / turmoil:
              Weak Safe Haven:   rho_{stress} <= 0
              Strong Safe Haven: rho_{stress} < 0
        - Capital Preservation Criterion (Operational Return Definition):
          Asset exhibits non-negative expected return conditioned on severe market tail declines:
              E[r_i | r_benchmark < Q_alpha] >= 0

    Testing Protocol:
        - Evaluated across multiple stress thresholds alpha in (1%, 5%, 10%).
        - Uses leave-one-market-out benchmark r_bench^{(-i)} to eliminate self-correlation.
        - Computes both conditional correlation rho_{stress} and conditional mean return.
        - Computes 95% moving-block bootstrap confidence intervals for both metrics.
        - Applies Holm-Bonferroni Family-Wise Error Rate adjustment across tested markets.
        - Explicit Scope Limitation: Conclusions apply strictly to the 8 tested European core
          equity indices, selected equity benchmarks, sample period, and operational definitions.
          Does not evaluate non-equity instruments (e.g. Bunds, Gold, CHF).

    Args:
        returns_df: Aligned daily returns DataFrame (dates x assets).
        stress_quantiles: Benchmark tail quantiles (e.g. 0.01, 0.05, 0.10).
        use_leave_one_out: Whether benchmark for asset i strictly excludes asset i.
        block_size: Block length in days for moving-block bootstrap (default 5).
        n_bootstraps: Resampling iterations for bootstrap CIs.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing comprehensive test results and conservative classifications.
    """
    if returns_df.shape[1] < 2:
        raise ValueError("At least 2 markets required for safe-haven analysis.")

    if use_leave_one_out:
        benchmarks_df = compute_leave_one_out_benchmarks(returns_df)
    else:
        ew = returns_df.mean(axis=1)
        benchmarks_df = pd.DataFrame({col: ew for col in returns_df.columns}, index=returns_df.index)

    records = []

    for q in stress_quantiles:
        p_vals_ret = []
        q_records = []

        for asset in returns_df.columns:
            r_asset = returns_df[asset].values
            r_bench = benchmarks_df[asset].values

            # Tail threshold determination
            q_thresh = float(np.percentile(r_bench, q * 100.0))
            stress_mask = r_bench <= q_thresh
            calm_mask = r_bench > q_thresh
            n_stress = int(np.sum(stress_mask))

            if n_stress < 4:
                q_records.append({
                    "market": asset,
                    "quantile_threshold": f"{int(q*100)}%",
                    "cutoff_return": round(q_thresh, 4),
                    "n_stress_obs": n_stress,
                    "mean_return": np.nan,
                    "median_return": np.nan,
                    "ret_ci_95_lower": np.nan,
                    "ret_ci_95_upper": np.nan,
                    "raw_p_value_ret": 1.0,
                    "corr_stress": np.nan,
                    "corr_stress_ci_lower": np.nan,
                    "corr_stress_ci_upper": np.nan,
                    "corr_calm": np.nan,
                    "delta_corr": np.nan,
                    "baur_lucey_status": "Insufficient Stress Observations",
                    "capital_preservation_status": "Insufficient Stress Observations",
                })
                p_vals_ret.append(1.0)
                continue

            cond_rets = r_asset[stress_mask]
            mean_ret = float(np.mean(cond_rets))
            median_ret = float(np.median(cond_rets))

            # Dependence-aware Moving-Block Bootstrap 95% CI for Conditional Return
            ret_ci_lower, ret_ci_upper = block_bootstrap_mean_ci(
                cond_rets,
                block_size=block_size,
                n_bootstraps=n_bootstraps,
                confidence_level=0.95,
                random_seed=random_seed,
            )

            # Conditional Correlation Analysis (Baur-Lucey exact definition)
            r_bench_stress = r_bench[stress_mask]
            corr_stress, corr_ci_lower, corr_ci_upper = block_bootstrap_corr_ci(
                cond_rets,
                r_bench_stress,
                block_size=block_size,
                n_bootstraps=n_bootstraps,
                confidence_level=0.95,
                random_seed=random_seed,
            )

            # Calm-period correlation
            r_asset_calm = r_asset[calm_mask]
            r_bench_calm = r_bench[calm_mask]
            corr_calm = float(stats.pearsonr(r_asset_calm, r_bench_calm)[0]) if len(r_asset_calm) > 5 else np.nan
            delta_corr = corr_stress - corr_calm if not np.isnan(corr_calm) else np.nan

            # One-sided t-test on conditional return: H_0: mean <= 0 vs H_1: mean > 0
            if np.std(cond_rets, ddof=1) > 1e-12:
                t_stat, p_two_sided = stats.ttest_1samp(cond_rets, 0.0)
                p_val_ret = float(p_two_sided / 2.0 if t_stat > 0 else 1.0 - (p_two_sided / 2.0))
            else:
                p_val_ret = 1.0 if mean_ret <= 0 else 0.0

            p_vals_ret.append(p_val_ret)

            # Baur-Lucey Safe Haven Classification:
            # Requires rho_stress <= 0
            if not np.isnan(corr_stress):
                if corr_ci_upper <= 0.0:
                    bl_status = "Candidate Safe Haven under tested definition (Strong, rho < 0)"
                elif corr_stress <= 0.0:
                    bl_status = "Candidate Safe Haven under tested definition (Weak, rho <= 0)"
                elif corr_stress < corr_calm:
                    bl_status = "Diversifier / Attenuator (Correlation Drops under Stress)"
                else:
                    bl_status = "Co-Moving / Contagion (Correlation Rises under Stress)"
            else:
                bl_status = "Inconclusive Correlation"

            # Capital Preservation Status
            cap_status = "Capital Preserving (Non-negative return)" if mean_ret >= 0 else "Loss Incurring (Negative return)"

            q_records.append({
                "market": asset,
                "quantile_threshold": f"{int(q*100)}%",
                "cutoff_return": round(q_thresh, 4),
                "n_stress_obs": n_stress,
                "mean_return": round(mean_ret, 5),
                "median_return": round(median_ret, 5),
                "ret_ci_95_lower": round(ret_ci_lower, 5),
                "ret_ci_95_upper": round(ret_ci_upper, 5),
                "raw_p_value_ret": p_val_ret,
                "corr_stress": round(corr_stress, 4) if not np.isnan(corr_stress) else np.nan,
                "corr_stress_ci_lower": round(corr_ci_lower, 4) if not np.isnan(corr_ci_lower) else np.nan,
                "corr_stress_ci_upper": round(corr_ci_upper, 4) if not np.isnan(corr_ci_upper) else np.nan,
                "corr_calm": round(corr_calm, 4) if not np.isnan(corr_calm) else np.nan,
                "delta_corr": round(delta_corr, 4) if not np.isnan(delta_corr) else np.nan,
                "baur_lucey_status": bl_status,
                "capital_preservation_status": cap_status,
            })

        # Apply Holm-Bonferroni correction across markets within this quantile threshold
        adj_p_vals = holm_bonferroni_correction(np.array(p_vals_ret))
        for rec, adj_p in zip(q_records, adj_p_vals):
            rec["holm_adj_p_value_ret"] = round(float(adj_p), 4)
            records.append(rec)

    df_sh = pd.DataFrame(records)
    return df_sh


# -----------------------------------------------------------------------------
# 4. In-Window vs. Out-of-Window Event Metric Comparison
# -----------------------------------------------------------------------------
def compare_crisis_window_metrics(
    metrics_df: pd.DataFrame,
    partition_df: pd.DataFrame,
    event_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
    block_size: int = 20,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compare coupling metrics inside crisis windows vs pre-event and calm baselines.

    Methodological Improvements for Serial Dependence:
        1. Overlapping rolling windows (window M=20, stride 1) induce strong MA(19) autocorrelation.
        2. Classical two-sample tests (like Kolmogorov-Smirnov) strictly assume i.i.d. draws.
           When applied to rolling series, their standard errors are severely deflated, producing
           spurious significance. Ordinary KS p-values are retained strictly as descriptive diagnostics.
        3. Statistical inference is conducted via moving-block bootstrap (block size = 20 days),
           generating dependence-aware 95% confidence intervals and empirical bootstrap p-values.
        4. Holm-Bonferroni step-down correction is applied across events to control Family-Wise
           Error Rate (FWER) without confounding it as a remedy for within-series autocorrelation.
        5. Windows with insufficient pre-event data (e.g. events near dataset boundaries) are explicitly flagged.

    Args:
        metrics_df: DataFrame of rolling metrics (mean_correlation, spectral_entropy, etc.).
        partition_df: Output of partition_event_windows.
        event_manifest: Event manifest dictionary.
        block_size: Block length for moving-block bootstrap (default 20 days).
        n_bootstraps: Resampling iterations.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame comparing in-event metrics against baseline with dependence-aware inference.
    """
    manifest = event_manifest or STRUCTURED_EVENT_MANIFEST
    aligned_metrics = metrics_df.reindex(partition_df.index).dropna()
    aligned_partition = partition_df.reindex(aligned_metrics.index)

    # Calm baseline stats
    calm_mask = aligned_partition["window_type"] == "calm_baseline"
    calm_corr_vals = aligned_metrics.loc[calm_mask, "mean_correlation"].values
    calm_ent_vals = aligned_metrics.loc[calm_mask, "spectral_entropy"].values

    calm_mu = float(np.mean(calm_corr_vals)) if len(calm_corr_vals) > 0 else np.nan
    calm_h = float(np.mean(calm_ent_vals)) if len(calm_ent_vals) > 0 else np.nan

    records = []
    boot_p_values = []

    for ev_id, ev in manifest.items():
        ev_mask = (aligned_partition["event_id"] == ev_id) & (aligned_partition["window_type"] == "in_event")
        pre_mask = (aligned_partition["event_id"] == ev_id) & (aligned_partition["window_type"] == "pre_event")

        n_in = int(ev_mask.sum())
        n_pre = int(pre_mask.sum())

        if n_in == 0:
            continue

        in_corr_vals = aligned_metrics.loc[ev_mask, "mean_correlation"].values
        in_ent_vals = aligned_metrics.loc[ev_mask, "spectral_entropy"].values

        in_mu = float(np.mean(in_corr_vals))
        in_h = float(np.mean(in_ent_vals))

        pre_corr_vals = aligned_metrics.loc[pre_mask, "mean_correlation"].values
        pre_mu = float(np.mean(pre_corr_vals)) if n_pre > 0 else np.nan

        # Data sufficiency audit
        pre_status = "Sufficient (N >= 20)" if n_pre >= 20 else f"Insufficient / Boundary (N={n_pre})"

        # Dependence-Aware Moving-Block Bootstrap for Difference in Correlation
        delta_corr, corr_ci_low, corr_ci_upp, boot_p_corr = block_bootstrap_diff_means(
            in_corr_vals,
            calm_corr_vals,
            block_size=block_size,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            random_seed=random_seed,
        )

        # Dependence-Aware Moving-Block Bootstrap for Difference in Spectral Entropy
        delta_ent, ent_ci_low, ent_ci_upp, _ = block_bootstrap_diff_means(
            in_ent_vals,
            calm_ent_vals,
            block_size=block_size,
            n_bootstraps=n_bootstraps,
            confidence_level=0.95,
            random_seed=random_seed + 1,
        )

        boot_p_values.append(boot_p_corr if not np.isnan(boot_p_corr) else 1.0)

        # Descriptive-only Kolmogorov-Smirnov test (clearly documented as violating independence)
        if n_in >= 5 and len(calm_corr_vals) >= 5:
            ks_stat, ks_p = stats.ks_2samp(in_corr_vals, calm_corr_vals)
        else:
            ks_stat, ks_p = np.nan, 1.0

        records.append({
            "event_id": ev_id,
            "event_name": ev["name"],
            "start": ev["start"],
            "end": ev["end"],
            "in_event_days": n_in,
            "calm_baseline_days": len(calm_corr_vals),
            "pre_event_days": n_pre,
            "pre_event_status": pre_status,
            "in_mean_corr": round(in_mu, 4),
            "calm_mean_corr": round(calm_mu, 4) if not np.isnan(calm_mu) else np.nan,
            "delta_corr_vs_calm": round(delta_corr, 4),
            "delta_corr_ci_95_lower": round(corr_ci_low, 4) if not np.isnan(corr_ci_low) else np.nan,
            "delta_corr_ci_95_upper": round(corr_ci_upp, 4) if not np.isnan(corr_ci_upp) else np.nan,
            "boot_p_value_corr": round(boot_p_corr, 4),
            "pre_mean_corr": round(pre_mu, 4) if not np.isnan(pre_mu) else np.nan,
            "in_spectral_entropy": round(in_h, 4),
            "calm_spectral_entropy": round(calm_h, 4) if not np.isnan(calm_h) else np.nan,
            "delta_entropy_vs_calm": round(delta_ent, 4),
            "delta_entropy_ci_95_lower": round(ent_ci_low, 4) if not np.isnan(ent_ci_low) else np.nan,
            "delta_entropy_ci_95_upper": round(ent_ci_upp, 4) if not np.isnan(ent_ci_upp) else np.nan,
            "ks_statistic": round(float(ks_stat), 4) if not np.isnan(ks_stat) else np.nan,
            "ks_p_value_descriptive_only": round(float(ks_p), 4) if not np.isnan(ks_p) else np.nan,
        })

    df_comp = pd.DataFrame(records)
    if not df_comp.empty and "boot_p_value_corr" in df_comp.columns:
        adj_boot_ps = holm_bonferroni_correction(np.array(boot_p_values))
        df_comp["holm_adj_boot_p_value"] = np.round(adj_boot_ps, 4)

    return df_comp


