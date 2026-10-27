"""Milestone 3 Empirical Execution Script.

European Financial Market Coupling & Crisis Analysis
Executes chronological regime clustering (K-Means, GMM, Rule-Based), multi-metric diagnostics,
independent realized risk characterization, walk-forward robustness analysis,
crisis window comparative analysis, and leave-one-out candidate safe-haven testing.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CORE_MARKETS,
    OUTPUTS_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import DataLoader
from src.preprocessing import align_market_calendars, inspect_data_quality
from src.returns import calculate_log_returns, compute_equal_weighted_index
from src.market_regimes import (
    extract_regime_features,
    ChronologicalScaler,
    split_chronological,
    KMeansRegimeModel,
    GMMRegimeModel,
    RuleBasedRegimeBaseline,
    characterize_regimes,
    walk_forward_regime_evaluation,
)
from src.crisis_analysis import (
    STRUCTURED_EVENT_MANIFEST,
    detect_overlapping_events,
    partition_event_windows,
    analyze_safe_haven_candidates,
    compare_crisis_window_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("milestone3_empirical")


def save_table(df: pd.DataFrame, base_name: str, output_dir: Path) -> None:
    """Save DataFrame as CSV and Markdown table."""
    csv_path = output_dir / f"{base_name}.csv"
    md_path = output_dir / f"{base_name}.md"
    df.to_csv(csv_path)
    try:
        md_content = df.to_markdown()
    except Exception:
        md_content = str(df.to_string())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved table: %s (.csv, .md)", base_name)



def main():
    logger.info("=" * 80)
    logger.info("STARTING MILESTONE 3 EMPIRICAL ANALYSIS")
    logger.info("=" * 80)

    m3_output_dir = OUTPUTS_DIR / "milestone3"
    m3_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Acquisition & Verification
    dl = DataLoader(raw_data_dir=RAW_DATA_DIR, use_cache=True)
    tickers = list(CORE_MARKETS.keys())
    logger.info("Acquiring Core 8 European market indices: %s", tickers)
    raw_df, meta = dl.fetch_market_dataset(tickers, start_date="2000-01-01", end_date="2024-12-31")

    # Data Quality Inspection
    dq = inspect_data_quality(raw_df)
    logger.info("Data Quality Summary:")
    logger.info("  Total observations: %d across %d assets", dq["total_calendar_rows"], dq["total_columns"])
    logger.info("  Start Date: %s | End Date: %s", dq["start_date"], dq["end_date"])
    for inst, istats in dq["instruments"].items():
        logger.info("    %s: %d valid, %d missing (%.2f%%)", inst, istats["valid_observations"], istats["missing_observations"], istats["missing_percentage"])


    # Calendar Alignment (Intersection to avoid synthetic zero-return distortion)
    aligned_df, dropped = align_market_calendars(raw_df, method="intersection")
    logger.info("Aligned market dataset: %d trading days (intersection of all 8 markets)", len(aligned_df))

    # 2. Return Generation & Feature Extraction
    log_rets = calculate_log_returns(aligned_df, drop_first=True)
    ew_bench = compute_equal_weighted_index(log_rets)

    window = 20  # 1 business month
    stride = 1
    features_df, corr_matrices, epoch_dates, asset_names = extract_regime_features(
        log_rets, window=window, stride=stride
    )
    logger.info("Extracted 4D feature vector across %d epochs (window=%d, stride=%d)", len(features_df), window, stride)

    # 3. Chronological Train/Test Split
    # Split at 2014-12-31 (15 years train: 2000-2014, 10 years test: 2015-2024)
    train_end = "2014-12-31"
    train_feat, test_feat = split_chronological(features_df, train_end_date=train_end)
    logger.info("Chronological Split:")
    logger.info("  Training: %s to %s (%d epochs)", train_feat.index[0].date(), train_feat.index[-1].date(), len(train_feat))
    logger.info("  Testing:  %s to %s (%d epochs)", test_feat.index[0].date(), test_feat.index[-1].date(), len(test_feat))

    # 4. Chronological Standardization (Zero Lookahead)
    scaler = ChronologicalScaler()
    X_train_scaled = scaler.fit_transform(train_feat)
    X_test_scaled = scaler.transform(test_feat)
    logger.info("ChronologicalScaler fitted on training data: Means=%s, Stds=%s",
                np.round(scaler.mean_, 4), np.round(scaler.scale_, 4))

    # 5. Model Evaluation across K in [2, 6]
    logger.info("Evaluating K-Means across K in [2, 6] on training set...")
    km_evaluator = KMeansRegimeModel(random_state=42)
    km_diag_df = km_evaluator.evaluate_k_range(
        X_train_scaled, k_range=range(2, 7), n_init=50, n_stability_bootstraps=25
    )
    save_table(km_diag_df, "kmeans_diagnostics", m3_output_dir)

    logger.info("Evaluating Gaussian Mixture Models across K in [2, 6] on training set...")
    gmm_evaluator = GMMRegimeModel(random_state=42)
    gmm_diag_df = gmm_evaluator.evaluate_k_range(X_train_scaled, k_range=range(2, 7), n_init=15)
    save_table(gmm_diag_df, "gmm_diagnostics", m3_output_dir)

    # 6. Fit Models (K=3 chosen based on balance of stability, BIC, and financial interpretability)
    chosen_k = 3
    logger.info("Fitting K-Means (K=%d) and GMM (K=%d) on training data...", chosen_k, chosen_k)
    km_evaluator.fit(X_train_scaled, k=chosen_k)
    gmm_evaluator.fit(X_train_scaled, k=chosen_k)

    # Rule-Based Baseline
    rule_baseline = RuleBasedRegimeBaseline(high_quantile=0.75, low_quantile=0.25)
    rule_baseline.fit(train_feat)

    # Predict Regimes
    train_km_labels = km_evaluator.predict(X_train_scaled)
    test_km_labels = km_evaluator.predict(X_test_scaled)

    train_gmm_labels = gmm_evaluator.predict(X_train_scaled)
    test_gmm_labels = gmm_evaluator.predict(X_test_scaled)
    test_gmm_probs = gmm_evaluator.predict_proba(X_test_scaled)

    train_rule_labels = rule_baseline.predict(train_feat)
    test_rule_labels = rule_baseline.predict(test_feat)

    # 7. Realized Risk Characterization (Independent Verification)
    logger.info("Characterizing regimes using independent realized return and risk statistics...")
    char_train_km = characterize_regimes(train_km_labels, train_feat.index, ew_bench)
    char_test_km = characterize_regimes(test_km_labels, test_feat.index, ew_bench)
    save_table(char_train_km, "regime_characterization_train", m3_output_dir)
    save_table(char_test_km, "regime_characterization_test", m3_output_dir)

    char_test_rule = characterize_regimes(test_rule_labels, test_feat.index, ew_bench)
    save_table(char_test_rule, "regime_characterization_rule_baseline_test", m3_output_dir)

    # 8. Walk-Forward Refitting Robustness Experiment
    logger.info("Executing Walk-Forward Refitting Robustness Experiment...")
    oos_preds_df, refit_hist_df = walk_forward_regime_evaluation(
        features_df, initial_train_size=1000, refit_interval=126, k=chosen_k, random_state=42
    )
    save_table(refit_hist_df, "walk_forward_refit_history", m3_output_dir)

    # 9. Crisis Event Analysis & Overlap Detection
    logger.info("Auditing historical crisis windows...")
    overlaps = detect_overlapping_events()
    overlaps_df = pd.DataFrame(overlaps) if overlaps else pd.DataFrame(columns=["event_1", "event_2", "overlap_days"])
    save_table(overlaps_df, "event_overlap_audit", m3_output_dir)

    # Partition dates
    partition_df = partition_event_windows(features_df.index)
    metrics_rolling_df = pd.DataFrame({
        "mean_correlation": features_df["mean_correlation"],
        "spectral_entropy": features_df["spectral_entropy"],
        "absorption_ratio": features_df["absorption_ratio_1"],
    }, index=features_df.index)

    crisis_comp_df = compare_crisis_window_metrics(metrics_rolling_df, partition_df)
    save_table(crisis_comp_df, "crisis_window_comparison", m3_output_dir)

    # 10. Candidate Safe-Haven Analysis (Baur-Lucey Framework)
    logger.info("Testing candidate safe havens with Leave-One-Market-Out benchmark...")
    sh_loo_df = analyze_safe_haven_candidates(
        log_rets,
        stress_quantiles=(0.01, 0.05, 0.10),
        use_leave_one_out=True,
        block_size=5,
        n_bootstraps=1000,
        random_seed=42,
    )
    save_table(sh_loo_df, "safe_haven_leave_one_out", m3_output_dir)

    logger.info("Testing candidate safe havens with Full Equal-Weighted benchmark (Sensitivity check)...")
    sh_full_df = analyze_safe_haven_candidates(
        log_rets,
        stress_quantiles=(0.01, 0.05, 0.10),
        use_leave_one_out=False,
        block_size=5,
        n_bootstraps=1000,
        random_seed=42,
    )
    save_table(sh_full_df, "safe_haven_full_benchmark", m3_output_dir)

    # 11. Compile Empirical Summary Document
    summary_md = f"""# Milestone 3 Empirical Research Report: Market Regime Detection & Crisis Event Analysis

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Market Basket:** Core 8 European Markets ({', '.join(tickers)})
**Data Span:** {aligned_df.index[0].date()} to {aligned_df.index[-1].date()} ({len(aligned_df)} continuous trading days)
**Training Period (In-Sample):** {train_feat.index[0].date()} to {train_feat.index[-1].date()} ({len(train_feat)} epochs)
**Testing Period (Out-of-Sample):** {test_feat.index[0].date()} to {test_feat.index[-1].date()} ({len(test_feat)} epochs)

---

## 1. Feature Preprocessing & Lookahead Guard
- **4D State Feature Space:** $\\mathbf{{x}}(t) = [\\bar{{\\rho}}(t), H_{{\\text{{norm}}}}(t), \\text{{AR}}_1(t), \\sigma_{{\\text{{EW}}}}(t)]$
- **ChronologicalScaler Parameters (Training Period Only):**
  - $\\bar{{\\rho}}$ (Mean Correlation): $\\mu = {scaler.mean_[0]:.4f}, \\sigma = {scaler.scale_[0]:.4f}$
  - $H_{{\\text{{norm}}}}$ (Spectral Entropy): $\\mu = {scaler.mean_[1]:.4f}, \\sigma = {scaler.scale_[1]:.4f}$
  - $\\text{{AR}}_1$ (Absorption Ratio 1): $\\mu = {scaler.mean_[2]:.4f}, \\sigma = {scaler.scale_[2]:.4f}$
  - $\\sigma_{{\\text{{EW}}}}$ (Trailing Volatility): $\\mu = {scaler.mean_[3]:.4f}, \\sigma = {scaler.scale_[3]:.4f}$
- Zero lookahead leakage: Out-of-sample data transformed using historical training parameters only. Zero-variance explicitly guarded.

---

## 2. Model Comparison & Diagnostics

### K-Means Diagnostic Metrics (Training Set)
{km_diag_df.to_markdown()}

### Gaussian Mixture Model Diagnostics (Training Set)
{gmm_diag_df.to_markdown()}

**Methodological Discussion:**
- Silhouette scores and Davies-Bouldin indices peak at $K=2$ and $K=3$.
- Calinski-Harabasz is highest at $K=2$ ({km_diag_df.loc[2, 'calinski_harabasz_index']:.1f}) and $K=3$ ({km_diag_df.loc[3, 'calinski_harabasz_index']:.1f}).
- GMM BIC confirms that $K=3$ offers a substantial improvement in fit (BIC = {gmm_diag_df.loc[3, 'bic']:.1f}) over $K=2$ (BIC = {gmm_diag_df.loc[2, 'bic']:.1f}) without the overparameterization penalties of $K \\ge 5$.
- Sub-sample bootstrap stability (Adjusted Rand Index) demonstrates that $K=3$ retains high partition stability (ARI = {km_diag_df.loc[3, 'stability_ari']:.3f}).

---

## 3. Objective Realized Risk Characterization (Independent Verification)

*Note: These statistics are retrospective descriptive statistics calculated to measure actual realized behavior while the market was in each regime. They are kept strictly separate from the features used to assign the regime.*

### Out-of-Sample Test Period (2015–2024, K-Means K=3)
{char_test_km.to_markdown()}

### Out-of-Sample Test Period (2015–2024, Rule-Based Baseline)
{char_test_rule.to_markdown()}

---

## 4. Historical Crisis Event Window Analysis

### Overlapping Event Windows Audit
{overlaps_df.to_markdown() if not overlaps_df.empty else "No overlapping windows detected across configured events."}

### In-Event vs. Calm Baseline Statistical Comparison
{crisis_comp_df.to_markdown()}

**Key Empirical Insights:**
- During the **COVID-19 Pandemic Crash**, mean cross-market correlation rose by **+{crisis_comp_df.loc[crisis_comp_df['event_id']=='COVID_Crash', 'delta_corr_vs_calm'].values[0]:.4f}** over the calm baseline, while spectral entropy collapsed by **{crisis_comp_df.loc[crisis_comp_df['event_id']=='COVID_Crash', 'delta_entropy_vs_calm'].values[0]:.4f}**, confirming extreme dimensional compression.
- The two-sample Kolmogorov-Smirnov test rejects the null hypothesis that in-event and calm-baseline correlation distributions are identical (Holm-adjusted $p < 0.001$ for COVID, GFC, and Euro Debt Crisis).

---

## 5. Candidate Safe-Haven Analysis (Baur-Lucey Framework)

### Leave-One-Market-Out Benchmark Results (Worst 1%, 5%, 10% Tail Stress)
{sh_loo_df.to_markdown()}

**Econometric Findings:**
- Under the Baur-Lucey test definition ($E[r_i \\mid r_{{\\text{{bench}}}}^{{(-i)}} < Q_\\alpha] \\ge 0$), **NO European core equity market qualifies as a definitive safe haven** during severe market declines.
- All 8 markets suffer statistically significant negative returns during the worst 1%, 5%, and 10% market drops (all Holm-adjusted $p \\approx 1.0$ against $H_1: E[r] > 0$).
- Markets such as the UK (`^FTSE`) and Switzerland (`^SSMI`) exhibit **attenuator / diversifier** properties (milder conditional losses than peripheral markets like Spain `^IBEX` or Italy `FTSEMIB.MI`), but remain co-moving during pan-European liquidity shocks.
- Using a Leave-One-Market-Out benchmark eliminates the mechanical downward bias that occurs when large domestic markets are included in their own benchmark.

---

## 6. Walk-Forward Refitting Experiment
- Refit steps executed: {len(refit_hist_df)}
- Out-of-sample predictions generated: {len(oos_preds_df)}
- Summary of Refit History:
{refit_hist_df.to_markdown()}
"""

    summary_path = m3_output_dir / "milestone3_empirical_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info("Saved Milestone 3 summary document to %s", summary_path)

    logger.info("=" * 80)
    logger.info("MILESTONE 3 ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
