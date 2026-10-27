"""Streamlit Interactive Dashboard & Application Entrypoint.

European Financial Market Coupling, Decoupling, and Crisis-State Analysis
Milestone 2: Rolling Correlations, Spectral Entropy, and Absorption Dynamics.
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from config.settings import (
    CORE_MARKETS,
    EXTENDED_MARKETS,
    HISTORICAL_CRISIS_WINDOWS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
)
from src.data_loader import load_market_data, DataLoader
from src.preprocessing import align_market_calendars, inspect_data_quality
from src.returns import calculate_log_returns, compute_annualized_summary, compute_equal_weighted_index
from src.correlation import (
    compute_rolling_correlation_matrices,
    power_map_correlation_matrix,
    compute_epoch_distance_matrix,
)
from src.metrics import (
    compute_rolling_metrics,
    compute_average_correlation,
    compute_spectral_entropy,
    compute_absorption_ratio,
)
from src.market_regimes import (
    extract_regime_features,
    ChronologicalScaler,
    split_chronological,
    KMeansRegimeModel,
    GMMRegimeModel,
    RuleBasedRegimeBaseline,
    characterize_regimes,
)
from src.crisis_analysis import (
    STRUCTURED_EVENT_MANIFEST,
    detect_overlapping_events,
    partition_event_windows,
    compute_leave_one_out_benchmarks,
    analyze_safe_haven_candidates,
    compare_crisis_window_metrics,
)
from src.visualization import (
    plot_correlation_heatmap,
    plot_spectral_phase_space,
    plot_regime_timeline,
    plot_safe_haven_scatter,
    plot_transition_matrix,
)

st.set_page_config(
    page_title="European Market Coupling & Crisis Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("🇪🇺 European Financial Market Coupling & Crisis Observatory")
st.markdown(
    """
    **A statistically rigorous quantitative finance framework.**  
    Empirical investigation of dynamic market coupling, cross-border correlation regimes, 
    spectral entropy transitions, and systemic co-movement across European equity markets.
    """
)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Market & Date Selection")

all_market_options = {**CORE_MARKETS, **EXTENDED_MARKETS}
default_selected = list(CORE_MARKETS.keys())

selected_tickers = st.sidebar.multiselect(
    "Select European Markets:",
    options=list(all_market_options.keys()),
    default=default_selected,
    format_func=lambda x: f"{all_market_options[x]['country']} — {all_market_options[x]['name']} ({x})",
)

col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start Date", value=datetime.date(2010, 1, 1))
end_date = col_d2.date_input("End Date", value=datetime.date.today())

alignment_method = st.sidebar.selectbox(
    "Calendar Alignment Strategy:",
    options=["intersection", "union_drop_all_nan", "ffill_strict"],
    index=0,
    help=(
        "'intersection': Keep only days where all markets are open. Prevents artificial zero returns.\n"
        "'union_drop_all_nan': Keep days where at least one market is open.\n"
        "'ffill_strict': Forward-fill prices for up to 1 day."
    ),
)

st.sidebar.header("📐 Rolling Metric Parameters")
window_m = st.sidebar.slider("Epoch Window M (Days)", min_value=10, max_value=60, value=20, step=5)
stride_delta = st.sidebar.slider("Epoch Stride Δτ (Days)", min_value=1, max_value=5, value=1, step=1)

use_cache = st.sidebar.checkbox("Use Local Raw Data Cache", value=True)

if not selected_tickers or len(selected_tickers) < 2:
    st.warning("Please select at least two European markets from the sidebar to analyze cross-market coupling.")
    st.stop()

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
with st.spinner("Acquiring and synchronizing European market data..."):
    try:
        raw_df, metadata = load_market_data(
            tickers=selected_tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            use_cache=use_cache,
        )
        aligned_df, align_rep = align_market_calendars(
            raw_df,
            method=alignment_method,
        )
        log_returns = calculate_log_returns(aligned_df)
        equal_weight_bench = compute_equal_weighted_index(log_returns)
        annualized_stats = compute_annualized_summary(log_returns)

        # Rolling Correlation & Metrics
        corr_mats, epoch_dates, asset_names = compute_rolling_correlation_matrices(
            log_returns,
            window=window_m,
            stride=stride_delta,
        )
        metrics_df = compute_rolling_metrics(corr_mats, epoch_dates, top_k_absorption=2)
    except Exception as exc:
        st.error(f"Error during data acquisition/processing: {exc}")
        st.stop()

# -----------------------------------------------------------------------------
# Dashboard Layout
# -----------------------------------------------------------------------------
tab_metrics, tab_regimes, tab_crisis, tab_robustness, tab_correlations, tab_overview, tab_returns, tab_quality, tab_metadata = st.tabs([
    "🌌 Spectral Entropy & Coupling",
    "🔮 Market Regimes & Clustering",
    "🛡️ Crisis Windows & Safe Havens",
    "⚖️ Statistical Robustness & Sensitivity",
    "🔬 Correlation Matrix Trajectory",
    "📊 Market Overview",
    "📈 Returns & Risk",
    "🔍 Data Quality Audit",
    "📋 Instrument Manifest",
])


with tab_metrics:
    st.subheader("Market-Wide Coupling Dynamics: Entropy H(τ) vs Mean Correlation μ(τ)")
    st.markdown(
        r"""
        - **Mean Correlation μ(τ):** Average of all off-diagonal pairwise correlations. Rises when markets co-move collectively.
        - **Normalized Spectral Entropy H(τ):** Von Neumann entropy of normalized correlation eigenvalues $p_i = \lambda_i / N$.  
          - $H \to 1.0$: Highly diversified, multi-factor market regime (risks dispersed across independent dimensions).  
          - $H \to 0.0$: Collective coupling / collapse into a single dominant factor ($\lambda_1 \approx N$, systemic crash).
        """
    )

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    latest_mu = metrics_df["mean_correlation"].iloc[-1]
    latest_h = metrics_df["spectral_entropy"].iloc[-1]
    latest_ar1 = metrics_df["absorption_ratio_k1"].iloc[-1]
    latest_ar2 = metrics_df["absorption_ratio_k2"].iloc[-1]

    col_m1.metric("Latest Mean Corr μ(τ)", f"{latest_mu:.3f}")
    col_m2.metric("Latest Spectral Entropy H_norm(τ)", f"{latest_h:.3f}")
    col_m3.metric("Absorption Ratio AR₁ (Top Eigenvalue)", f"{latest_ar1:.1%}")
    col_m4.metric("Absorption Ratio AR₂ (Top 2 Eigenvalues)", f"{latest_ar2:.1%}")

    # Plot metrics time series
    st.line_chart(metrics_df[["mean_correlation", "spectral_entropy"]])
    
    st.subheader("Absorption Ratio Dynamics (Variance Fraction in Latent Factors)")
    st.line_chart(metrics_df[["absorption_ratio_k1", "absorption_ratio_k2"]])

    st.subheader("Phase Separation Space: Spectral Entropy H(τ) vs Mean Correlation μ(τ)")
    st.caption("Visualizes market trajectories through Normalized Spectral Entropy and Mean Pairwise Correlation.")
    if not metrics_df.empty:
        fig_phase = plot_spectral_phase_space(metrics_df)
        st.pyplot(fig_phase)
        plt.close(fig_phase)

with tab_regimes:
    st.subheader("🔮 Market Regime Detection & Clustering (Lookahead-Free)")
    st.markdown(
        r"""
        **Methodological Design:**
        - **Approved 4D State Space:** $\mathbf{x}(t) = [\bar{\rho}(t), H_{\text{norm}}(t), \text{AR}_1(t), \sigma_{\text{EW}}(t)]$  
          *(Mean off-diagonal correlation, normalized spectral entropy, absorption ratio 1, trailing benchmark volatility)*
        - **Strictly Chronological Training/Testing:** Scaler mean and standard deviation are fitted **strictly on the historical training period** and applied forward. Zero lookahead leakage.
        - **Independent Characterization:** Discovered states are characterized using independent realized return, volatility, max drawdown, downside semi-variance, VaR, and Expected Shortfall.
        """
    )

    m3_out = Path(__file__).resolve().parent / "outputs" / "milestone3"
    m4_out = Path(__file__).resolve().parent / "outputs" / "milestone4"
    km_diag_file = m3_out / "kmeans_diagnostics.csv"
    gmm_diag_file = m3_out / "gmm_diagnostics.csv"
    m4_char_file = m4_out / "high_vol_regime_characterization_test.csv"
    char_test_file = m3_out / "regime_characterization_test.csv"
    char_rule_file = m3_out / "regime_characterization_rule_baseline_test.csv"

    if km_diag_file.exists() and gmm_diag_file.exists():
        km_diag_df = pd.read_csv(km_diag_file, index_col=0)
        gmm_diag_df = pd.read_csv(gmm_diag_file, index_col=0)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("### K-Means Diagnostic Metrics across K ∈ [2, 6]")
            st.dataframe(
                km_diag_df.style.format({
                    "inertia": "{:.1f}",
                    "silhouette_score": "{:.3f}",
                    "davies_bouldin_index": "{:.3f}",
                    "calinski_harabasz_index": "{:.1f}",
                    "stability_ari": "{:.3f}",
                }),
                use_container_width=True,
            )
            st.line_chart(km_diag_df[["silhouette_score", "davies_bouldin_index"]])

        with col_r2:
            st.markdown("### Gaussian Mixture Model Information Criteria")
            st.dataframe(
                gmm_diag_df.style.format({
                    "bic": "{:.1f}",
                    "aic": "{:.1f}",
                    "log_likelihood": "{:.3f}",
                }),
                use_container_width=True,
            )
            st.line_chart(gmm_diag_df[["bic", "aic"]])

        st.markdown("---")
        st.subheader("Objective Realized Risk Profile (Out-of-Sample Test Period 2015–2024)")
        st.caption("Realized metrics measured while the market occupied each state. Kept strictly separate from the features used to assign the regime.")

        if m4_char_file.exists():
            char_m4_df = pd.read_csv(m4_char_file, index_col=0)
            st.markdown("#### K-Means (K=3) Discovered Regimes — Rigorous Decomposition (Milestone 4)")
            st.dataframe(
                char_m4_df[[
                    "Sample_Pct", "Arithmetic_Annualized_Return", "Geometric_Annualized_Return",
                    "Annualized_Volatility", "Volatility_Drag", "Positive_Days_Pct",
                    "Contiguous_Max_Drawdown", "Concatenated_Max_Drawdown", "Forward_1D_Annualized_Return",
                    "VaR_0.05", "Expected_Shortfall_0.05"
                ]].style.format({
                    "Sample_Pct": "{:.1f}%",
                    "Arithmetic_Annualized_Return": "{:+.2%}",
                    "Geometric_Annualized_Return": "{:+.2%}",
                    "Annualized_Volatility": "{:.2%}",
                    "Volatility_Drag": "{:.2%}",
                    "Positive_Days_Pct": "{:.1f}%",
                    "Contiguous_Max_Drawdown": "{:.2%}",
                    "Concatenated_Max_Drawdown": "{:.2%}",
                    "Forward_1D_Annualized_Return": "{:+.2%}",
                    "VaR_0.05": "{:.2%}",
                    "Expected_Shortfall_0.05": "{:.2%}",
                }),
                use_container_width=True,
            )

            with st.expander("🔍 Deep-Dive: Investigation of Positive Return in High-Volatility State (+13.96%)"):
                st.markdown(
                    r"""
                    **Root Cause & Mathematical Verification:**
                    1. **Contemporaneous vs Forward:** State 1 (Turbulent Stress) is identified by trailing 20-day volatility and correlation. Because high volatility persists for weeks after market crashes, State 1 encompasses both the initial crash AND the explosive policy-relief rebound rallies.
                    2. **Explosive Rebounds Inside State 1:** Over 57% of days in State 1 are positive. The single largest up-days in European market history occurred while the market occupied State 1 (e.g. +8.24% on 2020-03-24 upon Fed liquidity backstops, +5.42% on 2022-03-09, +5.11% on 2020-11-09).
                    3. **Severe Volatility Drag:** Compounding drag ($\frac{1}{2}\sigma^2 = 3.87\%$) reduces the geometric compound return to **+10.55%**.
                    4. **Contiguous Episode Drawdown:** Within unbroken contiguous crisis stretches (such as Feb–Jul 2020), the portfolio suffered a catastrophic **-37.29% contiguous drawdown**.
                    """
                )
                rebound_file = m4_out / "state1_top_rebound_days.csv"
                if rebound_file.exists():
                    st.dataframe(pd.read_csv(rebound_file, index_col=0), use_container_width=True)

            strat_file = m4_out / "hypothetical_strategy_evaluation.csv"
            if strat_file.exists():
                with st.expander("📊 Hypothetical Forward Trading Strategy Simulation (Lookahead-Free Execution Timing)"):
                    st.markdown(
                        r"""
                        **Methodological Safeguards (Priority 2):**
                        - **Lookahead-Free Timing:** Features $\mathbf{x}(t)$ and regime $S_t$ are determined at date $t$ market close. Exposure $w_{t+1}$ applies strictly to forward return $r_{t+1}$.
                        - **Exposure Rules:** De-risk to cash ($w = 0.0$) in State 1 (Turbulent Stress); 100% equity in State 0 and State 2.
                        - **Friction:** 5 bps transaction costs per trade switch.
                        - **Academic Disclaimer:** *THIS ANALYSIS IS STRICTLY HYPOTHETICAL AND RETROSPECTIVE. It evaluates signal latency and forward risk mitigation properties, NOT live trading execution or an actionable investment recommendation.*
                        """
                    )
                    st.dataframe(pd.read_csv(strat_file, index_col=0), use_container_width=True)

        elif char_test_file.exists():

            char_test_df = pd.read_csv(char_test_file, index_col=0)
            st.dataframe(char_test_df, use_container_width=True)

        # Hungarian Walk-Forward Diagnostics
        trans_file = m4_out / "walk_forward_transition_matrix.csv"
        pers_file = m4_out / "walk_forward_persistence.csv"
        if trans_file.exists() and pers_file.exists():
            st.markdown("---")
            st.subheader("Walk-Forward Sequential Refitting & Hungarian Label Alignment")
            st.caption("Walk-forward refitting with Hungarian centroid matching (scipy.optimize.linear_sum_assignment) prevents sequential label switching.")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.markdown("#### Out-of-Sample Empirical Transition Matrix $P(S_{t+1}=j \mid S_t=i)$")
                trans_df = pd.read_csv(trans_file, index_col=0)
                fig_trans = plot_transition_matrix(trans_df)
                st.pyplot(fig_trans)
                plt.close(fig_trans)
                st.dataframe(trans_df.style.format("{:.2%}"), use_container_width=True)
            with col_w2:
                st.markdown("#### Regime Persistence & Episode Durations")
                st.dataframe(pd.read_csv(pers_file, index_col=0).style.format({
                    "Occupancy_Pct": "{:.1f}%",
                    "Mean_Duration_Days": "{:.1f}d",
                    "Median_Duration_Days": "{:.1f}d",
                }), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive Regime Timeline & Benchmark Trajectory")
        st.caption("Dynamic K-Means (K=3) state classification across the active date range with historical crisis overlays.")
        if len(corr_mats) >= 30:
            try:
                feat_df = extract_regime_features(corr_mats, epoch_dates, equal_weight_bench, window=window_m)
                km_model = KMeansRegimeModel(n_clusters=3, random_state=42)
                scaler_dyn = ChronologicalScaler()
                feat_scaled = scaler_dyn.fit_transform(feat_df)
                pred_regimes = pd.Series(km_model.fit_predict(feat_scaled), index=feat_df.index)
                fig_timeline = plot_regime_timeline(
                    pred_regimes,
                    benchmark_series=equal_weight_bench,
                    crisis_events=HISTORICAL_CRISIS_WINDOWS,
                    title=f"Discovered Macro Regimes (K=3) & Benchmark Trajectory ({start_date} to {end_date})",
                )
                st.pyplot(fig_timeline)
                plt.close(fig_timeline)
            except Exception as exc:
                st.caption(f"Timeline plot unavailable: {exc}")
    else:
        st.info("Run `python scripts/run_milestone4_audit.py` to generate complete empirical regime outputs.")

with tab_crisis:
    st.subheader("🛡️ Historical Crisis Windows & Safe-Haven Econometric Audit")
    st.markdown(
        r"""
        **Methodological & Econometric Audit (Milestone 4):**
        - **Baur & Lucey (2010) Exact Definition:** An asset is a safe haven if it is uncorrelated or negatively correlated with the market benchmark during tail stress ($\rho_{\text{stress}} \le 0$).
        - **Capital Preservation:** An asset protects capital if its conditional expected return is non-negative ($E[r_i \mid \text{stress}] \ge 0$).
        - **Leave-One-Market-Out Benchmark:** Asset $i$'s benchmark excludes asset $i$ to prevent mechanical self-correlation.
        - **Dependence-Aware Inference:** Moving-block bootstrap ($L=20$ days) accounts for $\text{MA}(19)$ autocorrelation from overlapping rolling windows. Ordinary KS p-values are retained strictly as supplementary descriptive diagnostics.
        """
    )

    m4_out = Path(__file__).resolve().parent / "outputs" / "milestone4"
    crisis_m4_file = m4_out / "crisis_window_dependence_aware.csv"
    sh_audit_file = m4_out / "safe_haven_audit_leave_one_out.csv"

    if crisis_m4_file.exists():
        crisis_df = pd.read_csv(crisis_m4_file, index_col=0)
        st.markdown("### Dependence-Aware Crisis Window Comparison (Moving-Block Bootstrap)")
        st.dataframe(
            crisis_df[[
                "event_name", "start", "end", "in_event_days", "pre_event_status",
                "in_mean_corr", "delta_corr_vs_calm", "delta_corr_ci_95_lower", "delta_corr_ci_95_upper",
                "holm_adj_boot_p_value", "delta_entropy_vs_calm", "ks_p_value_descriptive_only"
            ]].style.format({
                "in_mean_corr": "{:.3f}",
                "delta_corr_vs_calm": "{:+.3f}",
                "delta_corr_ci_95_lower": "{:+.3f}",
                "delta_corr_ci_95_upper": "{:+.3f}",
                "holm_adj_boot_p_value": "{:.4f}",
                "delta_entropy_vs_calm": "{:+.3f}",
                "ks_p_value_descriptive_only": "{:.4f}",
            }),
            use_container_width=True,
        )

    if sh_audit_file.exists():
        sh_df = pd.read_csv(sh_audit_file, index_col=0)
        st.markdown("### Candidate Safe-Haven Audit (Leave-One-Market-Out Benchmark)")
        
        col_sh1, col_sh2 = st.columns([1, 1])
        with col_sh1:
            q_sel = st.selectbox("Select Tail Stress Threshold:", ["1%", "5%", "10%"], index=0)
            m_sel = st.selectbox("Select Market to Visualize:", selected_tickers, index=0)
            q_val = 0.01 if q_sel == "1%" else (0.05 if q_sel == "5%" else 0.10)
            sh_sub = sh_df[sh_df["quantile_threshold"] == q_sel]

            st.dataframe(
                sh_sub[[
                    "market", "cutoff_return", "n_stress_obs", "mean_return",
                    "corr_stress", "corr_stress_ci_lower", "corr_stress_ci_upper",
                    "corr_calm", "delta_corr", "baur_lucey_status", "capital_preservation_status"
                ]].style.format({
                    "cutoff_return": "{:.2%}",
                    "mean_return": "{:.2%}",
                    "corr_stress": "{:.3f}",
                    "corr_stress_ci_lower": "{:.3f}",
                    "corr_stress_ci_upper": "{:.3f}",
                    "corr_calm": "{:.3f}",
                    "delta_corr": "{:+.3f}",
                }),
                use_container_width=True,
            )

        with col_sh2:
            try:
                loo_benchmarks = compute_leave_one_out_benchmarks(log_returns)
                if m_sel in loo_benchmarks:
                    fig_sh = plot_safe_haven_scatter(
                        log_returns[m_sel],
                        loo_benchmarks[m_sel],
                        stress_quantile=q_val,
                        market_name=all_market_options[m_sel]["country"],
                    )
                    st.pyplot(fig_sh)
                    plt.close(fig_sh)
            except Exception as exc:
                st.caption(f"Safe-haven scatter plot unavailable: {exc}")

        st.warning(
            "📌 **Econometric Verdict:** Under Baur & Lucey's (2010) exact definition ($\rho_{\text{stress}} \le 0$), "
            "**NO European core equity market qualifies as a safe haven** during tail stress. All conditional correlations remain strongly positive ($+0.65$ to $+0.92$), "
            "with 95% block-bootstrap confidence intervals bounded well above zero. Defensive markets (Switzerland `^SSMI`, UK `^FTSE`) function as "
            "**attenuators/diversifiers** (milder losses), but remain firmly co-moving. This conclusion is bounded to the 8 core equity indices analyzed."
        )

with tab_robustness:
    st.subheader("⚖️ Methodological Robustness & Sensitivity Audit")
    st.markdown(
        """
        Comprehensive sensitivity analysis evaluating the stability of empirical conclusions across rolling window lengths, 
        market basket definitions, walk-forward refit horizons, and alternative regime models.
        """
    )

    m4_out = Path(__file__).resolve().parent / "outputs" / "milestone4"
    w_sens_file = m4_out / "window_length_sensitivity.csv"
    b_sens_file = m4_out / "market_basket_sensitivity.csv"
    wf_sens_file = m4_out / "walk_forward_sensitivity.csv"
    mod_comp_file = m4_out / "model_comparison_summary.csv"

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if w_sens_file.exists():
            st.markdown("#### Rolling Window Parameter Sensitivity (M ∈ [15, 20, 30, 45, 60] Days)")
            st.dataframe(pd.read_csv(w_sens_file, index_col=0).style.format({
                "mean_correlation": "{:.3f}",
                "correlation_std": "{:.3f}",
                "mean_spectral_entropy": "{:.3f}",
                "mean_absorption_ratio": "{:.3f}",
                "state1_turbulent_vol": "{:.2%}",
                "state2_calm_vol": "{:.2%}",
            }), use_container_width=True)

    with col_s2:
        if b_sens_file.exists():
            st.markdown("#### Market Basket Sensitivity (Core 8 vs Core 8 + Sweden post-Nov 2008)")
            st.dataframe(pd.read_csv(b_sens_file, index_col=0).style.format({
                "mean_pairwise_corr": "{:.3f}",
                "mean_spectral_entropy": "{:.3f}",
                "mean_absorption_ratio": "{:.3f}",
                "annualized_volatility": "{:.2%}",
            }), use_container_width=True)

    col_s3, col_s4 = st.columns(2)
    with col_s3:
        if wf_sens_file.exists():
            st.markdown("#### Walk-Forward Horizon Sensitivity")
            st.dataframe(pd.read_csv(wf_sens_file, index_col=0).style.format({
                "mean_training_silhouette": "{:.3f}",
                "state0_occupancy_pct": "{:.1f}%",
                "state1_occupancy_pct": "{:.1f}%",
                "state2_occupancy_pct": "{:.1f}%",
                "state1_mean_duration_days": "{:.1f}d",
            }), use_container_width=True)

    with col_s4:
        if mod_comp_file.exists():
            st.markdown("#### Model Comparison Summary (K-Means vs GMM vs Rule-Based)")
            st.dataframe(pd.read_csv(mod_comp_file, index_col=0), use_container_width=True)

    st.markdown("---")
    st.markdown("### Structured Robustness Classification of Findings")
    st.markdown(
        """
        | Finding / Conclusion | Robustness Status | Evidence & Stability across Parameters | Methodological Dependency / Caveats |
        |---|:---:|---|---|
        | **Dimensional Collapse during Systemic Crises** | **ROBUST** | Stable across all window lengths (M=15 to 60), all models, and all basket specifications. Spectral entropy consistently drops by 30–60% during GFC, COVID, and Brexit. | Requires calendar intersection alignment to avoid artificial zero returns. |
        | **Absence of Equity Safe Havens in Core Europe** | **ROBUST** | Stable across all tested stress thresholds (1%, 5%, 10%), both benchmark definitions (Leave-One-Out vs Equal-Weighted), and all 8 markets. All rho_stress strictly positive (> 0.65). | Restricted to equity indices; non-equity safe havens (Bunds, Gold) not tested. |
        | **Pre-2004 vs Post-2004 Coupling Shift** | **ROBUST** | Both walk-forward refitting and crisis window analyses show European cross-market correlation was significantly lower during Dot-Com (0.68) than post-2007 crises (0.85–0.93). | Dependent on historical availability of cash indices. |
        | **High-Volatility State Positive Compounded Return** | **SENSITIVE / CONDITIONAL** | Arithmetic return is positive (+13.96%) due to violent policy rebound days, but compound geometric return suffers heavy volatility drag (10.55% vs 13.96%). Maximum drawdown within contiguous episodes reaches -37.29%. | Sensitive to whether returns are measured contemporaneously or forward (r_{t+1}). |
        | **Optimal Cluster Count (K=2 vs K=3 vs K=4)** | **METHODOLOGY-DEPENDENT** | Silhouette score peaks at K=2 (0.473), Davies-Bouldin is best at K=3 (0.768) and K=4 (0.701), while GMM BIC favors K=3 (-14,780.9). | Selecting K=3 balances likelihood fit against overparameterization and provides meaningful economic separation. |
        | **Exact Crisis Alert Lead Times** | **INCONCLUSIVE / NON-PREDICTIVE** | Changes in entropy or correlation before historical crisis dates represent retrospective occupancy, not genuine out-of-sample early warnings. | Do not claim predictive forecasting without forward horizon sequential testing. |
        """
    )

with tab_correlations:

    st.subheader("Pairwise Correlation Matrix Inspection")
    if len(epoch_dates) > 0:
        date_selection = st.select_slider(
            "Select Observation Date:",
            options=epoch_dates,
            value=epoch_dates[-1],
            format_func=lambda d: str(d.date()),
        )
        date_idx = list(epoch_dates).index(date_selection)
        C_selected = corr_mats[date_idx]

        corr_display_df = pd.DataFrame(C_selected, index=asset_names, columns=asset_names)
        
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            st.markdown(f"**Correlation Matrix for Epoch Ending: {date_selection.date()}** (Window = {window_m}d)")
            fig_corr = plot_correlation_heatmap(
                corr_display_df,
                title=f"Pairwise Correlation Matrix ({date_selection.date()}, M={window_m}d)",
            )
            st.pyplot(fig_corr)
            plt.close(fig_corr)
            st.dataframe(corr_display_df.style.background_gradient(cmap="coolwarm", vmin=-1.0, vmax=1.0).format("{:.3f}"), use_container_width=True)

        with col_c2:
            st.markdown("**Epoch Eigenvalue Spectrum:**")
            ev_selected = np.sort(np.linalg.eigvalsh(C_selected))[::-1]
            ev_df = pd.DataFrame({
                "Rank": [f"λ_{i+1}" for i in range(len(ev_selected))],
                "Eigenvalue": ev_selected,
                "Variance Share": ev_selected / len(ev_selected),
            })
            st.dataframe(ev_df.style.format({"Eigenvalue": "{:.3f}", "Variance Share": "{:.1%}"}), use_container_width=True)

with tab_overview:
    st.subheader("Normalized Cumulative Performance (Base = 100)")
    normalized_prices = (aligned_df / aligned_df.iloc[0]) * 100.0
    st.line_chart(normalized_prices)

    st.subheader("Equal-Weighted European Benchmark Index (Synthetic)")
    st.caption("Constructed as the daily arithmetic mean return of selected European markets. Rebalanced daily to equal weights.")
    bench_cum = (1.0 + (equal_weight_bench.cumsum())).apply(lambda x: max(x, 0)) * 100.0
    st.line_chart(bench_cum)

with tab_returns:
    st.subheader("Annualized Summary Statistics (252 Trading Days)")
    st.dataframe(
        annualized_stats.style.format({
            "Annualized Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Annualized Sharpe (Rf=0)": "{:.2f}",
            "Daily Skewness": "{:.3f}",
            "Daily Excess Kurtosis": "{:.3f}",
            "Observations": "{:d}",
        }),
        use_container_width=True,
    )

    st.subheader("Daily Logarithmic Returns")
    st.line_chart(log_returns)

with tab_quality:
    st.subheader("Preprocessing & Calendar Alignment Audit")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    q_col1.metric("Raw Calendar Rows", align_rep["initial_rows"])
    q_col2.metric("Aligned Trading Days", align_rep["final_rows"])
    q_col3.metric("Retention Rate", f"{align_rep['retention_rate_pct']:.1f}%")
    q_col4.metric("Synthetic Zeros Introduced", align_rep["synthetic_zero_returns_induced"])

    st.markdown("### Missing Data Diagnostics per Instrument")
    q_info = inspect_data_quality(raw_df)
    q_table = []
    for t in selected_tickers:
        t_stat = q_info["instruments"].get(t, {})
        q_table.append({
            "Ticker": t,
            "Market": all_market_options[t]["country"],
            "Valid Days": t_stat.get("valid_observations", 0),
            "Missing Days": t_stat.get("missing_observations", 0),
            "Missing %": f"{t_stat.get('missing_percentage', 0.0):.2f}%",
            "First Date": t_stat.get("first_valid_date", "N/A"),
            "Last Date": t_stat.get("last_valid_date", "N/A"),
        })
    st.dataframe(pd.DataFrame(q_table), use_container_width=True)

with tab_metadata:
    st.subheader("European Market Manifest & Return Convention Documentation")
    st.info(
        "ℹ️ **Note on Return Conventions:** DAX 40 (`^GDAXI`) is a Total Return (Performance) index that reinvests gross dividends. "
        "Other European benchmarks (`^FTSE`, `^FCHI`, etc.) are Price Return indices. Empirical audit shows correlation between DAX TR and PR is 0.9989, "
        "with an annualized dividend drift of ~2.89%."
    )
    meta_list = []
    for t in selected_tickers:
        m = all_market_options[t]
        meta_list.append({
            "Ticker": t,
            "Name": m.get("name"),
            "Country": m.get("country"),
            "Currency": m.get("currency"),
            "Index Type": m.get("index_type"),
            "Exchange": m.get("exchange"),
            "Coverage Start": m.get("earliest_available"),
            "Notes / Caveats": m.get("notes"),
        })
    st.dataframe(pd.DataFrame(meta_list), use_container_width=True)

    st.subheader("Historical Crisis Windows Configured")
    st.dataframe(pd.DataFrame(HISTORICAL_CRISIS_WINDOWS).T, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Milestone 4 Complete:** Statistical robustness, dependence-aware inference, safe-haven econometric audit, Hungarian walk-forward refitting, and full sensitivity suite.")

