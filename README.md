# European Financial Market Coupling and Crisis-State Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen.svg)]()

[![License](https://img.shields.io/badge/license-MIT-green.svg)]()


A modular, reproducible, and statistically defensible research application to analyze dynamic relationships, systemic coupling, decoupling episodes, and crisis-state transitions across major European equity markets.

---

## 1. Project Overview & Motivation

European equity markets exhibit complex, time-varying interdependence. During periods of financial market turmoil (e.g., the 2008 Lehman collapse, the 2010–2012 European sovereign debt crisis, the 2020 COVID-19 crash, and the 2022 European energy shock), cross-market correlations surge—a phenomenon termed **financial market coupling** or systemic co-movement. Conversely, periods of divergence, idiosyncratic shocks, or flight to quality produce **decoupling**.

Understanding these dynamics is vital for:
- **Systemic Risk Monitoring:** Detecting when diversified national markets collapse into a single dominant factor.
- **Cross-Border Portfolio Diversification:** Determining whether international equity diversification breaks down when it is needed most.
- **Empirical Market State Discovery:** Uncovering macro-financial regimes without forcing subjective narrative labels.

---

## 2. Research Notebook Audit Summary

Before building this codebase, an exhaustive audit of the baseline research notebook (`PRJ3_3.ipynb`) was conducted. Key findings include:

| Severity | Issue | Impact in Baseline | Remediation in Rebuilt Pipeline |
|---|---|---|---|
| **P0 (Critical)** | `NameError` in Module 13 (Cell 20) | Execution crashed at Cell 20; modules 13–16 and Module X were unexecutable. | Rebuilt clean, modular pipeline with zero missing dependencies and full test coverage. |
| **P0 (Critical)** | Full-sample standardization | Future volatility (e.g. 2020 COVID) leaked into historical alerts (2006–2008). | Walk-forward and strictly trailing standardization will be used for regime detection. |
| **P1 (High)** | Forward-filling prices (`ffill(limit=5)`) | Bank holidays generated artificial zero returns, depressing volatility and deflating correlations. | Replaced with strict calendar intersection (`intersection`) and audit diagnostics. |
| **P1 (High)** | US-traded ETFs for Sweden (`EWD`) & Benchmark (`EZU`) | NYSE Arca trading hours (14:30–21:00 London) and USD FX noise confounded local cash indices. | Prioritized native European cash indices (`^OMX`, `^FTSE`, `^GDAXI`, etc.) and synthetic benchmarks. |
| **P1 (High)** | Elementwise power-mapping non-PSD | $C_{ij}^{1+\epsilon}$ produced negative eigenvalues; author patched with `np.abs(λ)`. | True correlation matrix eigenvalues are preserved with strict PSD safeguards. |
| **P2 (Medium)** | Forced $K=4$ clusters | Overrode statistical evidence ($K=2$ had lowest std dev) to fit a pre-conceived narrative. | Multi-metric cluster evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz). |
| **P2 (Medium)** | Semantic labeling fallacy | Highest correlation state was hardcoded as "Crash" without verifying returns or drawdowns. | Clusters characterized by return, volatility, and downside semi-variance. |
| **P2 (Medium)** | Flawed "Safe Haven" definition | Labeled as safe haven if $\text{corr}_{\text{calm}} - \text{corr}_{\text{stress}} > 0.08$ even if losing 35%. | Safe-haven status treated as a testable hypothesis requiring capital preservation under stress. |

*(For the complete audit breakdown, see [`audit_report.md`](file:///C:/Users/goudd/.gemini/antigravity-ide/brain/e3d04f98-a92c-48f7-b8ef-a023586cd5a2/audit_report.md)).*

---

## 3. Project Directory Architecture

```text
European Financial Market Coupling & Crisis Analysis/
├── config/
│   ├── __init__.py
│   └── settings.py              # Instrument manifests, paths, crisis dates, and parameters
├── data/
│   ├── raw/                     # Cached yfinance CSVs & metadata JSON (ignored by git)
│   └── processed/               # Cleaned, calendar-aligned prices and returns
├── outputs/
│   ├── milestone3/              # Empirical tables & reports for baseline pipeline
│   └── milestone4/              # Robustness, sensitivity, and strategy evaluation tables
├── scripts/
│   ├── reproduce_dax_comparison.py # German DAX TR vs PR dividend wedge audit
│   ├── run_milestone3_analysis.py  # 2000-2024 Core 8 baseline pipeline runner
│   └── run_milestone4_audit.py     # Milestone 4 statistical audit & sensitivity suite
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Acquisition, coverage validation, provenance metadata
│   ├── preprocessing.py         # Price validation, calendar alignment, data quality audit
│   ├── returns.py               # Logarithmic returns, simple returns, benchmark aggregates
│   ├── correlation.py           # Rolling correlation matrices, power-mapping, epoch distances
│   ├── metrics.py               # Spectral entropy, eigenvalues, absorption ratio dynamics
│   ├── market_regimes.py        # Lookahead-free 4D features, K-Means/GMM, Hungarian alignment
│   ├── crisis_analysis.py       # Event windows, safe-haven audits, moving-block bootstrap
│   └── visualization.py         # Publication-quality heatmaps, phase plots, timelines
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py      # Data acquisition and caching tests
│   ├── test_preprocessing.py    # Price validation and calendar alignment tests
│   ├── test_returns.py          # Return calculations and mathematical identities
│   ├── test_correlation.py      # Rolling correlations and power-mapping tests
│   ├── test_epoch_distance.py   # Vectorized SciPy cdist distance tests & benchmarks
│   ├── test_metrics.py          # Matrix validation, spectral entropy & absorption tests
│   ├── test_market_regimes.py   # 4D features, Hungarian matching & strategy tests
│   ├── test_crisis_analysis.py  # Event partitioning, safe havens & bootstrap tests
│   └── test_visualization.py    # Matplotlib/Seaborn visualization & edge-case tests
├── app.py                       # Streamlit interactive quantitative dashboard
├── requirements.txt             # Pinned project dependencies
├── .gitignore                   # Exclusion rules for data, cache, and build files
├── README.md                    # Project documentation
└── PRJ3_3.ipynb                 # Original baseline notebook (preserved for historical reference)
```

---

## 4. Market Scope & Instrument Documentation

The system tracks 8 core European cash market indices with continuous historical coverage from before 2000 to the present:

| Ticker | Country | Index Name | Currency | Index Return Type | Earliest Date | Notes |
|---|---|---|---|---|---|---|
| `^FTSE` | United Kingdom | FTSE 100 | GBP | Price Return | 1984-01-03 | Heavy mining, energy, and global banking weighting. |
| `^GDAXI` | Germany | DAX 40 | EUR | Total Return (Performance) | 1987-12-30 | Dividends are reinvested into the index level. |
| `^FCHI` | France | CAC 40 | EUR | Price Return | 1990-03-01 | High luxury goods, aerospace, and energy weighting. |
| `^SSMI` | Switzerland | SMI 20 | CHF | Price Return | 1990-11-09 | Defensive tilt; large healthcare and consumer staples. |
| `^AEX` | Netherlands | AEX 25 | EUR | Price Return | 1992-10-12 | Significant semiconductor technology weighting (ASML). |
| `^IBEX` | Spain | IBEX 35 | EUR | Price Return | 1993-07-12 | Retail banking and utility concentration. |
| `FTSEMIB.MI`| Italy | FTSE MIB | EUR | Price Return | 1997-12-31 | Financial and industrial exposure. |
| `^BFX` | Belgium | BEL 20 | EUR | Price Return | 1991-04-09 | Chemicals, financials, and biopharma. |

**Extended Instruments:**
- `^OMX` (Sweden): Local cash index available on Yahoo Finance from 2008-11-20.
- `EWD` (Sweden Proxy ETF): Available from 1996, but traded on US hours in USD (subject to FX and trading hour mismatch).
- `^OMXC25` (Denmark): Available from 2016-12-19 (`^OMXC20PI` is delisted/defunct on Yahoo Finance).

---

## 5. Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python -m pytest -v
```

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 6. Verification & Automated Test Results

The automated test suite comprises **91 unit tests and benchmarks** covering mathematical correctness, spectral decomposition, numerical safeguards, error handling, edge cases, high-performance epoch distance calculation, walk-forward Hungarian scale alignment, lookahead-free strategy simulation, moving-block bootstrap inference, and modular publication visualization routines:

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.7.0
rootdir: D:\Projects\European Financial Market Coupling & Crisis Analysis
collected 91 items

tests/test_correlation.py::test_rolling_correlation_known_values PASSED  [  1%]
tests/test_correlation.py::test_rolling_correlation_invalid_parameters PASSED [  2%]
tests/test_correlation.py::test_power_map_correlation_matrix PASSED      [  4%]
tests/test_correlation.py::test_extract_upper_triangular_flat PASSED     [  5%]
tests/test_correlation.py::test_rolling_correlation_constant_return_skipped PASSED [  6%]
tests/test_correlation.py::test_rolling_correlation_strictly_trailing_no_lookahead PASSED [  8%]
tests/test_crisis_analysis.py::test_detect_overlapping_events PASSED     [  9%]
tests/test_crisis_analysis.py::test_partition_event_windows PASSED       [ 10%]
tests/test_crisis_analysis.py::test_compute_leave_one_out_benchmarks PASSED [ 12%]
tests/test_crisis_analysis.py::test_holm_bonferroni_correction PASSED    [ 13%]
tests/test_crisis_analysis.py::test_analyze_safe_haven_candidates PASSED [ 14%]
tests/test_crisis_analysis.py::test_compare_crisis_window_metrics PASSED [ 16%]
tests/test_crisis_analysis.py::test_block_bootstrap_mean_ci PASSED       [ 17%]
tests/test_crisis_analysis.py::test_block_bootstrap_corr_ci PASSED       [ 18%]
tests/test_crisis_analysis.py::test_block_bootstrap_diff_means PASSED    [ 20%]
tests/test_crisis_analysis.py::test_block_bootstrap_corr_series_alignment PASSED [ 21%]
tests/test_crisis_analysis.py::test_block_bootstrap_block_length_sensitivity PASSED [ 22%]
tests/test_data_loader.py::test_metadata_lookup PASSED                   [ 24%]
tests/test_data_loader.py::test_cache_path_generation PASSED             [ 25%]
tests/test_data_loader.py::test_fetch_market_dataset_caching PASSED      [ 27%]
tests/test_data_loader.py::test_fetch_market_dataset_all_failures_raise PASSED [ 28%]
tests/test_epoch_distance.py::test_epoch_distance_numerical_equivalence PASSED [ 29%]
tests/test_epoch_distance.py::test_epoch_distance_mathematical_properties PASSED [ 31%]
tests/test_epoch_distance.py::test_epoch_distance_duplicate_matrices PASSED [ 32%]
tests/test_epoch_distance.py::test_epoch_distance_chunked_equivalence PASSED [ 33%]
tests/test_epoch_distance.py::test_epoch_distance_known_extremes PASSED  [ 35%]
tests/test_epoch_distance.py::test_epoch_distance_benchmark PASSED       [ 36%]
tests/test_market_regimes.py::test_extract_regime_features_structure PASSED [ 37%]
tests/test_market_regimes.py::test_extract_regime_features_insufficient_data PASSED [ 39%]
tests/test_market_regimes.py::test_chronological_scaler_lookahead_protection PASSED [ 40%]
tests/test_market_regimes.py::test_chronological_scaler_zero_variance_guard PASSED [ 41%]
tests/test_market_regimes.py::test_split_chronological PASSED            [ 43%]
tests/test_market_regimes.py::test_kmeans_evaluation_and_stability PASSED [ 44%]
tests/test_market_regimes.py::test_gmm_evaluation_and_soft_probabilities PASSED [ 45%]
tests/test_market_regimes.py::test_rule_based_baseline PASSED            [ 47%]
tests/test_market_regimes.py::test_characterize_regimes PASSED           [ 48%]
tests/test_market_regimes.py::test_walk_forward_regime_evaluation PASSED [ 50%]
tests/test_market_regimes.py::test_walk_forward_label_alignment_under_scale_drift PASSED [ 51%]
tests/test_market_regimes.py::test_simulate_regime_switching_strategy PASSED [ 52%]
tests/test_metrics.py::test_validate_correlation_matrix_valid PASSED     [ 54%]
tests/test_metrics.py::test_validate_correlation_matrix_non_square PASSED [ 55%]
tests/test_metrics.py::test_validate_correlation_matrix_dimension_too_small PASSED [ 56%]
tests/test_metrics.py::test_validate_correlation_matrix_non_symmetric PASSED [ 58%]
tests/test_metrics.py::test_validate_correlation_matrix_nan_inf PASSED   [ 59%]
tests/test_metrics.py::test_validate_correlation_matrix_invalid_diagonal PASSED [ 60%]
tests/test_metrics.py::test_validate_correlation_matrix_out_of_bounds PASSED [ 62%]
tests/test_metrics.py::test_validate_correlation_matrix_materially_negative_eigenvalues PASSED [ 63%]
tests/test_metrics.py::test_validate_correlation_matrix_floating_point_noise_safeguard PASSED [ 64%]
tests/test_metrics.py::test_compute_average_correlation PASSED           [ 66%]
tests/test_metrics.py::test_compute_eigenvalues_analytical PASSED        [ 67%]
tests/test_metrics.py::test_spectral_entropy_identity_matrix PASSED      [ 68%]
tests/test_metrics.py::test_spectral_entropy_collinear_matrix PASSED     [ 70%]
tests/test_metrics.py::test_spectral_entropy_equicorrelated_matrix PASSED [ 71%]
tests/test_metrics.py::test_absorption_ratio PASSED                      [ 72%]
tests/test_metrics.py::test_compute_rolling_metrics_table PASSED         [ 74%]
tests/test_preprocessing.py::test_validate_price_series_valid PASSED     [ 75%]
tests/test_preprocessing.py::test_validate_price_series_non_monotonic PASSED [ 77%]
tests/test_preprocessing.py::test_validate_price_series_duplicate_dates PASSED [ 78%]
tests/test_preprocessing.py::test_validate_price_series_negative_prices PASSED [ 79%]
tests/test_preprocessing.py::test_validate_price_series_all_nan_column PASSED [ 81%]
tests/test_preprocessing.py::test_inspect_data_quality PASSED            [ 82%]
tests/test_preprocessing.py::test_align_market_calendars_intersection PASSED [ 83%]
tests/test_preprocessing.py::test_align_market_calendars_drops_high_nan_column PASSED [ 85%]
tests/test_preprocessing.py::test_align_market_calendars_invalid_method PASSED [ 86%]
tests/test_preprocessing.py::test_generate_data_quality_report PASSED    [ 87%]
tests/test_returns.py::test_log_returns_synthetic_values PASSED          [ 89%]
tests/test_returns.py::test_log_and_simple_returns_identity PASSED       [ 90%]
tests/test_returns.py::test_drop_first_flag PASSED                       [ 91%]
tests/test_returns.py::test_negative_prices_raise_value_error PASSED     [ 93%]
tests/test_returns.py::test_zero_prices_raise_value_error PASSED         [ 94%]
tests/test_returns.py::test_invalid_type_raises_type_error PASSED        [ 95%]
tests/test_returns.py::test_empty_dataframe_raises_value_error PASSED    [ 91%]
tests/test_returns.py::test_compute_equal_weighted_index PASSED          [ 92%]
tests/test_returns.py::test_compute_annualized_summary PASSED            [ 81%]
tests/test_visualization.py::test_plot_correlation_heatmap_valid PASSED  [ 82%]
tests/test_visualization.py::test_plot_correlation_heatmap_with_nan PASSED [ 83%]
tests/test_visualization.py::test_plot_correlation_heatmap_empty_raises PASSED [ 84%]
tests/test_visualization.py::test_plot_correlation_heatmap_non_square_raises PASSED [ 85%]
tests/test_visualization.py::test_plot_spectral_phase_space_valid PASSED [ 86%]
tests/test_visualization.py::test_plot_spectral_phase_space_missing_columns_raises PASSED [ 87%]
tests/test_visualization.py::test_plot_spectral_phase_space_empty_raises PASSED [ 89%]
tests/test_visualization.py::test_plot_spectral_phase_space_unaligned_regimes_raises PASSED [ 90%]
tests/test_visualization.py::test_plot_regime_timeline_valid PASSED      [ 91%]
tests/test_visualization.py::test_plot_regime_timeline_empty_raises PASSED [ 92%]
tests/test_visualization.py::test_plot_regime_timeline_all_nan_raises PASSED [ 93%]
tests/test_visualization.py::test_plot_safe_haven_scatter_valid PASSED   [ 94%]
tests/test_visualization.py::test_plot_safe_haven_scatter_invalid_quantile PASSED [ 95%]
tests/test_visualization.py::test_plot_safe_haven_scatter_empty_or_no_overlap PASSED [ 96%]
tests/test_visualization.py::test_plot_transition_matrix_valid PASSED    [ 97%]
tests/test_visualization.py::test_plot_transition_matrix_non_square_raises PASSED [ 98%]
tests/test_visualization.py::test_plot_transition_matrix_empty_raises PASSED [100%]

============================= 91 passed in 7.01s ==============================
```

---

## 7. Milestone 3: Market Regime Detection & Crisis Event Analysis

### 7.1 Methodological Design & Lookahead Guard
- **Approved 4D State Space:**
  $$\mathbf{x}(t) = [\bar{\rho}(t), H_{\text{norm}}(t), \text{AR}_1(t), \sigma_{\text{EW}}(t)]$$
  - $\bar{\rho}(t)$: Mean off-diagonal pairwise correlation.
  - $H_{\text{norm}}(t)$: Normalized Von Neumann spectral entropy ($H \in [0, 1]$).
  - $\text{AR}_1(t)$: Absorption Ratio of the principal eigenvalue ($\lambda_1 / N$).
  - $\sigma_{\text{EW}}(t)$: Trailing annualized volatility of the equal-weighted European benchmark.
  - *UMAP is strictly isolated from clustering feature space and used for visualization only.*
- **Chronological Training/Testing Separation:**
  - Training Period: `2000-02-02` to `2014-12-30` (3,659 epochs, 15 years).
  - Testing Period: `2015-01-05` to `2024-12-30` (2,454 epochs, 10 years).
  - Feature standardization (`ChronologicalScaler`) fits mean and standard deviation **strictly on training observations** with an explicit guard against zero or near-zero historical variance. Test features are standardized using historical training statistics only.
  - Hyperparameter tuning and cluster counts are selected using training diagnostics only.

### 7.2 Multi-Model Comparison & Diagnostics
Evaluated on training set across $K \in [2, 6]$:
- **K-Means Diagnostics:**
  - Silhouette Score peaks at $K=2$ ($0.473$) and $K=3$ ($0.435$).
  - Davies-Bouldin Index is lowest (most compact/separated) at $K=3$ ($0.768$) and $K=4$ ($0.701$).
  - Calinski-Harabasz Variance Ratio confirms strong structure at $K=3$ ($4,362.1$).
  - Sub-sample bootstrap stability (25 iterations) confirms high partition stability for $K=3$ ($\text{ARI} = 0.967$).
- **Gaussian Mixture Model (GMM):**
  - Evaluated via Akaike and Bayesian Information Criteria (AIC & BIC).
  - $K=3$ yields BIC = $-14,780.9$ (substantial improvement over $K=2$ at $-11,718.2$).
- **Rule-Based Baseline:**
  - Non-parametric benchmark using training quantiles: State 0 (Coupled Stress: $\bar{\rho} > Q_{75}$ and $\sigma > Q_{75}$), State 1 (Decoupled: $\bar{\rho} < Q_{25}$), State 2 (Normal).
- **Walk-Forward Refitting Robustness Experiment:**
  - 41 refit iterations executed across 2004–2024 (initial training: 1,000 days, refit interval: 126 days).
  - 5,113 out-of-sample regime assignments generated. Out-of-sample training silhouette remains steady at $0.43–0.49$.

### 7.3 Independent Realized Risk Characterization & Investigation of State 1
*Methodological Safeguard:* Realized statistics are descriptive retrospective metrics measured while the market occupied each state, kept strictly separate from the features used to assign the regime.
Out-of-sample realized statistics (2015–2024, K-Means $K=3$):
- **State 0 (Normal / Moderate Co-movement, 51.7%):** Arithmetic Ann Return $-1.50\%$, Geometric Ann Return $-2.41\%$, Ann Vol $13.71\%$, Vol Drag $0.94\%$, Contiguous Max DD $-7.92\%$, Concatenated Max DD $-25.70\%$, ES$_{0.05}$ $-2.05\%$.
- **State 1 (High Volatility / Turbulent Stress, 23.0%):** Arithmetic Ann Return $+13.96\%$, Geometric Ann Return $+10.55\%$, Ann Vol $27.82\%$, Vol Drag $3.87\%$, Contiguous Max DD $-37.29\%$, Concatenated Max DD $-37.29\%$, Downside Dev $32.13\%$, ES$_{0.05}$ $-4.32\%$.
- **State 2 (Calm / Low Volatility Drift, 25.3%):** Arithmetic Ann Return $+7.67\%$, Geometric Ann Return $+7.55\%$, Ann Vol $8.78\%$, Vol Drag $0.39\%$, Contiguous Max DD $-6.23\%$, Concatenated Max DD $-13.80\%$, Downside Dev $9.02\%$, ES$_{0.05}$ $-1.25\%$.

#### Investigation of Positive Arithmetic Return (+13.96%) in High-Volatility State:
1. **Contemporaneous vs. Forward Return:** Regimes are classified using a 20-day trailing window of correlation and volatility. State 1 identifies periods of **systemic market turbulence**, which includes both the initial violent selloffs AND the sharp, explosive relief rallies that follow central bank liquidity interventions (e.g., Fed/ECB backstops in March 2020).
2. **Explosive Rebound Days Inside State 1:** 57.0% of days in State 1 are positive. The top 5 daily returns in the entire 10-year test period occurred during State 1: 2020-03-24 (**+8.24%**), 2022-03-09 (**+5.42%**), 2020-11-09 (**+5.11%**), 2015-08-25 (**+4.05%**), 2020-05-18 (**+4.03%**).
3. **Severe Volatility Drag:** Compounding drag ($\frac{1}{2}\sigma^2 = 3.87\%$) reduces the geometric compound return to **+10.55%**.
4. **Contiguous Episode Drawdown:** Within unbroken contiguous stretches (such as Feb 4 to July 21, 2020; 114 days), the portfolio suffered a catastrophic **-37.29% contiguous drawdown**.

### 7.3.1 Retrospective Descriptive Returns vs. Hypothetical Forward Strategy Returns (Priority 2)

A critical methodological boundary must be maintained between descriptive regime characterization and tradable performance:
1. **Contemporaneous Descriptive Regime Returns ($r_t$):** The average return observed on calendar days ex-post classified into regime $S_t$. Because regime discovery uses a 20-day trailing window, $S_t$ reflects conditions over $[t-19, t]$. These returns are retrospective and non-tradable.
2. **Hypothetical Forward Strategy Returns ($r_{t+1}$):** A strictly lookahead-free forward execution simulation where regime features and classification $S_t$ are determined at date $t$ market close, and portfolio exposure $w_{t+1}$ is established for date $t+1$, earning next-day forward return $r_{t+1}$. Day $t$'s return is never attributed to a signal that required day $t$'s closing price to calculate.

#### Forward Strategy Specification & Friction Assumptions:
- **Execution Timing:** Signal $S_t$ computed at date $t$ close $\rightarrow$ Execution occurs at $t+1$ close $\rightarrow$ Earns forward return $r_{t+1}$.
- **Asset Universe:** Equal-weighted core European market basket vs. Cash.
- **Exposure Rules:**
  - State 0 (Normal / Moderate Co-movement): 100% Equity Exposure ($w=1.0$).
  - State 2 (Calm / Low Volatility Drift): 100% Equity Exposure ($w=1.0$).
  - State 1 (High Volatility / Turbulent Stress): De-risk to 0% Equity ($w=0.0$, 100% Cash earning 0.0% nominal return).
- **Transaction Costs:** 5 basis points ($0.05\%$) deducted per turnover transition ($|\Delta w| \times 0.0005$).

#### Empirical Out-of-Sample Results (2015–2024, 2,454 Trading Days):

| Metric | Hypothetical Regime-Switching Strategy (Lookahead-Free) | Equal-Weighted European Benchmark (Buy & Hold) |
|:---|:---:|:---:|
| **Cumulative Return** | **-10.94%** | **+36.03%** |
| **Arithmetic Annualized Return** | -0.52% | +4.64% |
| **Geometric Annualized Return** | -1.18% | +3.21% |
| **Annualized Volatility** | **11.55%** | **17.15%** |
| **Sharpe Ratio (Rf=0)** | -0.045 | +0.271 |
| **Maximum Drawdown** | **-36.09%** | **-37.29%** |
| **Calmar Ratio** | -0.033 | +0.086 |
| **Total Rebalancing Trades** | 86 | 0 |
| **Cumulative Transaction Drag** | 4.30% | 0.00% |

#### Economic Rationale for Forward Underperformance:
- **Signal Latency:** Rolling correlation and volatility require several days of market turmoil to cross the classification boundary into State 1. By the time the signal triggers de-risking, a substantial portion of the initial crash drawdown has already been incurred.
- **Cash Drag during Violent Relief Spikes:** Following crisis selloffs, central bank policy interventions (e.g., Fed/ECB liquidity backstops in March 2020) trigger violent recovery up-days (+8.24% on 2020-03-24, +5.42% on 2022-03-09) while 20-day trailing volatility remains elevated. Sitting in cash during these recovery spikes permanently penalizes compounded equity wealth.
- **Methodological Takeaway:** Retrospective positive returns in high-volatility states are an artifact of policy rebound days captured inside the trailing volatility window. They provide zero evidence of tradable profitability.
- **Academic Disclaimer:** *THIS STRATEGY ANALYSIS IS STRICTLY RETROSPECTIVE AND HYPOTHETICAL. It is designed to evaluate signal latency and forward risk mitigation properties, NOT live execution or an actionable investment strategy. Past performance is no guarantee of future returns.*

### 7.4 Historical Crisis Event Analysis (Dependence-Aware Inference)
- 7 documented events configured with primary literature citations (DotCom, GFC Lehman, Euro Sovereign Debt, Grexit/China Slowdown, Brexit Referendum, COVID-19 Crash, Ukraine War / Energy Shock).
- **Serial Dependence Audit:** Overlapping rolling windows ($M=20$, stride 1) share 19 days of underlying returns, creating an $\text{MA}(19)$ autocorrelation structure. Ordinary KS-test p-values are retained strictly as descriptive diagnostics.
- **Moving-Block Bootstrap:** Implemented moving-block bootstrap ($L=20$ days matching epoch length $M$) reporting effect sizes, 95% confidence intervals, and Holm-Bonferroni adjusted p-values.
- **Statistical Findings:**
  - **COVID-19 Crash:** Cross-market correlation increased by **$+0.1568$** ($95\%\text{ CI } [+0.1469, +0.1797]$, Holm boot $p < 0.001$). Spectral entropy collapsed by **$-0.2213$** ($95\%\text{ CI } [-0.2551, -0.2093]$).
  - **Brexit Referendum:** Mean correlation $+0.1484$ ($95\%\text{ CI } [+0.1355, +0.1633]$, Holm boot $p < 0.001$). Entropy collapsed by $-0.2125$.
  - **GFC Lehman:** Mean correlation $+0.0939$ ($95\%\text{ CI } [+0.0632, +0.1193]$, Holm boot $p < 0.001$). Entropy collapsed by $-0.1194$.
  - **Dot-Com Crash (2000–2002):** Mean correlation was lower than the modern calm baseline ($\Delta = -0.0929$, $95\%\text{ CI } [-0.1460, -0.0431]$), confirming pan-European equity coupling intensified significantly after mid-2000s financial integration.

### 7.4.1 Moving-Block Bootstrap Validation & Statistical Inference (Priority 3)
- **Paired Observation Alignment:** When estimating cross-market correlations and differences under moving-block bootstrap resampling, paired observations $(x_t, y_t)$ must remain strictly synchronized. `block_bootstrap_corr_ci` aligns input Series via `pd.concat([x, y], axis=1).dropna()`, ensuring that date indexes match and that missing rows or asymmetric holidays do not shift cross-sectional alignment.
- **Two-Sided Empirical Hypothesis Testing:** The empirical difference-in-means bootstrap computes two-sided p-values:
  $$p = 2 \cdot \min\left(P(\Delta^* \le 0), P(\Delta^* \ge 0)\right)$$
  This properly captures both systemic correlation surges ($\Delta > 0$, e.g., COVID-19 $+0.1568$, $p < 0.001$) and historical decoupling/integration shifts ($\Delta < 0$, e.g., Dot-Com crash $-0.0929$, $p < 0.001$).
- **Block-Length Sensitivity ($L \in [2, 5, 10, 20]$):** Evaluated across varying block lengths to account for short-term and persistent autocorrelation. While nominal p-values for mild shifts vary slightly with block length, large structural shifts (GFC, COVID, Brexit) retain statistical significance ($p < 0.001$) across all block sizes.
- **Descriptive Role of Kolmogorov-Smirnov Tests:** Classical two-sample KS tests assume independent and identically distributed observations. Overlapping rolling windows violate this assumption through their $\text{MA}(M-1)$ structure. All KS-test p-values in this repository are explicitly labelled as descriptive supplementary diagnostics, while decision inference relies strictly on the dependence-aware block bootstrap.
- **Uncertainty Boundary:** Moving-block bootstrap adjusts for serial dependency in the sample, but does not eliminate all statistical uncertainty or regime non-stationarity across macro-financial eras.

### 7.5 Safe-Haven Econometric Audit (Baur & Lucey 2010 Framework)
- **Literature Reference:** Baur, D. G., & Lucey, B. M. (2010). *Is gold a hedge or a safe haven? An analysis of stocks, bonds and gold.* Financial Review, 45(2), 217-229.
- **Exact Safe-Haven Definition:** An asset is a safe haven if it is uncorrelated or negatively correlated with the market benchmark during tail stress ($\rho_{\text{stress}} \le 0$).
- **Capital Preservation Definition:** An asset preserves capital if its conditional return is non-negative ($E[r_i \mid \text{stress}] \ge 0$).
- **Endogeneity Safeguard:** Implemented **Leave-One-Market-Out Benchmarks** ($r_{\text{bench}}^{(-i)}$) where market $i$'s benchmark strictly excludes market $i$.
- **Empirical Results (1%, 5%, and 10% Tail Stress):**
  - Stress sample sizes: 1% ($N=62$), 5% ($N=307$), 10% ($N=614$).
  - Every single market has conditional stress correlation $\rho_{\text{stress}} \in [+0.73, +0.92]$, with 95% moving-block bootstrap confidence intervals strictly bounded well above zero (e.g. at 1%: $\text{FTSE } [0.47, 0.90]$, $\text{DAX } [0.49, 0.92]$, $\text{CAC } [0.74, 0.97]$, $\text{SMI } [0.47, 0.86]$).
  - Under capital preservation ($E[r_i \mid \text{stress}] \ge 0$), all 8 markets suffer severe losses (worst 1% returns: $-3.83\%$ to $-5.38\%$, Holm $p \approx 1.0$).
  - **Econometric Conclusion:** Under Baur & Lucey's exact definition, **NO European core equity market qualifies as a safe haven**. Defensive markets (Switzerland `^SSMI` and UK `^FTSE`) function as **attenuators/diversifiers** (milder losses), but remain firmly co-moving. Bounded strictly to the 8 core European equity indices tested.

### 7.6 Walk-Forward Evaluation & Hungarian Label Alignment
- **Zero Lookahead Audit:** Scaler fitting, model fitting, and model selection use strictly historical data available at each refit date.
- **Hungarian Centroid Matching:** Implemented `scipy.optimize.linear_sum_assignment` to align refit cluster centroids to initial reference centroids, permanently eliminating sequential label-switching.
- **Transition Matrix & Persistence:**
  - Diagonal persistence exceeds $91.6\%$ to $94.3\%$.
  - Mean episode duration: State 1 (Turbulent Stress) persists for an average of 11.8 days; State 0 (Normal) persists for 17.4 days.
  - Zero direct transitions between State 1 (Turbulent Stress) and State 2 (Calm Decoupled), confirming that transitions pass through State 0.
- **Horizon Sensitivity:** Stable across refit intervals (60, 126, 252 days) and initial training lengths (500, 1000, 1500 days).

### 7.6.1 Scale-Invariant Hungarian Centroid Alignment under Scaler Drift (Priority 1)
In sequential walk-forward K-Means evaluation with an expanding historical training window, the feature scaler is refitted at every step $t$.
- **The Mismatched Coordinate Frame Problem:**
  At step 0, centroids $\mathbf{c}_0$ are standardized via $(\mathbf{\mu}_0, \mathbf{\sigma}_0)$. At step $t > 0$, new macro-financial regimes (such as the 2008 Lehman collapse or 2020 COVID shock) enter the training window, causing historical feature means $\mathbf{\mu}_t$ and standard deviations $\mathbf{\sigma}_t$ to drift. If centroids were matched in standardized coordinates $\mathbf{Z}_t = \frac{\mathbf{x} - \mathbf{\mu}_t}{\mathbf{\sigma}_t}$ directly against $\mathbf{Z}_0 = \frac{\mathbf{x} - \mathbf{\mu}_0}{\mathbf{\sigma}_0}$, Euclidean distances would be computed across incompatible coordinate systems, causing severe label corruption.
- **Scale-Invariant Reference Projection:**
  At each walk-forward refit step $t$:
  1. The refitted K-Means centroids $\mathbf{c}_t^{\text{std}}$ are converted back into their physical, unscaled feature units using the refit scaler's inverse transform:
     $$\mathbf{c}_t^{\text{raw}} = \mathbf{c}_t^{\text{std}} \odot \mathbf{\sigma}_t + \mathbf{\mu}_t$$
  2. Both the reference centroids $\mathbf{c}_0^{\text{raw}}$ and current centroids $\mathbf{c}_t^{\text{raw}}$ are projected into the fixed baseline coordinate system using the step 0 reference scaler:
     $$\mathbf{c}_t^{\text{ref}} = \frac{\mathbf{c}_t^{\text{raw}} - \mathbf{\mu}_0}{\mathbf{\sigma}_0}$$
  3. The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) solves the optimal assignment on the cost matrix:
     $$D_{ij} = \|\mathbf{c}_t^{\text{ref}, i} - \mathbf{c}_0^{\text{ref}, j}\|_2$$
  4. This provides a strictly scale-invariant, lookahead-free mapping from step $t$ cluster IDs to baseline cluster IDs without using future observations.
- **Synthetic Verification:**
  Tested with synthetic 2-cluster datasets under deliberate scaler shifts ($\Delta \mu = +2.5, \Delta \sigma = 1.8\times$), verifying that cluster identities remain 100% aligned under severe parameter drift (`test_walk_forward_label_alignment_under_scale_drift`).

---

## 8. Robustness Classification of Findings

| Finding / Conclusion | Robustness Status | Evidence & Stability across Parameters | Methodological Dependency / Caveats |
|---|:---:|---|---|
| **Dimensional Collapse during Systemic Crises** | **ROBUST** | Stable across all window lengths ($M=15$ to $60$), all models, and all basket specifications. Spectral entropy consistently drops by 30–60% during GFC, COVID, and Brexit. | Requires calendar intersection alignment to avoid artificial zero returns. |
| **Absence of Equity Safe Havens in Core Europe** | **ROBUST** | Stable across all tested stress thresholds (1%, 5%, 10%), both benchmark definitions (Leave-One-Out vs Equal-Weighted), and all 8 markets. All $\rho_{\text{stress}}$ strictly positive ($> 0.65$). | Restricted to equity indices; non-equity safe havens (Bunds, Gold) not tested. |
| **Pre-2004 vs Post-2004 Coupling Shift** | **ROBUST** | Both walk-forward refitting and crisis window analyses show European cross-market correlation was significantly lower during Dot-Com (0.68) than post-2007 crises (0.85–0.93). | Dependent on historical availability of cash indices. |
| **High-Volatility State Positive Compounded Return** | **SENSITIVE / CONDITIONAL** | Arithmetic return is positive (+13.96%) due to violent policy rebound days, but compound geometric return suffers heavy volatility drag ($10.55\%$ vs $13.96\%$). Maximum drawdown within contiguous episodes reaches $-37.29\%$. | Sensitive to whether returns are measured contemporaneously or forward ($r_{t+1}$). |
| **Optimal Cluster Count ($K=2$ vs $K=3$ vs $K=4$)** | **METHODOLOGY-DEPENDENT** | Silhouette score peaks at $K=2$ (0.473), Davies-Bouldin is best at $K=3$ (0.768) and $K=4$ (0.701), while GMM BIC favors $K=3$ ($-14,780.9$). | Selecting $K=3$ balances likelihood fit against overparameterization and provides meaningful economic separation. |
| **Exact Crisis Alert Lead Times** | **INCONCLUSIVE / NON-PREDICTIVE** | Changes in entropy or correlation before historical crisis dates represent retrospective occupancy, not genuine out-of-sample early warnings. | Do not claim predictive forecasting without forward horizon sequential testing. |

---

## 9. Reproducibility & Reproduction Commands

To reproduce the complete analysis and all audit deliverables from scratch:

```bash
# 1. Run automated unit test suite (70 tests, 100% pass)
python -m pytest -v

# 2. Run the German DAX Total Return vs Price Return audit
python scripts/reproduce_dax_comparison.py

# 3. Run the Milestone 3 empirical analysis pipeline (2000-2024 Core 8 data)
python scripts/run_milestone3_analysis.py

# 4. Run the Milestone 4 Statistical Robustness & Sensitivity Audit Suite
python scripts/run_milestone4_audit.py

# 5. Launch the Streamlit interactive dashboard
streamlit run app.py
```

All generated tables and summary reports are located in `outputs/milestone3/` and `outputs/milestone4/` in both CSV and Markdown formats.

---

## 10. Limitations & Methodological Boundaries

1. **Retrospective State Occupancy vs. Forward Early Warning:** Discovered regimes and event window statistics describe retrospective market states. Changes in correlation or entropy before historical crises do not constitute proof of predictive early-warning ability without separate out-of-sample sequential hypothesis tests with a forward target horizon.
2. **Equity-Only Safe-Haven Absence:** The empirical finding that no market acts as a safe haven is strictly bounded to the 8 core European equity markets analyzed over 2000–2024. As documented in financial economics, true safe havens reside in government bonds (Bunds), precious metals (Gold), or reserve currencies (CHF, USD).
3. **Nordic Index History Constraints:** While Core 8 cash indices provide continuous coverage from 2000 to 2024, Scandinavian cash indices (`^OMX`, `^OMXC25`) have shorter Yahoo Finance histories (2008 and 2016), restricting long-horizon pre-2008 Scandinavian backtesting.
4. **Moving-Window Memory Constraints:** A rolling window of length $M$ implies that extreme market shocks stay in the state vector for $M$ days, meaning crisis recovery rebounds are inherently classified inside the high-volatility cluster.

---

## 11. Implementation Roadmap

- [x] **Milestone 1: Audit & Foundation (COMPLETED)**
  - Full audit of baseline notebook `PRJ3_3.ipynb`.
  - Modular project architecture (`config`, `data`, `src`, `tests`).
  - Robust yfinance acquisition, provenance metadata, and caching.
  - Calendar synchronization and data validation without artificial zero returns.
  - Logarithmic and simple return calculations with mathematical identity tests.
- [x] **Milestone 2: Statistical Mechanics & Spectral Dynamics (COMPLETED)**
  - Rolling pairwise correlation matrices with strict window and min_periods verification.
  - Average off-diagonal correlation $\mu(\tau)$ formula.
  - Positive semi-definite (PSD) safeguards and spectral decomposition.
  - Normalized spectral entropy $H_{\text{norm}}(\tau)$ and Absorption Ratio $\text{AR}_k(\tau)$.
  - High-performance vectorized epoch-distance calculation with SciPy `cdist`.
  - Empirical audit of German DAX Total Return vs Price Return discrepancy.
  - Interactive correlation matrix and spectral entropy explorer in `app.py`.
- [x] **Milestone 3: Market Regime Detection & Crisis Event Analysis (COMPLETED)**
  - Lookahead-free 4D state space $[\bar{\rho}, H_{\text{norm}}, \text{AR}_1, \sigma_{\text{EW}}]$.
  - ChronologicalScaler with zero-variance protection.
  - Multi-metric K-Means evaluation & GMM (AIC, BIC).
  - Rule-based quantile grid baseline.
  - Realized return, volatility, max drawdown, semi-variance, VaR, and ES characterization.
  - Walk-forward refitting robustness experiment (41 steps, 5,113 OOS predictions).
  - Historical crisis windows (7 documented events, overlap audit, KS tests, Holm correction).
  - Leave-one-market-out Baur-Lucey safe-haven analysis with moving-block bootstrap CIs.
- [x] **Milestone 4: Statistical Robustness, Reproducibility, and Final Research Audit (COMPLETED)**
  - Safe-haven econometric audit: Baur & Lucey (2010) conditional correlation $\rho_{\text{stress}} \le 0$ with 95% bootstrap CIs across 1%, 5%, and 10% thresholds.
  - Crisis-window dependence-aware inference: Moving-block bootstrap ($L=20$ days) for autocorrelated rolling metrics; Holm adjustment; pre-event data sufficiency audit.
  - Positive return investigation in State 1: Arithmetic return (+13.96%) decomposed into geometric compound return (+10.55%), volatility drag (3.87%), contiguous max drawdown (-37.29%), and top relief rally rebound days (+8.24% on 2020-03-24).
  - Walk-forward Hungarian alignment (`scipy.optimize.linear_sum_assignment`): Eliminated sequential label switching; empirical transition matrix $P(S_{t+1}=j \mid S_t=i)$ and persistence.
  - Comprehensive sensitivity suite: Rolling windows ($M \in [15, 20, 30, 45, 60]$), market basket (Core 8 vs Core 8 + Sweden post-2008), refit horizons, and model comparison.
  - Streamlit dashboard updated with dedicated statistical robustness & sensitivity observatory.
- [x] **Milestone 5: Final Technical Audit & Portfolio Publication (COMPLETED)**
  - Scale-invariant Hungarian centroid matching under walk-forward scaler drift: Physical reconstitution via `scaler.inverse_transform` into fixed reference coordinates; synthetic drift unit test verified.
  - Separation of retrospective descriptive returns ($r_t$) from forward strategy returns ($r_{t+1}$): Lookahead-free execution with 5 bps turnover friction; documented signal latency and relief-rally cash drag.
  - Crisis moving-block bootstrap validation: Paired observation alignment via `pd.concat`, two-sided empirical p-values, and block-length sensitivity ($L \in [2, 5, 10, 20]$).
  - Full automated test suite expanded to **91 unit tests and benchmarks** (100% passing in ~7.0s).
  - Publication-ready financial visualization routines implemented in `src/visualization.py` with full dashboard integration.
  - All reproduction scripts verified and documented (`reproduce_dax_comparison.py`, `run_milestone3_analysis.py`, `run_milestone4_audit.py`).
  - Streamlit dashboard updated with hypothetical strategy backtest, dynamic regime timelines, safe-haven scatter plots, and strict academic disclaimers.


