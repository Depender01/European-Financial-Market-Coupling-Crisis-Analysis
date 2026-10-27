"""Unit tests for market regime detection and clustering module."""

import numpy as np
import pandas as pd
import pytest

from src.market_regimes import (
    extract_regime_features,
    ChronologicalScaler,
    split_chronological,
    KMeansRegimeModel,
    GMMRegimeModel,
    RuleBasedRegimeBaseline,
    characterize_regimes,
    walk_forward_regime_evaluation,
    simulate_regime_switching_strategy,
)


@pytest.fixture
def synthetic_market_returns() -> pd.DataFrame:
    """Fixture providing 100 days of synthetic returns for 4 assets."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    data = {
        "UK": np.random.normal(0.0005, 0.01, 100),
        "Germany": np.random.normal(0.0004, 0.012, 100),
        "France": np.random.normal(0.0003, 0.011, 100),
        "Switzerland": np.random.normal(0.0002, 0.008, 100),
    }
    return pd.DataFrame(data, index=dates)


def test_extract_regime_features_structure(synthetic_market_returns: pd.DataFrame):
    """Verify that feature extraction yields the expected 4D feature columns and dates."""
    features_df, corr_mats, dates, assets = extract_regime_features(
        synthetic_market_returns,
        window=20,
        stride=1,
    )

    expected_cols = [
        "mean_correlation",
        "spectral_entropy",
        "absorption_ratio_1",
        "benchmark_volatility",
    ]
    assert list(features_df.columns) == expected_cols
    assert len(features_df) == len(dates)
    assert len(features_df) == len(corr_mats)
    assert assets == ["UK", "Germany", "France", "Switzerland"]
    assert features_df.index.is_monotonic_increasing
    assert np.isfinite(features_df.values).all()


def test_extract_regime_features_insufficient_data():
    """Window larger than dataset must raise ValueError."""
    dates = pd.date_range("2026-01-01", periods=10)
    small_df = pd.DataFrame({"A": np.random.randn(10), "B": np.random.randn(10)}, index=dates)
    with pytest.raises(ValueError, match="Insufficient observations"):
        extract_regime_features(small_df, window=20)


def test_chronological_scaler_lookahead_protection():
    """Verify scaler parameters depend ONLY on training observations."""
    np.random.seed(42)
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(50, 4))
    X_test_base = np.random.normal(loc=20.0, scale=5.0, size=(30, 4))

    scaler = ChronologicalScaler()
    scaler.fit(X_train)

    # Recorded training parameters
    train_mean = scaler.mean_.copy()
    train_scale = scaler.scale_.copy()

    # Transforming X_test_base
    scaled_test1 = scaler.transform(X_test_base)

    # Drastically modifying test data should NOT alter the fitted scaler
    X_test_extreme = X_test_base * 1000.0
    scaled_test2 = scaler.transform(X_test_extreme)

    np.testing.assert_allclose(scaler.mean_, train_mean)
    np.testing.assert_allclose(scaler.scale_, train_scale)
    # The transform must be linear with respect to training parameters
    np.testing.assert_allclose(scaled_test1, (X_test_base - train_mean) / train_scale)


def test_chronological_scaler_zero_variance_guard():
    """Features with zero variance (std = 0) must be scaled by 1.0 to prevent division by zero."""
    X_constant = np.array([
        [1.0, 5.0],
        [2.0, 5.0],
        [3.0, 5.0],
    ])
    scaler = ChronologicalScaler()
    scaler.fit(X_constant)

    assert scaler.scale_[0] > 0
    assert scaler.scale_[1] == 1.0  # Safe fallback scale

    scaled = scaler.transform(X_constant)
    assert np.isfinite(scaled).all()
    assert np.all(scaled[:, 1] == 0.0)


def test_split_chronological(synthetic_market_returns: pd.DataFrame):
    """Verify chronological train/test split ordering and boundary enforcement."""
    # Split by ratio
    train_df, test_df = split_chronological(synthetic_market_returns, train_ratio=0.60)
    assert len(train_df) == 60
    assert len(test_df) == 40
    # Crucial: Strictly chronological (no interleaving or future leakage)
    assert train_df.index[-1] < test_df.index[0]

    # Split by explicit date
    cutoff = str(synthetic_market_returns.index[49].date())
    train_d, test_d = split_chronological(synthetic_market_returns, train_end_date=cutoff)
    assert train_d.index[-1] <= pd.Timestamp(cutoff)
    assert test_d.index[0] > pd.Timestamp(cutoff)


def test_kmeans_evaluation_and_stability(synthetic_market_returns: pd.DataFrame):
    """Test K-Means multi-metric evaluation across K and deterministic cluster prediction."""
    features_df, _, _, _ = extract_regime_features(synthetic_market_returns, window=20)
    scaler = ChronologicalScaler()
    X_scaled = scaler.fit_transform(features_df)

    km_model = KMeansRegimeModel(random_state=42)
    eval_df = km_model.evaluate_k_range(X_scaled, k_range=range(2, 5), n_init=10, n_stability_bootstraps=5)

    assert list(eval_df.index) == [2, 3, 4]
    assert "silhouette_score" in eval_df.columns
    assert "davies_bouldin_index" in eval_df.columns
    assert "calinski_harabasz_index" in eval_df.columns
    assert "stability_ari" in eval_df.columns

    # Stability ARI should be between -1 and 1
    assert (eval_df["stability_ari"] >= -1.0).all()
    assert (eval_df["stability_ari"] <= 1.0).all()

    # Fit final model with K=2
    km_model.fit(X_scaled, k=2)
    labels = km_model.predict(X_scaled)
    assert len(labels) == len(X_scaled)
    assert set(np.unique(labels)).issubset({0, 1})


def test_gmm_evaluation_and_soft_probabilities(synthetic_market_returns: pd.DataFrame):
    """Test Gaussian Mixture Model evaluation (AIC/BIC) and posterior state probabilities."""
    features_df, _, _, _ = extract_regime_features(synthetic_market_returns, window=20)
    scaler = ChronologicalScaler()
    X_scaled = scaler.fit_transform(features_df)

    gmm_model = GMMRegimeModel(random_state=42)
    eval_df = gmm_model.evaluate_k_range(X_scaled, k_range=range(2, 4), n_init=5)

    assert "aic" in eval_df.columns
    assert "bic" in eval_df.columns
    assert "converged" in eval_df.columns

    # Fit final GMM with K=2
    gmm_model.fit(X_scaled, k=2)
    hard_labels = gmm_model.predict(X_scaled)
    soft_probs = gmm_model.predict_proba(X_scaled)

    assert soft_probs.shape == (len(X_scaled), 2)
    # Probabilities must sum to 1.0 for each observation
    np.testing.assert_allclose(soft_probs.sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(soft_probs >= 0.0)
    # Hard labels must correspond to argmax of soft probabilities
    assert np.array_equal(hard_labels, np.argmax(soft_probs, axis=1))


def test_rule_based_baseline(synthetic_market_returns: pd.DataFrame):
    """Test transparent rule-based quantile grid assignments."""
    features_df, _, _, _ = extract_regime_features(synthetic_market_returns, window=20)
    train_feat, test_feat = split_chronological(features_df, train_ratio=0.6)

    baseline = RuleBasedRegimeBaseline(high_quantile=0.75, low_quantile=0.25)
    baseline.fit(train_feat)

    assert baseline.q_high_corr_ is not None
    assert baseline.q_low_corr_ is not None

    test_labels = baseline.predict(test_feat)
    assert len(test_labels) == len(test_feat)
    assert set(np.unique(test_labels)).issubset({0, 1, 2})


def test_characterize_regimes(synthetic_market_returns: pd.DataFrame):
    """Verify objective realized risk characterization for discovered clusters."""
    ew_rets = synthetic_market_returns.mean(axis=1)
    dates = synthetic_market_returns.index

    # Assign alternating dummy labels (State 0 and State 1)
    labels = np.array([0 if i % 2 == 0 else 1 for i in range(len(dates))])

    df_char = characterize_regimes(labels, dates, ew_rets)

    assert len(df_char) == 2
    assert set(df_char.index) == {"State_0", "State_1"}
    assert "Arithmetic_Annualized_Return" in df_char.columns
    assert "Geometric_Annualized_Return" in df_char.columns
    assert "Annualized_Volatility" in df_char.columns
    assert "Volatility_Drag" in df_char.columns
    assert "Positive_Days_Pct" in df_char.columns
    assert "Contiguous_Max_Drawdown" in df_char.columns
    assert "Concatenated_Max_Drawdown" in df_char.columns
    assert "Forward_1D_Annualized_Return" in df_char.columns
    assert "Downside_Deviation" in df_char.columns
    assert "VaR_0.05" in df_char.columns
    assert "Expected_Shortfall_0.05" in df_char.columns
    assert (df_char["Observations"] == 50).all()


def test_walk_forward_regime_evaluation(synthetic_market_returns: pd.DataFrame):
    """Test strictly chronological walk-forward refitting robustness experiment."""
    features_df, _, _, _ = extract_regime_features(synthetic_market_returns, window=20)
    # features_df has 81 rows (100 - 20 + 1)
    # Set initial_train_size=40, refit_interval=20
    oos_df, refit_df = walk_forward_regime_evaluation(
        features_df,
        initial_train_size=40,
        refit_interval=20,
        k=2,
        random_state=42,
    )

    assert not oos_df.empty
    assert "wf_regime" in oos_df.columns
    # Out of sample count should equal total features minus initial train size
    assert len(oos_df) == len(features_df) - 40
    # Dates must be strictly after the initial training cutoff
    assert oos_df.index[0] == features_df.index[40]

    assert not refit_df.empty
    assert "train_inertia" in refit_df.columns
    assert "train_silhouette" in refit_df.columns
    assert (refit_df["train_size"] >= 40).all()

    # Test with return_diagnostics=True
    oos_df_diag, refit_df_diag, trans_mat, pers_df = walk_forward_regime_evaluation(
        features_df,
        initial_train_size=40,
        refit_interval=20,
        k=2,
        random_state=42,
        return_diagnostics=True,
    )
    assert not trans_mat.empty
    assert trans_mat.shape == (2, 2)
    # Rows of transition matrix must sum to 1.0
    np.testing.assert_allclose(trans_mat.sum(axis=1).values, 1.0, rtol=1e-5)

    assert not pers_df.empty
    assert "Mean_Duration_Days" in pers_df.columns
    assert "Distinct_Episodes" in pers_df.columns


def test_walk_forward_label_alignment_under_scale_drift():
    """Verify that Hungarian alignment correctly preserves cluster identities when scaler parameters drift."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")

    # Construct two distinct clusters in physical space:
    # Cluster 0 (Calm): centered at [0.3, 0.3, 0.3, 0.1]
    # Cluster 1 (Turbulent): centered at [0.8, 0.8, 0.8, 0.4]
    c0 = rng.normal(loc=[0.3, 0.3, 0.3, 0.1], scale=0.03, size=(60, 4))
    c1 = rng.normal(loc=[0.8, 0.8, 0.8, 0.4], scale=0.03, size=(60, 4))

    # Interleave to create alternating periods
    data = np.vstack([c0[:30], c1[:30], c0[30:], c1[30:]])
    feat_df = pd.DataFrame(data, index=dates, columns=["feat1", "feat2", "feat3", "feat4"])

    # Initial train = 50, refit = 25
    # The training set in step 1 will include more high-volatility points, shifting scaler mean and std
    oos_df, refit_df = walk_forward_regime_evaluation(
        feat_df,
        initial_train_size=50,
        refit_interval=25,
        k=2,
        random_state=42,
    )

    assert len(oos_df) == 70
    # Check that labels are assigned consistently: points in cluster 1 should have identical label across steps
    # Compare raw values of feat1 with wf_regime
    high_feat_labels = oos_df.loc[feat_df.loc[oos_df.index, "feat1"] > 0.6, "wf_regime"]
    low_feat_labels = oos_df.loc[feat_df.loc[oos_df.index, "feat1"] < 0.5, "wf_regime"]

    # High feature points and low feature points must map to mutually exclusive labels across all refit steps
    high_mode = high_feat_labels.mode()[0]
    low_mode = low_feat_labels.mode()[0]
    assert high_mode != low_mode
    assert (high_feat_labels == high_mode).mean() >= 0.95
    assert (low_feat_labels == low_mode).mean() >= 0.95


def test_simulate_regime_switching_strategy():
    """Verify that regime switching strategy executes with 1-day lag and no lookahead."""
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    # Regimes: State 0 on days 0..4, State 1 (Stress) on days 5..7, State 0 on days 8..9
    labels = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 0, 0], index=dates)
    # Benchmark returns: positive on calm days, sharp -5% drop on day 6
    bench_returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, -0.05, -0.05, 0.02, 0.01, 0.01], index=dates)

    strat_res = simulate_regime_switching_strategy(
        labels,
        bench_returns,
        cash_annual_return=0.0,
        cost_per_trade_bps=0.0,
    )

    summary = strat_res["summary"]
    returns = strat_res["returns"]

    assert "Hypothetical Regime-Switching Strategy (Lookahead-Free)" in summary.columns
    assert "Equal-Weighted European Benchmark (Buy & Hold)" in summary.columns

    # Verify 1-day execution lag:
    # Day 5 had S_5 = 1 (State 1 observed at close of day 5).
    # Therefore, equity weight for Day 6 (w_6) must be 0.0 (Cash)!
    assert returns.loc[dates[6], "equity_weight"] == 0.0
    # Day 6 return for strategy should be cash (0.0), avoiding the -5% benchmark drop!
    assert returns.loc[dates[6], "strategy_net_return"] == 0.0
    assert returns.loc[dates[6], "benchmark_return"] == -0.05



