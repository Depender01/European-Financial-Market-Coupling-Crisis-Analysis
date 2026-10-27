"""Market Regime Detection and Clustering Module.

Implements chronological feature preprocessing (lookahead-free), multi-model regime detection
(K-Means, Gaussian Mixture Models, and Rule-Based Baseline), multi-metric diagnostics,
and independent realized risk characterization.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from scipy.optimize import linear_sum_assignment

from .correlation import compute_rolling_correlation_matrices
from .metrics import compute_rolling_metrics
from .returns import compute_equal_weighted_index

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. Feature Extraction
# -----------------------------------------------------------------------------
def extract_regime_features(
    returns_df: pd.DataFrame,
    window: int = 20,
    stride: int = 1,
    trading_days_per_year: int = 252,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex, List[str]]:
    """Extract the 4D market-state feature vector without dimensionality distortion:

    x(t) = [
        mean_off_diagonal_correlation,
        normalized_spectral_entropy,
        absorption_ratio_1,
        trailing_annualized_equal_weighted_benchmark_volatility
    ]

    Args:
        returns_df: Aligned daily log returns DataFrame (dates x assets).
        window: Rolling epoch duration M in trading days (default 20).
        stride: Rolling stride Delta tau in trading days (default 1).
        trading_days_per_year: Annualization factor (default 252).

    Returns:
        Tuple of:
            - features_df: DataFrame with 4 feature columns indexed by epoch end dates.
            - corr_matrices: 3D array of rolling correlation matrices (N_epochs, N, N).
            - epoch_dates: DatetimeIndex of epoch end dates.
            - asset_names: List of asset column names.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(returns_df).__name__}")

    if len(returns_df) < window:
        raise ValueError(f"Insufficient observations ({len(returns_df)}) for window {window}")

    # 1. Compute rolling correlation matrices and spectral metrics
    corr_matrices, epoch_dates, asset_names = compute_rolling_correlation_matrices(
        returns_df, window=window, stride=stride
    )
    metrics_df = compute_rolling_metrics(corr_matrices, epoch_dates, top_k_absorption=1)

    # 2. Compute trailing annualized equal-weighted benchmark volatility
    ew_returns = compute_equal_weighted_index(returns_df)
    rolling_vol_series = (
        ew_returns.rolling(window=window)
        .std(ddof=1)
        .reindex(epoch_dates)
        * np.sqrt(trading_days_per_year)
    )

    features_df = pd.DataFrame(
        {
            "mean_correlation": metrics_df["mean_correlation"],
            "spectral_entropy": metrics_df["spectral_entropy"],
            "absorption_ratio_1": metrics_df["absorption_ratio_k1"],
            "benchmark_volatility": rolling_vol_series,
        },
        index=epoch_dates,
    )

    # Drop any edge NaNs (e.g. from initial volatility calculation)
    features_df.dropna(inplace=True)
    valid_idx = features_df.index
    # Align corr_matrices to valid feature dates
    date_mask = np.isin(epoch_dates, valid_idx)
    corr_matrices_valid = corr_matrices[date_mask]

    logger.info("Extracted %d regime feature vectors across %d dimensions", len(features_df), features_df.shape[1])
    return features_df, corr_matrices_valid, valid_idx, asset_names


# -----------------------------------------------------------------------------
# 2. Chronological Standardization (Zero Lookahead Leakage)
# -----------------------------------------------------------------------------
class ChronologicalScaler:
    """Standardizes features strictly on historical / training data to prevent lookahead bias.

    Fits mean and scale on a specified training period [t_0, t_train_end], then transforms
    subsequent out-of-sample data using training statistics only. Explicitly guards
    against zero historical standard deviation.
    """

    def __init__(self, zero_scale_tol: float = 1e-12):
        self.zero_scale_tol = zero_scale_tol
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.n_features_in_: int = 0

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "ChronologicalScaler":
        """Compute training mean and standard deviation.

        Args:
            X: Training feature matrix (rows: observations, cols: features).

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            arr = X.values
        else:
            arr = np.asarray(X)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got ndim={arr.ndim}")

        if len(arr) < 2:
            raise ValueError("At least 2 observations required to compute sample standard deviation.")

        self.n_features_in_ = arr.shape[1]
        self.mean_ = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0, ddof=1)

        # Handle zero or near-zero variance features explicitly
        self.scale_ = np.where(stds <= self.zero_scale_tol, 1.0, stds)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Apply training parameters to transform incoming observations.

        Formula: z = (x - mean_train) / scale_train
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X)

        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature count mismatch: fitted on {self.n_features_in_}, received {arr.shape[1]}")

        scaled = (arr - self.mean_) / self.scale_
        return scaled.astype(np.float64)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fit to training data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Reverse standardizing transformation back to original feature units.

        Formula: x = z * scale_train + mean_train

        Args:
            X: Standardized feature matrix (Z-scores).

        Returns:
            np.ndarray: Reconstituted feature matrix in original physical units.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            arr = X.values
        else:
            arr = np.asarray(X)

        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature count mismatch: fitted on {self.n_features_in_}, received {arr.shape[1]}")

        raw = arr * self.scale_ + self.mean_
        return raw.astype(np.float64)



def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    train_end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame into strictly chronological training and testing sets.

    Args:
        df: Datetime-indexed DataFrame.
        train_ratio: Fraction of observations reserved for training if train_end_date is None.
        train_end_date: Explicit date threshold string (YYYY-MM-DD). If provided, rows <= date
            form the training set, and rows > date form the test set.

    Returns:
        Tuple of (train_df, test_df).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if train_end_date is not None:
        cutoff = pd.Timestamp(train_end_date)
        train_df = df[df.index <= cutoff].copy()
        test_df = df[df.index > cutoff].copy()
    else:
        n_train = int(len(df) * train_ratio)
        if n_train < 10:
            raise ValueError(f"Training split too small ({n_train} rows). Increase train_ratio.")
        train_df = df.iloc[:n_train].copy()
        test_df = df.iloc[n_train:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Chronological split produced empty subset (train: {len(train_df)}, test: {len(test_df)}). "
            f"Adjust cutoff date or ratio."
        )

    logger.info(
        "Chronological split: Train [%s to %s, %d obs], Test [%s to %s, %d obs]",
        train_df.index[0].date(), train_df.index[-1].date(), len(train_df),
        test_df.index[0].date(), test_df.index[-1].date(), len(test_df)
    )
    return train_df, test_df


# -----------------------------------------------------------------------------
# 3. Model 1: K-Means with Multi-Metric Evaluation & Stability
# -----------------------------------------------------------------------------
class KMeansRegimeModel:
    """K-Means clustering on standardized statistical features with comprehensive diagnostics."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_: Dict[int, KMeans] = {}
        self.fitted_k_: Optional[int] = None

    def evaluate_k_range(
        self,
        X_train_scaled: np.ndarray,
        k_range: range = range(2, 7),
        n_init: int = 50,
        n_stability_bootstraps: int = 20,
    ) -> pd.DataFrame:
        """Evaluate clustering quality across multiple values of K.

        Diagnostics:
            - Inertia (Sum of squared distances to closest center)
            - Silhouette Score (Separation and compactness, higher is better, in [-1, 1])
            - Davies-Bouldin Index (Similarity of clusters, lower is better, >= 0)
            - Calinski-Harabasz Index (Variance ratio criterion, higher is better)
            - Stability Score (Mean Adjusted Rand Index across bootstrapped sub-samples)

        Args:
            X_train_scaled: Scaled feature matrix.
            k_range: Range of cluster counts to test.
            n_init: Number of initializations per K.
            n_stability_bootstraps: Number of bootstrap iterations for stability.

        Returns:
            pd.DataFrame of diagnostic metrics indexed by K.
        """
        results = []
        N_samples = len(X_train_scaled)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=n_init)
            labels = km.fit_predict(X_train_scaled)
            self.models_[k] = km

            # Core diagnostics
            inertia = km.inertia_
            sil = float(silhouette_score(X_train_scaled, labels))
            db = float(davies_bouldin_score(X_train_scaled, labels))
            ch = float(calinski_harabasz_score(X_train_scaled, labels))

            # Cluster stability via sub-sample bootstrapping
            ari_scores = []
            rng = np.random.RandomState(self.random_state)
            sub_size = int(0.80 * N_samples)

            for b in range(n_stability_bootstraps):
                idx_b = rng.choice(N_samples, size=sub_size, replace=False)
                X_b = X_train_scaled[idx_b]
                km_b = KMeans(n_clusters=k, random_state=self.random_state + b + 1, n_init=10)
                labels_b = km_b.fit_predict(X_b)
                labels_orig_subset = labels[idx_b]
                ari_scores.append(adjusted_rand_score(labels_orig_subset, labels_b))

            stability_ari = float(np.mean(ari_scores))

            results.append({
                "k": k,
                "inertia": inertia,
                "silhouette_score": sil,
                "davies_bouldin_index": db,
                "calinski_harabasz_index": ch,
                "stability_ari": stability_ari,
            })

        df_eval = pd.DataFrame(results).set_index("k")
        return df_eval

    def fit(self, X_train_scaled: np.ndarray, k: int, n_init: int = 50) -> "KMeansRegimeModel":
        """Fit final model with chosen K on training data."""
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=n_init)
        km.fit(X_train_scaled)
        self.models_[k] = km
        self.fitted_k_ = k
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Assign cluster labels to new observations using fitted centroids."""
        if self.fitted_k_ is None or self.fitted_k_ not in self.models_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.models_[self.fitted_k_].predict(X_scaled)


# -----------------------------------------------------------------------------
# 4. Model 2: Gaussian Mixture Models (Probabilistic Soft Assignment)
# -----------------------------------------------------------------------------
class GMMRegimeModel:
    """Gaussian Mixture Model for probabilistic regime clustering and model selection."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_: Dict[int, GaussianMixture] = {}
        self.fitted_k_: Optional[int] = None

    def evaluate_k_range(
        self,
        X_train_scaled: np.ndarray,
        k_range: range = range(2, 7),
        n_init: int = 10,
    ) -> pd.DataFrame:
        """Evaluate GMM across K using Akaike and Bayesian Information Criteria.

        Returns:
            pd.DataFrame with AIC, BIC, Log-Likelihood, and Converged status.
        """
        results = []
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                random_state=self.random_state,
                n_init=n_init,
                covariance_type="full",
            )
            gmm.fit(X_train_scaled)
            self.models_[k] = gmm

            results.append({
                "k": k,
                "bic": float(gmm.bic(X_train_scaled)),
                "aic": float(gmm.aic(X_train_scaled)),
                "log_likelihood": float(gmm.score(X_train_scaled)),
                "converged": bool(gmm.converged_),
            })

        return pd.DataFrame(results).set_index("k")

    def fit(self, X_train_scaled: np.ndarray, k: int, n_init: int = 10) -> "GMMRegimeModel":
        """Fit final GMM with chosen K."""
        gmm = GaussianMixture(
            n_components=k,
            random_state=self.random_state,
            n_init=n_init,
            covariance_type="full",
        )
        gmm.fit(X_train_scaled)
        self.models_[k] = gmm
        self.fitted_k_ = k
        return self

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict most probable cluster assignment."""
        if self.fitted_k_ is None:
            raise RuntimeError("GMM has not been fitted. Call fit() first.")
        return self.models_[self.fitted_k_].predict(X_scaled)

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict posterior state probabilities gamma_k(tau) = P(S_tau = k | x_tau)."""
        if self.fitted_k_ is None:
            raise RuntimeError("GMM has not been fitted. Call fit() first.")
        return self.models_[self.fitted_k_].predict_proba(X_scaled)


# -----------------------------------------------------------------------------
# 5. Model 3: Non-Parametric Rule-Based Baseline
# -----------------------------------------------------------------------------
class RuleBasedRegimeBaseline:
    """Transparent non-parametric rule-based regime baseline.

    Uses historical training percentiles:
        - State 0 (Coupled Stress): mean_correlation > Q75 AND volatility > Q75
        - State 1 (Decoupled / Dispersed): mean_correlation < Q25
        - State 2 (Normal / Baseline Co-movement): all other periods
    """

    def __init__(self, high_quantile: float = 0.75, low_quantile: float = 0.25):
        self.high_quantile = high_quantile
        self.low_quantile = low_quantile
        self.q_high_corr_: Optional[float] = None
        self.q_low_corr_: Optional[float] = None
        self.q_high_vol_: Optional[float] = None

    def fit(self, train_features_df: pd.DataFrame) -> "RuleBasedRegimeBaseline":
        """Compute quantile thresholds from training data only."""
        corr = train_features_df["mean_correlation"]
        vol = train_features_df["benchmark_volatility"]

        self.q_high_corr_ = float(corr.quantile(self.high_quantile))
        self.q_low_corr_ = float(corr.quantile(self.low_quantile))
        self.q_high_vol_ = float(vol.quantile(self.high_quantile))
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Assign rule-based regimes based on training quantile thresholds."""
        if self.q_high_corr_ is None:
            raise RuntimeError("Baseline has not been fitted. Call fit() first.")

        corr = features_df["mean_correlation"].values
        vol = features_df["benchmark_volatility"].values
        N = len(features_df)

        labels = np.full(N, 2, dtype=int)  # Default: State 2 (Normal)

        # State 0: Coupled Stress
        stress_mask = (corr >= self.q_high_corr_) & (vol >= self.q_high_vol_)
        labels[stress_mask] = 0

        # State 1: Decoupled / Low Correlation
        decoupled_mask = (corr <= self.q_low_corr_) & ~stress_mask
        labels[decoupled_mask] = 1

        return labels


# -----------------------------------------------------------------------------
# 6. Objective Realized Risk Characterization (Independent Verification)
# -----------------------------------------------------------------------------
def characterize_regimes(
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    benchmark_returns: pd.Series,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Characterize discovered regimes using realized return, volatility, and tail-risk statistics.

    IMPORTANT METHODOLOGICAL DISTINCTION:
        These statistics are retrospective descriptive metrics calculated to examine what
        financial behavior occurred while the market occupied each state. They are kept
        strictly separate from the statistical moments used to assign the regime.
        No cluster is labeled 'Crash' or 'Calm' without verifying these realized metrics.

    Metrics calculated per regime:
        - Observation Count (days) & Sample Fraction (%)
        - Realized Arithmetic Annualized Return (mean * 252)
        - Realized Geometric (Compounded) Annualized Return ((1 + R)^(252/N) - 1)
        - Realized Annualized Volatility
        - Theoretical Volatility Drag (0.5 * sigma^2)
        - Percentage of Positive Return Days (explaining violent crisis relief rebounds)
        - Contiguous Episode Max Drawdown (drawdown within unbroken contiguous regime stretches)
        - Concatenated Max Drawdown (drawdown across concatenated disjoint days)
        - Forward 1-Day Annualized Return (E[r_{t+1} | S_t] * 252)
        - Downside Semi-Variance (annualized)
        - Empirical 5% Value-at-Risk (VaR_0.05)
        - Empirical 5% Expected Shortfall (CVaR / ES_0.05)

    Args:
        labels: 1D array of regime cluster labels.
        dates: DatetimeIndex corresponding to labels.
        benchmark_returns: Series of daily returns for the equal-weighted European benchmark.
        trading_days_per_year: Annualization factor (252).

    Returns:
        pd.DataFrame indexed by Regime Label.
    """
    bench_aligned = benchmark_returns.reindex(dates).dropna()
    common_idx = bench_aligned.index
    label_series = pd.Series(labels, index=dates).reindex(common_idx)

    # Next-day forward return aligned to regime date t
    fwd_returns = benchmark_returns.shift(-1).reindex(common_idx)

    # Identify continuous contiguous episodes of regimes
    episode_series = (label_series != label_series.shift()).cumsum()

    unique_labels = sorted(label_series.unique())
    records = []

    for state in unique_labels:
        mask = label_series == state
        rets = bench_aligned[mask].values
        n_obs = len(rets)

        if n_obs < 2:
            records.append({
                "Regime": f"State_{state}",
                "Observations": n_obs,
                "Sample_Pct": 0.0,
                "Arithmetic_Annualized_Return": np.nan,
                "Geometric_Annualized_Return": np.nan,
                "Annualized_Volatility": np.nan,
                "Volatility_Drag": np.nan,
                "Positive_Days_Pct": np.nan,
                "Contiguous_Max_Drawdown": np.nan,
                "Concatenated_Max_Drawdown": np.nan,
                "Forward_1D_Annualized_Return": np.nan,
                "Downside_Deviation": np.nan,
                "VaR_0.05": np.nan,
                "Expected_Shortfall_0.05": np.nan,
            })
            continue

        ann_ret_arith = float(np.mean(rets) * trading_days_per_year)
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(trading_days_per_year))
        vol_drag = float(0.5 * (ann_vol ** 2))

        # Geometric compound annualized return
        cumprod_wealth = float(np.prod(1.0 + rets))
        if cumprod_wealth > 0:
            ann_ret_geom = float((cumprod_wealth ** (trading_days_per_year / float(n_obs))) - 1.0)
        else:
            ann_ret_geom = -1.0

        pos_pct = float(round(np.mean(rets > 0) * 100.0, 1))

        # Downside semi-deviation (volatility of negative returns only)
        neg_rets = rets[rets < 0]
        downside_dev = (
            float(np.sqrt(np.mean(neg_rets**2)) * np.sqrt(trading_days_per_year))
            if len(neg_rets) > 0
            else 0.0
        )

        # 1. Concatenated disjoint drawdown
        cum_concat = np.cumprod(1.0 + rets)
        peak_concat = np.maximum.accumulate(cum_concat)
        dd_concat = (cum_concat - peak_concat) / peak_concat
        max_dd_concat = float(np.min(dd_concat)) if len(dd_concat) > 0 else 0.0

        # 2. Contiguous unbroken episode drawdown (eliminating disjoint concatenation artifact)
        contiguous_drawdowns = []
        state_episodes = episode_series[mask]
        for ep_id in state_episodes.unique():
            ep_rets = bench_aligned.loc[state_episodes[state_episodes == ep_id].index].values
            if len(ep_rets) > 0:
                ep_cum = np.cumprod(1.0 + ep_rets)
                ep_peak = np.maximum.accumulate(ep_cum)
                ep_dd = (ep_cum - ep_peak) / ep_peak
                contiguous_drawdowns.append(float(np.min(ep_dd)))

        max_dd_contiguous = float(np.min(contiguous_drawdowns)) if len(contiguous_drawdowns) > 0 else 0.0

        # Forward 1-day return
        fwd_state_rets = fwd_returns[mask].dropna().values
        fwd_ann_ret = float(np.mean(fwd_state_rets) * trading_days_per_year) if len(fwd_state_rets) > 0 else np.nan

        # Empirical tail risk
        var_05 = float(np.percentile(rets, 5))
        tail_losses = rets[rets <= var_05]
        es_05 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_05

        records.append({
            "Regime": f"State_{state}",
            "Observations": int(n_obs),
            "Sample_Pct": float(round(n_obs / len(bench_aligned) * 100.0, 1)),
            "Arithmetic_Annualized_Return": float(ann_ret_arith),
            "Geometric_Annualized_Return": float(ann_ret_geom),
            "Annualized_Volatility": float(ann_vol),
            "Volatility_Drag": float(vol_drag),
            "Positive_Days_Pct": float(pos_pct),
            "Contiguous_Max_Drawdown": float(max_dd_contiguous),
            "Concatenated_Max_Drawdown": float(max_dd_concat),
            "Forward_1D_Annualized_Return": float(fwd_ann_ret),
            "Downside_Deviation": float(downside_dev),
            "VaR_0.05": float(var_05),
            "Expected_Shortfall_0.05": float(es_05),
        })

    df_char = pd.DataFrame(records).set_index("Regime")
    return df_char


# -----------------------------------------------------------------------------
# 7. Walk-Forward Refitting Experiment (Robustness Analysis)
# -----------------------------------------------------------------------------
def walk_forward_regime_evaluation(
    features_df: pd.DataFrame,
    initial_train_size: int = 500,
    refit_interval: int = 60,
    k: int = 3,
    random_state: int = 42,
    return_diagnostics: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Execute a strictly chronological walk-forward refitting experiment with persistent label alignment.

    Robustness Protocol:
        1. Start with initial training period of `initial_train_size` observations.
        2. Fit ChronologicalScaler and KMeans(k) on expanding training set X[:t].
        3. Solve the assignment problem via the Hungarian algorithm (linear_sum_assignment)
           to match step s centroids to reference centroids, completely eliminating label switching!
        4. Predict regime labels out-of-sample on the subsequent test window X[t : t + refit_interval].
        5. Step forward by `refit_interval` and repeat until the end of the series.
        6. Zero lookahead leakage: Every out-of-sample prediction uses only preceding data.

    Args:
        features_df: Feature DataFrame indexed by DatetimeIndex (N x 4).
        initial_train_size: Number of initial observations in training burn-in (default 500 ~ 2 years).
        refit_interval: Out-of-sample forward step in days (default 60 ~ 1 quarter).
        k: Number of clusters (default 3).
        random_state: Seed for reproducibility.
        return_diagnostics: If True, also returns out-of-sample transition matrix and persistence.

    Returns:
        If return_diagnostics is False:
            Tuple of (oos_predictions_df, refit_history_df)
        If return_diagnostics is True:
            Tuple of (oos_predictions_df, refit_history_df, transition_matrix_df, persistence_df)
    """
    T = len(features_df)
    if T < initial_train_size + refit_interval:
        raise ValueError(
            f"Insufficient observations ({T}) for initial_train_size={initial_train_size} "
            f"and refit_interval={refit_interval}."
        )

    oos_dates = []
    oos_labels = []
    refit_records = []

    curr_idx = initial_train_size
    refit_step = 0
    ref_scaler: Optional[ChronologicalScaler] = None
    ref_centroids_raw: Optional[np.ndarray] = None

    while curr_idx < T:
        train_slice = features_df.iloc[:curr_idx]
        test_end = min(curr_idx + refit_interval, T)
        test_slice = features_df.iloc[curr_idx:test_end]

        scaler = ChronologicalScaler()
        X_train_scaled = scaler.fit_transform(train_slice)
        X_test_scaled = scaler.transform(test_slice)

        km = KMeans(n_clusters=k, random_state=random_state + refit_step, n_init=20)
        km.fit(X_train_scaled)

        # Centroids in current step's standardized coordinate system
        curr_centroids_scaled = km.cluster_centers_
        # Convert centroids back into original physical feature units
        curr_centroids_raw = scaler.inverse_transform(curr_centroids_scaled)

        # Hungarian Algorithm: Persistent Centroid & Label Alignment in Consistent Coordinate System
        if ref_centroids_raw is None or ref_scaler is None:
            # Step 0: Save baseline reference scaler and raw physical centroids
            ref_scaler = scaler
            ref_centroids_raw = curr_centroids_raw.copy()
            label_map = {i: i for i in range(k)}
        else:
            # METHODOLOGICAL SAFEGUARD (Priority 1):
            # Because scaler is refit on expanding training sets, current centroids (curr_centroids_scaled)
            # are expressed in step t's coordinates (Z_t = (x - mu_t) / sigma_t).
            # To compare centroids in a single consistent, scale-invariant coordinate system without distortion,
            # we transform curr_centroids_raw into the baseline reference coordinate system using ref_scaler:
            # Z_ref = (curr_centroids_raw - mu_0) / sigma_0.
            curr_centroids_in_ref = ref_scaler.transform(curr_centroids_raw)
            ref_centroids_in_ref = ref_scaler.transform(ref_centroids_raw)

            cost_matrix = np.linalg.norm(curr_centroids_in_ref[:, None, :] - ref_centroids_in_ref[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            label_map = {row: col for row, col in zip(row_ind, col_ind)}

        raw_test_preds = km.predict(X_test_scaled)
        aligned_test_preds = np.array([label_map[l] for l in raw_test_preds])

        oos_dates.extend(test_slice.index)
        oos_labels.extend(aligned_test_preds)

        raw_train_labels = km.predict(X_train_scaled)
        train_sil = (
            float(silhouette_score(X_train_scaled, raw_train_labels))
            if len(np.unique(raw_train_labels)) > 1
            else 0.0
        )

        refit_records.append({
            "step": refit_step,
            "refit_date": features_df.index[curr_idx],
            "train_size": curr_idx,
            "test_size": len(test_slice),
            "train_inertia": float(km.inertia_),
            "train_silhouette": train_sil,
        })

        curr_idx = test_end
        refit_step += 1

    oos_df = pd.DataFrame(
        {"wf_regime": oos_labels},
        index=pd.DatetimeIndex(oos_dates),
    )
    refit_df = pd.DataFrame(refit_records).set_index("step")

    logger.info(
        "Walk-forward evaluation completed: %d steps, %d out-of-sample predictions (Hungarian alignment applied).",
        refit_step, len(oos_df)
    )

    if not return_diagnostics:
        return oos_df, refit_df

    # Out-of-sample empirical transition matrix P(S_{t+1} = j | S_t = i)
    raw_crosstab = pd.crosstab(
        oos_df["wf_regime"].shift(1),
        oos_df["wf_regime"],
        normalize="index",
    )
    trans_matrix = raw_crosstab.reindex(index=range(k), columns=range(k), fill_value=0.0)
    for r in range(k):
        if trans_matrix.loc[r].sum() == 0.0:
            trans_matrix.loc[r, r] = 1.0  # absorbing if unvisited

    # Persistence / run-length statistics
    episodes = (oos_df["wf_regime"] != oos_df["wf_regime"].shift()).cumsum()
    pers_records = []
    for reg in range(k):
        sub_eps = episodes[oos_df["wf_regime"] == reg]
        if len(sub_eps) > 0:
            lengths = sub_eps.value_counts().values
            pers_records.append({
                "Regime": f"State_{reg}",
                "Total_Observations": len(sub_eps),
                "Occupancy_Pct": round(len(sub_eps) / len(oos_df) * 100.0, 1),
                "Distinct_Episodes": len(lengths),
                "Mean_Duration_Days": round(float(np.mean(lengths)), 1),
                "Median_Duration_Days": round(float(np.median(lengths)), 1),
                "Max_Duration_Days": int(np.max(lengths)),
            })
        else:
            pers_records.append({
                "Regime": f"State_{reg}",
                "Total_Observations": 0,
                "Occupancy_Pct": 0.0,
                "Distinct_Episodes": 0,
                "Mean_Duration_Days": np.nan,
                "Median_Duration_Days": np.nan,
                "Max_Duration_Days": 0,
            })

    persistence_df = pd.DataFrame(pers_records).set_index("Regime")
    return oos_df, refit_df, trans_matrix, persistence_df


# -----------------------------------------------------------------------------
# 8. Hypothetical Forward Trading Strategy Simulation (Priority 2)
# -----------------------------------------------------------------------------
def simulate_regime_switching_strategy(
    regime_labels: pd.Series,
    benchmark_returns: pd.Series,
    cash_annual_return: float = 0.0,
    cost_per_trade_bps: float = 5.0,
    trading_days_per_year: int = 252,
) -> Dict[str, Any]:
    """Simulate a hypothetical out-of-sample regime-switching strategy with strictly forward execution.

    CRITICAL METHODOLOGICAL DISTINCTION (Priority 2):
        1. Lookahead-Free Execution Timing:
           Features x(t) and regime label S(t) are calculated at the end of trading day t.
           Trading decisions established at day t close are executed to capture return r(t+1) on day t+1.
           Day t's return is NEVER attributed to the signal generated from day t's closing data.
        2. Exposure Rules:
           - Calm / Normal Regimes (State 0, State 2): 100% long benchmark equity (w_{t+1} = 1.0).
           - Turbulent Stress Regime (State 1): 0% equity, 100% cash (w_{t+1} = 0.0, de-risking).
        3. Transaction Costs:
           Every portfolio switch between equity and cash incurs a transaction cost of `cost_per_trade_bps` (default 5 bps).
        4. Academic & Retrospective Disclaimer:
           THIS ANALYSIS IS STRICTLY HYPOTHETICAL AND RETROSPECTIVE. It evaluates signal latency and forward
           risk mitigation properties, NOT live trading execution or an actionable investment recommendation.
           Past hypothetical backtests do not guarantee future profitability.

    Args:
        regime_labels: pd.Series of regime predictions S_t indexed by date.
        benchmark_returns: pd.Series of daily returns for the equal-weighted benchmark.
        cash_annual_return: Annualized return on cash (default 0.0).
        cost_per_trade_bps: Transaction cost per trade in basis points (1 bp = 0.0001).
        trading_days_per_year: Annualization factor (252).

    Returns:
        Dict containing:
            - 'summary': pd.DataFrame comparing Strategy vs Buy & Hold Benchmark.
            - 'returns': pd.DataFrame of daily strategy, benchmark, and cash returns.
    """
    # Align dates
    common_idx = regime_labels.index.intersection(benchmark_returns.index).sort_values()
    if len(common_idx) < 2:
        raise ValueError("Insufficient overlapping dates between regime labels and benchmark returns.")

    aligned_labels = regime_labels.loc[common_idx]
    aligned_bench = benchmark_returns.loc[common_idx]

    # Convert regime S_t into next-day weight w_{t+1}
    # State 1 = Turbulent Stress -> Cash (w = 0.0); Other states -> Equity (w = 1.0)
    target_weights = np.where(aligned_labels == 1, 0.0, 1.0)
    target_weights_series = pd.Series(target_weights, index=common_idx)

    # Shift weights by 1 day forward to ensure lookahead-free execution: w_{t+1} applies to r_{t+1}
    fwd_weights = target_weights_series.shift(1).dropna()
    fwd_bench_rets = aligned_bench.loc[fwd_weights.index]

    daily_cash_ret = (1.0 + cash_annual_return) ** (1.0 / trading_days_per_year) - 1.0
    turnover = np.abs(fwd_weights - fwd_weights.shift(1).fillna(1.0))
    tx_fee = turnover * (cost_per_trade_bps / 10000.0)

    gross_strat_rets = fwd_weights * fwd_bench_rets + (1.0 - fwd_weights) * daily_cash_ret
    net_strat_rets = gross_strat_rets - tx_fee

    # Compute comparative performance statistics
    def _compute_stats(rets: pd.Series) -> Dict[str, float]:
        r = rets.values
        n = len(r)
        cum_ret = float(np.prod(1.0 + r) - 1.0)
        ann_arith = float(np.mean(r) * trading_days_per_year)
        ann_geom = float(((1.0 + cum_ret) ** (trading_days_per_year / n)) - 1.0) if cum_ret > -1.0 else -1.0
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(trading_days_per_year))
        sharpe = float((ann_arith - cash_annual_return) / ann_vol) if ann_vol > 1e-8 else 0.0

        cum_wealth = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum_wealth)
        dd = (cum_wealth - peak) / peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
        calmar = float(ann_geom / abs(max_dd)) if abs(max_dd) > 1e-8 else 0.0

        return {
            "Cumulative_Return": round(cum_ret, 4),
            "Arithmetic_Annualized_Return": round(ann_arith, 4),
            "Geometric_Annualized_Return": round(ann_geom, 4),
            "Annualized_Volatility": round(ann_vol, 4),
            "Sharpe_Ratio": round(sharpe, 3),
            "Maximum_Drawdown": round(max_dd, 4),
            "Calmar_Ratio": round(calmar, 3),
        }

    bench_stats = _compute_stats(fwd_bench_rets)
    bench_stats["Total_Trades"] = 0
    bench_stats["Transaction_Drag_Pct"] = 0.0

    strat_stats = _compute_stats(net_strat_rets)
    strat_stats["Total_Trades"] = int(np.sum(turnover > 0))
    strat_stats["Transaction_Drag_Pct"] = round(float(np.sum(tx_fee)), 4)

    summary_df = pd.DataFrame({
        "Hypothetical Regime-Switching Strategy (Lookahead-Free)": strat_stats,
        "Equal-Weighted European Benchmark (Buy & Hold)": bench_stats,
    })

    returns_df = pd.DataFrame({
        "strategy_net_return": net_strat_rets,
        "strategy_gross_return": gross_strat_rets,
        "benchmark_return": fwd_bench_rets,
        "equity_weight": fwd_weights,
        "transaction_fee": tx_fee,
    })

    return {
        "summary": summary_df,
        "returns": returns_df,
    }



