"""Unit tests for visualization functions in src/visualization.py."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization import (
    plot_correlation_heatmap,
    plot_spectral_phase_space,
    plot_regime_timeline,
    plot_safe_haven_scatter,
    plot_transition_matrix,
)


@pytest.fixture
def sample_corr_matrix():
    data = [
        [1.0, 0.75, 0.60],
        [0.75, 1.0, 0.70],
        [0.60, 0.70, 1.0],
    ]
    return pd.DataFrame(data, index=["UK", "FR", "DE"], columns=["UK", "FR", "DE"])


@pytest.fixture
def sample_metrics_df():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "mean_correlation": np.random.uniform(0.4, 0.9, size=100),
            "spectral_entropy": np.random.uniform(0.2, 0.8, size=100),
        },
        index=dates,
    )


# -----------------------------------------------------------------------------
# plot_correlation_heatmap tests
# -----------------------------------------------------------------------------
def test_plot_correlation_heatmap_valid(sample_corr_matrix):
    fig = plot_correlation_heatmap(sample_corr_matrix)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_correlation_heatmap_with_nan():
    data = [
        [1.0, np.nan, 0.60],
        [np.nan, 1.0, 0.70],
        [0.60, 0.70, 1.0],
    ]
    df = pd.DataFrame(data, index=["A", "B", "C"], columns=["A", "B", "C"])
    fig = plot_correlation_heatmap(df, annot=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmap_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_correlation_heatmap(pd.DataFrame())

    with pytest.raises(ValueError, match="cannot be None"):
        plot_correlation_heatmap(None)


def test_plot_correlation_heatmap_non_square_raises():
    df = pd.DataFrame([[1.0, 0.5]], columns=["A", "B"])
    with pytest.raises(ValueError, match="must be square"):
        plot_correlation_heatmap(df)


# -----------------------------------------------------------------------------
# plot_spectral_phase_space tests
# -----------------------------------------------------------------------------
def test_plot_spectral_phase_space_valid(sample_metrics_df):
    fig1 = plot_spectral_phase_space(sample_metrics_df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    regimes = pd.Series(np.random.choice([0, 1, 2], size=len(sample_metrics_df)), index=sample_metrics_df.index)
    fig2 = plot_spectral_phase_space(sample_metrics_df, regime_series=regimes)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_plot_spectral_phase_space_missing_columns_raises(sample_metrics_df):
    bad_df = sample_metrics_df.drop(columns=["spectral_entropy"])
    with pytest.raises(ValueError, match="must contain column 'spectral_entropy'"):
        plot_spectral_phase_space(bad_df)


def test_plot_spectral_phase_space_empty_raises():
    with pytest.raises(ValueError, match="must be a non-empty DataFrame"):
        plot_spectral_phase_space(pd.DataFrame())


def test_plot_spectral_phase_space_unaligned_regimes_raises(sample_metrics_df):
    # Regimes on disjoint date index
    disjoint_regimes = pd.Series([0, 1, 2], index=pd.date_range("1990-01-01", periods=3, freq="D"))
    with pytest.raises(ValueError, match="No overlapping date indices"):
        plot_spectral_phase_space(sample_metrics_df, regime_series=disjoint_regimes)


# -----------------------------------------------------------------------------
# plot_regime_timeline tests
# -----------------------------------------------------------------------------
def test_plot_regime_timeline_valid(sample_metrics_df):
    regimes = pd.Series(np.random.choice([0, 1, 2], size=len(sample_metrics_df)), index=sample_metrics_df.index)
    bench = pd.Series(np.cumprod(1 + np.random.normal(0.0005, 0.01, size=len(sample_metrics_df))), index=sample_metrics_df.index)
    crisis_dict = {
        "Event1": ("2020-02-01", "2020-03-01"),
        "InvalidEvent": ("invalid_date", "2020-04-01"),  # Should handle gracefully
    }

    fig = plot_regime_timeline(regimes, benchmark_series=bench, crisis_events=crisis_dict)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_regime_timeline_empty_raises():
    with pytest.raises(ValueError, match="must be a non-empty pandas Series"):
        plot_regime_timeline(pd.Series(dtype=float))


def test_plot_regime_timeline_all_nan_raises():
    with pytest.raises(ValueError, match="contains only NaN values"):
        plot_regime_timeline(pd.Series([np.nan, np.nan]))


# -----------------------------------------------------------------------------
# plot_safe_haven_scatter tests
# -----------------------------------------------------------------------------
def test_plot_safe_haven_scatter_valid():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    bench = pd.Series(np.random.normal(0, 0.01, size=100), index=dates)
    asset = pd.Series(0.8 * bench + np.random.normal(0, 0.005, size=100), index=dates)

    fig = plot_safe_haven_scatter(asset, bench, stress_quantile=0.05, market_name="DAX")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_safe_haven_scatter_invalid_quantile():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.random.randn(10), index=dates)
    with pytest.raises(ValueError, match="stress_quantile must be strictly between 0 and 1"):
        plot_safe_haven_scatter(s, s, stress_quantile=1.5)


def test_plot_safe_haven_scatter_empty_or_no_overlap():
    s1 = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    s2 = pd.Series([0.01, 0.02], index=pd.date_range("2021-01-01", periods=2, freq="D"))
    with pytest.raises(ValueError, match="Fewer than 2 overlapping"):
        plot_safe_haven_scatter(s1, s2)


# -----------------------------------------------------------------------------
# plot_transition_matrix tests
# -----------------------------------------------------------------------------
def test_plot_transition_matrix_valid():
    data = [
        [0.90, 0.08, 0.02],
        [0.10, 0.85, 0.05],
        [0.05, 0.05, 0.90],
    ]
    df = pd.DataFrame(data, index=["State 0", "State 1", "State 2"], columns=["State 0", "State 1", "State 2"])
    fig = plot_transition_matrix(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_transition_matrix_non_square_raises():
    df = pd.DataFrame([[0.5, 0.5]], columns=["A", "B"])
    with pytest.raises(ValueError, match="must be square"):
        plot_transition_matrix(df)


def test_plot_transition_matrix_empty_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_transition_matrix(pd.DataFrame())
