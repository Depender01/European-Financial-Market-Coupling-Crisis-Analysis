"""Unit tests for rolling correlation and power-mapping module."""

import numpy as np
import pandas as pd
import pytest

from src.correlation import (
    compute_rolling_correlation_matrices,
    power_map_correlation_matrix,
    extract_upper_triangular_flat,
)


@pytest.fixture
def synthetic_correlated_returns() -> pd.DataFrame:
    """Fixture with 3 assets with known statistical relationships:

    Asset A: random standard normal
    Asset B: identical to A (rho = +1.0)
    Asset C: perfectly inverted to A (rho = -1.0)
    """
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    a = np.random.randn(50)
    data = {
        "Asset_A": a,
        "Asset_B": a.copy(),        # Correlated +1.0
        "Asset_C": -a.copy(),       # Correlated -1.0
    }
    return pd.DataFrame(data, index=dates)


def test_rolling_correlation_known_values(synthetic_correlated_returns: pd.DataFrame):
    """Test rolling correlation captures exact analytical correlations (+1.0 and -1.0)."""
    corr_mats, dates, assets = compute_rolling_correlation_matrices(
        synthetic_correlated_returns,
        window=20,
        stride=1,
    )

    assert len(corr_mats) == 31  # 50 - 20 + 1
    assert len(dates) == 31
    assert assets == ["Asset_A", "Asset_B", "Asset_C"]

    # In every window: corr(A, B) must be 1.0, corr(A, C) must be -1.0
    for C in corr_mats:
        assert pytest.approx(C[0, 1], rel=1e-5) == 1.0   # A vs B
        assert pytest.approx(C[0, 2], rel=1e-5) == -1.0  # A vs C
        assert pytest.approx(C[1, 2], rel=1e-5) == -1.0  # B vs C
        assert pytest.approx(C[0, 0], rel=1e-5) == 1.0   # Diagonal
        assert pytest.approx(C[1, 1], rel=1e-5) == 1.0
        assert pytest.approx(C[2, 2], rel=1e-5) == 1.0


def test_rolling_correlation_invalid_parameters(synthetic_correlated_returns: pd.DataFrame):
    """Test validation errors for invalid window, stride, and min_periods."""
    # Data length (50) < window (60)
    with pytest.raises(ValueError, match="Insufficient data"):
        compute_rolling_correlation_matrices(synthetic_correlated_returns, window=60)

    # min_periods > window
    with pytest.raises(ValueError, match="cannot exceed window"):
        compute_rolling_correlation_matrices(synthetic_correlated_returns, window=20, min_periods=25)

    # stride < 1
    with pytest.raises(ValueError, match="stride must be >= 1"):
        compute_rolling_correlation_matrices(synthetic_correlated_returns, window=20, stride=0)

    # Non-dataframe
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        compute_rolling_correlation_matrices([1, 2, 3])  # type: ignore


def test_power_map_correlation_matrix():
    """Verify power mapping transformation formula and diagonal preservation."""
    C = np.array([
        [1.0,  0.5, -0.5],
        [0.5,  1.0,  0.2],
        [-0.5, 0.2,  1.0],
    ])

    eps = 0.6  # Exponent = 1 + 0.6 = 1.6
    C_pm = power_map_correlation_matrix(C, epsilon=eps)

    # Diagonal must strictly remain 1.0
    assert np.all(np.diag(C_pm) == 1.0)

    # Symmetry preserved
    np.testing.assert_allclose(C_pm, C_pm.T, rtol=1e-7)

    # Positive value: 0.5^1.6 = 0.32987697769
    expected_pos = float(0.5 ** 1.6)
    assert pytest.approx(C_pm[0, 1], rel=1e-6) == expected_pos

    # Negative value: sign(-0.5) * |-0.5|^1.6 = -0.32987697769
    assert pytest.approx(C_pm[0, 2], rel=1e-6) == -expected_pos

    # 0.2^1.6 = 0.0761276
    expected_small = float(0.2 ** 1.6)
    assert pytest.approx(C_pm[1, 2], rel=1e-6) == expected_small


def test_extract_upper_triangular_flat():
    """Verify upper-triangular off-diagonal extraction shape and order."""
    N = 4
    P = N * (N - 1) // 2  # 6
    N_epochs = 5

    # Create dummy correlation matrices
    corr_mats = np.zeros((N_epochs, N, N))
    for ep in range(N_epochs):
        corr_mats[ep] = np.eye(N)
        corr_mats[ep, 0, 1] = corr_mats[ep, 1, 0] = 0.1 * (ep + 1)
        corr_mats[ep, 0, 2] = corr_mats[ep, 2, 0] = 0.2
        corr_mats[ep, 0, 3] = corr_mats[ep, 3, 0] = 0.3
        corr_mats[ep, 1, 2] = corr_mats[ep, 2, 1] = 0.4
        corr_mats[ep, 1, 3] = corr_mats[ep, 3, 1] = 0.5
        corr_mats[ep, 2, 3] = corr_mats[ep, 3, 2] = 0.6
        corr_mats[ep] = 0.5 * (corr_mats[ep] + corr_mats[ep].T)

    flat = extract_upper_triangular_flat(corr_mats)
    assert flat.shape == (N_epochs, P)
    assert pytest.approx(flat[0, 0]) == 0.1  # (0, 1) element of epoch 0
    assert pytest.approx(flat[1, 0]) == 0.2  # (0, 1) element of epoch 1


def test_rolling_correlation_constant_return_skipped():
    """Constant-return series (variance = 0) must be skipped rather than producing NaN matrices."""
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    df = pd.DataFrame({
        "A": np.random.randn(25),
        "B": np.zeros(25),  # 0 variance
    }, index=dates)

    # All windows have 0 variance in B, so no valid correlation matrix can be formed
    with pytest.raises(ValueError, match="No valid correlation matrices"):
        compute_rolling_correlation_matrices(df, window=20)


def test_rolling_correlation_strictly_trailing_no_lookahead():
    """Verify that correlation at epoch date tau contains ONLY data up to tau (no lookahead bias)."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    returns_base = pd.DataFrame({
        "A": np.random.randn(40),
        "B": np.random.randn(40),
    }, index=dates)

    # Compute correlation trajectory
    mats_base, dates_base, _ = compute_rolling_correlation_matrices(returns_base, window=20, stride=1)

    # Create modified returns where observations after day 25 are drastically altered (extreme shock)
    returns_mod = returns_base.copy()
    returns_mod.iloc[26:, :] = 999.0

    mats_mod, dates_mod, _ = compute_rolling_correlation_matrices(returns_mod, window=20, stride=1)

    # Find the epoch index corresponding to day 25 (dates_base[5] since 20 + 5 = 25)
    epoch_25_idx = 5  # End date is day 20 + 5 = 25
    assert dates_base[epoch_25_idx] == dates_mod[epoch_25_idx]

    # The correlation matrices up to and including day 25 must be BITWISE IDENTICAL
    for i in range(epoch_25_idx + 1):
        np.testing.assert_allclose(mats_base[i], mats_mod[i], rtol=1e-7)
