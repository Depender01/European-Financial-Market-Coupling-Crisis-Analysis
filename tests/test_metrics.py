"""Unit tests for financial metrics, spectral decomposition, and entropy."""

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    validate_correlation_matrix,
    compute_average_correlation,
    compute_eigenvalues,
    compute_spectral_entropy,
    compute_absorption_ratio,
    compute_rolling_metrics,
)


@pytest.fixture
def identity_matrix_4x4() -> np.ndarray:
    """4x4 Identity matrix representing 4 completely uncorrelated assets."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def perfectly_correlated_matrix_4x4() -> np.ndarray:
    """4x4 matrix of all ones representing 4 perfectly co-moving assets."""
    return np.ones((4, 4), dtype=np.float64)


@pytest.fixture
def equicorrelated_matrix_4x4() -> np.ndarray:
    """4x4 matrix with off-diagonal correlation rho = 0.5.

    Analytical eigenvalues for N=4, rho=0.5:
    lambda_1 = 1 + (N - 1) * rho = 1 + 3 * 0.5 = 2.5
    lambda_2 = lambda_3 = lambda_4 = 1 - rho = 0.5
    Sum of eigenvalues = 2.5 + 0.5 + 0.5 + 0.5 = 4.0 = N.
    """
    rho = 0.5
    C = (1.0 - rho) * np.eye(4) + rho * np.ones((4, 4))
    return C


# -----------------------------------------------------------------------------
# Correlation Matrix Validation Tests
# -----------------------------------------------------------------------------
def test_validate_correlation_matrix_valid(identity_matrix_4x4: np.ndarray):
    """Identity matrix must pass validation."""
    C_clean = validate_correlation_matrix(identity_matrix_4x4)
    np.testing.assert_allclose(C_clean, identity_matrix_4x4)


def test_validate_correlation_matrix_non_square():
    """Non-square matrices must raise ValueError."""
    with pytest.raises(ValueError, match="2D square array"):
        validate_correlation_matrix(np.ones((3, 4)))


def test_validate_correlation_matrix_dimension_too_small():
    """1x1 matrix must raise ValueError (minimum dimension is 2)."""
    with pytest.raises(ValueError, match="dimension N must be at least 2"):
        validate_correlation_matrix(np.array([[1.0]]))


def test_validate_correlation_matrix_non_symmetric():
    """Asymmetric matrix must raise ValueError."""
    C = np.eye(3)
    C[0, 1] = 0.8
    C[1, 0] = 0.2  # Asymmetric!
    with pytest.raises(ValueError, match="not symmetric"):
        validate_correlation_matrix(C)


def test_validate_correlation_matrix_nan_inf():
    """Matrix containing NaN or Inf must raise ValueError."""
    C_nan = np.eye(3)
    C_nan[0, 1] = C_nan[1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        validate_correlation_matrix(C_nan)

    C_inf = np.eye(3)
    C_inf[0, 1] = C_inf[1, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite values"):
        validate_correlation_matrix(C_inf)


def test_validate_correlation_matrix_invalid_diagonal():
    """Diagonal entries not equal to 1.0 must raise ValueError."""
    C = np.eye(3)
    C[1, 1] = 0.85
    with pytest.raises(ValueError, match="Diagonal entries must be 1.0"):
        validate_correlation_matrix(C)


def test_validate_correlation_matrix_out_of_bounds():
    """Off-diagonal values outside [-1, 1] must raise ValueError."""
    C = np.eye(3)
    C[0, 1] = C[1, 0] = 1.05
    with pytest.raises(ValueError, match="elements exceed"):
        validate_correlation_matrix(C)


def test_validate_correlation_matrix_materially_negative_eigenvalues():
    """Materially non-positive-definite matrices must be rejected."""
    # Construct an invalid correlation matrix with high pairwise correlations that are geometrically impossible:
    # corr(A, B) = 0.9, corr(B, C) = 0.9, corr(A, C) = -0.9
    C_invalid = np.array([
        [ 1.0,  0.9, -0.9],
        [ 0.9,  1.0,  0.9],
        [-0.9,  0.9,  1.0],
    ])
    # Min eigenvalue for this matrix is approximately -0.62!
    with pytest.raises(ValueError, match="not positive semi-definite"):
        validate_correlation_matrix(C_invalid)


def test_validate_correlation_matrix_floating_point_noise_safeguard():
    """Floating-point rounding errors producing near-zero negative eigenvalues must be safeguarded."""
    # Matrix that has theoretical eigenvalue = 0, but might numerically produce -1e-16
    C = np.array([
        [1.0, 1.0],
        [1.0, 1.0],
    ])
    C_clean = validate_correlation_matrix(C, psd_tol=1e-6)
    assert C_clean is not None


# -----------------------------------------------------------------------------
# Average Correlation Tests
# -----------------------------------------------------------------------------
def test_compute_average_correlation(
    identity_matrix_4x4: np.ndarray,
    perfectly_correlated_matrix_4x4: np.ndarray,
    equicorrelated_matrix_4x4: np.ndarray,
):
    """Verify average off-diagonal correlation for canonical matrices."""
    # Identity: all off-diagonal = 0.0
    assert compute_average_correlation(identity_matrix_4x4) == 0.0

    # Perfectly correlated: all off-diagonal = 1.0
    assert compute_average_correlation(perfectly_correlated_matrix_4x4) == 1.0

    # Equicorrelated with rho = 0.5
    assert pytest.approx(compute_average_correlation(equicorrelated_matrix_4x4), rel=1e-7) == 0.5


# -----------------------------------------------------------------------------
# Spectral Decomposition Tests
# -----------------------------------------------------------------------------
def test_compute_eigenvalues_analytical(
    identity_matrix_4x4: np.ndarray,
    perfectly_correlated_matrix_4x4: np.ndarray,
    equicorrelated_matrix_4x4: np.ndarray,
):
    """Verify eigenvalues match exact analytical solutions."""
    # 1. Identity: all lambda_i = 1.0
    ev_id = compute_eigenvalues(identity_matrix_4x4)
    np.testing.assert_allclose(ev_id, [1.0, 1.0, 1.0, 1.0], rtol=1e-7)

    # 2. Perfectly correlated: lambda_1 = 4.0, lambda_2..4 = 0.0
    ev_perf = compute_eigenvalues(perfectly_correlated_matrix_4x4)
    np.testing.assert_allclose(ev_perf, [4.0, 0.0, 0.0, 0.0], atol=1e-7)

    # 3. Equicorrelated (N=4, rho=0.5): lambda_1 = 2.5, lambda_2..4 = 0.5
    ev_equi = compute_eigenvalues(equicorrelated_matrix_4x4)
    np.testing.assert_allclose(ev_equi, [2.5, 0.5, 0.5, 0.5], rtol=1e-7)


# -----------------------------------------------------------------------------
# Spectral Entropy Tests
# -----------------------------------------------------------------------------
def test_spectral_entropy_identity_matrix(identity_matrix_4x4: np.ndarray):
    """Identity matrix must yield MAXIMUM entropy H = ln(N) and H_norm = 1.0."""
    N = 4
    h_unnorm = compute_spectral_entropy(identity_matrix_4x4, normalize=False)
    h_norm = compute_spectral_entropy(identity_matrix_4x4, normalize=True)

    expected_unnorm = np.log(float(N))
    assert pytest.approx(h_unnorm, rel=1e-7) == expected_unnorm
    assert pytest.approx(h_norm, rel=1e-7) == 1.0


def test_spectral_entropy_collinear_matrix(perfectly_correlated_matrix_4x4: np.ndarray):
    """Perfect correlation must yield MINIMUM entropy H = 0.0 and H_norm = 0.0."""
    h_unnorm = compute_spectral_entropy(perfectly_correlated_matrix_4x4, normalize=False)
    h_norm = compute_spectral_entropy(perfectly_correlated_matrix_4x4, normalize=True)

    assert pytest.approx(h_unnorm, abs=1e-7) == 0.0
    assert pytest.approx(h_norm, abs=1e-7) == 0.0


def test_spectral_entropy_equicorrelated_matrix(equicorrelated_matrix_4x4: np.ndarray):
    """Equicorrelated matrix (rho=0.5) must match analytical Shannon entropy."""
    # Eigenvalues: [2.5, 0.5, 0.5, 0.5]
    # Probabilities: p_1 = 2.5/4 = 0.625; p_2 = p_3 = p_4 = 0.5/4 = 0.125
    # H = - [0.625 * ln(0.625) + 3 * (0.125 * ln(0.125))]
    p1 = 0.625
    p2 = 0.125
    expected_h = -(p1 * np.log(p1) + 3 * p2 * np.log(p2))
    expected_hnorm = expected_h / np.log(4.0)

    h_unnorm = compute_spectral_entropy(equicorrelated_matrix_4x4, normalize=False)
    h_norm = compute_spectral_entropy(equicorrelated_matrix_4x4, normalize=True)

    assert pytest.approx(h_unnorm, rel=1e-7) == expected_h
    assert pytest.approx(h_norm, rel=1e-7) == expected_hnorm
    assert 0.0 < h_norm < 1.0


# -----------------------------------------------------------------------------
# Absorption Ratio Tests
# -----------------------------------------------------------------------------
def test_absorption_ratio(
    identity_matrix_4x4: np.ndarray,
    perfectly_correlated_matrix_4x4: np.ndarray,
    equicorrelated_matrix_4x4: np.ndarray,
):
    """Verify absorption ratio formula AR_k = sum(lambda_top_k) / N."""
    # Identity: all lambda = 1.0 -> AR_1 = 1/4 = 0.25; AR_2 = 2/4 = 0.50
    assert compute_absorption_ratio(identity_matrix_4x4, k=1) == 0.25
    assert compute_absorption_ratio(identity_matrix_4x4, k=2) == 0.50

    # Perfect correlation: lambda_1 = 4.0 -> AR_1 = 4/4 = 1.0
    assert pytest.approx(compute_absorption_ratio(perfectly_correlated_matrix_4x4, k=1), rel=1e-7) == 1.0

    # Equicorrelated (2.5, 0.5, 0.5, 0.5): AR_1 = 2.5/4 = 0.625; AR_2 = (2.5+0.5)/4 = 0.75
    assert pytest.approx(compute_absorption_ratio(equicorrelated_matrix_4x4, k=1), rel=1e-7) == 0.625
    assert pytest.approx(compute_absorption_ratio(equicorrelated_matrix_4x4, k=2), rel=1e-7) == 0.75

    # Invalid k
    with pytest.raises(ValueError, match="k must satisfy"):
        compute_absorption_ratio(identity_matrix_4x4, k=0)
    with pytest.raises(ValueError, match="k must satisfy"):
        compute_absorption_ratio(identity_matrix_4x4, k=5)


# -----------------------------------------------------------------------------
# Rolling Metrics Aggregation Tests
# -----------------------------------------------------------------------------
def test_compute_rolling_metrics_table():
    """Verify rolling metrics DataFrame generation across a sequence of matrices."""
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    corr_mats = np.array([
        np.eye(3),
        np.ones((3, 3)),
        (1 - 0.4) * np.eye(3) + 0.4 * np.ones((3, 3)),
    ])

    metrics_df = compute_rolling_metrics(corr_mats, dates, top_k_absorption=2)

    assert isinstance(metrics_df, pd.DataFrame)
    assert len(metrics_df) == 3
    assert set(metrics_df.columns) == {
        "mean_correlation",
        "spectral_entropy",
        "unnormalized_entropy",
        "absorption_ratio_k1",
        "absorption_ratio_k2",
        "max_eigenvalue",
        "min_eigenvalue",
    }

    # Epoch 0 (Identity): mu=0.0, H_norm=1.0, AR_1=1/3
    assert metrics_df.iloc[0]["mean_correlation"] == 0.0
    assert pytest.approx(metrics_df.iloc[0]["spectral_entropy"], rel=1e-5) == 1.0
    assert pytest.approx(metrics_df.iloc[0]["absorption_ratio_k1"], rel=1e-5) == 1.0 / 3.0

    # Epoch 1 (Collinear): mu=1.0, H_norm=0.0, AR_1=1.0
    assert metrics_df.iloc[1]["mean_correlation"] == 1.0
    assert pytest.approx(metrics_df.iloc[1]["spectral_entropy"], abs=1e-5) == 0.0
    assert pytest.approx(metrics_df.iloc[1]["absorption_ratio_k1"], rel=1e-5) == 1.0
