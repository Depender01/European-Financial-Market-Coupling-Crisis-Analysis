"""Unit tests and benchmarks for epoch distance calculation."""

import time
import numpy as np
import pytest

from src.correlation import (
    compute_epoch_distance_matrix,
    compute_epoch_distance_reference,
)


@pytest.fixture
def synthetic_correlation_epochs() -> np.ndarray:
    """Fixture providing a series of 20 correlation matrices (N=5 assets)."""
    np.random.seed(42)
    N_epochs = 20
    N = 5
    mats = np.zeros((N_epochs, N, N), dtype=np.float64)

    for i in range(N_epochs):
        # Generate random covariance via Wishart-like matrix
        A = np.random.randn(N, 10)
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        mats[i] = 0.5 * (corr + corr.T)

    return mats


def test_epoch_distance_numerical_equivalence(synthetic_correlation_epochs: np.ndarray):
    """Verify that the optimized SciPy cdist implementation is numerically identical to reference."""
    dist_opt = compute_epoch_distance_matrix(synthetic_correlation_epochs)
    dist_ref = compute_epoch_distance_reference(synthetic_correlation_epochs)

    assert dist_opt.shape == dist_ref.shape
    # Must be bitwise / float-equivalent within floating precision
    np.testing.assert_allclose(dist_opt, dist_ref, rtol=1e-5, atol=1e-6)


def test_epoch_distance_mathematical_properties(synthetic_correlation_epochs: np.ndarray):
    """Verify distance metric properties: non-negativity, symmetry, and zero diagonal."""
    zeta = compute_epoch_distance_matrix(synthetic_correlation_epochs)
    N_epochs = len(synthetic_correlation_epochs)

    assert zeta.shape == (N_epochs, N_epochs)

    # 1. Zero diagonal: dist(tau, tau) = 0
    np.testing.assert_allclose(np.diag(zeta), 0.0, atol=1e-7)

    # 2. Symmetry: dist(tau_1, tau_2) == dist(tau_2, tau_1)
    np.testing.assert_allclose(zeta, zeta.T, rtol=1e-6)

    # 3. Non-negativity: dist(tau_1, tau_2) >= 0
    assert (zeta >= 0.0).all()


def test_epoch_distance_duplicate_matrices():
    """Identical correlation matrices must have strictly 0 distance."""
    N = 4
    m1 = np.eye(N)
    m2 = np.eye(N)
    mats = np.array([m1, m2])
    zeta = compute_epoch_distance_matrix(mats)
    assert zeta[0, 1] == 0.0
    assert zeta[1, 0] == 0.0


def test_epoch_distance_chunked_equivalence(synthetic_correlation_epochs: np.ndarray):
    """Chunked calculation must be bitwise identical to unchunked calculation."""
    zeta_full = compute_epoch_distance_matrix(synthetic_correlation_epochs, chunk_size=None)
    zeta_chunked = compute_epoch_distance_matrix(synthetic_correlation_epochs, chunk_size=5)

    np.testing.assert_allclose(zeta_full, zeta_chunked, rtol=1e-7)


def test_epoch_distance_known_extremes():
    """Verify distance between known extreme matrices: Identity (rho=0) vs All-Ones (rho=1)."""
    N = 4
    id_mat = np.eye(N)
    ones_mat = np.ones((N, N))

    mats = np.array([id_mat, ones_mat])
    zeta = compute_epoch_distance_matrix(mats)

    assert zeta[0, 0] == 0.0
    assert zeta[1, 1] == 0.0
    assert pytest.approx(zeta[0, 1], rel=1e-6) == 1.0
    assert pytest.approx(zeta[1, 0], rel=1e-6) == 1.0


def test_epoch_distance_benchmark():
    """Benchmark the optimized SciPy cdist method against the reference implementation on 200 epochs."""
    np.random.seed(123)
    N_epochs = 200
    N = 8
    P = N * (N - 1) // 2

    # Create dummy correlation matrices
    corr_mats = np.zeros((N_epochs, N, N), dtype=np.float64)
    for i in range(N_epochs):
        corr_mats[i] = np.eye(N) + 0.05 * np.random.randn(N, N)
        corr_mats[i] = 0.5 * (corr_mats[i] + corr_mats[i].T)
        np.fill_diagonal(corr_mats[i], 1.0)
        corr_mats[i] = np.clip(corr_mats[i], -1.0, 1.0)

    # Benchmark optimized
    t0 = time.perf_counter()
    dist_opt = compute_epoch_distance_matrix(corr_mats)
    t_opt = time.perf_counter() - t0

    # Benchmark reference loop
    t0 = time.perf_counter()
    dist_ref = compute_epoch_distance_reference(corr_mats)
    t_ref = time.perf_counter() - t0

    # Ensure equivalence
    np.testing.assert_allclose(dist_opt, dist_ref, rtol=1e-5, atol=1e-6)

    # Optimized should be significantly faster or at worst comparable on small batches
    assert t_opt < 5.0  # Must complete in under 5 seconds
    print(f"\n[BENCHMARK] N_epochs={N_epochs}, Assets={N}: Optimized={t_opt*1000:.2f}ms, Reference={t_ref*1000:.2f}ms (Speedup: {t_ref/max(t_opt, 1e-6):.1f}x)")
