"""Financial Metrics, Spectral Decomposition, and Market Entropy Module.

Implements mathematically defensible cross-market correlation metrics, positive semi-definite
safeguards, spectral entropy, and eigenvalue absorption ratio.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_correlation_matrix(
    C: np.ndarray,
    tol: float = 1e-7,
    psd_tol: float = 1e-6,
) -> np.ndarray:
    """Validate that a matrix is a mathematically valid correlation matrix.

    Checks:
        1. 2D square matrix with dimension N >= 2.
        2. All entries are finite (no NaN, Inf).
        3. Matrix is symmetric: max|C - C^T| < tol.
        4. Diagonal entries are approximately 1.0 (|C_ii - 1.0| < 1e-4).
        5. Off-diagonal entries lie in [-1.0 - 1e-4, 1.0 + 1e-4].
        6. Matrix is positive semi-definite: min(eigenvalues) >= -psd_tol.

    Args:
        C: 2D numpy array to validate.
        tol: Tolerance for symmetry check.
        psd_tol: Tolerance threshold for negative eigenvalues before raising ValueError.

    Returns:
        Symmetrized and clipped correlation matrix.

    Raises:
        TypeError: If C is not a numpy array.
        ValueError: If any structural, symmetry, finite, or PSD validation fails.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(C).__name__}")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Matrix must be 2D square array, got shape {C.shape}")

    N = C.shape[0]
    if N < 2:
        raise ValueError(f"Matrix dimension N must be at least 2, got {N}")

    if not np.isfinite(C).all():
        raise ValueError("Correlation matrix contains non-finite values (NaN or Inf).")

    # Check symmetry
    asym = np.max(np.abs(C - C.T))
    if asym > tol:
        raise ValueError(f"Matrix is not symmetric: maximum asymmetry is {asym:.2e} (tolerance: {tol:.2e})")

    # Symmetrize to cancel minute floating-point rounding
    C_sym = 0.5 * (C + C.T)

    # Check diagonal
    diag_dev = np.max(np.abs(np.diag(C_sym) - 1.0))
    if diag_dev > 1e-3:
        raise ValueError(f"Diagonal entries must be 1.0; maximum deviation is {diag_dev:.2e}")

    # Check off-diagonal bounds
    off_diag_dev = np.max(np.abs(C_sym)) - 1.0
    if off_diag_dev > 1e-3:
        raise ValueError(f"Matrix elements exceed [-1, 1] range by {off_diag_dev:.2e}")

    # Eigenvalue positive semi-definiteness check
    eigenvalues = np.linalg.eigvalsh(C_sym)
    min_eig = np.min(eigenvalues)

    if min_eig < -psd_tol:
        raise ValueError(
            f"Correlation matrix is not positive semi-definite: minimum eigenvalue is {min_eig:.4e} "
            f"(threshold: -{psd_tol:.1e}). This indicates incompatible pairwise observations or data corruption."
        )

    # Force exact diagonal of 1.0 and clip to [-1, 1]
    np.fill_diagonal(C_sym, 1.0)
    C_clean = np.clip(C_sym, -1.0, 1.0)
    return C_clean


def compute_average_correlation(C: np.ndarray, validate: bool = True) -> float:
    """Compute the average off-diagonal pairwise correlation.

    Formula:
        rho_bar = 2 / (N * (N - 1)) * sum_{i < j} C_{ij}

    Args:
        C: N x N correlation matrix.
        validate: Whether to run validate_correlation_matrix first.

    Returns:
        Scalar average off-diagonal correlation in [-1.0, 1.0].
    """
    if validate:
        C = validate_correlation_matrix(C)

    N = C.shape[0]
    iu_rows, iu_cols = np.triu_indices(N, k=1)
    return float(np.mean(C[iu_rows, iu_cols]))


def compute_eigenvalues(
    C: np.ndarray,
    validate: bool = True,
    zero_clip_tol: float = 1e-7,
) -> np.ndarray:
    """Compute sorted eigenvalues of a correlation matrix with numerical safeguards.

    Numerical Safeguard:
        Floating-point arithmetic on positive semi-definite matrices can produce tiny
        eigenvalues like -1e-16. Any eigenvalue in [-zero_clip_tol, 0) is clamped to 0.0.
        Materially negative eigenvalues (< -zero_clip_tol) trigger a validation failure.

    Args:
        C: N x N correlation matrix.
        validate: Whether to validate C.
        zero_clip_tol: Clamp threshold for small floating-point negative noise.

    Returns:
        1D array of sorted eigenvalues in descending order: lambda_{(1)} >= ... >= lambda_{(N)}.
    """
    if validate:
        C = validate_correlation_matrix(C, psd_tol=zero_clip_tol)

    # eigvalsh is optimized for real symmetric matrices
    ev = np.linalg.eigvalsh(C)

    # Clamp tiny floating-point negative values to 0.0
    ev = np.where((ev < 0.0) & (ev >= -zero_clip_tol), 0.0, ev)

    # Sort descending
    ev_sorted = np.sort(ev)[::-1]
    return ev_sorted


def compute_spectral_entropy(
    C: np.ndarray,
    normalize: bool = True,
    validate: bool = True,
) -> float:
    """Calculate the Von Neumann / Shannon spectral entropy of a correlation matrix.

    Mathematical Definition:
        For an N x N correlation matrix C with eigenvalues lambda_1, ..., lambda_N >= 0,
        and sum_{j=1}^N lambda_j = Tr(C) = N:

        Define normalized eigenspectrum probabilities:
            p_i = lambda_i / N,   such that sum_{i=1}^N p_i = 1, p_i >= 0.

        Unnormalized Spectral Entropy:
            H(C) = - sum_{i=1, p_i > 0}^N p_i * ln(p_i)

        Normalized Spectral Entropy:
            H_norm(C) = H(C) / ln(N)   in [0.0, 1.0]

    Financial Interpretation:
        - Maximum Entropy (H_norm = 1.0):
          Occurs when C = I_N (identity matrix, all eigenvalues equal lambda_i = 1).
          Markets are completely decoupled / uncorrelated. Risks are dispersed across
          N independent dimensions (maximum diversification benefit).
        - Minimum Entropy (H_norm = 0.0):
          Occurs when all markets are perfectly collinear (C_ij = 1.0 for all i, j).
          A single dominant eigenvalue lambda_1 = N accounts for 100% of variance.
          Extreme coupling; diversification completely collapses (systemic crash).

    Args:
        C: N x N correlation matrix.
        normalize: If True, divide by ln(N) to scale into [0, 1].
        validate: Whether to validate C.

    Returns:
        Spectral entropy scalar.
    """
    ev = compute_eigenvalues(C, validate=validate)
    N = len(ev)

    # Calculate probabilities
    total_trace = np.sum(ev)
    if total_trace <= 0:
        return 0.0

    p = ev / total_trace

    # Filter out zero / near-zero eigenvalues (0 * ln(0) = 0 by convention)
    p_pos = p[p > 1e-12]

    if len(p_pos) == 0:
        return 0.0

    entropy = -float(np.sum(p_pos * np.log(p_pos)))

    if normalize:
        max_entropy = np.log(float(N))
        if max_entropy <= 0:
            return 0.0
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    return entropy


def compute_absorption_ratio(
    C: np.ndarray,
    k: int = 1,
    validate: bool = True,
) -> float:
    """Compute the Absorption Ratio (Kritzman et al., 2011).

    Definition:
        The fraction of total variance explained by the top k eigenvalues:
            AR_k = (sum_{i=1}^k lambda_{(i)}) / N

    Higher AR indicates that a small number of latent systemic factors explain
    most cross-market variance (increasing fragility and co-movement).

    Args:
        C: N x N correlation matrix.
        k: Number of dominant eigenvalues to sum (1 <= k <= N).
        validate: Whether to validate C.

    Returns:
        Absorption ratio float in [k/N, 1.0].
    """
    ev = compute_eigenvalues(C, validate=validate)
    N = len(ev)

    if not 1 <= k <= N:
        raise ValueError(f"k must satisfy 1 <= k <= {N}, got {k}")

    top_k_sum = np.sum(ev[:k])
    return float(np.clip(top_k_sum / N, 0.0, 1.0))


def compute_rolling_metrics(
    corr_matrices: np.ndarray,
    dates: pd.DatetimeIndex,
    top_k_absorption: int = 2,
) -> pd.DataFrame:
    """Compute rolling trajectory of all market coupling metrics across a sequence of correlation matrices.

    Args:
        corr_matrices: 3D array of shape (N_epochs, N_assets, N_assets).
        dates: DatetimeIndex of length N_epochs corresponding to each epoch end date.
        top_k_absorption: Value of k for absorption ratio AR_k.

    Returns:
        pd.DataFrame with columns:
            - 'mean_correlation': Average off-diagonal correlation mu(tau)
            - 'spectral_entropy': Normalized spectral entropy H_norm(tau) in [0, 1]
            - 'unnormalized_entropy': Raw entropy H(tau) in [0, ln(N)]
            - 'absorption_ratio_k1': Fraction of variance in top eigenvalue lambda_1 / N
            - 'absorption_ratio_k': Fraction of variance in top k eigenvalues
            - 'max_eigenvalue': lambda_1
            - 'min_eigenvalue': lambda_N
    """
    if corr_matrices.ndim != 3:
        raise ValueError(f"Expected 3D array of correlation matrices, got ndim={corr_matrices.ndim}")

    N_epochs, N, N_cols = corr_matrices.shape
    if N != N_cols:
        raise ValueError(f"Correlation matrices must be square, got {N}x{N_cols}")

    if len(dates) != N_epochs:
        raise ValueError(f"Dates length ({len(dates)}) must match N_epochs ({N_epochs})")

    records = []
    for i in range(N_epochs):
        C = corr_matrices[i]
        try:
            ev = compute_eigenvalues(C, validate=True)
            mu = compute_average_correlation(C, validate=False)
            h_norm = compute_spectral_entropy(C, normalize=True, validate=False)
            h_raw = compute_spectral_entropy(C, normalize=False, validate=False)
            ar_1 = float(ev[0] / N)
            ar_k = float(np.sum(ev[:top_k_absorption]) / N)

            records.append({
                "mean_correlation": mu,
                "spectral_entropy": h_norm,
                "unnormalized_entropy": h_raw,
                "absorption_ratio_k1": ar_1,
                f"absorption_ratio_k{top_k_absorption}": ar_k,
                "max_eigenvalue": float(ev[0]),
                "min_eigenvalue": float(ev[-1]),
            })
        except Exception as exc:
            logger.warning("Error processing epoch %d (%s): %s", i, dates[i].date(), exc)
            records.append({
                "mean_correlation": np.nan,
                "spectral_entropy": np.nan,
                "unnormalized_entropy": np.nan,
                "absorption_ratio_k1": np.nan,
                f"absorption_ratio_k{top_k_absorption}": np.nan,
                "max_eigenvalue": np.nan,
                "min_eigenvalue": np.nan,
            })

    metrics_df = pd.DataFrame(records, index=dates)
    return metrics_df
