"""Rolling Correlation and Epoch Distance Analysis Module.

Implements sliding-window pairwise Pearson correlation estimation, non-PSD power-mapping
heuristics, and memory-efficient vectorized epoch-distance calculation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .metrics import validate_correlation_matrix

logger = logging.getLogger(__name__)


def compute_rolling_correlation_matrices(
    returns_df: pd.DataFrame,
    window: int = 20,
    min_periods: Optional[int] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """Compute rolling pairwise Pearson correlation matrices across sliding epochs.

    Args:
        returns_df: DataFrame of asset returns (dates x assets).
        window: Sliding epoch duration in trading days (M). Default 20 days (~1 month).
        min_periods: Minimum valid observations required in an epoch window.
            Defaults to window (strict full window).
        stride: Window step size (Delta tau). Default 1 day (continuous daily trajectory).

    Returns:
        Tuple of:
            - 3D numpy array of shape (N_epochs, N_assets, N_assets)
            - DatetimeIndex of length N_epochs corresponding to the end date of each epoch
            - List of asset column names

    Raises:
        ValueError: If returns_df has fewer rows than window, or contains non-numeric data.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, received {type(returns_df).__name__}")

    if len(returns_df) < window:
        raise ValueError(
            f"Insufficient data: returns_df has {len(returns_df)} rows, but window is {window}."
        )

    if min_periods is None:
        min_periods = window

    if min_periods > window:
        raise ValueError(f"min_periods ({min_periods}) cannot exceed window ({window}).")

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    T, N = returns_df.shape
    asset_names = list(returns_df.columns)
    dates = returns_df.index

    matrices_list: List[np.ndarray] = []
    epoch_dates: List[pd.Timestamp] = []

    t = 0
    while t + window <= T:
        sub_df = returns_df.iloc[t : t + window]
        end_date = dates[t + window - 1]

        # Check observation count
        valid_obs = sub_df.dropna(how="any")
        if len(valid_obs) >= min_periods:
            # Check for zero or near-zero variance (constant-return series)
            stds = valid_obs.std(ddof=1)
            if (stds <= 1e-12).any():
                zero_var_cols = stds[stds <= 1e-12].index.tolist()
                logger.warning(
                    "Skipped window ending %s: zero or near-zero variance in asset(s) %s",
                    end_date, zero_var_cols
                )
                t += stride
                continue

            # Compute Pearson correlation matrix
            C_raw = valid_obs.corr().values

            if not np.isfinite(C_raw).all():
                logger.warning("Skipped window ending %s: non-finite correlation values.", end_date)
                t += stride
                continue

            # Enforce symmetry and diagonal 1.0
            C_sym = 0.5 * (C_raw + C_raw.T)
            np.fill_diagonal(C_sym, 1.0)
            C_clipped = np.clip(C_sym, -1.0, 1.0)

            # Validate for downstream spectral analysis
            try:
                C_valid = validate_correlation_matrix(C_clipped)
                matrices_list.append(C_valid.astype(np.float64))
                epoch_dates.append(end_date)
            except ValueError as val_err:
                logger.warning("Skipped window ending %s: invalid matrix (%s)", end_date, val_err)
        else:
            logger.debug(
                "Skipped window ending %s: valid observations (%d) < min_periods (%d)",
                end_date, len(valid_obs), min_periods
            )

        t += stride

    if not matrices_list:
        raise ValueError(
            f"No valid correlation matrices could be computed. Check missing data or min_periods ({min_periods})."
        )

    corr_array = np.array(matrices_list, dtype=np.float64)
    epoch_dates_idx = pd.DatetimeIndex(epoch_dates)

    logger.info(
        "Computed %d correlation matrices (window=%d, stride=%d, assets=%d)",
        len(corr_array), window, stride, N
    )
    return corr_array, epoch_dates_idx, asset_names


def power_map_correlation_matrix(
    C: np.ndarray,
    epsilon: float = 0.6,
) -> np.ndarray:
    """Apply non-linear power-mapping transformation to an individual correlation matrix.

    Mathematical Formula:
        C_{ij}^{(eps)} = sign(C_{ij}) * |C_{ij}|^(1 + eps),   for i != j
        C_{ii}^{(eps)} = 1.0

    CRITICAL METHODOLOGICAL WARNING:
        Elementwise power transformation generally DESTROYS the positive semi-definite (PSD)
        property of a correlation matrix. The transformed matrix C^{(eps)} can have negative
        eigenvalues and CANNOT be used for spectral entropy, principal component analysis,
        or risk modeling. It is strictly preserved as a legacy visualization heuristic.

    Args:
        C: N x N correlation matrix.
        epsilon: Power exponent (default 0.6 => exponent 1.6).

    Returns:
        Transformed symmetric matrix with diagonal 1.0.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"Expected 2D square matrix, got shape {C.shape}")

    C_pm = np.sign(C) * (np.abs(C) ** (1.0 + epsilon))
    # Preserve exact diagonal = 1.0
    np.fill_diagonal(C_pm, 1.0)
    # Ensure symmetry
    C_pm = 0.5 * (C_pm + C_pm.T)
    return np.clip(C_pm, -1.0, 1.0)


def extract_upper_triangular_flat(corr_matrices: np.ndarray) -> np.ndarray:
    """Extract flattened upper-triangular off-diagonal elements for a sequence of correlation matrices.

    Args:
        corr_matrices: 3D array of shape (N_epochs, N_assets, N_assets).

    Returns:
        2D array of shape (N_epochs, P), where P = N * (N - 1) / 2.
    """
    if corr_matrices.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {corr_matrices.shape}")

    N_epochs, N, N_cols = corr_matrices.shape
    if N != N_cols:
        raise ValueError(f"Matrices must be square, got {N}x{N_cols}")

    iu_rows, iu_cols = np.triu_indices(N, k=1)
    # Reshape and extract upper triangle in a single vectorized index operation
    flat = corr_matrices[:, iu_rows, iu_cols]
    return flat


def compute_epoch_distance_matrix(
    corr_matrices: np.ndarray,
    power_mapped: bool = False,
    epsilon: float = 0.6,
    chunk_size: Optional[int] = 1000,
) -> np.ndarray:
    """Compute pairwise epoch distance matrix zeta(tau_1, tau_2) using optimized C-level distance.

    Mathematical Definition:
        zeta(tau_1, tau_2) = (1 / P) * sum_{k=1}^P |u_k(tau_1) - u_k(tau_2)|

        where u(tau) is the flattened vector of P = N(N - 1)/2 upper-triangular correlation pairs.

    High-Performance Implementation:
        Notice that zeta(tau_1, tau_2) is mathematically identical to the Manhattan / Cityblock
        distance (L1 norm) divided by P.
        Using scipy.spatial.distance.cdist(flat, flat, metric='cityblock') / P computes the full
        symmetric distance matrix in compiled C. When chunk_size is provided, row blocks are
        computed incrementally to limit peak working memory.

    Args:
        corr_matrices: 3D array of shape (N_epochs, N_assets, N_assets).
        power_mapped: If True, apply power_map_correlation_matrix first (legacy comparison).
        epsilon: Power mapping exponent if power_mapped is True.
        chunk_size: Row block size for memory-conscious calculation (default 1000).

    Returns:
        2D symmetric array of shape (N_epochs, N_epochs) of pairwise epoch distances.
    """
    if power_mapped:
        mats = np.array([power_map_correlation_matrix(C, epsilon) for C in corr_matrices])
    else:
        mats = corr_matrices

    flat_epochs = extract_upper_triangular_flat(mats)  # (N_epochs, P)
    N_epochs, P = flat_epochs.shape

    if P == 0:
        raise ValueError("Matrix dimension must be >= 2 (P = 0).")

    dist_matrix = np.zeros((N_epochs, N_epochs), dtype=np.float32)

    if chunk_size is not None and chunk_size > 0 and N_epochs > chunk_size:
        # Memory-conscious chunked C computation
        for i in range(0, N_epochs, chunk_size):
            chunk = flat_epochs[i : i + chunk_size]
            dist_matrix[i : i + chunk_size] = (cdist(chunk, flat_epochs, metric="cityblock") / float(P)).astype(np.float32)
    else:
        dist_matrix = (cdist(flat_epochs, flat_epochs, metric="cityblock") / float(P)).astype(np.float32)

    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def compute_epoch_distance_reference(corr_matrices: np.ndarray) -> np.ndarray:
    """Naive reference implementation of epoch distance for verification and benchmarking.

    Uses an explicit loop to compute:
        zeta[i, j] = mean(|C_i[upper] - C_j[upper]|)
    """
    flat = extract_upper_triangular_flat(corr_matrices)
    N_epochs = len(flat)
    zeta = np.zeros((N_epochs, N_epochs), dtype=np.float32)

    for i in range(N_epochs):
        diff = np.abs(flat[i : i + 1] - flat)
        zeta[i] = diff.mean(axis=1)

    np.fill_diagonal(zeta, 0.0)
    return zeta
