"""Financial Market Visualization and Diagnostics Module.

Provides modular, publication-quality plotting routines using Matplotlib and Seaborn
for European financial market coupling, spectral dynamics, macro regimes, and safe-haven audits.
All routines validate inputs, handle edge cases defensively, and return Matplotlib Figure objects.
"""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    corr_matrix: Union[pd.DataFrame, np.ndarray],
    title: str = "Pairwise Equity Market Correlation Matrix",
    cmap: str = "coolwarm",
    annot: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot an annotated heatmap for a pairwise cross-market correlation matrix.

    Args:
        corr_matrix: N x N square correlation DataFrame or numpy array.
        title: Plot title.
        cmap: Matplotlib colormap name.
        annot: Whether to annotate values inside cells.
        vmin: Colormap minimum value.
        vmax: Colormap maximum value.
        figsize: Figure dimensions (width, height).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ValueError: If input is None, empty, not 2D, or not square.
    """
    if corr_matrix is None:
        raise ValueError("Correlation matrix cannot be None.")
    
    if not isinstance(corr_matrix, pd.DataFrame):
        corr_matrix = pd.DataFrame(corr_matrix)

    if corr_matrix.empty or corr_matrix.shape[0] == 0:
        raise ValueError("Correlation matrix cannot be empty.")

    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(
            f"Correlation matrix must be square (N x N), got shape {corr_matrix.shape}."
        )

    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom annotation formatting to handle potential NaNs safely
    if annot:
        annot_labels = corr_matrix.map(
            lambda v: f"{v:.2f}" if pd.notna(v) and np.isfinite(v) else "N/A"
        )
        sns.heatmap(
            corr_matrix,
            annot=annot_labels,
            fmt="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            square=True,
            linewidths=0.5,
            ax=ax,
        )
    else:
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            square=True,
            linewidths=0.5,
            ax=ax,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_spectral_phase_space(
    metrics_df: pd.DataFrame,
    regime_series: Optional[pd.Series] = None,
    title: str = "Phase Separation Space: Spectral Entropy H(τ) vs Mean Correlation μ(τ)",
    figsize: Tuple[int, int] = (9, 6),
) -> plt.Figure:
    """Plot the phase separation scatter diagram between Spectral Entropy and Mean Correlation.

    Args:
        metrics_df: DataFrame containing 'mean_correlation' and 'spectral_entropy'.
        regime_series: Optional categorical Series of regime labels aligned with metrics_df.
        title: Plot title.
        figsize: Figure dimensions (width, height).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ValueError: If metrics_df is None, empty, or missing required columns.
    """
    if metrics_df is None or not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty:
        raise ValueError("metrics_df must be a non-empty DataFrame.")

    for col in ["mean_correlation", "spectral_entropy"]:
        if col not in metrics_df.columns:
            raise ValueError(f"metrics_df must contain column '{col}'. Available: {list(metrics_df.columns)}")

    clean_metrics = metrics_df[["mean_correlation", "spectral_entropy"]].dropna()
    if clean_metrics.empty:
        raise ValueError("metrics_df contains no valid non-null correlation/entropy rows.")

    fig, ax = plt.subplots(figsize=figsize)

    if regime_series is not None and not regime_series.empty:
        aligned = clean_metrics.join(regime_series.rename("Regime").dropna(), how="inner")
        if aligned.empty:
            raise ValueError("No overlapping date indices between metrics_df and regime_series.")
        aligned["Regime"] = aligned["Regime"].astype(str)

        sns.scatterplot(
            data=aligned,
            x="mean_correlation",
            y="spectral_entropy",
            hue="Regime",
            palette="tab10",
            alpha=0.6,
            s=25,
            edgecolor=None,
            ax=ax,
        )
    else:
        ax.scatter(
            clean_metrics["mean_correlation"],
            clean_metrics["spectral_entropy"],
            color="steelblue",
            alpha=0.5,
            s=20,
            edgecolors="none",
        )

    ax.set_xlabel(r"Mean Off-Diagonal Correlation $\bar{\rho}(\tau)$", fontsize=11)
    ax.set_ylabel(r"Normalized Spectral Entropy $H_{\text{norm}}(\tau)$", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_regime_timeline(
    regime_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    crisis_events: Optional[Dict[str, Tuple[str, str]]] = None,
    title: str = "European Equity Regime Evolution with Event Overlays",
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot regime classifications over time with optional price trajectory and historical event spans.

    Args:
        regime_series: Time series of regime classifications (integers or strings).
        benchmark_series: Optional price or index level series for background visualization.
        crisis_events: Optional dictionary mapping event names to (start_date, end_date) tuples.
        title: Plot title.
        figsize: Figure dimensions (width, height).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ValueError: If regime_series is None, empty, or contains only NaNs.
    """
    if regime_series is None or not isinstance(regime_series, pd.Series) or regime_series.empty:
        raise ValueError("regime_series must be a non-empty pandas Series.")

    regime_aligned = regime_series.dropna()
    if regime_aligned.empty:
        raise ValueError("regime_series contains only NaN values.")

    # Ensure index is datetime
    if not isinstance(regime_aligned.index, pd.DatetimeIndex):
        try:
            regime_aligned.index = pd.to_datetime(regime_aligned.index)
        except Exception as exc:
            raise ValueError(f"regime_series index could not be converted to DatetimeIndex: {exc}")

    fig, ax1 = plt.subplots(figsize=figsize)

    unique_regimes = sorted(regime_aligned.unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_regimes), 3)))
    regime_color_map = {reg: colors[i] for i, reg in enumerate(unique_regimes)}

    dates = regime_aligned.index
    for i in range(len(dates) - 1):
        reg = regime_aligned.iloc[i]
        ax1.axvspan(
            dates[i],
            dates[i + 1],
            color=regime_color_map[reg],
            alpha=0.35,
            linewidth=0,
        )

    # Plot benchmark level if provided
    if benchmark_series is not None and not benchmark_series.empty:
        bench_clean = benchmark_series.dropna()
        if not isinstance(bench_clean.index, pd.DatetimeIndex):
            try:
                bench_clean.index = pd.to_datetime(bench_clean.index)
            except Exception:
                pass
        bench_aligned = bench_clean.reindex(dates).ffill().bfill()
        if not bench_aligned.empty and bench_aligned.notna().any():
            ax2 = ax1.twinx()
            ax2.plot(bench_aligned.index, bench_aligned.values, color="navy", lw=1.5, label="Benchmark Level")
            ax2.set_ylabel("Equal-Weighted European Benchmark Level", color="navy", fontsize=11)
            ax2.tick_params(axis="y", labelcolor="navy")

    # Add crisis event spans defensively
    if crisis_events:
        for event_name, (start_dt, end_dt) in crisis_events.items():
            try:
                start = pd.Timestamp(start_dt)
                end = pd.Timestamp(end_dt)
                if start <= dates[-1] and end >= dates[0]:
                    ax1.axvspan(start, end, color="crimson", alpha=0.15, linestyle=":", linewidth=1)
                    mid_date = start + (end - start) / 2
                    ax1.text(
                        mid_date,
                        ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] != 0 else 0.95,
                        event_name,
                        rotation=90,
                        fontsize=8,
                        color="darkred",
                        ha="center",
                        va="top",
                        fontweight="bold",
                    )
            except Exception:
                continue

    # Legend for regimes
    patches = [mpatches.Patch(color=regime_color_map[r], label=f"State {r}") for r in unique_regimes]
    ax1.legend(handles=patches, loc="upper left", framealpha=0.85, fontsize=9)

    ax1.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Regime State Occupancy", fontsize=11)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    return fig


def plot_safe_haven_scatter(
    market_returns: pd.Series,
    benchmark_returns: pd.Series,
    stress_quantile: float = 0.05,
    market_name: str = "Market",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot conditional stress return scatter highlighting Baur & Lucey tail stress observations.

    Args:
        market_returns: Asset return series.
        benchmark_returns: Market benchmark return series.
        stress_quantile: Lower tail quantile threshold (strictly between 0 and 1).
        market_name: Name of the asset being evaluated.
        figsize: Figure dimensions (width, height).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ValueError: If inputs are None, empty, invalid quantile, or fewer than 2 common dates.
    """
    if market_returns is None or benchmark_returns is None:
        raise ValueError("Return series cannot be None.")
    if market_returns.empty or benchmark_returns.empty:
        raise ValueError("Return series cannot be empty.")
    if not (0.0 < stress_quantile < 1.0):
        raise ValueError(f"stress_quantile must be strictly between 0 and 1, got {stress_quantile}.")

    aligned = pd.concat([market_returns.rename("asset"), benchmark_returns.rename("bench")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError(
            f"Fewer than 2 overlapping non-null return observations found ({len(aligned)})."
        )

    threshold = aligned["bench"].quantile(stress_quantile)
    stress_mask = aligned["bench"] <= threshold

    fig, ax = plt.subplots(figsize=figsize)
    # Calm days
    calm_sub = aligned.loc[~stress_mask]
    if not calm_sub.empty:
        ax.scatter(
            calm_sub["bench"] * 100,
            calm_sub["asset"] * 100,
            color="steelblue",
            alpha=0.35,
            s=15,
            label=f"Normal/Calm Days (N={len(calm_sub)})",
        )
    # Stress days
    stress_sub = aligned.loc[stress_mask]
    if not stress_sub.empty:
        ax.scatter(
            stress_sub["bench"] * 100,
            stress_sub["asset"] * 100,
            color="crimson",
            alpha=0.85,
            s=30,
            label=f"Tail Stress Days (Bench <= {stress_quantile:.0%}, N={len(stress_sub)})",
        )

    ax.axvline(threshold * 100, color="crimson", linestyle="--", lw=1.2, label=f"Cutoff ({threshold*100:.2f}%)")
    ax.axhline(0, color="gray", linestyle="-", lw=0.8)
    ax.axvline(0, color="gray", linestyle="-", lw=0.8)

    ax.set_xlabel("European Benchmark Daily Return (%)", fontsize=11)
    ax.set_ylabel(f"{market_name} Daily Return (%)", fontsize=11)
    ax.set_title(f"Safe-Haven Tail Stress Analysis: {market_name} vs Benchmark", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    return fig


def plot_transition_matrix(
    transition_matrix: Union[pd.DataFrame, np.ndarray],
    title: str = r"Empirical Walk-Forward Regime Transition Probabilities $P(S_{t+1}=j \mid S_t=i)$",
    figsize: Tuple[int, int] = (7, 5),
) -> plt.Figure:
    """Plot an annotated heatmap of the empirical regime transition matrix.

    Args:
        transition_matrix: Square matrix or DataFrame of transition probabilities.
        title: Plot title.
        figsize: Figure dimensions (width, height).

    Returns:
        matplotlib.figure.Figure object.

    Raises:
        ValueError: If input is None, empty, not 2D, or not square.
    """
    if transition_matrix is None:
        raise ValueError("Transition matrix cannot be None.")

    if not isinstance(transition_matrix, pd.DataFrame):
        transition_matrix = pd.DataFrame(transition_matrix)

    if transition_matrix.empty or transition_matrix.shape[0] == 0:
        raise ValueError("Transition matrix cannot be empty.")

    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError(
            f"Transition matrix must be square (K x K), got shape {transition_matrix.shape}."
        )

    fig, ax = plt.subplots(figsize=figsize)
    
    annot_labels = transition_matrix.map(
        lambda v: f"{v:.1%}" if pd.notna(v) and np.isfinite(v) else "N/A"
    )

    sns.heatmap(
        transition_matrix,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=0.8,
        ax=ax,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(r"Next Regime $S_{t+1}$", fontsize=10)
    ax.set_ylabel(r"Current Regime $S_t$", fontsize=10)
    plt.tight_layout()
    return fig
