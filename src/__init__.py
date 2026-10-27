"""Source package for European Financial Market Coupling and Crisis Analysis."""

from .data_loader import DataLoader, load_market_data
from .preprocessing import (
    validate_price_series,
    align_market_calendars,
    inspect_data_quality,
    generate_data_quality_report,
)
from .returns import (
    calculate_log_returns,
    calculate_simple_returns,
    compute_equal_weighted_index,
    compute_annualized_summary,
)
from .correlation import (
    compute_rolling_correlation_matrices,
    power_map_correlation_matrix,
    compute_epoch_distance_matrix,
    compute_epoch_distance_reference,
    extract_upper_triangular_flat,
)
from .metrics import (
    validate_correlation_matrix,
    compute_average_correlation,
    compute_eigenvalues,
    compute_spectral_entropy,
    compute_absorption_ratio,
    compute_rolling_metrics,
)
from .market_regimes import (
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
from .crisis_analysis import (
    STRUCTURED_EVENT_MANIFEST,
    detect_overlapping_events,
    partition_event_windows,
    compute_leave_one_out_benchmarks,
    holm_bonferroni_correction,
    block_bootstrap_mean_ci,
    block_bootstrap_corr_ci,
    block_bootstrap_diff_means,
    analyze_safe_haven_candidates,
    compare_crisis_window_metrics,
)
from .visualization import (
    plot_correlation_heatmap,
    plot_spectral_phase_space,
    plot_regime_timeline,
    plot_safe_haven_scatter,
    plot_transition_matrix,
)

__all__ = [
    "DataLoader",
    "load_market_data",
    "validate_price_series",
    "align_market_calendars",
    "inspect_data_quality",
    "generate_data_quality_report",
    "calculate_log_returns",
    "calculate_simple_returns",
    "compute_equal_weighted_index",
    "compute_annualized_summary",
    "compute_rolling_correlation_matrices",
    "power_map_correlation_matrix",
    "compute_epoch_distance_matrix",
    "compute_epoch_distance_reference",
    "extract_upper_triangular_flat",
    "validate_correlation_matrix",
    "compute_average_correlation",
    "compute_eigenvalues",
    "compute_spectral_entropy",
    "compute_absorption_ratio",
    "compute_rolling_metrics",
    "extract_regime_features",
    "ChronologicalScaler",
    "split_chronological",
    "KMeansRegimeModel",
    "GMMRegimeModel",
    "RuleBasedRegimeBaseline",
    "characterize_regimes",
    "walk_forward_regime_evaluation",
    "simulate_regime_switching_strategy",
    "STRUCTURED_EVENT_MANIFEST",
    "detect_overlapping_events",
    "partition_event_windows",
    "compute_leave_one_out_benchmarks",
    "holm_bonferroni_correction",
    "block_bootstrap_mean_ci",
    "block_bootstrap_corr_ci",
    "block_bootstrap_diff_means",
    "analyze_safe_haven_candidates",
    "compare_crisis_window_metrics",
    "plot_correlation_heatmap",
    "plot_spectral_phase_space",
    "plot_regime_timeline",
    "plot_safe_haven_scatter",
    "plot_transition_matrix",
]


