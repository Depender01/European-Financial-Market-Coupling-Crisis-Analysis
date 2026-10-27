"""Data Acquisition and Caching Module.

Retrieves historical financial data from yfinance for European equity indices and proxies,
validates retrieval coverage, handles errors gracefully, and caches raw data with provenance metadata.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    CORE_MARKETS,
    EXTENDED_MARKETS,
    BENCHMARK_CONFIG,
    RAW_DATA_DIR,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Manages acquisition, validation, and local caching of financial time series data."""

    def __init__(
        self,
        raw_data_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """Initialize the DataLoader.

        Args:
            raw_data_dir: Path to raw data directory. Defaults to RAW_DATA_DIR in settings.
            use_cache: Whether to load cached raw data if available.
        """
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else RAW_DATA_DIR
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

    def get_instrument_metadata(self, ticker: str) -> Dict[str, Any]:
        """Look up metadata for a given instrument ticker.

        Args:
            ticker: Yahoo Finance symbol.

        Returns:
            Dictionary with name, country, currency, index type, and notes.
        """
        if ticker in CORE_MARKETS:
            return CORE_MARKETS[ticker]
        if ticker in EXTENDED_MARKETS:
            return EXTENDED_MARKETS[ticker]
        if ticker in BENCHMARK_CONFIG:
            return BENCHMARK_CONFIG[ticker]
        return {
            "name": ticker,
            "country": "Unknown",
            "currency": "Unknown",
            "index_type": "Unknown",
            "notes": "Custom / Unregistered instrument",
        }

    def _get_cache_paths(
        self, tickers: List[str], start_date: str, end_date: Optional[str]
    ) -> Tuple[Path, Path]:
        """Generate deterministic cache file paths for data and metadata."""
        end_str = end_date or "latest"
        tickers_hash = "_".join(sorted([t.replace("^", "").replace(".", "_") for t in tickers]))
        # Truncate hash if too long for filesystem
        if len(tickers_hash) > 60:
            tickers_hash = f"basket_{len(tickers)}_tickers"
        base_name = f"raw_{tickers_hash}_{start_date}_{end_str}"
        return (
            self.raw_data_dir / f"{base_name}.csv",
            self.raw_data_dir / f"{base_name}_meta.json",
        )

    def download_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: Optional[str] = None,
        auto_adjust: bool = True,
    ) -> Optional[pd.Series]:
        """Download price data for an individual ticker with error handling.

        Args:
            ticker: Ticker symbol.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD) or None.
            auto_adjust: Whether to auto-adjust for splits and dividends.

        Returns:
            pd.Series of closing prices or None if download failed.
        """
        try:
            logger.info("Downloading ticker: %s (%s to %s)", ticker, start_date, end_date or "latest")
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=auto_adjust,
                progress=False,
            )
            if data is None or data.empty:
                logger.warning("Empty data returned for ticker: %s", ticker)
                return None

            # Handle MultiIndex columns if returned by yfinance
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.levels[0]:
                    series = data["Close"]
                    if isinstance(series, pd.DataFrame):
                        series = series.squeeze()
                else:
                    series = data.iloc[:, 0]
            elif "Close" in data.columns:
                series = data["Close"]
            else:
                series = data.iloc[:, 0]

            if not isinstance(series, pd.Series):
                series = series.squeeze()

            series.name = ticker
            # Normalize index to timezone-naive date
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            series.index = pd.to_datetime(series.index).normalize()
            series = series.dropna()
            return series

        except Exception as exc:
            logger.error("Failed to download %s: %s", ticker, str(exc))
            return None

    def fetch_market_dataset(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = DEFAULT_END_DATE,
        auto_adjust: bool = True,
        force_download: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch raw prices for a list of tickers, using cache if permitted.

        Args:
            tickers: List of ticker symbols. Defaults to CORE_MARKETS keys.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD) or None.
            auto_adjust: Adjust prices for dividends/splits.
            force_download: If True, bypass cache.

        Returns:
            Tuple of:
                - pd.DataFrame of closing prices (rows: dates, columns: tickers)
                - Metadata dictionary detailing retrieval date, coverage, and failures.
        """
        if tickers is None:
            tickers = list(CORE_MARKETS.keys())

        csv_path, meta_path = self._get_cache_paths(tickers, start_date, end_date)

        # Check cache
        if self.use_cache and not force_download and csv_path.exists() and meta_path.exists():
            try:
                cached_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata["loaded_from_cache"] = True
                logger.info("Loaded dataset from cache: %s", csv_path.name)
                return cached_df, metadata
            except Exception as exc:
                logger.warning("Cache read failed (%s); downloading fresh data.", exc)

        retrieval_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        series_dict: Dict[str, pd.Series] = {}
        failed_tickers: List[str] = []
        coverage_report: Dict[str, Any] = {}

        for t in tickers:
            s = self.download_ticker(t, start_date=start_date, end_date=end_date, auto_adjust=auto_adjust)
            if s is not None and len(s) > 0:
                series_dict[t] = s
                coverage_report[t] = {
                    "start_date": str(s.index[0].date()),
                    "end_date": str(s.index[-1].date()),
                    "total_observations": int(len(s)),
                    "first_price": float(s.iloc[0]),
                    "last_price": float(s.iloc[-1]),
                }
            else:
                failed_tickers.append(t)
                coverage_report[t] = {
                    "error": "Download failed or empty series returned",
                    "total_observations": 0,
                }

        if not series_dict:
            raise ValueError(
                f"No market data could be downloaded for any of the requested tickers: {tickers}. "
                f"Check internet connection or ticker symbols."
            )

        # Combine into DataFrame without forward filling (raw representation)
        raw_prices = pd.DataFrame(series_dict)
        raw_prices.index = pd.to_datetime(raw_prices.index).normalize()
        raw_prices.sort_index(inplace=True)

        metadata: Dict[str, Any] = {
            "source": "Yahoo Finance (via yfinance)",
            "retrieval_timestamp_utc": retrieval_timestamp,
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "requested_tickers": tickers,
            "succeeded_tickers": list(series_dict.keys()),
            "failed_tickers": failed_tickers,
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "effective_start_date": str(raw_prices.index[0].date()),
            "effective_end_date": str(raw_prices.index[-1].date()),
            "total_raw_days": int(len(raw_prices)),
            "coverage_per_ticker": coverage_report,
            "loaded_from_cache": False,
        }

        # Cache results to disk
        try:
            raw_prices.to_csv(csv_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Saved raw dataset and metadata to cache: %s", csv_path.name)
        except Exception as exc:
            logger.warning("Failed to write cache files: %s", exc)

        return raw_prices, metadata


def load_market_data(
    tickers: Optional[List[str]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
    use_cache: bool = True,
    force_download: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Functional convenience wrapper around DataLoader.fetch_market_dataset."""
    loader = DataLoader(use_cache=use_cache)
    return loader.fetch_market_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        force_download=force_download,
    )
