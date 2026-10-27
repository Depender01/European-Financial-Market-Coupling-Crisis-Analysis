"""Unit tests for data acquisition and caching module."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader
from config.settings import CORE_MARKETS, EXTENDED_MARKETS


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary directory for data loader tests."""
    cache_dir = tmp_path / "raw_test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_metadata_lookup():
    """Verify correct metadata lookup for core and extended instruments."""
    loader = DataLoader()

    dax_meta = loader.get_instrument_metadata("^GDAXI")
    assert dax_meta["country"] == "Germany"
    assert dax_meta["index_type"] == "Total Return / Performance"

    ftse_meta = loader.get_instrument_metadata("^FTSE")
    assert ftse_meta["country"] == "United Kingdom"
    assert ftse_meta["currency"] == "GBP"

    ewd_meta = loader.get_instrument_metadata("EWD")
    assert "WARNING" in ewd_meta["notes"]

    unknown_meta = loader.get_instrument_metadata("NONEXISTENT_TICKER")
    assert unknown_meta["country"] == "Unknown"


def test_cache_path_generation(temp_cache_dir: Path):
    """Verify deterministic cache file naming."""
    loader = DataLoader(raw_data_dir=temp_cache_dir)
    csv_path, meta_path = loader._get_cache_paths(["^FTSE", "^GDAXI"], "2020-01-01", "2021-01-01")

    assert csv_path.parent == temp_cache_dir
    assert meta_path.parent == temp_cache_dir
    assert csv_path.suffix == ".csv"
    assert meta_path.suffix == ".json"
    assert "2020-01-01" in csv_path.name


def test_fetch_market_dataset_caching(temp_cache_dir: Path):
    """Test caching behavior: second call loads from disk without re-downloading."""
    loader = DataLoader(raw_data_dir=temp_cache_dir, use_cache=True)

    # Mock download_ticker to return synthetic data without hitting Yahoo Finance
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    mock_series_ftse = pd.Series([7000.0, 7050.0, 7020.0, 7100.0, 7120.0], index=dates, name="^FTSE")
    mock_series_dax = pd.Series([15000.0, 15100.0, 15050.0, 15200.0, 15250.0], index=dates, name="^GDAXI")

    def mock_download(ticker, *args, **kwargs):
        if ticker == "^FTSE":
            return mock_series_ftse.copy()
        if ticker == "^GDAXI":
            return mock_series_dax.copy()
        return None

    with patch.object(loader, "download_ticker", side_effect=mock_download) as mock_dl:
        # First call: should download and save cache
        df1, meta1 = loader.fetch_market_dataset(
            tickers=["^FTSE", "^GDAXI"],
            start_date="2026-01-01",
            end_date="2026-01-08",
        )
        assert mock_dl.call_count == 2
        assert meta1["loaded_from_cache"] is False
        assert set(df1.columns) == {"^FTSE", "^GDAXI"}
        assert len(df1) == 5

        # Second call: should read directly from disk cache without calling download_ticker
        mock_dl.reset_mock()
        df2, meta2 = loader.fetch_market_dataset(
            tickers=["^FTSE", "^GDAXI"],
            start_date="2026-01-01",
            end_date="2026-01-08",
        )
        assert mock_dl.call_count == 0  # Not called!
        assert meta2["loaded_from_cache"] is True
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)


def test_fetch_market_dataset_all_failures_raise(temp_cache_dir: Path):
    """If all tickers fail to download, raise ValueError."""
    loader = DataLoader(raw_data_dir=temp_cache_dir, use_cache=False)

    with patch.object(loader, "download_ticker", return_value=None):
        with pytest.raises(ValueError, match="No market data could be downloaded"):
            loader.fetch_market_dataset(tickers=["INVALID_1", "INVALID_2"])
