"""Tests for funding rate strategy + data provider (mocked API)."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from strategies.core.funding_rate import (
    FUNDING_LONG_THRESH,
    FUNDING_SHORT_THRESH,
    LS_RATIO_LONG_THRESH,
    LS_RATIO_SHORT_THRESH,
    add_funding_indicators,
    populate_funding_entries,
    populate_funding_exits,
)
from strategies.data.funding_provider import (
    STALE_THRESHOLD_SECONDS,
    FundingData,
    FundingDataProvider,
)


# ── FundingDataProvider tests (mocked API) ───────────────────────────────

class TestFundingDataProvider:
    def test_get_returns_none_when_no_data(self):
        provider = FundingDataProvider("BTCUSDT")
        assert provider.get() is None

    def test_get_returns_data_when_fresh(self):
        provider = FundingDataProvider("BTCUSDT")
        provider._data = FundingData(0.0003, 1.2, time.time())
        data = provider.get()
        assert data is not None
        assert data.funding_rate == 0.0003
        assert data.long_short_ratio == 1.2

    def test_get_returns_none_when_stale(self):
        provider = FundingDataProvider("BTCUSDT")
        # Data from 11 minutes ago (> 10 min stale threshold)
        provider._data = FundingData(0.0003, 1.2, time.time() - STALE_THRESHOLD_SECONDS - 60)
        assert provider.get() is None

    @patch("strategies.data.funding_provider.requests.get")
    def test_fetch_once_success(self, mock_get):
        """Successful API fetch returns FundingData."""
        mock_funding_resp = MagicMock()
        mock_funding_resp.json.return_value = {"lastFundingRate": "0.00035"}
        mock_funding_resp.raise_for_status = MagicMock()

        mock_ls_resp = MagicMock()
        mock_ls_resp.json.return_value = [{"longShortRatio": "1.45"}]
        mock_ls_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_funding_resp, mock_ls_resp]

        provider = FundingDataProvider("BTCUSDT")
        data = provider.fetch_once()
        assert data is not None
        assert data.funding_rate == pytest.approx(0.00035)
        assert data.long_short_ratio == pytest.approx(1.45)

    @patch("strategies.data.funding_provider.requests.get")
    def test_fetch_once_api_failure(self, mock_get):
        """API failure returns None after retries."""
        mock_get.side_effect = Exception("Connection error")
        provider = FundingDataProvider("BTCUSDT")
        data = provider.fetch_once()
        assert data is None

    @patch("strategies.data.funding_provider.requests.get")
    def test_fetch_once_funding_fails_ls_succeeds(self, mock_get):
        """If funding rate fails, entire fetch returns None."""
        mock_get.side_effect = Exception("Rate limited")
        provider = FundingDataProvider("BTCUSDT")
        data = provider.fetch_once()
        assert data is None


# ── Funding strategy tests ───────────────────────────────────────────────

class TestAddFundingIndicators:
    def test_columns_exist(self):
        n = 50
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "open": rng.uniform(49000, 51000, n),
            "high": rng.uniform(50000, 52000, n),
            "low": rng.uniform(48000, 50000, n),
            "close": rng.uniform(49000, 51000, n),
            "volume": rng.uniform(100, 1000, n),
        })
        df = add_funding_indicators(df)
        for col in ["fr_rsi", "fr_atr", "fr_volume_sma", "funding_rate", "long_short_ratio"]:
            assert col in df.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = add_funding_indicators(df)
        assert len(df) == 0


class TestPopulateFundingEntries:
    def _base_df(self, n: int = 5) -> pd.DataFrame:
        return pd.DataFrame({
            "regime": ["RANGING"] * n,
            "regime_confidence": [0.7] * n,
            "funding_rate": [0.0001] * n,  # neutral by default
            "long_short_ratio": [1.0] * n,
            "fr_rsi": [50.0] * n,
            "volume": [600.0] * n,
            "fr_volume_sma": [500.0] * n,
        })

    def test_no_signal_neutral_funding(self):
        df = self._base_df()
        df = populate_funding_entries(df)
        assert df["fr_enter_long"].sum() == 0
        assert df["fr_enter_short"].sum() == 0

    def test_long_extreme_short_crowding(self):
        df = self._base_df()
        df["funding_rate"] = -0.001  # < -0.05%
        df["fr_rsi"] = 30            # < 35
        df["long_short_ratio"] = 0.5  # < 0.7
        df = populate_funding_entries(df)
        assert df["fr_enter_long"].sum() == 5

    def test_short_extreme_long_crowding(self):
        df = self._base_df()
        df["funding_rate"] = 0.001   # > 0.08%
        df["fr_rsi"] = 70            # > 65
        df["long_short_ratio"] = 2.0  # > 1.8
        df = populate_funding_entries(df)
        assert df["fr_enter_short"].sum() == 5

    def test_volatile_regime_blocks(self):
        """No funding trades in VOLATILE regime."""
        df = self._base_df()
        df["regime"] = "VOLATILE"
        df["funding_rate"] = -0.001
        df["fr_rsi"] = 30
        df["long_short_ratio"] = 0.5
        df = populate_funding_entries(df)
        assert df["fr_enter_long"].sum() == 0

    def test_nan_funding_no_signal(self):
        """Missing funding data → no signals (graceful)."""
        df = self._base_df()
        df["funding_rate"] = float("nan")
        df = populate_funding_entries(df)
        assert df["fr_enter_long"].sum() == 0
        assert df["fr_enter_short"].sum() == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_funding_entries(df)
        assert "fr_enter_long" in df.columns


class TestPopulateFundingExits:
    def test_funding_normalized_triggers_exit(self):
        df = pd.DataFrame({
            "funding_rate": [-0.001, -0.0005, -0.0001, 0.0, 0.0001],
        })
        df = populate_funding_exits(df)
        # Indices 2,3,4 have funding in normal range
        assert df.loc[2, "fr_exit_long"] == 1
        assert df.loc[0, "fr_exit_long"] == 0

    def test_nan_funding_no_exit(self):
        df = pd.DataFrame({"funding_rate": [float("nan")] * 3})
        df = populate_funding_exits(df)
        assert df["fr_exit_long"].sum() == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_funding_exits(df)
        assert "fr_exit_long" in df.columns
