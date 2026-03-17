"""Tests for trend following strategy — tiered confluence (3 gates + 2-of-4 scoring)."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.trend_following import (
    add_trend_indicators,
    populate_trend_entries,
    populate_trend_exits,
)


def _golden_path_long(n: int = 6) -> pd.DataFrame:
    """Create a DataFrame where all hard gates + full confluence are met at last row.

    Hard gates: Supertrend bullish, close > EMA50, RSI in 30-70
    Scoring: EMA9 > EMA21, ADX > 20, volume > 1.5x SMA, close > EMA200
    """
    return pd.DataFrame({
        # Gate 1: Supertrend bullish
        "supertrend_direction": [1] * n,
        # Gate 2: Close > EMA50
        "close": [51000, 51100, 51200, 51300, 51400, 51500],
        "ema_50": [50500] * n,
        # Gate 3: RSI between 30-70
        "rsi_14": [50, 50, 50, 50, 50, 55],
        # Score A: EMA9 > EMA21
        "ema_9": [51200] * n,
        "ema_21": [51000] * n,
        # Score B: ADX > 20 (BTC default)
        "tf_adx": [18, 19, 20, 22, 24, 25],
        # Score C: Volume > 1.5x SMA
        "volume": [500, 500, 500, 500, 500, 700],
        "volume_sma_20": [400] * n,   # 700 > 400 * 1.5 = 600 ✓
        # Score D: Close > EMA200
        "ema_200": [50000] * n,
    })


def _golden_path_short(n: int = 6) -> pd.DataFrame:
    """Mirror of golden path for short entries."""
    return pd.DataFrame({
        "supertrend_direction": [-1] * n,
        "close": [49000, 48900, 48800, 48700, 48600, 48500],
        "ema_50": [49500] * n,
        "rsi_14": [50, 50, 50, 50, 50, 45],
        "ema_9": [48800] * n,
        "ema_21": [49000] * n,     # EMA9 < EMA21 ✓
        "tf_adx": [18, 19, 20, 22, 24, 25],
        "volume": [500, 500, 500, 500, 500, 700],
        "volume_sma_20": [400] * n,
        "ema_200": [50000] * n,    # close < EMA200 ✓
    })


class TestAddTrendIndicators:
    def test_columns_exist(self):
        n = 300
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 100, n)) + 50000
        df = pd.DataFrame({
            "open": closes + rng.normal(0, 50, n),
            "high": closes + rng.uniform(50, 200, n),
            "low": closes - rng.uniform(50, 200, n),
            "close": closes,
            "volume": rng.uniform(100, 1000, n),
        })
        df = add_trend_indicators(df)
        expected = ["supertrend_direction", "supertrend_value", "tf_adx",
                    "ema_9", "ema_21", "ema_50", "ema_200",
                    "stochrsi_k", "stochrsi_d",
                    "atr_14", "volume_sma_20", "rsi_14",
                    "bb_upper", "bb_lower", "bb_middle"]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = add_trend_indicators(df)
        assert len(df) == 0

    def test_per_pair_supertrend(self):
        """SOL pair should use different Supertrend settings."""
        n = 300
        rng = np.random.default_rng(42)
        closes = np.cumsum(rng.normal(0, 100, n)) + 50000
        df = pd.DataFrame({
            "open": closes + rng.normal(0, 50, n),
            "high": closes + rng.uniform(50, 200, n),
            "low": closes - rng.uniform(50, 200, n),
            "close": closes,
            "volume": rng.uniform(100, 1000, n),
        })
        # Should not crash with different pair
        df = add_trend_indicators(df, pair="SOL/USDT:USDT")
        assert "supertrend_direction" in df.columns


class TestPopulateTrendEntries:
    def test_all_gates_plus_full_confluence_long(self):
        """All 3 gates TRUE + 4/4 confluence = signal fires."""
        df = _golden_path_long()
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1
        assert df.loc[5, "tf_signal_tag"] == "trend_following"

    def test_all_gates_plus_min_confluence_long(self):
        """All 3 gates TRUE + exactly 2/4 confluence = signal fires."""
        df = _golden_path_long()
        # Disable score C (volume) and score D (EMA200)
        df["volume"] = [100] * 6                # below 1.5x SMA → score C = 0
        df["ema_200"] = [55000] * 6             # close below → score D = 0
        # Score A (EMA9>EMA21) = 1, Score B (ADX>20) = 1 → total = 2 ✓
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1

    def test_all_gates_but_insufficient_confluence(self):
        """All 3 gates TRUE but only 1/4 confluence = NO signal."""
        df = _golden_path_long()
        # Disable B, C, D — only score A remains
        df["tf_adx"] = [10] * 6                 # below 20 → score B = 0
        df["volume"] = [100] * 6                # below 1.5x → score C = 0
        df["ema_200"] = [55000] * 6             # close below → score D = 0
        # Score A (EMA9>EMA21) = 1 → total = 1 < 2
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 0

    def test_no_signal_when_supertrend_bearish(self):
        """Gate 1 fails: Supertrend bearish blocks long entry."""
        df = _golden_path_long()
        df["supertrend_direction"] = -1
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_below_ema50(self):
        """Gate 2 fails: close below EMA50."""
        df = _golden_path_long()
        df["ema_50"] = [55000] * 6
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_rsi_overbought(self):
        """Gate 3 fails: RSI above 70."""
        df = _golden_path_long()
        df["rsi_14"] = [75, 75, 75, 75, 75, 75]
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_rsi_oversold(self):
        """Gate 3 fails: RSI below 30."""
        df = _golden_path_long()
        df["rsi_14"] = [25, 25, 25, 25, 25, 25]
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_all_conditions_met_short(self):
        """Golden path short: all gates + confluence met."""
        df = _golden_path_short()
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_short"] == 1
        assert df.loc[5, "tf_signal_tag"] == "trend_following"

    def test_per_pair_adx_threshold(self):
        """SOL uses ADX threshold 25 instead of BTC's 20."""
        df = _golden_path_long()
        df["tf_adx"] = [18, 19, 20, 21, 22, 22]  # above 20 (BTC), below 25 (SOL)
        # Disable other scores so only ADX matters
        df["ema_9"] = [49000] * 6     # below EMA21 → score A = 0
        df["volume"] = [100] * 6       # score C = 0
        # Score D = 1 (close > EMA200), Score B = 1 for BTC, 0 for SOL

        # BTC: ADX 22 > 20 → score B = 1, total = 1+1 = 2 ✓
        df_btc = df.copy()
        df_btc = populate_trend_entries(df_btc, pair="BTC/USDT:USDT")
        assert df_btc.loc[5, "tf_enter_long"] == 1

        # SOL: ADX 22 < 25 → score B = 0, total = 0+1 = 1 < 2
        df_sol = df.copy()
        df_sol = populate_trend_entries(df_sol, pair="SOL/USDT:USDT")
        assert df_sol.loc[5, "tf_enter_long"] == 0

    def test_signal_tag_column_exists(self):
        df = _golden_path_long()
        df = populate_trend_entries(df)
        assert "tf_signal_tag" in df.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = populate_trend_entries(df)
        assert "tf_enter_long" in df.columns
        assert "tf_signal_tag" in df.columns


class TestPopulateTrendExits:
    def test_adx_death_triggers_exit(self):
        """ADX dropping below 18 should trigger exit."""
        df = pd.DataFrame({
            "tf_adx": [30, 25, 20, 15, 12],
            "supertrend_direction": [1, 1, 1, 1, 1],
        })
        df = populate_trend_exits(df)
        assert df.loc[3, "tf_exit_long"] == 1
        assert df.loc[4, "tf_exit_long"] == 1
        assert df.loc[0, "tf_exit_long"] == 0

    def test_supertrend_flip_triggers_exit(self):
        """Supertrend flipping from bull to bear should exit long."""
        df = pd.DataFrame({
            "tf_adx": [30, 30, 30, 30, 30],
            "supertrend_direction": [1, 1, 1, -1, -1],
        })
        df = populate_trend_exits(df)
        assert df.loc[3, "tf_exit_long"] == 1
        assert df.loc[2, "tf_exit_long"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_trend_exits(df)
        assert "tf_exit_long" in df.columns
