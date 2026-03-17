"""Tests for trend following strategy — pullback + trend quality filters.

Architecture: 5 hard gates + 1-of-3 pullback + 2-of-4 confluence.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.core.trend_following import (
    add_trend_indicators,
    populate_trend_entries,
    populate_trend_exits,
)


def _golden_path_long(n: int = 6) -> pd.DataFrame:
    """DataFrame where all gates + pullback + confluence met at last row.

    Hard gates: Supertrend bullish, close>EMA50 + slope>0.001, bullish candle, RSI 30-60
    Pullback: RSI dipped below 45, low near EMA21
    Confluence: ADX rising, volume > 1.5x SMA, close > EMA200, EMA50 slope strong
    """
    return pd.DataFrame({
        # Gate 1: Supertrend bullish
        "supertrend_direction": [1] * n,
        # Gate 2: Close > EMA50 + EMA50 slope > 0.001
        "open": [50800, 50900, 51000, 51100, 51200, 51300],
        "close": [51000, 51100, 51200, 51300, 51400, 51500],
        "high": [51200, 51300, 51400, 51500, 51600, 51700],
        "low": [50700, 50800, 50900, 51000, 51100, 51200],
        "ema_50": [50500] * n,
        "ema_50_slope": [0.002] * n,       # trending up (> 0.001)
        # Gate 4: RSI between 30-60
        "rsi_14": [42, 38, 44, 48, 50, 55],    # dips below 45 → pullback B ✓
        # Pullback A: low near EMA21
        "ema_21": [51000] * n,
        "ema_9": [51200] * n,
        # Confluence A: ADX rising
        "tf_adx": [18, 19, 20, 22, 24, 25],    # rising over last 3 ✓
        # Confluence B: Volume > 1.5x SMA
        "volume": [500, 500, 500, 500, 500, 700],
        "volume_sma_20": [400] * n,
        # Confluence C: Close > EMA200
        "ema_200": [50000] * n,
        # BB middle for pullback C
        "bb_middle": [51100] * n,
    })


def _golden_path_short(n: int = 6) -> pd.DataFrame:
    """Mirror of golden path for short entries."""
    return pd.DataFrame({
        "supertrend_direction": [-1] * n,
        "open": [49200, 49100, 49000, 48900, 48800, 48700],
        "close": [49000, 48900, 48800, 48700, 48600, 48500],
        "high": [49300, 49200, 49100, 49000, 48900, 48800],
        "low": [48800, 48700, 48600, 48500, 48400, 48300],
        "ema_50": [49500] * n,
        "ema_50_slope": [-0.002] * n,      # trending down
        "rsi_14": [58, 62, 56, 52, 50, 45],
        "ema_21": [49000] * n,
        "ema_9": [48800] * n,
        "tf_adx": [18, 19, 20, 22, 24, 25],
        "volume": [500, 500, 500, 500, 500, 700],
        "volume_sma_20": [400] * n,
        "ema_200": [50000] * n,
        "bb_middle": [48900] * n,
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
                    "ema_9", "ema_21", "ema_50", "ema_200", "ema_50_slope",
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
        df = add_trend_indicators(df, pair="SOL/USDT:USDT")
        assert "supertrend_direction" in df.columns


class TestPopulateTrendEntries:
    def test_full_confluence_long(self):
        """All gates + pullback + full confluence = signal fires."""
        df = _golden_path_long()
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1
        assert df.loc[5, "tf_signal_tag"] == "trend_following"

    def test_min_confluence_long(self):
        """All gates + pullback + exactly 2/4 confluence = signal fires."""
        df = _golden_path_long()
        # Disable confluence C (EMA200) and D (slope strength)
        df["ema_200"] = [55000] * 6
        df["ema_50_slope"] = [0.0015] * 6   # above 0.001 gate but below 0.002 scoring
        # Confluence A (ADX rising) + B (vol > 1.5x) = 2 ✓
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1

    def test_insufficient_confluence_blocks(self):
        """All gates + pullback but only 1/4 confluence = NO signal."""
        df = _golden_path_long()
        # Disable A (ADX not rising), B (low vol), C (below EMA200)
        df["tf_adx"] = [25, 25, 25, 25, 25, 25]  # flat, not rising
        df["volume"] = [100] * 6
        df["ema_200"] = [55000] * 6
        df["ema_50_slope"] = [0.0015] * 6   # below 0.002 → score D = 0
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

    def test_no_signal_when_ema50_flat(self):
        """Gate 2 fails: EMA50 slope is flat (choppy market)."""
        df = _golden_path_long()
        df["ema_50_slope"] = [0.0005] * 6   # below 0.001 threshold
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_bearish_candle(self):
        """Gate 3 fails: bearish candle blocks long."""
        df = _golden_path_long()
        df["open"] = [52000] * 6
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_rsi_overbought(self):
        """Gate 4 fails: RSI above 60 (long ceiling)."""
        df = _golden_path_long()
        df["rsi_14"] = [65, 65, 65, 65, 65, 65]
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_rsi_oversold(self):
        """Gate 4 fails: RSI below 30."""
        df = _golden_path_long()
        df["rsi_14"] = [25, 25, 25, 25, 25, 25]
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_without_pullback(self):
        """Gates and confluence met but NO pullback = NO signal."""
        df = _golden_path_long()
        df["rsi_14"] = [55, 55, 55, 55, 55, 55]
        df["low"] = [52000] * 6
        df["bb_middle"] = [49000] * 6
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 0

    def test_pullback_ema21_only(self):
        """Pullback A (EMA21 test) sufficient with gates + confluence."""
        df = _golden_path_long()
        df["rsi_14"] = [55, 55, 55, 55, 55, 55]
        df["bb_middle"] = [49000] * 6
        # low=50700 vs ema_21=51000 → ratio=0.994 < 1.01 ✓
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1

    def test_pullback_rsi_only(self):
        """Pullback B (RSI dip) sufficient."""
        df = _golden_path_long()
        df["low"] = [52000] * 6
        df["bb_middle"] = [49000] * 6
        # RSI dips below 45 within 3-candle window (rows 3,4,5 at index 5)
        df["rsi_14"] = [55, 55, 55, 42, 50, 55]
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1

    def test_pullback_bb_middle_only(self):
        """Pullback C (BB middle test) sufficient."""
        df = _golden_path_long()
        df["low"] = [52000] * 6
        df["rsi_14"] = [55, 55, 55, 55, 55, 55]
        df["bb_middle"] = [52900] * 6
        df["close"] = [51000, 51100, 51200, 51300, 51400, 53000]
        df["open"] = [50800, 50900, 51000, 51100, 51200, 52800]
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1

    def test_adx_rising_as_confluence(self):
        """ADX rising catches early trends (ADX low but increasing)."""
        df = _golden_path_long()
        df["tf_adx"] = [12, 13, 14, 15, 16, 17]   # below old threshold 20, but RISING
        df["volume"] = [100] * 6                     # no volume score
        df["ema_200"] = [55000] * 6                  # no EMA200 score
        # ADX rising ✓ (17 > 14), EMA50 slope strong ✓ (0.003 > 0.002) → 2 confluence
        df["ema_50_slope"] = [0.003] * 6             # above 2x threshold for score D
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1

    def test_short_golden_path(self):
        """Golden path short: all gates + pullback + confluence."""
        df = _golden_path_short()
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_short"] == 1
        assert df.loc[5, "tf_signal_tag"] == "trend_following"

    def test_short_no_signal_when_bullish_candle(self):
        """Gate 3 fails for short: bullish candle blocks."""
        df = _golden_path_short()
        df["open"] = [48000] * 6
        df = populate_trend_entries(df)
        assert df["tf_enter_short"].sum() == 0

    def test_short_ema50_flat_blocks(self):
        """Gate 2 fails for short: EMA50 not trending down."""
        df = _golden_path_short()
        df["ema_50_slope"] = [0.0] * 6     # flat
        df = populate_trend_entries(df)
        assert df["tf_enter_short"].sum() == 0

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
        """ADX dropping below 12 should trigger exit (only truly dead trends)."""
        df = pd.DataFrame({
            "tf_adx": [30, 25, 20, 15, 10],
            "supertrend_direction": [1, 1, 1, 1, 1],
        })
        df = populate_trend_exits(df)
        assert df.loc[4, "tf_exit_long"] == 1   # ADX=10 < 12
        assert df.loc[3, "tf_exit_long"] == 0   # ADX=15 >= 12, no exit
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
