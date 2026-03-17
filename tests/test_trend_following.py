"""Tests for trend following strategy — OR-of-simple-groups entry logic."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.trend_following import (
    add_trend_indicators,
    populate_trend_entries,
    populate_trend_exits,
)


def _make_ohlcv(n: int = 300, base: float = 50000.0, trend: float = 0.0,
                noise: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = np.cumsum(rng.normal(trend, noise, n)) + base
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes + rng.normal(0, 50, n)
    volumes = rng.uniform(100, 1000, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


def _add_regime_columns(df: pd.DataFrame, regime: str = "TRENDING_BULL",
                        confidence: float = 0.8) -> pd.DataFrame:
    """Add regime columns that the trend strategy expects."""
    df["regime"] = regime
    df["regime_confidence"] = confidence
    return df


class TestAddTrendIndicators:
    def test_columns_exist(self):
        df = _make_ohlcv(300)
        df = add_trend_indicators(df)
        expected = ["supertrend_direction", "supertrend_value", "tf_adx",
                    "ema_9", "ema_50", "ema_200", "stochrsi_k", "stochrsi_d",
                    "atr_14", "volume_sma_20", "rsi_14",
                    "bb_upper", "bb_lower", "bb_middle"]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = add_trend_indicators(df)
        assert len(df) == 0


class TestPopulateTrendEntries:
    def test_path_a_supertrend_breakout_long(self):
        """Path A: Supertrend flip up + close > EMA50 + volume."""
        n = 5
        df = pd.DataFrame({
            "supertrend_direction": [-1, -1, -1, -1, 1],  # flips at idx 4
            "close": [51000, 51100, 51200, 51300, 51400],
            "ema_50": [50500] * n,
            "ema_9": [50000] * n,  # below ema_50 (so Path B won't fire)
            "tf_adx": [15, 15, 15, 15, 15],  # low ADX (so Path B/C won't fire)
            "bb_upper": [52000] * n,  # close below BB (so Path C won't fire)
            "bb_lower": [48000] * n,
            "rsi_14": [50] * n,
            "volume": [500, 500, 500, 500, 600],
            "volume_sma_20": [400] * n,
        })
        df = populate_trend_entries(df)
        assert df.loc[4, "tf_enter_long"] == 1
        assert df.loc[4, "tf_signal_tag"] == "tf_supertrend"

    def test_path_a_supertrend_breakout_short(self):
        """Path A short: Supertrend flip down + close < EMA50 + volume."""
        n = 5
        df = pd.DataFrame({
            "supertrend_direction": [1, 1, 1, 1, -1],  # flips at idx 4
            "close": [49000, 48900, 48800, 48700, 48600],
            "ema_50": [49500] * n,
            "ema_9": [50000] * n,
            "tf_adx": [15, 15, 15, 15, 15],
            "bb_upper": [52000] * n,
            "bb_lower": [48000] * n,
            "rsi_14": [50] * n,
            "volume": [500, 500, 500, 500, 600],
            "volume_sma_20": [400] * n,
        })
        df = populate_trend_entries(df)
        assert df.loc[4, "tf_enter_short"] == 1
        assert df.loc[4, "tf_signal_tag"] == "tf_supertrend"

    def test_path_b_ema_momentum_long(self):
        """Path B: EMA9 > EMA50 + ADX > 25 + RSI in range."""
        n = 3
        df = pd.DataFrame({
            "supertrend_direction": [1, 1, 1],
            "close": [51000, 51100, 51200],
            "ema_9": [51000, 51100, 51200],   # above ema_50
            "ema_50": [50500, 50500, 50500],
            "tf_adx": [20, 25, 30],  # > 25 at idx 2
            "bb_upper": [52000] * n,
            "bb_lower": [48000] * n,
            "rsi_14": [50, 55, 55],  # between 40-70
            "volume": [300, 300, 300],
            "volume_sma_20": [400] * n,  # vol below avg (Path A won't fire)
        })
        df = populate_trend_entries(df)
        assert df.loc[2, "tf_enter_long"] == 1

    def test_path_b_ema_momentum_short(self):
        """Path B short: EMA9 < EMA50 + ADX > 25 + RSI in range."""
        n = 3
        df = pd.DataFrame({
            "supertrend_direction": [-1, -1, -1],
            "close": [49000, 48900, 48800],
            "ema_9": [49000, 48900, 48800],   # below ema_50
            "ema_50": [49500, 49500, 49500],
            "tf_adx": [20, 25, 30],
            "bb_upper": [52000] * n,
            "bb_lower": [48000] * n,
            "rsi_14": [50, 45, 45],  # between 30-60 (mirror)
            "volume": [300, 300, 300],
            "volume_sma_20": [400] * n,
        })
        df = populate_trend_entries(df)
        assert df.loc[2, "tf_enter_short"] == 1

    def test_path_c_bb_breakout_long(self):
        """Path C: Close > BB upper + ADX rising + volume spike."""
        n = 5
        df = pd.DataFrame({
            "supertrend_direction": [1, 1, 1, 1, 1],
            "close": [50000, 50100, 50200, 50300, 52100],  # breaks above BB at idx 4
            "ema_9": [50000] * n,
            "ema_50": [50500] * n,  # close < ema50 (Path A won't fire)
            "tf_adx": [20, 22, 24, 26, 28],  # rising, but check shift(3)
            "bb_upper": [52000] * n,  # close > bb_upper at idx 4
            "bb_lower": [48000] * n,
            "rsi_14": [75, 75, 75, 75, 75],  # outside Path B range (40-70)
            "volume": [300, 300, 300, 300, 800],
            "volume_sma_20": [400] * n,  # 800 > 400*1.5=600 → vol spike
        })
        df = populate_trend_entries(df)
        assert df.loc[4, "tf_enter_long"] == 1

    def test_or_logic_any_path_fires(self):
        """Any single path firing should produce a long signal."""
        # Path B scenario: EMA cross + ADX + RSI
        n = 3
        df = pd.DataFrame({
            "supertrend_direction": [1, 1, 1],
            "close": [51000, 51100, 51200],
            "ema_9": [51000, 51100, 51200],
            "ema_50": [50500] * n,
            "tf_adx": [20, 25, 30],
            "bb_upper": [52000] * n,
            "bb_lower": [48000] * n,
            "rsi_14": [50, 55, 55],
            "volume": [300, 300, 300],
            "volume_sma_20": [400] * n,
        })
        df = populate_trend_entries(df)
        # Path B fires at idx 2 even though Path A doesn't (no ST flip, low vol)
        assert df.loc[2, "tf_enter_long"] == 1

    def test_no_path_fires_when_conditions_not_met(self):
        """No signals when no path conditions are met."""
        n = 3
        df = pd.DataFrame({
            "supertrend_direction": [1, 1, 1],  # no flip
            "close": [50500, 50500, 50500],
            "ema_9": [50000] * n,  # ema9 < ema50
            "ema_50": [50500] * n,
            "tf_adx": [15, 15, 15],  # ADX too low for B/C
            "bb_upper": [52000] * n,  # close not above BB
            "bb_lower": [48000] * n,
            "rsi_14": [50] * n,
            "volume": [300, 300, 300],
            "volume_sma_20": [400] * n,
        })
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0
        assert df["tf_enter_short"].sum() == 0

    def test_signal_tag_column_exists(self):
        """populate_trend_entries should add tf_signal_tag column."""
        df = pd.DataFrame({
            "supertrend_direction": [1, 1],
            "close": [50500, 50500],
            "ema_9": [50000] * 2,
            "ema_50": [50500] * 2,
            "tf_adx": [15] * 2,
            "bb_upper": [52000] * 2,
            "bb_lower": [48000] * 2,
            "rsi_14": [50] * 2,
            "volume": [300, 300],
            "volume_sma_20": [400] * 2,
        })
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
