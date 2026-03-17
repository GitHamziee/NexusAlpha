"""Tests for mean reversion strategy — OR-of-simple-groups entry logic."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.mean_reversion import (
    add_mr_indicators,
    populate_mr_entries,
    populate_mr_exits,
)


def _make_ohlcv(n: int = 300, base: float = 50000.0, noise: float = 100.0,
                seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = np.cumsum(rng.normal(0, noise, n)) + base
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes + rng.normal(0, 50, n)
    volumes = rng.uniform(100, 1000, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestAddMRIndicators:
    def test_columns_exist(self):
        df = _make_ohlcv(300)
        df = add_mr_indicators(df)
        expected = ["mr_bb_upper", "mr_bb_lower", "mr_bb_middle",
                    "mr_rsi", "mr_macd_hist", "mr_volume_sma",
                    "mr_atr", "mr_ema_200", "mr_ema_200_slope"]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = add_mr_indicators(df)
        assert len(df) == 0


class TestPopulateMREntries:
    def test_path_a_bb_bounce_long(self):
        """Path A: Close <= BB lower + RSI < 35."""
        n = 3
        df = pd.DataFrame({
            "close": [49600, 49500, 49400],  # at/below BB lower
            "open": [49700, 49600, 49500],   # bearish (won't trigger Path B/C)
            "mr_bb_lower": [49600] * n,
            "mr_bb_upper": [50400] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [40, 34, 30],  # < 35 at idx 1,2
            "mr_macd_hist": [-50, -50, -50],  # not turning up
            "volume": [300, 300, 300],
            "mr_volume_sma": [400] * n,  # low volume
        })
        df = populate_mr_entries(df)
        assert df.loc[1, "mr_enter_long"] == 1
        assert df.loc[1, "mr_signal_tag"] == "mr_bb_bounce"

    def test_path_a_bb_bounce_short(self):
        """Path A short: Close >= BB upper + RSI > 65."""
        n = 3
        df = pd.DataFrame({
            "close": [50400, 50500, 50600],
            "open": [50300, 50400, 50700],  # bullish at idx 0,1 → won't fire Path C
            "mr_bb_lower": [49600] * n,
            "mr_bb_upper": [50400] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [60, 66, 70],  # > 65 at idx 1,2
            "mr_macd_hist": [50, 50, 50],
            "volume": [300, 300, 300],
            "mr_volume_sma": [400] * n,
        })
        df = populate_mr_entries(df)
        assert df.loc[1, "mr_enter_short"] == 1
        assert df.loc[1, "mr_signal_tag"] == "mr_bb_bounce"

    def test_path_b_macd_reversal_long(self):
        """Path B: Close within 2% of BB lower + MACD turning up + bullish candle."""
        n = 5
        df = pd.DataFrame({
            "close": [49800, 49700, 49600, 49500, 49700],  # idx 4 is bullish
            "open": [49850, 49750, 49650, 49550, 49500],   # close > open at idx 4
            "mr_bb_lower": [49000] * n,  # 49700 <= 49000 * 1.02 = 49980 → within 2%
            "mr_bb_upper": [51000] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [45, 45, 45, 45, 45],  # > 35, so Path A won't fire
            "mr_macd_hist": [-50, -40, -30, -20, -10],  # turning up
            "volume": [300, 300, 300, 300, 300],
            "mr_volume_sma": [400] * n,  # low vol, Path C won't fire
        })
        df = populate_mr_entries(df)
        assert df.loc[4, "mr_enter_long"] == 1
        assert df.loc[4, "mr_signal_tag"] == "mr_macd_reversal"

    def test_path_c_rsi_bounce_long(self):
        """Path C: RSI < 30 + bullish candle + volume > SMA."""
        n = 3
        df = pd.DataFrame({
            "close": [50500, 50500, 50600],   # idx 2 is bullish
            "open": [50600, 50600, 50400],     # close > open at idx 2
            "mr_bb_lower": [49000] * n,        # close not near BB lower (Path A/B won't fire)
            "mr_bb_upper": [51000] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [35, 28, 25],            # < 30 at idx 1,2
            "mr_macd_hist": [-50, -50, -50],   # not turning up (Path B blocked)
            "volume": [300, 300, 600],
            "mr_volume_sma": [400] * n,        # 600 > 400 * 1.0 = 400 at idx 2
        })
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 1
        assert df.loc[2, "mr_signal_tag"] == "mr_rsi_bounce"

    def test_or_logic_any_path_fires(self):
        """Any single path firing should produce a long signal."""
        # Path A setup: close at BB lower + RSI < 35
        n = 3
        df = pd.DataFrame({
            "close": [49600, 49500, 49400],
            "open": [49700, 49600, 49500],
            "mr_bb_lower": [49600] * n,
            "mr_bb_upper": [50400] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [40, 33, 28],
            "mr_macd_hist": [-50, -50, -50],
            "volume": [300, 300, 300],
            "mr_volume_sma": [400] * n,
        })
        df = populate_mr_entries(df)
        assert df.loc[1, "mr_enter_long"] == 1  # Path A fires

    def test_no_path_fires_when_conditions_not_met(self):
        """No signals when no path conditions are met."""
        n = 3
        df = pd.DataFrame({
            "close": [50000, 50000, 50000],
            "open": [50000, 50000, 50000],
            "mr_bb_lower": [49000] * n,
            "mr_bb_upper": [51000] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [50, 50, 50],  # not oversold or overbought
            "mr_macd_hist": [-50, -50, -50],
            "volume": [300, 300, 300],
            "mr_volume_sma": [400] * n,
        })
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0
        assert df["mr_enter_short"].sum() == 0

    def test_signal_tag_column_exists(self):
        df = pd.DataFrame({
            "close": [50000], "open": [50000],
            "mr_bb_lower": [49000], "mr_bb_upper": [51000], "mr_bb_middle": [50000],
            "mr_rsi": [50], "mr_macd_hist": [0],
            "volume": [300], "mr_volume_sma": [400],
        })
        df = populate_mr_entries(df)
        assert "mr_signal_tag" in df.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_mr_entries(df)
        assert "mr_enter_long" in df.columns
        assert "mr_signal_tag" in df.columns


class TestPopulateMRExits:
    def test_bb_middle_tp_long(self):
        """Price reaching BB middle should trigger long exit."""
        df = pd.DataFrame({
            "close": [49500, 49700, 49900, 50000, 50100],
            "mr_bb_middle": [50000] * 5,
            "regime": ["RANGING"] * 5,
        })
        df = populate_mr_exits(df)
        assert df.loc[3, "mr_exit_long"] == 1
        assert df.loc[0, "mr_exit_long"] == 0

    def test_regime_change_triggers_exit(self):
        """Regime shifting to TRENDING should exit immediately."""
        df = pd.DataFrame({
            "close": [49500, 49600, 49700, 49800, 49900],
            "mr_bb_middle": [50500] * 5,
            "regime": ["RANGING", "RANGING", "RANGING", "TRENDING_BULL", "TRENDING_BULL"],
        })
        df = populate_mr_exits(df)
        assert df.loc[3, "mr_exit_long"] == 1
        assert df.loc[2, "mr_exit_long"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_mr_exits(df)
        assert "mr_exit_long" in df.columns
