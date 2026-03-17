"""Tests for mean reversion strategy."""

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
    def test_no_signals_wrong_regime(self):
        df = _make_ohlcv(300)
        df = add_mr_indicators(df)
        df["regime"] = "TRENDING_BULL"
        df["regime_confidence"] = 0.8
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0
        assert df["mr_enter_short"].sum() == 0

    def test_no_signals_low_confidence(self):
        df = _make_ohlcv(300)
        df = add_mr_indicators(df)
        df["regime"] = "RANGING"
        df["regime_confidence"] = 0.4
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_synthetic_perfect_long_setup(self):
        """All 9 long conditions manually satisfied."""
        n = 5
        df = pd.DataFrame({
            "regime": ["RANGING"] * n,
            "regime_confidence": [0.7] * n,
            "close": [49800, 49700, 49600, 49500, 49550],  # last is bullish
            "open": [49850, 49750, 49650, 49550, 49450],   # close > open at idx 4
            "mr_bb_lower": [49600] * n,  # close <= lower * 1.001 at idx 4: 49550 <= 49600*1.001=49649.6
            "mr_bb_upper": [50400] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [40, 35, 30, 28, 29],  # < 32 at idx 3,4
            "mr_macd_hist": [-50, -40, -30, -20, -10],  # turning up: hist > hist[1] and hist[1] < 0
            "volume": [500, 500, 500, 500, 600],
            "mr_volume_sma": [500] * n,  # 600 > 500 * 1.1 = 550
            "mr_ema_200": [49000] * n,  # close > ema_200
            "mr_ema_200_slope": [0.0001] * n,
        })
        df = populate_mr_entries(df)
        assert df.loc[4, "mr_enter_long"] == 1

    def test_synthetic_perfect_short_setup(self):
        """All 9 short conditions manually satisfied."""
        n = 5
        df = pd.DataFrame({
            "regime": ["RANGING"] * n,
            "regime_confidence": [0.7] * n,
            "close": [50200, 50300, 50400, 50450, 50420],  # last is bearish
            "open": [50150, 50250, 50350, 50400, 50500],   # close < open at idx 4
            "mr_bb_upper": [50400] * n,  # close >= upper * 0.999 at idx 4: 50420 >= 50400*0.999=50349.6
            "mr_bb_lower": [49600] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [60, 65, 70, 72, 71],  # > 68 at idx 3,4
            "mr_macd_hist": [50, 40, 30, 20, 10],  # turning down: hist < hist[1] and hist[1] > 0
            "volume": [500, 500, 500, 500, 600],
            "mr_volume_sma": [500] * n,
            "mr_ema_200": [51000] * n,  # close < ema_200
            "mr_ema_200_slope": [0.0001] * n,
        })
        df = populate_mr_entries(df)
        assert df.loc[4, "mr_enter_short"] == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_mr_entries(df)
        assert "mr_enter_long" in df.columns


class TestPopulateMRExits:
    def test_bb_middle_tp_long(self):
        """Price reaching BB middle should trigger long exit."""
        df = pd.DataFrame({
            "close": [49500, 49700, 49900, 50000, 50100],
            "mr_bb_middle": [50000] * 5,
            "regime": ["RANGING"] * 5,
        })
        df = populate_mr_exits(df)
        assert df.loc[3, "mr_exit_long"] == 1  # close >= bb_middle
        assert df.loc[0, "mr_exit_long"] == 0

    def test_regime_change_triggers_exit(self):
        """Regime shifting to TRENDING should exit immediately."""
        df = pd.DataFrame({
            "close": [49500, 49600, 49700, 49800, 49900],
            "mr_bb_middle": [50500] * 5,  # price hasn't reached middle
            "regime": ["RANGING", "RANGING", "RANGING", "TRENDING_BULL", "TRENDING_BULL"],
        })
        df = populate_mr_exits(df)
        assert df.loc[3, "mr_exit_long"] == 1
        assert df.loc[2, "mr_exit_long"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_mr_exits(df)
        assert "mr_exit_long" in df.columns
