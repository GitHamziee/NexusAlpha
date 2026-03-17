"""Tests for mean reversion strategy — simplified 3+1 tiered confluence."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.mean_reversion import (
    add_mr_indicators,
    populate_mr_entries,
    populate_mr_exits,
)


def _golden_path_long(n: int = 3) -> pd.DataFrame:
    """Create a DataFrame where all 3 hard gates + full confluence (3/3) are met.

    Hard gates: close <= BB lower * mult, RSI < oversold, bullish candle
    Scoring: volume > 1.2x SMA, MACD turning positive, close > EMA200
    """
    return pd.DataFrame({
        # Gate 1: close <= BB lower * 1.001 (BTC default)
        "close": [49700, 49600, 49550],   # 49550 <= 49600 * 1.001 = 49649.6 ✓
        # Gate 3: bullish candle (close > open)
        "open": [49800, 49700, 49400],    # idx 2: 49550 > 49400 → bullish ✓
        "mr_bb_lower": [49600] * n,
        "mr_bb_upper": [50400] * n,
        "mr_bb_middle": [50000] * n,
        # Gate 2: RSI < 35 (BTC oversold threshold)
        "mr_rsi": [40, 33, 28],           # idx 2: 28 < 35 ✓
        # Score A: volume > 1.2x SMA
        "volume": [500, 500, 550],
        "mr_volume_sma": [400] * n,       # 550 > 400 * 1.2 = 480 ✓
        # Score B: MACD histogram turning positive
        "mr_macd_hist": [-100, -80, -50], # -50 > -80 AND -80 < 0 ✓
        # Score C: close > EMA200
        "mr_ema_200": [49000] * n,        # 49550 > 49000 ✓
        "mr_ema_200_slope": [0.002] * n,
    })


def _golden_path_short(n: int = 3) -> pd.DataFrame:
    """Mirror of golden path for short entries."""
    return pd.DataFrame({
        # Gate 1: close >= BB upper * 0.999 (BTC default)
        "close": [50300, 50500, 50450],   # 50450 >= 50400 * 0.999 = 50349.6 ✓
        # Gate 3: bearish candle (close < open)
        "open": [50200, 50400, 50600],    # idx 2: 50450 < 50600 → bearish ✓
        "mr_bb_lower": [49600] * n,
        "mr_bb_upper": [50400] * n,
        "mr_bb_middle": [50000] * n,
        # Gate 2: RSI > 65 (BTC overbought threshold)
        "mr_rsi": [60, 66, 72],           # idx 2: 72 > 65 ✓
        # Score A: volume > 1.2x SMA
        "volume": [500, 500, 550],
        "mr_volume_sma": [400] * n,       # 550 > 480 ✓
        # Score B: MACD turning negative
        "mr_macd_hist": [100, 80, 50],    # 50 < 80 AND 80 > 0 ✓
        # Score C: close < EMA200
        "mr_ema_200": [51000] * n,        # 50450 < 51000 ✓
        "mr_ema_200_slope": [0.002] * n,
    })


class TestAddMRIndicators:
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

    def test_per_pair_bb_std(self):
        """SOL pair should use wider BB (std=2.5 vs BTC's 2.0)."""
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
        df = add_mr_indicators(df, pair="SOL/USDT:USDT")
        assert "mr_bb_lower" in df.columns


class TestPopulateMREntries:
    def test_all_gates_plus_full_confluence_long(self):
        """All 3 gates TRUE + 3/3 confluence = signal fires."""
        df = _golden_path_long()
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 1
        assert df.loc[2, "mr_signal_tag"] == "mean_reversion"

    def test_all_gates_plus_min_confluence_long(self):
        """All 3 gates TRUE + exactly 1/3 confluence = signal fires."""
        df = _golden_path_long()
        # Disable score B (MACD) and score C (EMA200)
        df["mr_macd_hist"] = [-50, -50, -50]    # not rising → score B = 0
        df["mr_ema_200"] = [55000] * 3           # close below → score C needs flat
        df["mr_ema_200_slope"] = [-0.01] * 3     # not flat → score C = 0
        # Score A (volume) = 1 → total = 1 ≥ 1 ✓
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 1

    def test_all_gates_but_zero_confluence(self):
        """All 3 gates TRUE but 0/3 confluence = NO signal."""
        df = _golden_path_long()
        # Disable all scores
        df["volume"] = [100] * 3                 # below 1.2x SMA → score A = 0
        df["mr_macd_hist"] = [-50, -50, -50]    # not rising → score B = 0
        df["mr_ema_200"] = [55000] * 3           # close below
        df["mr_ema_200_slope"] = [-0.01] * 3     # not flat → score C = 0
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 0

    def test_no_signal_when_above_bb_lower(self):
        """Gate 1 fails: close well above lower BB."""
        df = _golden_path_long()
        df["close"] = [50000, 50000, 50000]     # well above bb_lower
        df["open"] = [49900, 49900, 49900]       # keep bullish
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_rsi_not_oversold(self):
        """Gate 2 fails: RSI above oversold threshold."""
        df = _golden_path_long()
        df["mr_rsi"] = [50, 50, 50]             # not oversold (> 35)
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_bearish_candle(self):
        """Gate 3 fails: bearish candle (close < open)."""
        df = _golden_path_long()
        df["open"] = [49800, 49700, 49700]       # close 49550 < open 49700 → bearish
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_all_conditions_met_short(self):
        """Golden path short: all gates + confluence met."""
        df = _golden_path_short()
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_short"] == 1
        assert df.loc[2, "mr_signal_tag"] == "mean_reversion"

    def test_per_pair_rsi_threshold(self):
        """SOL uses RSI oversold=25 instead of BTC's 35."""
        df = _golden_path_long()
        df["mr_rsi"] = [40, 33, 30]             # above 25 (SOL), below 35 (BTC)

        # BTC: RSI 30 < 35 → gate 2 passes
        df_btc = df.copy()
        df_btc = populate_mr_entries(df_btc, pair="BTC/USDT:USDT")
        assert df_btc.loc[2, "mr_enter_long"] == 1

        # SOL: RSI 30 > 25 is False (30 > 25 is True, but we need RSI < 25)
        # RSI 30 < 25 is False → gate 2 fails
        df_sol = df.copy()
        df_sol = populate_mr_entries(df_sol, pair="SOL/USDT:USDT")
        assert df_sol.loc[2, "mr_enter_long"] == 0

    def test_below_ema200_but_flat_slope(self):
        """Score C: below EMA200 but slope is flat → still counts."""
        df = _golden_path_long()
        df["mr_ema_200"] = [55000] * 3          # close below EMA200
        df["mr_ema_200_slope"] = [0.0005] * 3   # flat slope (< 0.001)
        # Disable other scores
        df["volume"] = [100] * 3                 # score A = 0
        df["mr_macd_hist"] = [-50, -50, -50]    # score B = 0
        # Score C = 1 (flat slope) → total = 1 ≥ 1 ✓
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 1

    def test_no_conditions_met(self):
        """No signals when nothing is met."""
        n = 3
        df = pd.DataFrame({
            "close": [50000, 50000, 50000],
            "open": [50000, 50000, 50000],
            "mr_bb_lower": [49000] * n,
            "mr_bb_upper": [51000] * n,
            "mr_bb_middle": [50000] * n,
            "mr_rsi": [50, 50, 50],
            "mr_macd_hist": [-50, -50, -50],
            "volume": [300, 300, 300],
            "mr_volume_sma": [400] * n,
            "mr_ema_200": [49000] * n,
            "mr_ema_200_slope": [0.001] * n,
        })
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0
        assert df["mr_enter_short"].sum() == 0

    def test_signal_tag_column_exists(self):
        df = _golden_path_long()
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
