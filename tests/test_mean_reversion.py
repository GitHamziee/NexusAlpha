"""Tests for mean reversion strategy — 9-AND confluence entry logic."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.mean_reversion import (
    add_mr_indicators,
    populate_mr_entries,
    populate_mr_exits,
)


def _golden_path_long(n: int = 3) -> pd.DataFrame:
    """Create a DataFrame where ALL 6 entry conditions (3-8) are met at the last row.

    This is the base fixture: every test disables exactly one condition
    and verifies the signal disappears.
    """
    return pd.DataFrame({
        # Condition 3: Close at/below lower BB (close <= bb_lower * 1.001)
        "close": [49700, 49600, 49500],  # last row at BB lower
        # Condition 7: Bullish candle (close > open)
        "open": [49800, 49700, 49600],   # open > close except last: 49500 < 49600? No.
        # Need close > open for bullish candle at last row
        # So: close=49500, open=49400 → bullish at idx 2
        # But close needs to be <= bb_lower * 1.001
        # Let's adjust:
        "mr_bb_lower": [49600] * n,
        "mr_bb_upper": [50400] * n,
        "mr_bb_middle": [50000] * n,
        # Condition 4: RSI < 32
        "mr_rsi": [35, 30, 28],
        # Condition 5: MACD hist turning positive (hist > hist[1] AND hist[1] < 0)
        "mr_macd_hist": [-100, -80, -50],  # -80 < 0 and -50 > -80 → turning positive
        # Condition 6: Volume > SMA * 1.1
        "volume": [500, 500, 550],
        "mr_volume_sma": [400] * n,  # 550 > 400 * 1.1 = 440 ✓
        # Condition 8: Close > EMA200 OR EMA200 slope flat
        "mr_ema_200": [49000] * n,  # close 49500 > 49000 ✓
        "mr_ema_200_slope": [0.001] * n,
    })


def _fix_golden_path_long() -> pd.DataFrame:
    """Corrected golden path where close > open (bullish candle) at last row."""
    n = 3
    return pd.DataFrame({
        "close": [49700, 49500, 49550],   # idx 2: close=49550 > open=49400 → bullish
        "open": [49800, 49600, 49400],    # idx 2: bullish candle
        "mr_bb_lower": [49600] * n,       # 49550 <= 49600 * 1.001 = 49649.6 ✓
        "mr_bb_upper": [50400] * n,
        "mr_bb_middle": [50000] * n,
        "mr_rsi": [35, 30, 28],           # < 32 at idx 2 ✓
        "mr_macd_hist": [-100, -80, -50], # -50 > -80 AND -80 < 0 ✓
        "volume": [500, 500, 550],
        "mr_volume_sma": [400] * n,       # 550 > 440 ✓
        "mr_ema_200": [49000] * n,        # 49550 > 49000 ✓
        "mr_ema_200_slope": [0.001] * n,
    })


def _golden_path_short(n: int = 3) -> pd.DataFrame:
    """Mirror of golden path for short entries."""
    return pd.DataFrame({
        "close": [50300, 50500, 50450],   # idx 2: close=50450 < open=50600 → bearish
        "open": [50200, 50400, 50600],    # idx 2: bearish candle
        "mr_bb_lower": [49600] * n,
        "mr_bb_upper": [50400] * n,       # 50450 >= 50400 * 0.999 = 50349.6 ✓
        "mr_bb_middle": [50000] * n,
        "mr_rsi": [65, 70, 72],           # > 68 at idx 2 ✓
        "mr_macd_hist": [100, 80, 50],    # 50 < 80 AND 80 > 0 ✓
        "volume": [500, 500, 550],
        "mr_volume_sma": [400] * n,       # 550 > 440 ✓
        "mr_ema_200": [51000] * n,        # 50450 < 51000 ✓
        "mr_ema_200_slope": [0.001] * n,
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


class TestPopulateMREntries:
    def test_all_conditions_met_long(self):
        """Golden path: all 6 conditions met → signal fires."""
        df = _fix_golden_path_long()
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 1
        assert df.loc[2, "mr_signal_tag"] == "mean_reversion"

    def test_all_conditions_met_short(self):
        """Golden path short: all 6 conditions met → signal fires."""
        df = _golden_path_short()
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_short"] == 1
        assert df.loc[2, "mr_signal_tag"] == "mean_reversion"

    def test_no_signal_when_above_bb_lower(self):
        """Condition 3 fails: close well above lower BB."""
        df = _fix_golden_path_long()
        df["close"] = [50000, 50000, 50000]  # well above bb_lower
        df["open"] = [49900, 49900, 49900]   # keep bullish
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_rsi_not_oversold(self):
        """Condition 4 fails: RSI above 32."""
        df = _fix_golden_path_long()
        df["mr_rsi"] = [50, 50, 50]  # not oversold
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_macd_not_turning(self):
        """Condition 5 fails: MACD histogram not turning positive."""
        df = _fix_golden_path_long()
        df["mr_macd_hist"] = [-50, -80, -100]  # falling, not turning
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_macd_prev_not_negative(self):
        """Condition 5 fails: previous MACD histogram was positive."""
        df = _fix_golden_path_long()
        df["mr_macd_hist"] = [10, 20, 30]  # rising but prev was positive
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_low_volume(self):
        """Condition 6 fails: volume below SMA * 1.1."""
        df = _fix_golden_path_long()
        df["volume"] = [100, 100, 100]  # well below 400 * 1.1 = 440
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_bearish_candle(self):
        """Condition 7 fails: bearish candle (close < open)."""
        df = _fix_golden_path_long()
        df["open"] = [49800, 49700, 49700]  # close 49550 < open 49700 → bearish
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_no_signal_when_below_ema200_declining(self):
        """Condition 8 fails: close below EMA200 AND slope is steep (not flat)."""
        df = _fix_golden_path_long()
        df["mr_ema_200"] = [55000] * 3        # close well below EMA200
        df["mr_ema_200_slope"] = [-0.01] * 3   # steep decline (not flat)
        df = populate_mr_entries(df)
        assert df["mr_enter_long"].sum() == 0

    def test_signal_when_below_ema200_but_flat(self):
        """Condition 8 allows: below EMA200 but slope is flat."""
        df = _fix_golden_path_long()
        df["mr_ema_200"] = [55000] * 3          # close below EMA200
        df["mr_ema_200_slope"] = [0.0005] * 3   # flat slope (< 0.001)
        df = populate_mr_entries(df)
        assert df.loc[2, "mr_enter_long"] == 1

    def test_no_path_fires_when_conditions_not_met(self):
        """No signals when no conditions are met."""
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
        df = _fix_golden_path_long()
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
