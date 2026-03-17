"""Tests for trend following strategy — 9-AND confluence entry logic."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.trend_following import (
    add_trend_indicators,
    populate_trend_entries,
    populate_trend_exits,
)


def _golden_path_long(n: int = 6) -> pd.DataFrame:
    """Create a DataFrame where ALL 6 entry conditions (3-8) are met at the last row.

    This is the base fixture: every test disables exactly one condition
    and verifies the signal disappears.
    """
    return pd.DataFrame({
        # Condition 3: Supertrend bullish
        "supertrend_direction": [1] * n,
        # Condition 4: ADX > 28 AND rising (idx[-1] > idx[-4])
        "tf_adx": [20, 22, 24, 26, 28, 32],
        # Condition 5: Close > EMA200 and EMA50
        "close": [51000, 51100, 51200, 51300, 51400, 51500],
        "ema_200": [50000] * n,
        "ema_50": [50500] * n,
        # Condition 6: StochRSI K > D, was recently oversold
        "stochrsi_k": [20, 15, 10, 25, 40, 55],  # was < 45 within lookback, now K > D
        "stochrsi_d": [30, 25, 20, 30, 35, 50],   # K > D at last row
        # Condition 7: Volume above average
        "volume": [500, 500, 500, 500, 500, 600],
        "volume_sma_20": [400] * n,
        # Condition 8: RSI not overbought
        "rsi_14": [55, 55, 55, 55, 55, 60],
    })


def _golden_path_short(n: int = 6) -> pd.DataFrame:
    """Mirror of golden path for short entries."""
    return pd.DataFrame({
        "supertrend_direction": [-1] * n,
        "tf_adx": [20, 22, 24, 26, 28, 32],
        "close": [49000, 48900, 48800, 48700, 48600, 48500],
        "ema_200": [50000] * n,
        "ema_50": [49500] * n,
        "stochrsi_k": [80, 85, 90, 75, 60, 45],  # was > 55 within lookback, now K < D
        "stochrsi_d": [70, 75, 80, 70, 65, 50],   # K < D at last row
        "volume": [500, 500, 500, 500, 500, 600],
        "volume_sma_20": [400] * n,
        "rsi_14": [45, 45, 45, 45, 45, 40],
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
    def test_all_conditions_met_long(self):
        """Golden path: all 6 conditions met → signal fires."""
        df = _golden_path_long()
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_long"] == 1
        assert df.loc[5, "tf_signal_tag"] == "trend_following"

    def test_all_conditions_met_short(self):
        """Golden path short: all 6 conditions met → signal fires."""
        df = _golden_path_short()
        df = populate_trend_entries(df)
        assert df.loc[5, "tf_enter_short"] == 1
        assert df.loc[5, "tf_signal_tag"] == "trend_following"

    def test_no_signal_when_supertrend_bearish(self):
        """Condition 3 fails: Supertrend bearish blocks long entry."""
        df = _golden_path_long()
        df["supertrend_direction"] = -1  # bearish
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_adx_below_threshold(self):
        """Condition 4 fails: ADX below 28 blocks entry."""
        df = _golden_path_long()
        df["tf_adx"] = [15, 16, 17, 18, 19, 20]  # all below 28
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_adx_not_rising(self):
        """Condition 4 fails: ADX not rising (lower than 3 candles ago)."""
        df = _golden_path_long()
        df["tf_adx"] = [35, 34, 33, 32, 31, 30]  # above 28 but falling
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_below_ema200(self):
        """Condition 5 fails: close below EMA200."""
        df = _golden_path_long()
        df["ema_200"] = [55000] * 6  # close is below EMA200
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_below_ema50(self):
        """Condition 5 fails: close below EMA50."""
        df = _golden_path_long()
        df["ema_50"] = [55000] * 6  # close is below EMA50
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_stochrsi_not_oversold(self):
        """Condition 6 fails: StochRSI was never oversold recently."""
        df = _golden_path_long()
        df["stochrsi_k"] = [60, 65, 70, 75, 80, 85]  # never below 45
        df["stochrsi_d"] = [55, 60, 65, 70, 75, 80]   # K > D but no oversold
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_low_volume(self):
        """Condition 7 fails: volume below SMA."""
        df = _golden_path_long()
        df["volume"] = [100] * 6  # below SMA of 400
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signal_when_rsi_overbought(self):
        """Condition 8 fails: RSI above 75 blocks long entry."""
        df = _golden_path_long()
        df["rsi_14"] = [78, 78, 78, 78, 78, 78]  # above 75
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

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
