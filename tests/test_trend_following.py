"""Tests for trend following strategy — synthetic data for entry/exit conditions."""

import numpy as np
import pandas as pd
import pytest

from strategies.core.trend_following import (
    ADX_DEATH_LEVEL,
    ADX_ENTRY_THRESH,
    RSI_OB_GUARD,
    RSI_OS_GUARD,
    STOCHRSI_OVERSOLD,
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
                    "atr_14", "volume_sma_20", "rsi_14"]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = add_trend_indicators(df)
        assert len(df) == 0


class TestPopulateTrendEntries:
    def test_no_signals_wrong_regime(self):
        """No entries when regime is RANGING."""
        df = _make_ohlcv(300)
        df = add_trend_indicators(df)
        df = _add_regime_columns(df, regime="RANGING")
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0
        assert df["tf_enter_short"].sum() == 0

    def test_no_signals_low_confidence(self):
        """No entries when confidence < 0.6."""
        df = _make_ohlcv(300)
        df = add_trend_indicators(df)
        df = _add_regime_columns(df, confidence=0.4)
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0

    def test_no_signals_transition(self):
        df = _make_ohlcv(300)
        df = add_trend_indicators(df)
        df = _add_regime_columns(df, regime="TRANSITION", confidence=0.0)
        df = populate_trend_entries(df)
        assert df["tf_enter_long"].sum() == 0
        assert df["tf_enter_short"].sum() == 0

    def test_synthetic_perfect_long_setup(self):
        """Manually construct a row that satisfies all 9 long conditions."""
        n = 5
        df = pd.DataFrame({
            "regime": ["TRENDING_BULL"] * n,
            "regime_confidence": [0.8] * n,
            "supertrend_direction": [1] * n,
            "tf_adx": [15, 20, 25, 30, 35],  # rising and > 28 at index 3,4
            "close": [51000, 51100, 51200, 51300, 51400],
            "ema_200": [50000] * n,
            "ema_50": [50500] * n,
            # StochRSI: K crosses above D from below 30
            "stochrsi_k": [25, 20, 15, 28, 35],
            "stochrsi_d": [30, 25, 20, 30, 30],
            "volume": [500, 500, 500, 500, 600],
            "volume_sma_20": [400] * n,
            "rsi_14": [55, 55, 55, 55, 60],
        })
        df = populate_trend_entries(df)
        # Index 4: ADX=35>28, ADX[4]>ADX[1] (rising), ST=1, close>ema200,
        # close>ema50, K(35)>D(30) & K_prev(28)<D_prev(30) & K_prev(28)<30,
        # vol>sma, RSI<75
        assert df.loc[4, "tf_enter_long"] == 1

    def test_synthetic_perfect_short_setup(self):
        """Manually construct a row that satisfies all 9 short conditions."""
        n = 5
        df = pd.DataFrame({
            "regime": ["TRENDING_BEAR"] * n,
            "regime_confidence": [0.8] * n,
            "supertrend_direction": [-1] * n,
            "tf_adx": [15, 20, 25, 30, 35],
            "close": [49000, 48900, 48800, 48700, 48600],
            "ema_200": [50000] * n,
            "ema_50": [49500] * n,
            # StochRSI: K crosses below D from above 70
            "stochrsi_k": [75, 80, 85, 72, 65],
            "stochrsi_d": [70, 75, 80, 70, 70],
            "volume": [500, 500, 500, 500, 600],
            "volume_sma_20": [400] * n,
            "rsi_14": [45, 45, 45, 45, 40],
        })
        df = populate_trend_entries(df)
        # Index 4: ADX=35>28, rising, ST=-1, close<ema200&50,
        # K(65)<D(70) & K_prev(72)>=D_prev(70) & K_prev(72)>70,
        # vol>sma, RSI>25
        assert df.loc[4, "tf_enter_short"] == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = populate_trend_entries(df)
        assert "tf_enter_long" in df.columns

    def test_rsi_guard_blocks_long(self):
        """RSI above 75 should block long entry even if everything else aligns."""
        n = 5
        df = pd.DataFrame({
            "regime": ["TRENDING_BULL"] * n,
            "regime_confidence": [0.8] * n,
            "supertrend_direction": [1] * n,
            "tf_adx": [15, 20, 25, 30, 35],
            "close": [51000, 51100, 51200, 51300, 51400],
            "ema_200": [50000] * n,
            "ema_50": [50500] * n,
            "stochrsi_k": [25, 20, 15, 28, 35],
            "stochrsi_d": [30, 25, 20, 30, 30],
            "volume": [500, 500, 500, 500, 600],
            "volume_sma_20": [400] * n,
            "rsi_14": [55, 55, 55, 55, 80],  # RSI 80 > 75 → blocked
        })
        df = populate_trend_entries(df)
        assert df.loc[4, "tf_enter_long"] == 0


class TestPopulateTrendExits:
    def test_adx_death_triggers_exit(self):
        """ADX dropping below 18 should trigger exit."""
        df = pd.DataFrame({
            "tf_adx": [30, 25, 20, 15, 12],
            "supertrend_direction": [1, 1, 1, 1, 1],
        })
        df = populate_trend_exits(df)
        # Index 3 and 4 have ADX < 18
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
        assert df.loc[3, "tf_exit_long"] == 1  # flip happened here
        assert df.loc[2, "tf_exit_long"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        df = populate_trend_exits(df)
        assert "tf_exit_long" in df.columns
