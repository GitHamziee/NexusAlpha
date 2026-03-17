"""
Tests for the regime detector — synthetic data for each of the 5 regimes.

Each test constructs indicator values that should trigger a specific regime
classification, verifying both the label and confidence bounds.
"""

import math

import numpy as np
import pandas as pd
import pytest

from strategies.core.regime_detector import (
    RANGING,
    TRANSITION,
    TRENDING_BEAR,
    TRENDING_BULL,
    VOLATILE,
    add_regime_indicators,
    apply_regime,
    classify_regime,
    confirm_regime_multitf,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 250, base: float = 50000.0, trend: float = 0.0,
                noise: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    *trend*: per-bar drift (positive = uptrend).
    *noise*: standard deviation of random walk.
    """
    rng = np.random.default_rng(seed)
    closes = np.cumsum(rng.normal(trend, noise, n)) + base
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes + rng.normal(0, 50, n)
    volumes = rng.uniform(100, 1000, n)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ── classify_regime unit tests ───────────────────────────────────────────

class TestClassifyRegime:
    """Direct tests on the pure classify_regime function."""

    def test_trending_bull(self):
        regime, conf = classify_regime(
            adx=35, plus_di=30, minus_di=15,
            bb_width=0.04, bb_width_sma=0.03,
            ema_slope=0.01, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=25,
        )
        assert regime == TRENDING_BULL
        assert 0.0 < conf <= 0.9

    def test_trending_bear(self):
        regime, conf = classify_regime(
            adx=35, plus_di=12, minus_di=28,
            bb_width=0.04, bb_width_sma=0.03,
            ema_slope=-0.01, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=25,
        )
        assert regime == TRENDING_BEAR
        assert 0.0 < conf <= 0.9

    def test_ranging(self):
        """Low ADX + narrow BB → RANGING."""
        regime, conf = classify_regime(
            adx=14, plus_di=15, minus_di=16,
            bb_width=0.02, bb_width_sma=0.035,  # bb_width < sma * 0.8
            ema_slope=0.0, atr_ratio=0.008, atr_ratio_sma=0.008,
            adx_baseline=22,  # range_thresh = 22 - 4 = 18, adx=14 < 18
        )
        assert regime == RANGING
        assert 0.0 < conf <= 0.8

    def test_volatile_atr_spike(self):
        """ATR ratio > 2x its SMA → VOLATILE (overrides everything)."""
        regime, conf = classify_regime(
            adx=40, plus_di=30, minus_di=10,
            bb_width=0.04, bb_width_sma=0.03,
            ema_slope=0.02, atr_ratio=0.05, atr_ratio_sma=0.02,
            adx_baseline=25,
        )
        assert regime == VOLATILE
        assert conf == 0.3

    def test_volatile_bb_explosion(self):
        """BB width > 2.5x its SMA → VOLATILE."""
        regime, conf = classify_regime(
            adx=15, plus_di=15, minus_di=16,
            bb_width=0.10, bb_width_sma=0.03,  # 0.10 > 0.03 * 2.5
            ema_slope=0.0, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=22,
        )
        assert regime == VOLATILE

    def test_transition(self):
        """ADX between range and trend thresholds → TRANSITION (hard gate, conf=0.0)."""
        # baseline=22, range_thresh = 18, trend_thresh = 28
        regime, conf = classify_regime(
            adx=23, plus_di=18, minus_di=17,
            bb_width=0.035, bb_width_sma=0.03,  # not narrow enough for RANGING
            ema_slope=0.001, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=22,
        )
        assert regime == TRANSITION
        assert conf == pytest.approx(0.0)

    def test_nan_returns_transition(self):
        """Any NaN input should safely return TRANSITION with zero conf."""
        regime, conf = classify_regime(
            adx=float("nan"), plus_di=20, minus_di=15,
            bb_width=0.03, bb_width_sma=0.03,
            ema_slope=0.0, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=22,
        )
        assert regime == TRANSITION
        assert conf == 0.0

    def test_confidence_never_exceeds_0_9(self):
        """Even extremely high ADX should cap confidence at 0.9."""
        regime, conf = classify_regime(
            adx=60, plus_di=40, minus_di=10,
            bb_width=0.04, bb_width_sma=0.03,
            ema_slope=0.02, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=25,
        )
        assert conf <= 0.9

    def test_adaptive_threshold_high_baseline(self):
        """When ADX baseline is high, trend threshold rises accordingly."""
        # baseline=30 → trend_thresh = min(35, 30+3) = 33
        # ADX=33 is now at the boundary — 33 <= 33, so TRANSITION
        regime, conf = classify_regime(
            adx=33, plus_di=25, minus_di=15,
            bb_width=0.04, bb_width_sma=0.03,
            ema_slope=0.01, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=30,
        )
        assert regime == TRANSITION
        assert conf == pytest.approx(0.0)

    def test_adaptive_threshold_low_baseline(self):
        """When ADX baseline is low, trend threshold drops to floor."""
        # baseline=15 → trend_thresh = max(22, 15+6) = 22
        # ADX=24 is now trending
        regime, conf = classify_regime(
            adx=24, plus_di=25, minus_di=15,
            bb_width=0.04, bb_width_sma=0.03,
            ema_slope=0.01, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=15,
        )
        assert regime == TRENDING_BULL

    def test_fallback_ranging(self):
        """ADX below range thresh but BB not narrow → fallback RANGING 0.5."""
        # baseline=22 → range_thresh=18, adx=15 < 18 but bb_width NOT < sma*0.8
        regime, conf = classify_regime(
            adx=15, plus_di=15, minus_di=16,
            bb_width=0.035, bb_width_sma=0.03,  # 0.035 > 0.03*0.8=0.024, so not narrow
            ema_slope=0.0, atr_ratio=0.01, atr_ratio_sma=0.01,
            adx_baseline=22,
        )
        # ADX 15 < range_thresh 18, but bb_width not narrow → falls through
        # range_thresh(18) <= adx(15) is False, so TRANSITION check fails
        # Falls to default RANGING 0.5
        assert regime == RANGING
        assert conf == 0.5


# ── multi-timeframe confirmation tests ───────────────────────────────────

class TestMultiTFConfirmation:
    def test_agreement_keeps_confidence(self):
        r, c = confirm_regime_multitf(TRENDING_BULL, 0.8, TRENDING_BULL, 0.7)
        assert r == TRENDING_BULL
        assert c == 0.8

    def test_disagreement_reduces_confidence(self):
        r, c = confirm_regime_multitf(TRENDING_BULL, 0.8, RANGING, 0.6)
        assert r == TRENDING_BULL
        assert c == pytest.approx(0.64)  # 0.8 * 0.80

    def test_1h_volatile_overrides(self):
        """1H VOLATILE overrides regardless of 15m."""
        r, c = confirm_regime_multitf(TRENDING_BULL, 0.9, VOLATILE, 0.3)
        assert r == VOLATILE
        assert c == 0.3

    def test_transition_on_15m(self):
        r, c = confirm_regime_multitf(TRANSITION, 0.0, RANGING, 0.6)
        assert r == TRANSITION
        assert c == pytest.approx(0.0)  # TRANSITION conf=0.0, penalty irrelevant


# ── full pipeline tests (add_regime_indicators → apply_regime) ───────────

class TestFullPipeline:
    def test_add_regime_indicators_columns_exist(self):
        """All expected indicator columns are produced."""
        df = _make_ohlcv(250)
        df = add_regime_indicators(df)
        expected = ["adx_14", "plus_di", "minus_di", "adx_baseline",
                    "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_width_sma",
                    "ema_50", "ema_slope", "atr_14", "atr_ratio", "atr_ratio_sma"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_apply_regime_produces_columns(self):
        """apply_regime adds regime + regime_confidence columns."""
        df = _make_ohlcv(250)
        df = add_regime_indicators(df)
        df = apply_regime(df)
        assert "regime" in df.columns
        assert "regime_confidence" in df.columns

    def test_regime_values_are_valid(self):
        """All regime values are one of the 5 known labels."""
        df = _make_ohlcv(250)
        df = add_regime_indicators(df)
        df = apply_regime(df)
        valid = {TRENDING_BULL, TRENDING_BEAR, RANGING, VOLATILE, TRANSITION}
        unique = set(df["regime"].dropna().unique())
        assert unique.issubset(valid), f"Unexpected regime values: {unique - valid}"

    def test_confidence_range(self):
        """Confidence is always between 0.0 and 0.9."""
        df = _make_ohlcv(250)
        df = add_regime_indicators(df)
        df = apply_regime(df)
        valid = df["regime_confidence"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 0.9).all()

    def test_empty_dataframe(self):
        """Empty DataFrame doesn't crash."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = add_regime_indicators(df)
        df = apply_regime(df)
        assert len(df) == 0

    def test_strong_uptrend_data(self):
        """Synthetic strong uptrend should produce some TRENDING_BULL labels."""
        df = _make_ohlcv(300, trend=50.0, noise=30.0, seed=7)
        df = add_regime_indicators(df)
        df = apply_regime(df)
        # After warmup, we expect at least some trending bull
        tail = df.tail(50)
        assert TRENDING_BULL in tail["regime"].values

    def test_strong_downtrend_data(self):
        """Synthetic strong downtrend should produce some TRENDING_BEAR labels."""
        df = _make_ohlcv(300, base=60000, trend=-50.0, noise=30.0, seed=8)
        df = add_regime_indicators(df)
        df = apply_regime(df)
        tail = df.tail(50)
        assert TRENDING_BEAR in tail["regime"].values

    def test_missing_columns_raises(self):
        """apply_regime raises ValueError if indicators haven't been computed."""
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match="Missing indicator columns"):
            apply_regime(df)
