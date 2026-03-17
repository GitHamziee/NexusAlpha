"""
Regime Detector — Layer 1 of the 6-layer signal filter.

Classifies market into 5 regimes using adaptive thresholds so the system
only runs strategies that match the current market state.  This single
filter eliminates ~60 % of losing trades.

Adaptive thresholds: instead of fixed ADX 18/28, we track a 100-period
SMA of ADX and derive the range/trend boundaries from that baseline.
This lets the detector adjust to structurally different volatility
environments (e.g. 2021 bull vs 2022 bear).
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# ── Regime labels ────────────────────────────────────────────────────────
TRENDING_BULL = "TRENDING_BULL"
TRENDING_BEAR = "TRENDING_BEAR"
RANGING = "RANGING"
VOLATILE = "VOLATILE"
TRANSITION = "TRANSITION"

# ── Adaptive threshold offsets from ADX baseline ─────────────────────────
# ADX baseline = SMA(ADX, 100).  Range ceiling = baseline - 4,
# trend floor = baseline + 6.  Gap between them is the TRANSITION zone.
ADX_BASELINE_PERIOD = 100
ADX_RANGE_OFFSET = -4      # below baseline → ranging
ADX_TREND_OFFSET = 6       # above baseline → trending

# Hard floors/ceilings so thresholds stay sensible
ADX_RANGE_FLOOR = 12
ADX_RANGE_CEIL = 22
ADX_TREND_FLOOR = 22
ADX_TREND_CEIL = 35

# ── Volatility spike multipliers ─────────────────────────────────────────
ATR_SPIKE_MULT = 2.0       # ATR > 2x its SMA → volatile
BB_SPIKE_MULT = 2.5        # BB width > 2.5x its SMA → volatile

# ── Indicator periods ────────────────────────────────────────────────────
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
EMA_SLOPE_PERIOD = 50
EMA_SLOPE_LOOKBACK = 10    # rate-of-change window for EMA slope
ATR_PERIOD = 14
VOLATILITY_SMA_PERIOD = 100


def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed for regime classification.

    Operates in-place on *df* and returns it.  Expects standard OHLCV
    columns (open, high, low, close, volume).
    """
    if df.empty:
        return df

    # ADX + Directional Indicators
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)
    if adx_df is not None and not adx_df.empty:
        df["adx_14"] = adx_df[f"ADX_{ADX_PERIOD}"]
        df["plus_di"] = adx_df[f"DMP_{ADX_PERIOD}"]
        df["minus_di"] = adx_df[f"DMN_{ADX_PERIOD}"]
    else:
        df["adx_14"] = float("nan")
        df["plus_di"] = float("nan")
        df["minus_di"] = float("nan")

    # Adaptive ADX baseline (100-period SMA of ADX)
    df["adx_baseline"] = df["adx_14"].rolling(window=ADX_BASELINE_PERIOD, min_periods=30).mean()

    # Bollinger Bands → width
    bbands = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
    if bbands is not None and not bbands.empty:
        # pandas-ta column names vary by version; find them dynamically
        bbu_col = [c for c in bbands.columns if c.startswith("BBU_")][0]
        bbl_col = [c for c in bbands.columns if c.startswith("BBL_")][0]
        bbm_col = [c for c in bbands.columns if c.startswith("BBM_")][0]
        df["bb_upper"] = bbands[bbu_col]
        df["bb_lower"] = bbands[bbl_col]
        df["bb_middle"] = bbands[bbm_col]
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    else:
        df["bb_upper"] = float("nan")
        df["bb_lower"] = float("nan")
        df["bb_middle"] = float("nan")
        df["bb_width"] = float("nan")

    df["bb_width_sma"] = df["bb_width"].rolling(window=VOLATILITY_SMA_PERIOD, min_periods=20).mean()

    # EMA(50) slope — 10-period rate of change
    df["ema_50"] = ta.ema(df["close"], length=EMA_SLOPE_PERIOD)
    df["ema_slope"] = df["ema_50"].pct_change(periods=EMA_SLOPE_LOOKBACK)

    # ATR(14) + normalized ratio + SMA
    atr = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["atr_14"] = atr if atr is not None else float("nan")
    df["atr_ratio"] = df["atr_14"] / df["close"]
    df["atr_ratio_sma"] = df["atr_ratio"].rolling(window=VOLATILITY_SMA_PERIOD, min_periods=20).mean()

    return df


def classify_regime(
    adx: float,
    plus_di: float,
    minus_di: float,
    bb_width: float,
    bb_width_sma: float,
    ema_slope: float,
    atr_ratio: float,
    atr_ratio_sma: float,
    adx_baseline: float,
) -> Tuple[str, float]:
    """Classify a single row into one of 5 regimes with a confidence score.

    Uses adaptive thresholds derived from *adx_baseline* (100-period SMA of
    ADX) so the detector tracks structural changes in volatility.

    Returns (regime_label, confidence) where confidence is in [0.0, 0.9].
    TRANSITION always returns confidence 0.0 to block all trades.
    """
    # ── bail on missing data ─────────────────────────────────────────
    if _any_nan(adx, plus_di, minus_di, bb_width, bb_width_sma,
                ema_slope, atr_ratio, atr_ratio_sma, adx_baseline):
        return TRANSITION, 0.0

    # ── adaptive thresholds ──────────────────────────────────────────
    range_thresh = _clamp(adx_baseline + ADX_RANGE_OFFSET, ADX_RANGE_FLOOR, ADX_RANGE_CEIL)
    trend_thresh = _clamp(adx_baseline + ADX_TREND_OFFSET, ADX_TREND_FLOOR, ADX_TREND_CEIL)

    # ── VOLATILE — checked first (overrides everything) ──────────────
    if atr_ratio > atr_ratio_sma * ATR_SPIKE_MULT or bb_width > bb_width_sma * BB_SPIKE_MULT:
        return VOLATILE, 0.3

    # ── STRONG TREND ─────────────────────────────────────────────────
    if adx > trend_thresh:
        if plus_di > minus_di and ema_slope > 0:
            confidence = min(0.9, adx / 45.0)
            return TRENDING_BULL, confidence
        if minus_di > plus_di and ema_slope < 0:
            confidence = min(0.9, adx / 45.0)
            return TRENDING_BEAR, confidence

    # ── CLEAR RANGING ────────────────────────────────────────────────
    if adx < range_thresh and bb_width < bb_width_sma * 0.8:
        confidence = min(0.8, (range_thresh - adx) / 12.0)
        return RANGING, confidence

    # ── TRANSITION — DO NOT TRADE ────────────────────────────────────
    if range_thresh <= adx <= trend_thresh:
        return TRANSITION, 0.0

    # ── fallback: if nothing matched, treat as ranging with low conf ─
    return RANGING, 0.5


def apply_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised regime classification over a whole DataFrame.

    Adds ``regime`` and ``regime_confidence`` columns.  Requires
    `add_regime_indicators` to have been called first.
    """
    if df.empty:
        df["regime"] = pd.Series(dtype=str)
        df["regime_confidence"] = pd.Series(dtype=float)
        return df

    required = ["adx_14", "plus_di", "minus_di", "bb_width", "bb_width_sma",
                "ema_slope", "atr_ratio", "atr_ratio_sma", "adx_baseline"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing indicator columns: {missing}")

    regimes = []
    confidences = []
    for row in df[required].itertuples(index=False):
        regime, conf = classify_regime(*row)
        regimes.append(regime)
        confidences.append(conf)

    df["regime"] = regimes
    df["regime_confidence"] = confidences
    return df


def confirm_regime_multitf(
    regime_15m: str,
    conf_15m: float,
    regime_1h: str,
    conf_1h: float,
) -> Tuple[str, float]:
    """Apply multi-timeframe confirmation (Layer 2).

    Rules from the spec:
    - Both agree → confidence as-is
    - Disagree → cut confidence by 50 %
    - 1H says VOLATILE → override to VOLATILE regardless of 15m
    """
    # 1H VOLATILE overrides everything (higher TF wins for danger)
    if regime_1h == VOLATILE:
        return VOLATILE, 0.3

    # Agreement
    if regime_15m == regime_1h:
        return regime_15m, conf_15m

    # Disagreement → halve confidence (likely drops below 0.6 threshold)
    adjusted = conf_15m * 0.5
    return regime_15m, adjusted


def apply_multitf_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust regime columns using 1H confirmation data already merged.

    Expects ``regime_1h`` and ``regime_confidence_1h`` columns produced by
    running the regime detector on the 1H informative DataFrame and then
    using Freqtrade's ``merge_informative_pair``.
    """
    if df.empty:
        return df

    # If 1H columns are missing, leave 15m regime as-is (graceful degradation)
    if "regime_1h" not in df.columns or "regime_confidence_1h" not in df.columns:
        logger.warning("1H regime columns missing — skipping multi-TF confirmation")
        return df

    final_regimes = []
    final_confs = []
    for _, row in df.iterrows():
        r, c = confirm_regime_multitf(
            row["regime"],
            row["regime_confidence"],
            row.get("regime_1h", TRANSITION),
            row.get("regime_confidence_1h", 0.0),
        )
        final_regimes.append(r)
        final_confs.append(c)

    df["regime"] = final_regimes
    df["regime_confidence"] = final_confs
    return df


# ── private helpers ──────────────────────────────────────────────────────

def _any_nan(*values: float) -> bool:
    """Return True if any value is NaN."""
    import math
    return any(math.isnan(v) for v in values)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
