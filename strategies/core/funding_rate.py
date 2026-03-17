"""
Funding Rate Arbitrage Strategy — supplemental income from crowded positioning.

Active when: ALL regimes except VOLATILE.
Extreme funding = crowded positioning = mean reversion pressure.

Always half-size (0.5% risk), 3x ATR stops (wider), extreme thresholds only.
If funding data is unavailable, strategy simply doesn't fire.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

# ── indicator parameters ─────────────────────────────────────────────────
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 20

# ── entry thresholds (EXTREME — by design) ───────────────────────────────
FUNDING_LONG_THRESH = -0.0005    # funding < -0.05% (short crowding)
FUNDING_SHORT_THRESH = 0.0008    # funding > 0.08% (long crowding)
LS_RATIO_LONG_THRESH = 0.7       # crowd very short
LS_RATIO_SHORT_THRESH = 1.8      # crowd very long
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# ── exit parameters ──────────────────────────────────────────────────────
STOP_ATR_MULT = 3.0              # wider stops for funding trades
FUNDING_NORMAL_LOW = -0.0001     # funding normalized range
FUNDING_NORMAL_HIGH = 0.0001
MIN_HOLD_CANDLES = 32            # 1 funding period = 8 hours = 32 × 15m
MAX_HOLD_CANDLES = 192           # 48 hours


def add_funding_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators needed by the funding rate strategy."""
    if df.empty:
        return df

    rsi = ta.rsi(df["close"], length=RSI_PERIOD)
    df["fr_rsi"] = rsi if rsi is not None else float("nan")

    atr = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["fr_atr"] = atr if atr is not None else float("nan")

    df["fr_volume_sma"] = df["volume"].rolling(window=VOLUME_SMA_PERIOD).mean()

    # Funding columns are injected externally from the FundingDataProvider
    if "funding_rate" not in df.columns:
        df["funding_rate"] = float("nan")
    if "long_short_ratio" not in df.columns:
        df["long_short_ratio"] = float("nan")

    return df


def populate_funding_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Add funding rate strategy entry signals.

    Adds columns: fr_enter_long, fr_enter_short.
    Graceful: if funding_rate or long_short_ratio columns are all NaN,
    no signals fire.
    """
    if df.empty:
        df["fr_enter_long"] = pd.Series(dtype=int)
        df["fr_enter_short"] = pd.Series(dtype=int)
        return df

    vol_ok = df["volume"] > df["fr_volume_sma"]
    not_volatile = df["regime"] != "VOLATILE"

    # ── LONG — extreme short crowding ────────────────────────────────
    long_cond = (
        (df["funding_rate"] < FUNDING_LONG_THRESH) &
        (df["fr_rsi"] < RSI_OVERSOLD) &
        (df["long_short_ratio"] < LS_RATIO_LONG_THRESH) &
        vol_ok &
        not_volatile
    )

    # ── SHORT — extreme long crowding ────────────────────────────────
    short_cond = (
        (df["funding_rate"] > FUNDING_SHORT_THRESH) &
        (df["fr_rsi"] > RSI_OVERBOUGHT) &
        (df["long_short_ratio"] > LS_RATIO_SHORT_THRESH) &
        vol_ok &
        not_volatile
    )

    df["fr_enter_long"] = long_cond.astype(int).fillna(0).astype(int)
    df["fr_enter_short"] = short_cond.astype(int).fillna(0).astype(int)
    return df


def populate_funding_exits(df: pd.DataFrame) -> pd.DataFrame:
    """Add funding rate exit signals.

    Funding normalizes → exit.  Min/max hold enforced externally in
    confirm_trade_exit.
    """
    if df.empty:
        df["fr_exit_long"] = pd.Series(dtype=int)
        df["fr_exit_short"] = pd.Series(dtype=int)
        return df

    funding_normalized = (
        (df["funding_rate"] >= FUNDING_NORMAL_LOW) &
        (df["funding_rate"] <= FUNDING_NORMAL_HIGH)
    )

    df["fr_exit_long"] = funding_normalized.astype(int).fillna(0).astype(int)
    df["fr_exit_short"] = funding_normalized.astype(int).fillna(0).astype(int)
    return df
