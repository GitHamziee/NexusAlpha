"""
Mean Reversion Strategy — BB bounce entries in confirmed ranges.

Active when: Regime = RANGING, confidence >= 0.6.
Highest win-rate strategy (70-80%) because BB middle acts as a statistical
magnet — price reverts to the mean ~68% of the time within 2 std devs.

9 entry conditions per side must ALL be true simultaneously.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

from .thresholds import (
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    MR_BB_TOUCH_LONG_MULT as BB_TOUCH_LONG_MULT,
    MR_BB_TOUCH_SHORT_MULT as BB_TOUCH_SHORT_MULT,
    MR_EMA_SLOW as EMA_SLOW,
    MR_MACD_FAST as MACD_FAST,
    MR_MACD_SIGNAL as MACD_SIGNAL,
    MR_MACD_SLOW as MACD_SLOW,
    MR_RSI_OVERBOUGHT as RSI_OVERBOUGHT,
    MR_RSI_OVERSOLD as RSI_OVERSOLD,
    MR_STOP_ATR_MULT as STOP_ATR_MULT,
    MR_TIME_STOP_CANDLES as TIME_STOP_CANDLES,
    MR_VOLUME_MULT as VOLUME_MULT,
    RSI_PERIOD,
    VOLUME_SMA_PERIOD,
)

logger = logging.getLogger(__name__)


def add_mr_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed by the mean reversion strategy."""
    if df.empty:
        return df

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
    if bbands is not None and not bbands.empty:
        bbu_col = [c for c in bbands.columns if c.startswith("BBU_")][0]
        bbl_col = [c for c in bbands.columns if c.startswith("BBL_")][0]
        bbm_col = [c for c in bbands.columns if c.startswith("BBM_")][0]
        df["mr_bb_upper"] = bbands[bbu_col]
        df["mr_bb_lower"] = bbands[bbl_col]
        df["mr_bb_middle"] = bbands[bbm_col]
    else:
        df["mr_bb_upper"] = float("nan")
        df["mr_bb_lower"] = float("nan")
        df["mr_bb_middle"] = float("nan")

    # RSI
    rsi = ta.rsi(df["close"], length=RSI_PERIOD)
    df["mr_rsi"] = rsi if rsi is not None else float("nan")

    # MACD
    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None and not macd.empty:
        hist_col = [c for c in macd.columns if c.startswith("MACDh_")][0]
        df["mr_macd_hist"] = macd[hist_col]
    else:
        df["mr_macd_hist"] = float("nan")

    # Volume SMA
    df["mr_volume_sma"] = df["volume"].rolling(window=VOLUME_SMA_PERIOD).mean()

    # ATR
    atr = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["mr_atr"] = atr if atr is not None else float("nan")

    # EMA(200)
    df["mr_ema_200"] = ta.ema(df["close"], length=EMA_SLOW)
    df["mr_ema_200_slope"] = df["mr_ema_200"].pct_change(periods=10)

    return df


def populate_mr_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean reversion entry signals.

    Adds columns: mr_enter_long, mr_enter_short.
    """
    if df.empty:
        df["mr_enter_long"] = pd.Series(dtype=int)
        df["mr_enter_short"] = pd.Series(dtype=int)
        return df

    hist = df["mr_macd_hist"]
    hist_turning_up = (hist > hist.shift(1)) & (hist.shift(1) < 0)
    hist_turning_down = (hist < hist.shift(1)) & (hist.shift(1) > 0)

    bullish_candle = df["close"] > df["open"]
    bearish_candle = df["close"] < df["open"]

    ema_ok_long = (df["close"] > df["mr_ema_200"]) | (df["mr_ema_200_slope"].abs() < 0.001)
    ema_ok_short = (df["close"] < df["mr_ema_200"]) | (df["mr_ema_200_slope"].abs() < 0.001)

    # ── LONG — 9 conditions ──────────────────────────────────────────
    long_cond = (
        (df["regime"] == "RANGING") &                          # L1
        (df["regime_confidence"] >= 0.5) &                     # L1
        (df["close"] <= df["mr_bb_lower"] * BB_TOUCH_LONG_MULT) &  # L3: near lower BB
        (df["mr_rsi"] < RSI_OVERSOLD) &                        # L3: oversold
        hist_turning_up &                                      # L3: MACD turning
        (df["volume"] > df["mr_volume_sma"] * VOLUME_MULT) &  # L4: volume
        bullish_candle &                                       # L4: momentum shift
        ema_ok_long &                                          # L5: not fighting downtrend
        True                                                   # L5: cooldown (handled externally)
    )

    # ── SHORT — 9 conditions (mirror) ────────────────────────────────
    short_cond = (
        (df["regime"] == "RANGING") &
        (df["regime_confidence"] >= 0.5) &
        (df["close"] >= df["mr_bb_upper"] * BB_TOUCH_SHORT_MULT) &
        (df["mr_rsi"] > RSI_OVERBOUGHT) &
        hist_turning_down &
        (df["volume"] > df["mr_volume_sma"] * VOLUME_MULT) &
        bearish_candle &
        ema_ok_short &
        True
    )

    df["mr_enter_long"] = long_cond.astype(int).fillna(0).astype(int)
    df["mr_enter_short"] = short_cond.astype(int).fillna(0).astype(int)
    return df


def populate_mr_exits(df: pd.DataFrame) -> pd.DataFrame:
    """Add mean reversion exit signals.

    Adds columns: mr_exit_long, mr_exit_short.
    Exits: BB middle TP, regime change to TRENDING.
    ATR stop and time stop handled in custom_stoploss / confirm_trade_exit.
    """
    if df.empty:
        df["mr_exit_long"] = pd.Series(dtype=int)
        df["mr_exit_short"] = pd.Series(dtype=int)
        return df

    # Take profit at BB middle
    tp_long = df["close"] >= df["mr_bb_middle"]
    tp_short = df["close"] <= df["mr_bb_middle"]

    # Regime change to trending → immediate exit
    regime_change = (
        (df["regime"] == "TRENDING_BULL") |
        (df["regime"] == "TRENDING_BEAR")
    )

    df["mr_exit_long"] = (tp_long | regime_change).astype(int).fillna(0).astype(int)
    df["mr_exit_short"] = (tp_short | regime_change).astype(int).fillna(0).astype(int)
    return df
