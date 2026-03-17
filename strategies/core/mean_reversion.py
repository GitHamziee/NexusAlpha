"""
Mean Reversion Strategy — 9-AND confluence for high-selectivity entries.

ALL conditions must be simultaneously true for an entry signal.
Conditions 1-2 (regime=RANGING, MTF) and 9 (cooldown) are handled externally.
This module implements conditions 3-8:

  3. Close <= BB Lower x 1.001 (at or below lower band)
  4. RSI(14) < 32 (oversold)
  5. MACD histogram turning positive (hist > hist[1] AND hist[1] < 0)
  6. Volume > Volume SMA(20) x 1.1 (above-average volume on bounce)
  7. Bullish candle (close > open)
  8. Close > EMA(200) OR EMA(200) slope is flat
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
    MR_BB_TOUCH_LONG_MULT,
    MR_BB_TOUCH_SHORT_MULT,
    MR_EMA_SLOW as EMA_SLOW,
    MR_EMA200_FLAT_SLOPE,
    MR_MACD_FAST as MACD_FAST,
    MR_MACD_SIGNAL as MACD_SIGNAL,
    MR_MACD_SLOW as MACD_SLOW,
    MR_RSI_OVERBOUGHT,
    MR_RSI_OVERSOLD,
    MR_STOP_ATR_MULT as STOP_ATR_MULT,
    MR_TIME_STOP_CANDLES as TIME_STOP_CANDLES,
    MR_VOLUME_MULT,
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
    """Add mean reversion entry signals using 9-AND confluence.

    Adds columns: mr_enter_long, mr_enter_short, mr_signal_tag.
    ALL conditions (3-8) must be true simultaneously.
    Conditions 1-2 (regime=RANGING, MTF) and 9 (cooldown) are handled externally.
    """
    if df.empty:
        df["mr_enter_long"] = pd.Series(dtype=int)
        df["mr_enter_short"] = pd.Series(dtype=int)
        df["mr_signal_tag"] = pd.Series(dtype=str)
        return df

    # ── LONG CONDITIONS (all must be true) ─────────────────────────────

    # Condition 3: Close at or below lower BB
    cond_3_long = df["close"] <= df["mr_bb_lower"] * MR_BB_TOUCH_LONG_MULT

    # Condition 4: RSI oversold
    cond_4_long = df["mr_rsi"] < MR_RSI_OVERSOLD

    # Condition 5: MACD histogram turning positive (was negative, now rising)
    cond_5_long = (
        (df["mr_macd_hist"] > df["mr_macd_hist"].shift(1)) &
        (df["mr_macd_hist"].shift(1) < 0)
    )

    # Condition 6: Volume above average
    cond_6_long = df["volume"] > df["mr_volume_sma"] * MR_VOLUME_MULT

    # Condition 7: Bullish candle (momentum shifting)
    cond_7_long = df["close"] > df["open"]

    # Condition 8: Above EMA(200) OR EMA(200) slope is flat
    ema_flat = df["mr_ema_200_slope"].abs() < MR_EMA200_FLAT_SLOPE
    cond_8_long = (df["close"] > df["mr_ema_200"]) | ema_flat

    # 9-AND: ALL must be true
    long_cond = cond_3_long & cond_4_long & cond_5_long & cond_6_long & cond_7_long & cond_8_long

    # ── SHORT CONDITIONS (mirror) ──────────────────────────────────────

    # Condition 3: Close at or above upper BB
    cond_3_short = df["close"] >= df["mr_bb_upper"] * MR_BB_TOUCH_SHORT_MULT

    # Condition 4: RSI overbought
    cond_4_short = df["mr_rsi"] > MR_RSI_OVERBOUGHT

    # Condition 5: MACD histogram turning negative (was positive, now falling)
    cond_5_short = (
        (df["mr_macd_hist"] < df["mr_macd_hist"].shift(1)) &
        (df["mr_macd_hist"].shift(1) > 0)
    )

    # Condition 6: Volume above average
    cond_6_short = cond_6_long  # same for both sides

    # Condition 7: Bearish candle
    cond_7_short = df["close"] < df["open"]

    # Condition 8: Below EMA(200) OR EMA(200) slope is flat
    cond_8_short = (df["close"] < df["mr_ema_200"]) | ema_flat

    # 9-AND: ALL must be true
    short_cond = cond_3_short & cond_4_short & cond_5_short & cond_6_short & cond_7_short & cond_8_short

    # ── OUTPUT ─────────────────────────────────────────────────────────
    df["mr_enter_long"] = long_cond.astype(int).fillna(0).astype(int)
    df["mr_enter_short"] = short_cond.astype(int).fillna(0).astype(int)

    df["mr_signal_tag"] = ""
    df.loc[long_cond.fillna(False), "mr_signal_tag"] = "mean_reversion"
    df.loc[short_cond.fillna(False), "mr_signal_tag"] = "mean_reversion"

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
