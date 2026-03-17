"""
Mean Reversion Strategy — 3 independent paths for mean reversion entries.

Uses OR-of-simple-groups pattern: any ONE signal path can trigger an entry.
Each path has only 2-3 conditions (vs. the old 9-condition AND approach).

Path A — BB Bounce: close at/below lower BB + RSI oversold
Path B — MACD Reversal: close near lower BB + MACD turning + bullish candle
Path C — RSI Bounce: deep RSI oversold + bullish candle + volume

Regime is a soft gate (affects sizing in NexusAlpha.py, not signal blocking).
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
    MR_EMA_SLOW as EMA_SLOW,
    MR_MACD_FAST as MACD_FAST,
    MR_MACD_SIGNAL as MACD_SIGNAL,
    MR_MACD_SLOW as MACD_SLOW,
    MR_PATH_A_RSI as PATH_A_RSI,
    MR_PATH_B_BB_PROXIMITY as PATH_B_BB_PROXIMITY,
    MR_PATH_B_RSI_FILTER as PATH_B_RSI_FILTER,
    MR_PATH_C_RSI as PATH_C_RSI,
    MR_PATH_C_VOLUME_MULT as PATH_C_VOLUME_MULT,
    MR_STOP_ATR_MULT as STOP_ATR_MULT,
    MR_TIME_STOP_CANDLES as TIME_STOP_CANDLES,
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
    """Add mean reversion entry signals using 3 OR'd signal paths.

    Adds columns: mr_enter_long, mr_enter_short, mr_signal_tag.
    Any ONE path firing is sufficient for an entry signal.
    Regime is NOT checked here — it's a soft gate handled in NexusAlpha.py.
    """
    if df.empty:
        df["mr_enter_long"] = pd.Series(dtype=int)
        df["mr_enter_short"] = pd.Series(dtype=int)
        df["mr_signal_tag"] = pd.Series(dtype=str)
        return df

    hist = df["mr_macd_hist"]
    hist_turning_up = hist > hist.shift(1)
    hist_turning_down = hist < hist.shift(1)
    bullish_candle = df["close"] > df["open"]
    bearish_candle = df["close"] < df["open"]
    vol_ok = df["volume"] > df["mr_volume_sma"] * PATH_C_VOLUME_MULT

    # ── PATH A — BB Bounce (classic mean reversion) ─────────────────
    # Price at/below lower BB + RSI oversold
    path_a_long = (
        (df["close"] <= df["mr_bb_lower"]) &
        (df["mr_rsi"] < PATH_A_RSI)
    )
    path_a_short = (
        (df["close"] >= df["mr_bb_upper"]) &
        (df["mr_rsi"] > (100 - PATH_A_RSI))
    )

    # ── PATH B — MACD Reversal — DISABLED ────────────────────────────
    # Backtesting showed this path fires ~900 times in 273 days on BTC 15m
    # with negative expectancy. The MACD histogram oscillates too frequently
    # near BB on 15m timeframe, producing noise not signal.
    path_b_long = pd.Series(False, index=df.index)
    path_b_short = pd.Series(False, index=df.index)

    # ── PATH C — RSI Bounce — DISABLED ──────────────────────────────
    # Backtesting showed 26 trades in 2 months with 11.5% win rate on BTC 15m.
    # RSI < 30 without BB proximity catches falling knives in declining markets.
    # All profitable MR trades come from Path A (BB Bounce) which requires
    # price at the BB band, providing a structural support level.
    path_c_long = pd.Series(False, index=df.index)
    path_c_short = pd.Series(False, index=df.index)

    # ── COMBINE with OR ─────────────────────────────────────────────
    long_cond = path_a_long | path_b_long | path_c_long
    short_cond = path_a_short | path_b_short | path_c_short

    df["mr_enter_long"] = long_cond.astype(int).fillna(0).astype(int)
    df["mr_enter_short"] = short_cond.astype(int).fillna(0).astype(int)

    # Signal tag for enter_tag routing (priority: A > B > C)
    df["mr_signal_tag"] = ""
    df.loc[path_c_long.fillna(False), "mr_signal_tag"] = "mr_rsi_bounce"
    df.loc[path_b_long.fillna(False), "mr_signal_tag"] = "mr_macd_reversal"
    df.loc[path_a_long.fillna(False), "mr_signal_tag"] = "mr_bb_bounce"
    df.loc[path_c_short.fillna(False), "mr_signal_tag"] = "mr_rsi_bounce"
    df.loc[path_b_short.fillna(False), "mr_signal_tag"] = "mr_macd_reversal"
    df.loc[path_a_short.fillna(False), "mr_signal_tag"] = "mr_bb_bounce"

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
