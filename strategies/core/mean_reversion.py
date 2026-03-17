"""
Mean Reversion Strategy — Simplified 3+1 tiered confluence.

Architecture: 3 hard gates (ALL must be true) + 1-of-3 confluence scoring.
Per-pair parameters loaded from thresholds.PAIR_CONFIGS.

Hard gates:
  3. Close <= BB Lower x mult (at or below lower band)
  4. RSI(14) < oversold threshold
  5. Bullish candle (close > open)

Confluence score (need >= 1 of 3):
  A. Volume > threshold x SMA(20)
  B. MACD histogram turning positive (was negative, now rising)
  C. Close > EMA(200) or EMA(200) slope is flat

Conditions 1-2 (regime=RANGING, MTF) and cooldown are handled externally.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

from .thresholds import (
    ATR_PERIOD,
    BB_PERIOD,
    MR_EMA_SLOW as EMA_SLOW,
    MR_EMA200_FLAT_SLOPE,
    MR_MACD_FAST as MACD_FAST,
    MR_MACD_SIGNAL as MACD_SIGNAL,
    MR_MACD_SLOW as MACD_SLOW,
    RSI_PERIOD,
    VOLUME_SMA_PERIOD,
    get_pair_config,
)

logger = logging.getLogger(__name__)


def add_mr_indicators(df: pd.DataFrame, pair: str = "BTC/USDT:USDT") -> pd.DataFrame:
    """Compute all indicators needed by the mean reversion strategy."""
    if df.empty:
        return df

    cfg = get_pair_config(pair)

    # Bollinger Bands (per-pair std deviation)
    bbands = ta.bbands(df["close"], length=BB_PERIOD, std=cfg["bb_std"])
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


def populate_mr_entries(df: pd.DataFrame, pair: str = "BTC/USDT:USDT") -> pd.DataFrame:
    """Add mean reversion entry signals using 3+1 tiered confluence.

    Adds columns: mr_enter_long, mr_enter_short, mr_signal_tag.
    3 hard gates (ALL must be true) + 1-of-3 confluence scoring.
    Regime gate and cooldown are handled externally.
    """
    if df.empty:
        df["mr_enter_long"] = pd.Series(dtype=int)
        df["mr_enter_short"] = pd.Series(dtype=int)
        df["mr_signal_tag"] = pd.Series(dtype=str)
        return df

    cfg = get_pair_config(pair)

    # ── LONG ─────────────────────────────────────────────────────────────

    # Hard gates (ALL must be true)
    gate_1 = df["close"] <= df["mr_bb_lower"] * cfg["mr_bb_long_mult"]   # at/below lower BB
    gate_2 = df["mr_rsi"] < cfg["mr_rsi_oversold"]                       # RSI oversold
    gate_3 = df["close"] > df["open"]                                     # bullish candle

    hard_gate_long = gate_1 & gate_2 & gate_3

    # Confluence scoring (need >= 1 of 3)
    score_a = (df["volume"] > df["mr_volume_sma"] * cfg["mr_volume_mult"]).astype(int)
    score_b = ((df["mr_macd_hist"] > df["mr_macd_hist"].shift(1)) &
               (df["mr_macd_hist"].shift(1) < 0)).astype(int)            # MACD turning positive
    ema_flat = df["mr_ema_200_slope"].abs() < MR_EMA200_FLAT_SLOPE
    score_c = ((df["close"] > df["mr_ema_200"]) | ema_flat).astype(int)  # above EMA200 or flat

    confluence_long = score_a + score_b + score_c
    has_confluence_long = confluence_long >= 1

    long_cond = hard_gate_long & has_confluence_long

    # ── SHORT (mirror) ───────────────────────────────────────────────────

    gate_1s = df["close"] >= df["mr_bb_upper"] * cfg["mr_bb_short_mult"]  # at/above upper BB
    gate_2s = df["mr_rsi"] > cfg["mr_rsi_overbought"]                     # RSI overbought
    gate_3s = df["close"] < df["open"]                                     # bearish candle

    hard_gate_short = gate_1s & gate_2s & gate_3s

    score_bs = ((df["mr_macd_hist"] < df["mr_macd_hist"].shift(1)) &
                (df["mr_macd_hist"].shift(1) > 0)).astype(int)            # MACD turning negative
    score_cs = ((df["close"] < df["mr_ema_200"]) | ema_flat).astype(int)

    confluence_short = score_a + score_bs + score_cs
    has_confluence_short = confluence_short >= 1

    short_cond = hard_gate_short & has_confluence_short

    # ── OUTPUT ───────────────────────────────────────────────────────────
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
