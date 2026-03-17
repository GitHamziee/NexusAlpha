"""
Trend Following Strategy — catches established trends via 3 independent paths.

Uses OR-of-simple-groups pattern: any ONE signal path can trigger an entry.
Each path has only 2-3 conditions (vs. the old 9-condition AND approach).

Path A — Supertrend Breakout: ST flips bullish + close > EMA9 + volume
Path B — EMA Momentum: EMA9 > EMA50 + ADX strong + RSI in range
Path C — BB Breakout: close > BB upper + ADX rising + volume spike

Regime is a soft gate (affects sizing in NexusAlpha.py, not signal blocking).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

from .thresholds import (
    ADX_PERIOD,
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    RSI_PERIOD,
    STOCHRSI_RSI_PERIOD,
    STOCHRSI_SMOOTH,
    STOCHRSI_STOCH_PERIOD,
    SUPERTREND_MULT,
    SUPERTREND_PERIOD,
    TF_ADX_DEATH_LEVEL as ADX_DEATH_LEVEL,
    TF_EMA_FAST as EMA_FAST,
    TF_EMA_MID as EMA_MID,
    TF_EMA_SLOW as EMA_SLOW,
    TF_PATH_A_VOLUME_MULT as PATH_A_VOLUME_MULT,
    TF_PATH_B_ADX_THRESH as PATH_B_ADX_THRESH,
    TF_PATH_B_RSI_HIGH as PATH_B_RSI_HIGH,
    TF_PATH_B_RSI_LOW as PATH_B_RSI_LOW,
    TF_PATH_C_ADX_LOOKBACK as PATH_C_ADX_LOOKBACK,
    TF_PATH_C_VOLUME_SPIKE as PATH_C_VOLUME_SPIKE,
    TF_STOP_ATR_MULT as STOP_ATR_MULT,
    TF_TIME_STOP_CANDLES as TIME_STOP_CANDLES,
    TF_TP1_ATR_MULT as TP1_ATR_MULT,
    TF_TP2_ATR_MULT as TP2_ATR_MULT,
    VOLUME_SMA_PERIOD,
)

logger = logging.getLogger(__name__)


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed by the trend following strategy."""
    if df.empty:
        return df

    # Supertrend
    st = ta.supertrend(df["high"], df["low"], df["close"],
                       length=SUPERTREND_PERIOD, multiplier=SUPERTREND_MULT)
    if st is not None and not st.empty:
        st_dir_col = [c for c in st.columns if c.startswith("SUPERTd_")][0]
        st_val_col = [c for c in st.columns if c.startswith("SUPERT_") and "d_" not in c][0]
        df["supertrend_direction"] = st[st_dir_col]  # 1=bull, -1=bear
        df["supertrend_value"] = st[st_val_col]
    else:
        df["supertrend_direction"] = float("nan")
        df["supertrend_value"] = float("nan")

    # ADX (may already exist from regime detector, recompute here for safety)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)
    if adx_df is not None:
        df["tf_adx"] = adx_df[f"ADX_{ADX_PERIOD}"]
    else:
        df["tf_adx"] = float("nan")

    # EMAs
    df["ema_9"] = ta.ema(df["close"], length=EMA_FAST)
    df["ema_50"] = ta.ema(df["close"], length=EMA_MID)
    df["ema_200"] = ta.ema(df["close"], length=EMA_SLOW)

    # StochRSI
    stochrsi = ta.stochrsi(df["close"], length=STOCHRSI_RSI_PERIOD,
                           rsi_length=STOCHRSI_RSI_PERIOD,
                           k=STOCHRSI_STOCH_PERIOD, d=STOCHRSI_SMOOTH)
    if stochrsi is not None and not stochrsi.empty:
        k_col = [c for c in stochrsi.columns if c.startswith("STOCHRSIk_")][0]
        d_col = [c for c in stochrsi.columns if c.startswith("STOCHRSId_")][0]
        df["stochrsi_k"] = stochrsi[k_col]
        df["stochrsi_d"] = stochrsi[d_col]
    else:
        df["stochrsi_k"] = float("nan")
        df["stochrsi_d"] = float("nan")

    # ATR
    atr = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["atr_14"] = atr if atr is not None else float("nan")

    # Volume SMA
    df["volume_sma_20"] = df["volume"].rolling(window=VOLUME_SMA_PERIOD).mean()

    # RSI
    rsi = ta.rsi(df["close"], length=RSI_PERIOD)
    df["rsi_14"] = rsi if rsi is not None else float("nan")

    # Bollinger Bands (for Path C — BB Breakout)
    # Reuse regime detector's BB columns if present, otherwise compute
    if "bb_upper" not in df.columns:
        bbands = ta.bbands(df["close"], length=BB_PERIOD, std=BB_STD)
        if bbands is not None and not bbands.empty:
            bbu_col = [c for c in bbands.columns if c.startswith("BBU_")][0]
            bbl_col = [c for c in bbands.columns if c.startswith("BBL_")][0]
            bbm_col = [c for c in bbands.columns if c.startswith("BBM_")][0]
            df["bb_upper"] = bbands[bbu_col]
            df["bb_lower"] = bbands[bbl_col]
            df["bb_middle"] = bbands[bbm_col]
        else:
            df["bb_upper"] = float("nan")
            df["bb_lower"] = float("nan")
            df["bb_middle"] = float("nan")

    return df


def populate_trend_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend following entry signals using 3 OR'd signal paths.

    Adds columns: tf_enter_long, tf_enter_short, tf_signal_tag.
    Any ONE path firing is sufficient for an entry signal.
    Regime is NOT checked here — it's a soft gate handled in NexusAlpha.py.
    """
    if df.empty:
        df["tf_enter_long"] = pd.Series(dtype=int)
        df["tf_enter_short"] = pd.Series(dtype=int)
        df["tf_signal_tag"] = pd.Series(dtype=str)
        return df

    # Pre-compute shared conditions
    adx = df["tf_adx"]
    adx_rising = adx > adx.shift(PATH_C_ADX_LOOKBACK)
    vol_ok = df["volume"] > df["volume_sma_20"] * PATH_A_VOLUME_MULT
    vol_spike = df["volume"] > df["volume_sma_20"] * PATH_C_VOLUME_SPIKE

    # ── PATH A — Supertrend Breakout (trend initiation) ─────────────
    # Supertrend just flipped bullish + above EMA9 (short-term momentum) + volume
    st_flip_up = (df["supertrend_direction"] == 1) & (df["supertrend_direction"].shift(1) == -1)
    st_flip_down = (df["supertrend_direction"] == -1) & (df["supertrend_direction"].shift(1) == 1)

    path_a_long = st_flip_up & (df["close"] > df["ema_9"]) & vol_ok
    path_a_short = st_flip_down & (df["close"] < df["ema_9"]) & vol_ok

    # ── PATH B — EMA Momentum (trend continuation) ──────────────────
    # EMA9 CROSSES above EMA50 + ADX strong + RSI in healthy range
    # Crossover requirement prevents firing every candle in a trend
    path_b_long = (
        (df["ema_9"] > df["ema_50"]) &
        (df["ema_9"].shift(1) <= df["ema_50"].shift(1)) &  # crossover event
        (adx > PATH_B_ADX_THRESH) &
        (df["rsi_14"] > PATH_B_RSI_LOW) &
        (df["rsi_14"] < PATH_B_RSI_HIGH)
    )
    path_b_short = (
        (df["ema_9"] < df["ema_50"]) &
        (df["ema_9"].shift(1) >= df["ema_50"].shift(1)) &  # crossover event
        (adx > PATH_B_ADX_THRESH) &
        (df["rsi_14"] > (100 - PATH_B_RSI_HIGH)) &
        (df["rsi_14"] < (100 - PATH_B_RSI_LOW))
    )

    # ── PATH C — BB Breakout (volatility expansion) ─────────────────
    # Price breaks above/below BB + ADX rising + volume spike
    path_c_long = (
        (df["close"] > df["bb_upper"]) &
        adx_rising &
        vol_ok
    )
    path_c_short = (
        (df["close"] < df["bb_lower"]) &
        adx_rising &
        vol_ok
    )

    # ── COMBINE with OR ─────────────────────────────────────────────
    long_cond = path_a_long | path_b_long | path_c_long
    short_cond = path_a_short | path_b_short | path_c_short

    df["tf_enter_long"] = long_cond.astype(int).fillna(0).astype(int)
    df["tf_enter_short"] = short_cond.astype(int).fillna(0).astype(int)

    # Signal tag for enter_tag routing (priority: A > C > B)
    df["tf_signal_tag"] = ""
    df.loc[path_b_long.fillna(False), "tf_signal_tag"] = "tf_ema_momentum"
    df.loc[path_c_long.fillna(False), "tf_signal_tag"] = "tf_bb_breakout"
    df.loc[path_a_long.fillna(False), "tf_signal_tag"] = "tf_supertrend"
    df.loc[path_b_short.fillna(False), "tf_signal_tag"] = "tf_ema_momentum"
    df.loc[path_c_short.fillna(False), "tf_signal_tag"] = "tf_bb_breakout"
    df.loc[path_a_short.fillna(False), "tf_signal_tag"] = "tf_supertrend"

    return df


def populate_trend_exits(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend following exit signals.

    Adds columns: tf_exit_long, tf_exit_short.
    Exit triggers: ADX death (< 18), Supertrend flip, or BB middle reversion.
    ATR-based stop/TP are handled in custom_stoploss and confirm_trade_exit.
    """
    if df.empty:
        df["tf_exit_long"] = pd.Series(dtype=int)
        df["tf_exit_short"] = pd.Series(dtype=int)
        return df

    # ADX death exit: trend is dying
    adx_death = df["tf_adx"] < ADX_DEATH_LEVEL

    # Supertrend flip exits
    st_flip_bear = (df["supertrend_direction"] == -1) & (df["supertrend_direction"].shift(1) == 1)
    st_flip_bull = (df["supertrend_direction"] == 1) & (df["supertrend_direction"].shift(1) == -1)

    df["tf_exit_long"] = (adx_death | st_flip_bear).astype(int).fillna(0).astype(int)
    df["tf_exit_short"] = (adx_death | st_flip_bull).astype(int).fillna(0).astype(int)
    return df
