"""
Trend Following Strategy — 9-AND confluence for high-selectivity entries.

ALL conditions must be simultaneously true for an entry signal.
Conditions 1-2 (regime gate, MTF) and 9 (cooldown) are handled externally.
This module implements conditions 3-8:

  3. Supertrend bullish (price above Supertrend line)
  4. ADX > 28 AND rising (ADX > ADX[3])
  5. Price structure: close > EMA(200) AND close > EMA(50)
  6. StochRSI K crosses above D from below oversold zone
  7. Volume > Volume SMA(20) x 1.0
  8. RSI(14) < 75 (not overbought)
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
    TF_ADX_ENTRY_THRESH,
    TF_EMA_FAST as EMA_FAST,
    TF_EMA_MID as EMA_MID,
    TF_EMA_SLOW as EMA_SLOW,
    TF_RSI_OB_GUARD,
    TF_RSI_OS_GUARD,
    TF_STOCHRSI_LOOKBACK,
    TF_STOCHRSI_OVERBOUGHT,
    TF_STOCHRSI_OVERSOLD,
    TF_STOP_ATR_MULT as STOP_ATR_MULT,
    TF_TIME_STOP_CANDLES as TIME_STOP_CANDLES,
    TF_VOLUME_MULT,
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

    # Bollinger Bands (used by regime detector, computed here if missing)
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
    """Add trend following entry signals using 9-AND confluence.

    Adds columns: tf_enter_long, tf_enter_short, tf_signal_tag.
    ALL conditions (3-8) must be true simultaneously.
    Conditions 1-2 (regime, MTF) and 9 (cooldown) are handled externally.
    """
    if df.empty:
        df["tf_enter_long"] = pd.Series(dtype=int)
        df["tf_enter_short"] = pd.Series(dtype=int)
        df["tf_signal_tag"] = pd.Series(dtype=str)
        return df

    # ── LONG CONDITIONS (all must be true) ─────────────────────────────

    # Condition 3: Supertrend bullish
    cond_3_long = df["supertrend_direction"] == 1

    # Condition 4: ADX > 28 AND rising (higher than 3 candles ago)
    cond_4_long = (df["tf_adx"] > TF_ADX_ENTRY_THRESH) & (df["tf_adx"] > df["tf_adx"].shift(3))

    # Condition 5: Close > EMA(200) AND Close > EMA(50)
    cond_5_long = (df["close"] > df["ema_200"]) & (df["close"] > df["ema_50"])

    # Condition 6: StochRSI K crosses above D from oversold zone
    stochrsi_was_oversold = df["stochrsi_k"].rolling(
        window=TF_STOCHRSI_LOOKBACK, min_periods=1
    ).min() < TF_STOCHRSI_OVERSOLD
    cond_6_long = (df["stochrsi_k"] > df["stochrsi_d"]) & stochrsi_was_oversold

    # Condition 7: Volume above average
    cond_7_long = df["volume"] > df["volume_sma_20"] * TF_VOLUME_MULT

    # Condition 8: RSI not overbought
    cond_8_long = df["rsi_14"] < TF_RSI_OB_GUARD

    # 9-AND: ALL must be true
    long_cond = cond_3_long & cond_4_long & cond_5_long & cond_6_long & cond_7_long & cond_8_long

    # ── SHORT CONDITIONS (mirror) ──────────────────────────────────────

    # Condition 3: Supertrend bearish
    cond_3_short = df["supertrend_direction"] == -1

    # Condition 4: ADX > 28 AND rising
    cond_4_short = cond_4_long  # ADX conditions are the same for both sides

    # Condition 5: Close < EMA(200) AND Close < EMA(50)
    cond_5_short = (df["close"] < df["ema_200"]) & (df["close"] < df["ema_50"])

    # Condition 6: StochRSI K crosses below D from overbought zone
    stochrsi_was_overbought = df["stochrsi_k"].rolling(
        window=TF_STOCHRSI_LOOKBACK, min_periods=1
    ).max() > TF_STOCHRSI_OVERBOUGHT
    cond_6_short = (df["stochrsi_k"] < df["stochrsi_d"]) & stochrsi_was_overbought

    # Condition 7: Volume above average
    cond_7_short = cond_7_long  # same for both sides

    # Condition 8: RSI not oversold
    cond_8_short = df["rsi_14"] > TF_RSI_OS_GUARD

    # 9-AND: ALL must be true
    short_cond = cond_3_short & cond_4_short & cond_5_short & cond_6_short & cond_7_short & cond_8_short

    # ── OUTPUT ─────────────────────────────────────────────────────────
    df["tf_enter_long"] = long_cond.astype(int).fillna(0).astype(int)
    df["tf_enter_short"] = short_cond.astype(int).fillna(0).astype(int)

    df["tf_signal_tag"] = ""
    df.loc[long_cond.fillna(False), "tf_signal_tag"] = "trend_following"
    df.loc[short_cond.fillna(False), "tf_signal_tag"] = "trend_following"

    return df


def populate_trend_exits(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend following exit signals.

    Adds columns: tf_exit_long, tf_exit_short.
    Exit triggers: ADX death (< 18), Supertrend flip.
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
