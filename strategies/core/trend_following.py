"""
Trend Following Strategy — Pullback entry with trend quality filters.

Architecture: 5 hard gates + 1-of-3 pullback + 2-of-4 confluence.
Per-pair parameters loaded from thresholds.PAIR_CONFIGS.

Hard gates (ALL must be true):
  1. Supertrend direction (bullish for longs, bearish for shorts)
  2. Close vs EMA(50) + EMA(50) slope confirms direction (no choppy markets)
  3. Candle confirmation (bullish candle for longs, bearish for shorts)
  4. RSI band (30-60 for longs, 40-70 for shorts — not overbought/oversold)

Pullback detection (need >= 1 of 3):
  A. EMA(21) test: low touched EMA(21) within last 3 candles + bounced
  B. RSI dip: RSI dropped below 45 within last 3 candles + recovering
  C. BB middle test: low touched BB middle within last 3 candles + bounced

Confluence (need >= 2 of 4):
  A. ADX rising (trend strengthening, catches early trends)
  B. Volume > threshold x SMA(20) (participation)
  C. Close vs EMA(200) (major trend aligned)
  D. EMA(50) slope strength (trend is established)

Conditions 1-2 (regime gate, MTF) and cooldown are handled externally.
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
    EMA_50,
    RSI_PERIOD,
    STOCHRSI_RSI_PERIOD,
    STOCHRSI_SMOOTH,
    STOCHRSI_STOCH_PERIOD,
    TF_ADX_DEATH_LEVEL as ADX_DEATH_LEVEL,
    TF_ADX_RISING_LOOKBACK,
    TF_EMA_FAST as EMA_FAST,
    TF_EMA_MID as EMA_MID,
    TF_EMA_SLOW as EMA_SLOW,
    TF_EMA50_MIN_SLOPE,
    TF_EMA50_SLOPE_LOOKBACK,
    TF_RSI_PULLBACK_LONG,
    TF_RSI_PULLBACK_SHORT,
    VOLUME_SMA_PERIOD,
    get_pair_config,
)

logger = logging.getLogger(__name__)


def add_trend_indicators(df: pd.DataFrame, pair: str = "BTC/USDT:USDT") -> pd.DataFrame:
    """Compute all indicators needed by the trend following strategy."""
    if df.empty:
        return df

    cfg = get_pair_config(pair)

    # Supertrend (per-pair multiplier and period)
    st = ta.supertrend(df["high"], df["low"], df["close"],
                       length=cfg["supertrend_period"],
                       multiplier=cfg["supertrend_mult"])
    if st is not None and not st.empty:
        st_dir_col = [c for c in st.columns if c.startswith("SUPERTd_")][0]
        st_val_col = [c for c in st.columns if c.startswith("SUPERT_") and "d_" not in c][0]
        df["supertrend_direction"] = st[st_dir_col]  # 1=bull, -1=bear
        df["supertrend_value"] = st[st_val_col]
    else:
        df["supertrend_direction"] = float("nan")
        df["supertrend_value"] = float("nan")

    # ADX
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)
    if adx_df is not None:
        df["tf_adx"] = adx_df[f"ADX_{ADX_PERIOD}"]
    else:
        df["tf_adx"] = float("nan")

    # EMAs: 9, 21, 50, 200
    df["ema_9"] = ta.ema(df["close"], length=EMA_FAST)
    df["ema_21"] = ta.ema(df["close"], length=EMA_MID)
    df["ema_50"] = ta.ema(df["close"], length=EMA_50)
    df["ema_200"] = ta.ema(df["close"], length=EMA_SLOW)

    # EMA50 slope — measures if EMA50 is actually trending or flat
    df["ema_50_slope"] = (
        (df["ema_50"] - df["ema_50"].shift(TF_EMA50_SLOPE_LOOKBACK))
        / df["ema_50"].shift(TF_EMA50_SLOPE_LOOKBACK)
    )

    # StochRSI (kept for logging/ML, not hard-gated)
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


def populate_trend_entries(df: pd.DataFrame, pair: str = "BTC/USDT:USDT") -> pd.DataFrame:
    """Add trend following entry signals using pullback + trend quality filters.

    Adds columns: tf_enter_long, tf_enter_short, tf_signal_tag.
    5 hard gates + 1-of-3 pullback detection + 2-of-4 confluence.
    Regime gate and cooldown are handled externally.
    """
    if df.empty:
        df["tf_enter_long"] = pd.Series(dtype=int)
        df["tf_enter_short"] = pd.Series(dtype=int)
        df["tf_signal_tag"] = pd.Series(dtype=str)
        return df

    cfg = get_pair_config(pair)
    lookback = cfg.get("tf_pullback_lookback", 3)
    ema21_pct = cfg.get("tf_ema21_pullback_pct", 0.01)

    # EMA50 slope (computed in add_trend_indicators, fallback if missing)
    ema50_slope = df.get("ema_50_slope", pd.Series(0.0, index=df.index))

    # ── LONG ─────────────────────────────────────────────────────────────

    # Hard gates (ALL must be true)
    gate_1 = df["supertrend_direction"] == 1                           # Supertrend bullish
    gate_2 = (df["close"] > df["ema_50"]) & \
             (ema50_slope > TF_EMA50_MIN_SLOPE)                        # above EMA50 + trending up
    gate_3 = df["close"] > df["open"]                                  # bullish candle (momentum)
    gate_4 = (df["rsi_14"] > cfg["tf_rsi_low"]) & \
             (df["rsi_14"] < cfg.get("tf_rsi_long_ceil", 60))         # RSI 30-60 (room to run)

    hard_gate_long = gate_1 & gate_2 & gate_3 & gate_4

    # Pullback detection (need >= 1 of 3)
    # A: EMA(21) test — low touched EMA21 within lookback + bounced above
    low_to_ema21 = df["low"] / df["ema_21"]
    pullback_a = (low_to_ema21.rolling(window=lookback, min_periods=1).min()
                  <= 1.0 + ema21_pct) & (df["close"] > df["ema_21"])

    # B: RSI dip — RSI dropped below threshold within lookback + recovering
    pullback_b = (df["rsi_14"].rolling(window=lookback, min_periods=1).min()
                  < TF_RSI_PULLBACK_LONG) & (df["rsi_14"] > 40)

    # C: BB middle test — low touched BB middle within lookback + bounced
    bb_mid = df.get("bb_middle", pd.Series(dtype=float))
    if bb_mid.notna().any():
        low_to_bbm = df["low"] / bb_mid
        pullback_c = (low_to_bbm.rolling(window=lookback, min_periods=1).min()
                      <= 1.0) & (df["close"] > bb_mid)
    else:
        pullback_c = pd.Series(False, index=df.index)

    has_pullback_long = pullback_a | pullback_b | pullback_c

    # Confluence scoring (need >= 2 of 4)
    # A: ADX rising — trend is strengthening (catches early trends)
    score_a = (df["tf_adx"] > df["tf_adx"].shift(TF_ADX_RISING_LOOKBACK)).astype(int)
    score_b = (df["volume"] > df["volume_sma_20"] * cfg["tf_volume_mult"]).astype(int)
    score_c = (df["close"] > df["ema_200"]).astype(int)                # major trend aligned
    # D: EMA50 slope strength — trend is well established
    score_d = (ema50_slope > TF_EMA50_MIN_SLOPE * 2).astype(int)       # slope > 0.2%

    confluence_long = score_a + score_b + score_c + score_d
    has_confluence_long = confluence_long >= 2

    long_cond = hard_gate_long & has_pullback_long & has_confluence_long

    # ── SHORT (mirror) ───────────────────────────────────────────────────

    gate_1s = df["supertrend_direction"] == -1                          # Supertrend bearish
    gate_2s = (df["close"] < df["ema_50"]) & \
              (ema50_slope < -TF_EMA50_MIN_SLOPE)                       # below EMA50 + trending dn
    gate_3s = df["close"] < df["open"]                                  # bearish candle
    gate_4s = (df["rsi_14"] > cfg.get("tf_rsi_short_floor", 40)) & \
              (df["rsi_14"] < cfg["tf_rsi_high"])                       # RSI 40-70

    hard_gate_short = gate_1s & gate_2s & gate_3s & gate_4s

    # Pullback: price bounced UP to resistance then resumed down
    high_to_ema21 = df["high"] / df["ema_21"]
    pullback_as = (high_to_ema21.rolling(window=lookback, min_periods=1).max()
                   >= 1.0 - ema21_pct) & (df["close"] < df["ema_21"])

    pullback_bs = (df["rsi_14"].rolling(window=lookback, min_periods=1).max()
                   > TF_RSI_PULLBACK_SHORT) & (df["rsi_14"] < 60)

    if bb_mid.notna().any():
        high_to_bbm = df["high"] / bb_mid
        pullback_cs = (high_to_bbm.rolling(window=lookback, min_periods=1).max()
                       >= 1.0) & (df["close"] < bb_mid)
    else:
        pullback_cs = pd.Series(False, index=df.index)

    has_pullback_short = pullback_as | pullback_bs | pullback_cs

    score_cs = (df["close"] < df["ema_200"]).astype(int)                # below major trend
    score_ds = (ema50_slope < -TF_EMA50_MIN_SLOPE * 2).astype(int)     # downtrend established
    confluence_short = score_a + score_b + score_cs + score_ds
    has_confluence_short = confluence_short >= 2

    short_cond = hard_gate_short & has_pullback_short & has_confluence_short

    # ── OUTPUT ───────────────────────────────────────────────────────────
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
