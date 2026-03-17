"""
Centralized Thresholds — single source of truth for all tunable parameters.

Edit this file to fine-tune backtesting. All strategy modules import from here.
Organized by category for quick scanning.

Architecture: Tiered Confluence (3 hard gates + N-of-M scoring)
Per-pair parameters for BTC, ETH, SOL.
"""

from __future__ import annotations

from typing import Dict

# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR PERIODS (shared across modules)
# ═══════════════════════════════════════════════════════════════════════════

ADX_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0                          # default; overridden per pair
RSI_PERIOD = 14
EMA_SLOPE_PERIOD = 50
EMA_SLOPE_LOOKBACK = 10               # rate-of-change window for EMA slope
VOLUME_SMA_PERIOD = 20
VOLATILITY_SMA_PERIOD = 100           # SMA period for BB width and ATR ratio

# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Adaptive ADX baseline
ADX_BASELINE_PERIOD = 100              # SMA period for ADX baseline

# Offsets from ADX baseline to define regime zones
ADX_RANGE_OFFSET = -2                  # below baseline → ranging
ADX_TREND_OFFSET = 3                   # above baseline → trending

# Hard floors/ceilings so thresholds stay sensible
ADX_RANGE_FLOOR = 12
ADX_RANGE_CEIL = 22
ADX_TREND_FLOOR = 22
ADX_TREND_CEIL = 35

# Volatility spike multipliers (override to VOLATILE)
ATR_SPIKE_MULT = 2.0                   # ATR > 2x its SMA → volatile
BB_SPIKE_MULT = 2.5                    # BB width > 2.5x its SMA → volatile

# ═══════════════════════════════════════════════════════════════════════════
# TREND FOLLOWING — shared constants
# ═══════════════════════════════════════════════════════════════════════════

# Supertrend defaults (per-pair overrides in PAIR_CONFIGS)
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

# EMAs
TF_EMA_FAST = 9
TF_EMA_MID = 21                        # short-term trend (was 50 in 9-AND)
TF_EMA_SLOW = 200
EMA_50 = 50                            # medium-term trend (hard gate)

# StochRSI (kept for indicator computation, no longer hard-gated)
STOCHRSI_RSI_PERIOD = 14
STOCHRSI_STOCH_PERIOD = 3
STOCHRSI_SMOOTH = 3

# Pullback detection (shared across all pairs)
TF_RSI_PULLBACK_LONG = 45              # RSI below this = pullback for longs
TF_RSI_PULLBACK_SHORT = 55             # RSI above this = pullback for shorts

# Trend quality filters
TF_ADX_RISING_LOOKBACK = 3             # ADX must be rising over this many candles
TF_EMA50_MIN_SLOPE = 0.001             # EMA50 must change >=0.1% over 10 candles
TF_EMA50_SLOPE_LOOKBACK = 10           # candles to measure EMA50 slope

# Exit thresholds (shared across all pairs)
TF_ADX_DEATH_LEVEL = 12                # ADX death → exit only truly dead trends

# ═══════════════════════════════════════════════════════════════════════════
# MEAN REVERSION — shared constants
# ═══════════════════════════════════════════════════════════════════════════

# MACD
MR_MACD_FAST = 12
MR_MACD_SLOW = 26
MR_MACD_SIGNAL = 9

# EMA
MR_EMA_SLOW = 200

# EMA200 flat slope threshold (shared)
MR_EMA200_FLAT_SLOPE = 0.001

# ═══════════════════════════════════════════════════════════════════════════
# FUNDING RATE
# ═══════════════════════════════════════════════════════════════════════════

# Entry thresholds (EXTREME — by design)
FR_FUNDING_LONG_THRESH = -0.0005       # funding < -0.05% = short crowding
FR_FUNDING_SHORT_THRESH = 0.0008       # funding > 0.08% = long crowding
FR_LS_RATIO_LONG_THRESH = 0.7          # crowd very short
FR_LS_RATIO_SHORT_THRESH = 1.8         # crowd very long
FR_RSI_OVERSOLD = 35
FR_RSI_OVERBOUGHT = 65

# Exit thresholds
FR_STOP_ATR_MULT = 3.0                 # wider stops for funding trades
FR_FUNDING_NORMAL_LOW = -0.0001        # exit when funding normalizes
FR_FUNDING_NORMAL_HIGH = 0.0001
FR_MIN_HOLD_CANDLES = 32               # 8 hours = 32 × 15m
FR_MAX_HOLD_CANDLES = 192              # 48 hours

# ═══════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

MAX_RISK_PER_TRADE = 0.01              # 1% of account
FUNDING_RISK_PER_TRADE = 0.005         # 0.5% for funding rate trades
MAX_OPEN_TRADES = 5                    # 3 pairs need headroom (was 3)
MAX_LEVERAGE = 3.0
MAX_DAILY_DRAWDOWN = 0.03              # 3% daily limit
MAX_TOTAL_DRAWDOWN = 0.15              # 15% total limit
COOLDOWN_CANDLES = 4
PAIR_LOCKOUT_LOSSES = 3
PAIR_LOCKOUT_CANDLES = 16              # 4 hours on 15m
RISK_ATR_SPIKE_THRESHOLD = 1.5         # ATR > 1.5x SMA → widen stops
MAX_BALANCE_FRACTION = 0.33            # max 33% of balance per trade

# ═══════════════════════════════════════════════════════════════════════════
# REGIME HARD GATE
# ═══════════════════════════════════════════════════════════════════════════

REGIME_TRANSITION_CONFIDENCE = 0.25    # TRANSITION regime allows quarter-size
REGIME_MTF_PENALTY = 0.80              # 20% penalty when 1H disagrees
CONFIRM_MIN_CONFIDENCE = 0.50          # minimum confidence to allow any trade

# ═══════════════════════════════════════════════════════════════════════════
# PER-PAIR CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

PAIR_CONFIGS: Dict[str, dict] = {
    "BTC/USDT:USDT": {
        # Supertrend
        "supertrend_mult": 3.0,
        "supertrend_period": 10,
        # Trend following entry
        "tf_adx_thresh": 20,
        "tf_rsi_low": 30,
        "tf_rsi_high": 70,
        "tf_rsi_long_ceil": 60,             # longs: RSI must be below (not overbought)
        "tf_rsi_short_floor": 40,           # shorts: RSI must be above (not oversold)
        "tf_volume_mult": 1.5,
        # Pullback detection
        "tf_pullback_lookback": 3,          # candles to scan for pullback
        "tf_ema21_pullback_pct": 0.01,      # within 1% of EMA21 = pullback
        # Trend following exit
        "tf_stop_atr_mult": 3.0,        # was 2.0 — wider to survive noise
        "tf_tp1_atr_mult": 2.0,         # was 1.5 — give winners more room
        "tf_tp2_atr_mult": 4.0,         # was 3.0
        "tf_time_stop": 48,             # was 32 — 12 hours (patient with pullback entries)
        # Mean reversion entry
        "mr_rsi_oversold": 35,
        "mr_rsi_overbought": 65,
        "mr_bb_long_mult": 1.001,
        "mr_bb_short_mult": 0.999,
        "mr_volume_mult": 1.2,
        # Mean reversion exit
        "mr_stop_atr_mult": 2.5,        # was 1.5
        "mr_time_stop": 16,             # was 12
        # BB
        "bb_std": 2.0,
        # Stop floors/ceilings
        "min_stoploss_pct": -0.02,       # was -0.015 — wider to survive noise
        "max_stoploss_pct": -0.05,
    },
    "ETH/USDT:USDT": {
        "supertrend_mult": 3.5,
        "supertrend_period": 10,
        "tf_adx_thresh": 22,
        "tf_rsi_low": 30,
        "tf_rsi_high": 70,
        "tf_rsi_long_ceil": 60,
        "tf_rsi_short_floor": 40,
        "tf_volume_mult": 1.5,
        "tf_pullback_lookback": 3,
        "tf_ema21_pullback_pct": 0.012,     # slightly wider for ETH volatility
        "tf_stop_atr_mult": 3.0,        # was 2.0
        "tf_tp1_atr_mult": 2.0,         # was 1.5
        "tf_tp2_atr_mult": 3.5,         # was 2.5
        "tf_time_stop": 40,             # was 28 — 10 hours
        "mr_rsi_oversold": 30,
        "mr_rsi_overbought": 70,
        "mr_bb_long_mult": 1.002,
        "mr_bb_short_mult": 0.998,
        "mr_volume_mult": 1.2,
        "mr_stop_atr_mult": 2.5,        # was 1.5
        "mr_time_stop": 14,             # was 10
        "bb_std": 2.0,
        "min_stoploss_pct": -0.025,      # was -0.02 — wider for ETH
        "max_stoploss_pct": -0.06,
    },
    "SOL/USDT:USDT": {
        "supertrend_mult": 4.5,
        "supertrend_period": 12,
        "tf_adx_thresh": 25,
        "tf_rsi_low": 30,
        "tf_rsi_high": 70,
        "tf_rsi_long_ceil": 58,             # tighter for volatile SOL
        "tf_rsi_short_floor": 42,
        "tf_volume_mult": 1.8,
        "tf_pullback_lookback": 3,
        "tf_ema21_pullback_pct": 0.015,     # wider for SOL volatility
        "tf_stop_atr_mult": 3.5,        # was 2.5
        "tf_tp1_atr_mult": 2.5,         # was 2.0
        "tf_tp2_atr_mult": 5.0,         # was 3.5
        "tf_time_stop": 36,             # was 24 — 9 hours
        "mr_rsi_oversold": 25,
        "mr_rsi_overbought": 75,
        "mr_bb_long_mult": 1.005,
        "mr_bb_short_mult": 0.995,
        "mr_volume_mult": 1.5,
        "mr_stop_atr_mult": 3.0,        # was 2.0
        "mr_time_stop": 10,             # was 8
        "bb_std": 2.5,
        "min_stoploss_pct": -0.03,       # was -0.025 — wider for volatile SOL
        "max_stoploss_pct": -0.08,
    },
}

DEFAULT_PAIR_CONFIG = PAIR_CONFIGS["BTC/USDT:USDT"]


def get_pair_config(pair: str) -> dict:
    """Return per-pair parameter dictionary; falls back to BTC defaults."""
    return PAIR_CONFIGS.get(pair, DEFAULT_PAIR_CONFIG)
