"""
Centralized Thresholds — single source of truth for all tunable parameters.

Edit this file to fine-tune backtesting. All strategy modules import from here.
Organized by category for quick scanning.
"""

# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR PERIODS (shared across modules)
# ═══════════════════════════════════════════════════════════════════════════

ADX_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
EMA_SLOPE_PERIOD = 50
EMA_SLOPE_LOOKBACK = 10          # rate-of-change window for EMA slope
VOLUME_SMA_PERIOD = 20
VOLATILITY_SMA_PERIOD = 100      # SMA period for BB width and ATR ratio

# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Adaptive ADX baseline
ADX_BASELINE_PERIOD = 100         # SMA period for ADX baseline

# Offsets from ADX baseline to define regime zones
ADX_RANGE_OFFSET = -2             # below baseline → ranging
ADX_TREND_OFFSET = 3              # above baseline → trending

# Hard floors/ceilings so thresholds stay sensible
ADX_RANGE_FLOOR = 12
ADX_RANGE_CEIL = 22
ADX_TREND_FLOOR = 22
ADX_TREND_CEIL = 35

# Volatility spike multipliers (override to VOLATILE)
ATR_SPIKE_MULT = 2.0              # ATR > 2x its SMA → volatile
BB_SPIKE_MULT = 2.5               # BB width > 2.5x its SMA → volatile

# ═══════════════════════════════════════════════════════════════════════════
# TREND FOLLOWING
# ═══════════════════════════════════════════════════════════════════════════

# Supertrend
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

# EMAs
TF_EMA_FAST = 9
TF_EMA_MID = 50
TF_EMA_SLOW = 200

# StochRSI
STOCHRSI_RSI_PERIOD = 14
STOCHRSI_STOCH_PERIOD = 3
STOCHRSI_SMOOTH = 3

# Entry thresholds
TF_ADX_ENTRY_THRESH = 22          # minimum ADX for trend entry
TF_STOCHRSI_OVERSOLD = 45         # StochRSI oversold zone (wider for 15m BTC)
TF_STOCHRSI_OVERBOUGHT = 55       # StochRSI overbought zone
TF_STOCHRSI_LOOKBACK = 5          # candles to look back for recent oversold/overbought
TF_RSI_OB_GUARD = 78              # block long if RSI above this
TF_RSI_OS_GUARD = 22              # block short if RSI below this
TF_VOLUME_MULT = 1.0              # minimum volume ratio for entry

# Exit thresholds
TF_STOP_ATR_MULT = 2.0            # ATR multiplier for stop loss
TF_TP1_ATR_MULT = 1.5             # first take profit
TF_TP2_ATR_MULT = 3.0             # second take profit
TF_ADX_DEATH_LEVEL = 18           # ADX death → exit trend
TF_TIME_STOP_CANDLES = 20         # 5 hours on 15m

# ═══════════════════════════════════════════════════════════════════════════
# MEAN REVERSION
# ═══════════════════════════════════════════════════════════════════════════

# MACD
MR_MACD_FAST = 12
MR_MACD_SLOW = 26
MR_MACD_SIGNAL = 9

# EMA
MR_EMA_SLOW = 200

# Entry thresholds
MR_BB_TOUCH_LONG_MULT = 1.01      # close <= bb_lower * this (within 1% of lower BB)
MR_BB_TOUCH_SHORT_MULT = 0.99     # close >= bb_upper * this (within 1% of upper BB)
MR_RSI_OVERSOLD = 42              # RSI oversold for long entry
MR_RSI_OVERBOUGHT = 58            # RSI overbought for short entry
MR_VOLUME_MULT = 0.9              # volume multiplier for entry

# Exit thresholds
MR_STOP_ATR_MULT = 2.5            # ATR multiplier for stop loss
MR_TIME_STOP_CANDLES = 12         # 3 hours on 15m

# ═══════════════════════════════════════════════════════════════════════════
# FUNDING RATE
# ═══════════════════════════════════════════════════════════════════════════

# Entry thresholds (EXTREME — by design)
FR_FUNDING_LONG_THRESH = -0.0005   # funding < -0.05% = short crowding
FR_FUNDING_SHORT_THRESH = 0.0008   # funding > 0.08% = long crowding
FR_LS_RATIO_LONG_THRESH = 0.7     # crowd very short
FR_LS_RATIO_SHORT_THRESH = 1.8    # crowd very long
FR_RSI_OVERSOLD = 35
FR_RSI_OVERBOUGHT = 65

# Exit thresholds
FR_STOP_ATR_MULT = 3.0            # wider stops for funding trades
FR_FUNDING_NORMAL_LOW = -0.0001   # exit when funding normalizes
FR_FUNDING_NORMAL_HIGH = 0.0001
FR_MIN_HOLD_CANDLES = 32          # 8 hours = 32 × 15m
FR_MAX_HOLD_CANDLES = 192         # 48 hours

# ═══════════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

MAX_RISK_PER_TRADE = 0.01          # 1% of account
FUNDING_RISK_PER_TRADE = 0.005     # 0.5% for funding rate trades
MAX_OPEN_TRADES = 3
MAX_LEVERAGE = 3.0
MAX_DAILY_DRAWDOWN = 0.03          # 3% daily limit
MAX_TOTAL_DRAWDOWN = 0.15          # 15% total limit
COOLDOWN_CANDLES = 4
PAIR_LOCKOUT_LOSSES = 3
PAIR_LOCKOUT_CANDLES = 16          # 4 hours on 15m
RISK_ATR_SPIKE_THRESHOLD = 1.5    # ATR > 1.5x SMA → widen stops
MAX_BALANCE_FRACTION = 0.33        # max 33% of balance per trade

# ═══════════════════════════════════════════════════════════════════════════
# TREND FOLLOWING — OR SIGNAL PATHS
# ═══════════════════════════════════════════════════════════════════════════

# Path A: Supertrend Breakout
TF_PATH_A_VOLUME_MULT = 1.0          # volume > sma * this

# Path B: EMA Momentum
TF_PATH_B_ADX_THRESH = 25            # ADX > this for momentum confirmation
TF_PATH_B_RSI_LOW = 40               # RSI floor (avoid oversold fakeouts)
TF_PATH_B_RSI_HIGH = 70              # RSI ceiling (avoid overbought entries)

# Path C: BB Breakout
TF_PATH_C_ADX_LOOKBACK = 3           # ADX must be higher than N candles ago
TF_PATH_C_VOLUME_SPIKE = 1.5         # volume > sma * this for breakout

# ═══════════════════════════════════════════════════════════════════════════
# MEAN REVERSION — OR SIGNAL PATHS
# ═══════════════════════════════════════════════════════════════════════════

# Path A: BB Bounce
MR_PATH_A_RSI = 35                    # RSI < this for long, > (100-this) for short

# Path B: MACD Reversal
MR_PATH_B_BB_PROXIMITY = 0.02        # close within 2% of BB lower/upper

# Path C: RSI Bounce
MR_PATH_C_RSI = 30                    # deep RSI oversold for long
MR_PATH_C_VOLUME_MULT = 1.0          # volume > sma * this

# ═══════════════════════════════════════════════════════════════════════════
# REGIME SOFT GATE
# ═══════════════════════════════════════════════════════════════════════════

REGIME_TRANSITION_CONFIDENCE = 0.35   # was 0.0 — no longer blocks all trades
REGIME_MTF_PENALTY = 0.85            # was 0.75 — softer disagreement penalty
CONFIRM_MIN_CONFIDENCE = 0.25         # minimum confidence to allow any trade
