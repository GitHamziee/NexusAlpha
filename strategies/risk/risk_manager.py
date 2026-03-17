"""
Risk Manager — position sizing, drawdown circuit breakers, ATR stop scaling.

Every trade decision runs through here.  This module matters more than all
strategies combined: survive first, profit second.

Non-negotiable rules (from spec Section VIII):
- Max 1% risk per trade (0.5% for funding)
- Max 3 open trades
- Max 3x leverage
- 3% daily drawdown → stop trading today
- 15% total drawdown → full system review
- 4-candle cooldown after every loss
- Confidence-scaled sizing: >=0.8 → 1%, 0.6-0.8 → 0.5%, <0.6 → NO TRADE

Dynamic ATR stop scaling: when ATR > 1.5x its 100-period SMA, widen stops
proportionally and shrink position so dollar risk stays constant.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────
MAX_RISK_PER_TRADE = 0.01       # 1 % of account
FUNDING_RISK_PER_TRADE = 0.005  # 0.5 % for funding rate trades
MAX_OPEN_TRADES = 3
MAX_LEVERAGE = 3.0
MAX_DAILY_DRAWDOWN = 0.03       # 3 %
MAX_TOTAL_DRAWDOWN = 0.15       # 15 %
COOLDOWN_CANDLES = 4
PAIR_LOCKOUT_LOSSES = 3
PAIR_LOCKOUT_CANDLES = 16
ATR_SPIKE_THRESHOLD = 1.5       # ATR > 1.5x SMA triggers dynamic scaling
MAX_BALANCE_FRACTION = 0.33     # never risk more than 33% balance in one trade


def get_risk_percent(regime_confidence: float, is_funding: bool = False) -> float:
    """Return per-trade risk as a decimal based on regime confidence.

    Funding rate trades are always half-size regardless of confidence.
    Confidence < 0.6 → 0.0 (no trade allowed).
    """
    if regime_confidence < 0.6:
        return 0.0

    if is_funding:
        return FUNDING_RISK_PER_TRADE

    if regime_confidence >= 0.8:
        return MAX_RISK_PER_TRADE  # 1.0%
    # 0.6 <= confidence < 0.8
    return MAX_RISK_PER_TRADE / 2  # 0.5%


def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    stop_distance: float,
) -> float:
    """Fixed-fractional position sizing: risk_amount / stop_distance.

    Caps at MAX_BALANCE_FRACTION of account to avoid concentration.
    Returns 0.0 if inputs are invalid.
    """
    if account_balance <= 0 or risk_pct <= 0 or stop_distance <= 0:
        return 0.0

    risk_amount = account_balance * risk_pct
    position_size = risk_amount / stop_distance
    max_allowed = account_balance * MAX_BALANCE_FRACTION
    return min(position_size, max_allowed)


def scale_atr_stop(
    base_atr_mult: float,
    current_atr: float,
    atr_sma: float,
) -> float:
    """Widen ATR stop multiplier when volatility spikes.

    When current ATR > 1.5x its 100-period SMA, the stop multiplier is
    scaled up proportionally so the stop distance tracks actual vol.
    The caller must then shrink position size to keep dollar risk constant.
    """
    if current_atr <= 0 or atr_sma <= 0:
        return base_atr_mult

    ratio = current_atr / atr_sma
    if ratio > ATR_SPIKE_THRESHOLD:
        return base_atr_mult * (ratio / ATR_SPIKE_THRESHOLD)
    return base_atr_mult


def check_daily_drawdown(
    current_balance: float,
    day_start_balance: float,
) -> bool:
    """Return True if daily drawdown limit is breached (should stop trading).

    Breach = current balance has dropped >= 3% from day start.
    """
    if day_start_balance <= 0:
        return True  # safety: if we don't know starting balance, halt
    drawdown = (day_start_balance - current_balance) / day_start_balance
    return drawdown >= MAX_DAILY_DRAWDOWN


def check_total_drawdown(
    current_balance: float,
    peak_balance: float,
) -> bool:
    """Return True if total drawdown >= 15% from peak (needs system review)."""
    if peak_balance <= 0:
        return True
    drawdown = (peak_balance - current_balance) / peak_balance
    return drawdown >= MAX_TOTAL_DRAWDOWN


def check_max_open_trades(open_trade_count: int) -> bool:
    """Return True if we've hit the max open trades limit."""
    return open_trade_count >= MAX_OPEN_TRADES


def can_trade(
    regime_confidence: float,
    open_trade_count: int,
    current_balance: float,
    day_start_balance: float,
    peak_balance: float,
    is_funding: bool = False,
) -> bool:
    """Master gate: returns True only if all risk checks pass."""
    if get_risk_percent(regime_confidence, is_funding) == 0.0:
        return False
    if check_max_open_trades(open_trade_count):
        return False
    if check_daily_drawdown(current_balance, day_start_balance):
        return False
    if check_total_drawdown(current_balance, peak_balance):
        return False
    return True


def get_leverage() -> float:
    """Always 3x. No exceptions."""
    return MAX_LEVERAGE
