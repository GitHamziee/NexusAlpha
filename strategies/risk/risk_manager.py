"""
Risk Manager — position sizing, drawdown circuit breakers, ATR stop scaling.

Every trade decision runs through here.  This module matters more than all
strategies combined: survive first, profit second.

Rules:
- Max 1% risk per trade (0.5% for funding)
- Max 5 open trades (3 pairs need headroom)
- Max 3x leverage
- 3% daily drawdown → stop trading today
- 15% total drawdown → full system review
- 4-candle cooldown after every loss
- Confidence-scaled sizing:
    >= 0.8 → 1.0%
    0.6-0.8 → 0.5%
    0.5-0.6 → 0.25% (quarter risk — transition/penalized regimes)
    < 0.5 → NO TRADE

Dynamic ATR stop scaling: when ATR > 1.5x its 100-period SMA, widen stops
proportionally and shrink position so dollar risk stays constant.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from core.thresholds import (  # noqa: F401  (Docker / freqtrade)
        CONFIRM_MIN_CONFIDENCE,
        COOLDOWN_CANDLES,
        FUNDING_RISK_PER_TRADE,
        MAX_BALANCE_FRACTION,
        MAX_DAILY_DRAWDOWN,
        MAX_LEVERAGE,
        MAX_OPEN_TRADES,
        MAX_RISK_PER_TRADE,
        MAX_TOTAL_DRAWDOWN,
        PAIR_LOCKOUT_CANDLES,
        PAIR_LOCKOUT_LOSSES,
        RISK_ATR_SPIKE_THRESHOLD as ATR_SPIKE_THRESHOLD,
    )
except ModuleNotFoundError:
    from strategies.core.thresholds import (  # noqa: F401  (pytest)
        CONFIRM_MIN_CONFIDENCE,
        COOLDOWN_CANDLES,
        FUNDING_RISK_PER_TRADE,
        MAX_BALANCE_FRACTION,
        MAX_DAILY_DRAWDOWN,
        MAX_LEVERAGE,
        MAX_OPEN_TRADES,
        MAX_RISK_PER_TRADE,
        MAX_TOTAL_DRAWDOWN,
        PAIR_LOCKOUT_CANDLES,
        PAIR_LOCKOUT_LOSSES,
        RISK_ATR_SPIKE_THRESHOLD as ATR_SPIKE_THRESHOLD,
    )

logger = logging.getLogger(__name__)


def get_risk_percent(regime_confidence: float, is_funding: bool = False) -> float:
    """Return per-trade risk as a decimal based on regime confidence.

    Funding rate trades are always half-size regardless of confidence.
    Confidence < CONFIRM_MIN_CONFIDENCE (0.50) → 0.0 (no trade allowed).
    """
    if regime_confidence < CONFIRM_MIN_CONFIDENCE:
        return 0.0

    if is_funding:
        return FUNDING_RISK_PER_TRADE

    if regime_confidence >= 0.8:
        return MAX_RISK_PER_TRADE       # 1.0%
    if regime_confidence >= 0.6:
        return MAX_RISK_PER_TRADE / 2   # 0.5%
    # 0.50 <= confidence < 0.6 — quarter risk (transition/penalized)
    return MAX_RISK_PER_TRADE / 4       # 0.25%


# Regime-signal alignment map: which regimes "match" which signal types
_REGIME_SIGNAL_MATCH = {
    "TRENDING_BULL": "tf",
    "TRENDING_BEAR": "tf",
    "RANGING": "mr",
    "TRANSITION": None,  # no natural match — half size
    "VOLATILE": None,    # quarter size
}


def get_regime_adjusted_risk(
    regime: str,
    regime_confidence: float,
    signal_type: str,
    is_funding: bool = False,
) -> float:
    """Return regime-aware risk percent.

    - Matching regime+signal (e.g. TRENDING + tf) → full confidence-tier risk
    - Mismatched (e.g. RANGING + tf) → half of confidence-tier risk
    - VOLATILE → quarter of confidence-tier risk
    - Below CONFIRM_MIN_CONFIDENCE → 0.0 (no trade)
    """
    base_risk = get_risk_percent(regime_confidence, is_funding)
    if base_risk == 0.0:
        return 0.0

    # Funding always gets its fixed rate
    if is_funding:
        return base_risk

    # VOLATILE → always quarter size
    if regime == "VOLATILE":
        return base_risk / 4

    # Check regime-signal alignment
    matched_signal = _REGIME_SIGNAL_MATCH.get(regime)
    if matched_signal == signal_type:
        return base_risk  # full size
    else:
        return base_risk / 2  # half size for mismatch or TRANSITION


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
