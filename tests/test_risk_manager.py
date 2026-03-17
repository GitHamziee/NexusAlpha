"""Tests for the risk manager — every non-negotiable rule verified."""

import pytest

from strategies.risk.risk_manager import (
    MAX_BALANCE_FRACTION,
    MAX_RISK_PER_TRADE,
    FUNDING_RISK_PER_TRADE,
    calculate_position_size,
    can_trade,
    check_daily_drawdown,
    check_max_open_trades,
    check_total_drawdown,
    get_leverage,
    get_regime_adjusted_risk,
    get_risk_percent,
    scale_atr_stop,
)


# ── get_risk_percent ─────────────────────────────────────────────────────

class TestGetRiskPercent:
    def test_high_confidence(self):
        assert get_risk_percent(0.85) == MAX_RISK_PER_TRADE  # 1%

    def test_medium_confidence(self):
        assert get_risk_percent(0.7) == MAX_RISK_PER_TRADE / 2  # 0.5%

    def test_low_confidence_blocks(self):
        """Confidence < 0.6 → NO TRADE (hard gate)."""
        assert get_risk_percent(0.5) == 0.0
        assert get_risk_percent(0.3) == 0.0

    def test_edge_0_6(self):
        """Exactly 0.6 should allow trading at half risk."""
        assert get_risk_percent(0.6) == MAX_RISK_PER_TRADE / 2

    def test_edge_0_8(self):
        """Exactly 0.8 should give full risk."""
        assert get_risk_percent(0.8) == MAX_RISK_PER_TRADE

    def test_funding_always_half(self):
        """Funding trades always use 0.5% regardless of confidence."""
        assert get_risk_percent(0.9, is_funding=True) == FUNDING_RISK_PER_TRADE
        assert get_risk_percent(0.7, is_funding=True) == FUNDING_RISK_PER_TRADE

    def test_funding_blocked_below_06(self):
        """Funding trades also blocked below 0.6 confidence."""
        assert get_risk_percent(0.3, is_funding=True) == 0.0

    def test_funding_allowed_at_06(self):
        """Funding trades allowed at 0.6 confidence."""
        assert get_risk_percent(0.65, is_funding=True) == FUNDING_RISK_PER_TRADE

    def test_below_min_confidence_blocks(self):
        """Confidence below 0.6 blocks all trades (hard gate)."""
        assert get_risk_percent(0.5) == 0.0
        assert get_risk_percent(0.2) == 0.0
        assert get_risk_percent(0.1) == 0.0


# ── calculate_position_size ──────────────────────────────────────────────

class TestCalculatePositionSize:
    def test_basic(self):
        # $10000 balance, 1% risk = $100, stop = $500 → size = $100/$500 = 0.2
        size = calculate_position_size(10000, 0.01, 500)
        assert size == pytest.approx(0.2)

    def test_caps_at_max_fraction(self):
        # Tiny stop → huge position, but capped at 33% of balance
        size = calculate_position_size(10000, 0.01, 0.001)
        assert size == pytest.approx(10000 * MAX_BALANCE_FRACTION)

    def test_zero_balance(self):
        assert calculate_position_size(0, 0.01, 100) == 0.0

    def test_zero_risk(self):
        assert calculate_position_size(10000, 0, 100) == 0.0

    def test_zero_stop(self):
        assert calculate_position_size(10000, 0.01, 0) == 0.0

    def test_negative_inputs(self):
        assert calculate_position_size(-1000, 0.01, 100) == 0.0


# ── scale_atr_stop ───────────────────────────────────────────────────────

class TestScaleAtrStop:
    def test_normal_vol_no_change(self):
        """When ATR is near its SMA, multiplier unchanged."""
        result = scale_atr_stop(2.0, 100, 100)
        assert result == 2.0

    def test_spike_widens_stop(self):
        """ATR 2x SMA → stop should widen proportionally."""
        # ratio = 2.0, threshold = 1.5 → scale = 2.0/1.5 = 1.333
        result = scale_atr_stop(2.0, 200, 100)
        assert result == pytest.approx(2.0 * (2.0 / 1.5))

    def test_below_threshold_no_change(self):
        result = scale_atr_stop(2.0, 140, 100)  # ratio = 1.4 < 1.5
        assert result == 2.0

    def test_exactly_at_threshold(self):
        result = scale_atr_stop(2.0, 150, 100)  # ratio = 1.5, not > 1.5
        assert result == 2.0

    def test_zero_atr_sma(self):
        result = scale_atr_stop(2.0, 100, 0)
        assert result == 2.0


# ── drawdown checks ─────────────────────────────────────────────────────

class TestDrawdownChecks:
    def test_daily_drawdown_not_breached(self):
        assert check_daily_drawdown(9800, 10000) is False  # 2% < 3%

    def test_daily_drawdown_breached(self):
        assert check_daily_drawdown(9700, 10000) is True  # 3% = 3%

    def test_daily_drawdown_zero_start(self):
        assert check_daily_drawdown(9700, 0) is True  # safety halt

    def test_total_drawdown_not_breached(self):
        assert check_total_drawdown(9000, 10000) is False  # 10% < 15%

    def test_total_drawdown_breached(self):
        assert check_total_drawdown(8500, 10000) is True  # 15%

    def test_total_drawdown_zero_peak(self):
        assert check_total_drawdown(8500, 0) is True


# ── max open trades ──────────────────────────────────────────────────────

class TestMaxOpenTrades:
    def test_under_limit(self):
        assert check_max_open_trades(2) is False

    def test_at_limit(self):
        assert check_max_open_trades(3) is True

    def test_over_limit(self):
        assert check_max_open_trades(5) is True


# ── can_trade master gate ────────────────────────────────────────────────

class TestCanTrade:
    def _defaults(self, **overrides):
        params = dict(
            regime_confidence=0.8,
            open_trade_count=1,
            current_balance=9900,
            day_start_balance=10000,
            peak_balance=10000,
            is_funding=False,
        )
        params.update(overrides)
        return params

    def test_all_ok(self):
        assert can_trade(**self._defaults()) is True

    def test_low_confidence_blocks(self):
        """Confidence below 0.6 blocks trades (hard gate)."""
        assert can_trade(**self._defaults(regime_confidence=0.5)) is False
        assert can_trade(**self._defaults(regime_confidence=0.3)) is False

    def test_confidence_at_06_allows(self):
        """Confidence at 0.6 allows trading."""
        assert can_trade(**self._defaults(regime_confidence=0.6)) is True

    def test_max_trades_blocks(self):
        assert can_trade(**self._defaults(open_trade_count=3)) is False

    def test_daily_drawdown_blocks(self):
        assert can_trade(**self._defaults(current_balance=9600)) is False

    def test_total_drawdown_blocks(self):
        assert can_trade(**self._defaults(current_balance=8400, peak_balance=10000)) is False

    def test_funding_trade_allowed(self):
        assert can_trade(**self._defaults(is_funding=True, regime_confidence=0.7)) is True


# ── regime-adjusted risk ─────────────────────────────────────────────────

class TestRegimeAdjustedRisk:
    def test_matching_regime_signal_full_risk(self):
        """TRENDING + tf signal → full risk."""
        risk = get_regime_adjusted_risk("TRENDING_BULL", 0.8, "tf")
        assert risk == MAX_RISK_PER_TRADE

    def test_mismatched_regime_signal_half_risk(self):
        """RANGING + tf signal → half risk."""
        risk = get_regime_adjusted_risk("RANGING", 0.8, "tf")
        assert risk == MAX_RISK_PER_TRADE / 2

    def test_volatile_quarter_risk(self):
        """VOLATILE → quarter risk regardless of signal type."""
        risk = get_regime_adjusted_risk("VOLATILE", 0.8, "tf")
        assert risk == MAX_RISK_PER_TRADE / 4

    def test_transition_half_risk(self):
        """TRANSITION → half risk (no natural match)."""
        risk = get_regime_adjusted_risk("TRANSITION", 0.8, "tf")
        assert risk == MAX_RISK_PER_TRADE / 2

    def test_ranging_mr_full_risk(self):
        """RANGING + mr signal → full risk (match)."""
        risk = get_regime_adjusted_risk("RANGING", 0.8, "mr")
        assert risk == MAX_RISK_PER_TRADE

    def test_below_min_confidence_blocks(self):
        """Confidence below 0.6 → 0 risk (hard gate)."""
        risk = get_regime_adjusted_risk("TRENDING_BULL", 0.5, "tf")
        assert risk == 0.0

    def test_funding_always_fixed(self):
        """Funding trades always return fixed rate."""
        risk = get_regime_adjusted_risk("TRENDING_BULL", 0.8, "fr", is_funding=True)
        assert risk == FUNDING_RISK_PER_TRADE


# ── leverage ─────────────────────────────────────────────────────────────

class TestLeverage:
    def test_always_3x(self):
        assert get_leverage() == 3.0
