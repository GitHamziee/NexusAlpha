"""
NexusAlpha — Regime-Adaptive Crypto Trading Strategy

Main Freqtrade IStrategy orchestrating:
1. Regime detection (5 regimes with adaptive thresholds)
2. Strategy selection (trend following / mean reversion / funding rate)
3. Signal generation with 5-layer confluence filtering
4. Risk management (position sizing, circuit breakers, ATR stop scaling)

Target: 70-80% win rate through extreme selectivity (200 signals → ~13 trades/month).
ML enhancement planned for Phase 2 after 3+ months of live data.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)

from .core.funding_rate import (
    STOP_ATR_MULT as FR_STOP_ATR_MULT,
    add_funding_indicators,
    populate_funding_entries,
    populate_funding_exits,
)
from .core.mean_reversion import (
    STOP_ATR_MULT as MR_STOP_ATR_MULT,
    TIME_STOP_CANDLES as MR_TIME_STOP,
    add_mr_indicators,
    populate_mr_entries,
    populate_mr_exits,
)
from .core.regime_detector import (
    add_regime_indicators,
    apply_multitf_confirmation,
    apply_regime,
)
from .core.trend_following import (
    STOP_ATR_MULT as TF_STOP_ATR_MULT,
    TIME_STOP_CANDLES as TF_TIME_STOP,
    add_trend_indicators,
    populate_trend_entries,
    populate_trend_exits,
)
from .risk.risk_manager import (
    calculate_position_size,
    get_risk_percent,
    scale_atr_stop,
)

logger = logging.getLogger(__name__)

# ── Signal logging ───────────────────────────────────────────────────────
SIGNAL_LOG_DIR = Path("logs/signals")
TRADE_LOG_DIR = Path("logs/trades")
SCHEMA_VERSION = 1
SIGNAL_COLUMNS = [
    "timestamp", "schema_version",
    "rsi_14", "macd_histogram", "bb_percent_b", "adx_14",
    "supertrend_direction", "stochrsi_k", "atr_ratio",
    "volume_ratio",
    "regime", "regime_confidence",
    "funding_rate", "long_short_ratio",
    "return_5", "return_15", "return_60",
    "volatility_20",
    "hour_of_day", "day_of_week",
    "strategy_name", "entry_side", "entry_price",
    "signal_taken",
]
TRADE_COLUMNS = [
    "trade_id", "pair", "strategy", "side",
    "open_date", "close_date", "duration_minutes",
    "open_rate", "close_rate",
    "profit_ratio", "profit_abs",
    "stake_amount", "leverage",
    "exit_reason",
    "regime_at_entry", "regime_confidence_at_entry",
    "regime_at_exit", "regime_confidence_at_exit",
    "atr_at_entry", "atr_at_exit",
    "rsi_at_entry", "rsi_at_exit",
    "adx_at_entry", "adx_at_exit",
    "schema_version",
]


class NexusAlpha(IStrategy):
    """
    Regime-adaptive strategy using 3 sub-strategies with risk management.

    Filtering pipeline:
        Layer 1: Regime gate (kills ~60%)
        Layer 2: Multi-TF confirmation (kills ~40%)
        Layer 3: Confluence (3+ indicators align)
        Layer 4: Volume + momentum confirmation
        Layer 5: Risk gate (drawdown, cooldown, position sizing)
    """

    # ─── Freqtrade Configuration ───────────────────────────────────────

    INTERFACE_VERSION = 3
    timeframe = "15m"
    informative_timeframes = ["1h"]
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.05
    trailing_stop = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 210

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # ─── Hyperoptable Parameters ───────────────────────────────────────

    adx_trending_threshold = IntParameter(25, 35, default=30, space="buy")
    adx_ranging_threshold = IntParameter(15, 22, default=20, space="buy")
    mr_rsi_oversold = IntParameter(25, 35, default=30, space="buy")
    mr_rsi_overbought = IntParameter(65, 75, default=70, space="buy")
    mr_volume_mult = DecimalParameter(1.0, 1.5, default=1.2, space="buy")
    tf_adx_threshold = IntParameter(22, 30, default=25, space="buy")
    tf_stochrsi_low = IntParameter(15, 25, default=20, space="buy")
    tf_stochrsi_high = IntParameter(75, 85, default=80, space="buy")

    # ─── Circuit Breaker Protections ───────────────────────────────────

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {"method": "StoplossGuard", "trade_limit": 3,
             "stop_duration_candles": 16, "only_per_pair": True},
            {"method": "MaxDrawdown", "trade_back": "day",
             "max_allowed_drawdown": 0.03},
            {"method": "LowProfitPairs", "trade_back": "day",
             "trade_limit": 3, "required_profit": -0.02},
        ]

    # ─── Informative Pairs ─────────────────────────────────────────────

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, "1h") for pair in pairs]

    # ─── populate_indicators ───────────────────────────────────────────

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute ALL indicators, run regime detector, merge 1H data."""

        # 1) Regime indicators (ADX, BB, EMA slope, ATR) + classify
        dataframe = add_regime_indicators(dataframe)
        dataframe = apply_regime(dataframe)

        # 2) Trend following indicators
        dataframe = add_trend_indicators(dataframe)

        # 3) Mean reversion indicators
        dataframe = add_mr_indicators(dataframe)

        # 4) Funding rate indicators (funding_rate column injected by bot_loop_start)
        dataframe = add_funding_indicators(dataframe)

        # 5) Merge 1H informative data for multi-TF confirmation
        if self.dp:
            for pair in self.dp.current_whitelist():
                inf_df = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
                if not inf_df.empty:
                    inf_df = add_regime_indicators(inf_df)
                    inf_df = apply_regime(inf_df)
                    dataframe = merge_informative_pair(
                        dataframe, inf_df, self.timeframe, "1h",
                        ffill=True,
                    )

        # 6) Apply multi-TF regime confirmation (adjusts regime + confidence)
        dataframe = apply_multitf_confirmation(dataframe)

        # 7) Helper columns for signal logging
        dataframe["return_5"] = dataframe["close"].pct_change(5)
        dataframe["return_15"] = dataframe["close"].pct_change(15)
        dataframe["return_60"] = dataframe["close"].pct_change(60)
        dataframe["volatility_20"] = dataframe["close"].rolling(20).std() / dataframe["close"]

        # ATR SMA for dynamic stop scaling
        dataframe["atr_14_sma"] = dataframe["atr_14"].rolling(window=100, min_periods=20).mean()

        return dataframe

    # ─── populate_entry_trend ──────────────────────────────────────────

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Layer 1-4 filtering: run all strategy entries, tag with strategy name."""

        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0
        dataframe.loc[:, "enter_tag"] = ""

        # Strategy entry signals (each checks regime internally)
        dataframe = populate_trend_entries(dataframe)
        dataframe = populate_mr_entries(dataframe)
        dataframe = populate_funding_entries(dataframe)

        # Merge signals — first strategy to fire wins (priority: trend > MR > funding)
        # Trend Following
        long_tf = dataframe["tf_enter_long"] == 1
        short_tf = dataframe["tf_enter_short"] == 1
        dataframe.loc[long_tf, "enter_long"] = 1
        dataframe.loc[long_tf, "enter_tag"] = "trend_following_long"
        dataframe.loc[short_tf, "enter_short"] = 1
        dataframe.loc[short_tf & (dataframe["enter_tag"] == ""), "enter_tag"] = "trend_following_short"

        # Mean Reversion (only if no trend signal on same candle)
        long_mr = (dataframe["mr_enter_long"] == 1) & (dataframe["enter_long"] == 0)
        short_mr = (dataframe["mr_enter_short"] == 1) & (dataframe["enter_short"] == 0)
        dataframe.loc[long_mr, "enter_long"] = 1
        dataframe.loc[long_mr, "enter_tag"] = "mean_reversion_long"
        dataframe.loc[short_mr, "enter_short"] = 1
        dataframe.loc[short_mr, "enter_tag"] = "mean_reversion_short"

        # Funding Rate (lowest priority)
        long_fr = (dataframe["fr_enter_long"] == 1) & (dataframe["enter_long"] == 0)
        short_fr = (dataframe["fr_enter_short"] == 1) & (dataframe["enter_short"] == 0)
        dataframe.loc[long_fr, "enter_long"] = 1
        dataframe.loc[long_fr, "enter_tag"] = "funding_rate_long"
        dataframe.loc[short_fr, "enter_short"] = 1
        dataframe.loc[short_fr, "enter_tag"] = "funding_rate_short"

        return dataframe

    # ─── populate_exit_trend ───────────────────────────────────────────

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Strategy-specific exits based on enter_tag."""

        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        dataframe.loc[:, "exit_tag"] = ""

        # Compute all exit signals
        dataframe = populate_trend_exits(dataframe)
        dataframe = populate_mr_exits(dataframe)
        dataframe = populate_funding_exits(dataframe)

        # Trend following exits
        dataframe.loc[dataframe["tf_exit_long"] == 1, "exit_long"] = 1
        dataframe.loc[dataframe["tf_exit_long"] == 1, "exit_tag"] = "tf_adx_death_or_st_flip"
        dataframe.loc[dataframe["tf_exit_short"] == 1, "exit_short"] = 1
        dataframe.loc[dataframe["tf_exit_short"] == 1, "exit_tag"] = "tf_adx_death_or_st_flip"

        # Mean reversion exits
        mr_exit_l = (dataframe["mr_exit_long"] == 1) & (dataframe["exit_long"] == 0)
        mr_exit_s = (dataframe["mr_exit_short"] == 1) & (dataframe["exit_short"] == 0)
        dataframe.loc[mr_exit_l, "exit_long"] = 1
        dataframe.loc[mr_exit_l, "exit_tag"] = "mr_bb_middle_or_regime_change"
        dataframe.loc[mr_exit_s, "exit_short"] = 1
        dataframe.loc[mr_exit_s, "exit_tag"] = "mr_bb_middle_or_regime_change"

        # Funding exits
        fr_exit_l = (dataframe["fr_exit_long"] == 1) & (dataframe["exit_long"] == 0)
        fr_exit_s = (dataframe["fr_exit_short"] == 1) & (dataframe["exit_short"] == 0)
        dataframe.loc[fr_exit_l, "exit_long"] = 1
        dataframe.loc[fr_exit_l, "exit_tag"] = "fr_funding_normalized"
        dataframe.loc[fr_exit_s, "exit_short"] = 1
        dataframe.loc[fr_exit_s, "exit_tag"] = "fr_funding_normalized"

        return dataframe

    # ─── custom_stoploss ───────────────────────────────────────────────

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """ATR-based stop per strategy + dynamic ATR scaling in high vol."""

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.03

        last = dataframe.iloc[-1]
        atr = last.get("atr_14", 0)
        atr_sma = last.get("atr_14_sma", atr)
        entry_rate = trade.open_rate

        if entry_rate <= 0 or atr <= 0:
            return -0.03

        tag = trade.enter_tag or ""

        # Base multiplier per strategy
        if "trend_following" in tag:
            base_mult = TF_STOP_ATR_MULT  # 2.0
        elif "mean_reversion" in tag:
            base_mult = MR_STOP_ATR_MULT  # 1.5
        elif "funding_rate" in tag:
            base_mult = FR_STOP_ATR_MULT  # 3.0
        else:
            base_mult = 2.0

        # Dynamic scaling: widen stop when vol spikes
        scaled_mult = scale_atr_stop(base_mult, atr, atr_sma)

        stop_distance = scaled_mult * atr
        stop_pct = -(stop_distance / entry_rate)

        # Time stop: if trade has been open too long without profit
        trade_candles = (current_time - trade.open_date).total_seconds() / 900  # 15m candles
        if "trend_following" in tag and trade_candles > TF_TIME_STOP and current_profit < 0.005:
            return -0.001  # force close
        if "mean_reversion" in tag and trade_candles > MR_TIME_STOP and current_profit < 0.005:
            return -0.001

        return max(stop_pct, -0.05)  # never wider than 5%

    # ─── confirm_trade_entry ───────────────────────────────────────────

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """Layer 5: risk checks + signal feature logging."""

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False

        last = dataframe.iloc[-1]

        # Risk check: confidence threshold
        confidence = last.get("regime_confidence", 0)
        is_funding = "funding_rate" in (entry_tag or "")
        risk_pct = get_risk_percent(confidence, is_funding)
        if risk_pct == 0.0:
            logger.info("Trade rejected: confidence %.2f too low", confidence)
            return False

        # Log signal features for future ML training
        self._log_signal(last, entry_tag, side, rate, taken=True, current_time=current_time)

        # Stash entry-time context so _log_trade can compare entry vs exit
        # Freqtrade's Trade object allows setting custom attributes
        self._pending_entry_context = {
            "regime": last.get("regime", None),
            "confidence": last.get("regime_confidence", None),
            "atr": last.get("atr_14", None),
            "rsi": last.get("rsi_14", None),
            "adx": last.get("adx_14", None),
        }

        return True

    # ─── order_filled (attach entry context to Trade) ───────────────────

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order: dict,
        current_time: datetime,
        **kwargs,
    ) -> None:
        """Attach entry-time market context to the Trade for later logging."""
        if order.get("ft_order_side") == "buy" or order.get("side") == "buy":
            ctx = getattr(self, "_pending_entry_context", {})
            if ctx:
                trade._regime_at_entry = ctx.get("regime")
                trade._conf_at_entry = ctx.get("confidence")
                trade._atr_at_entry = ctx.get("atr")
                trade._rsi_at_entry = ctx.get("rsi")
                trade._adx_at_entry = ctx.get("adx")
                self._pending_entry_context = {}

    # ─── confirm_trade_exit (Trade Journal) ──────────────────────────────

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Log every trade outcome for learning. Always allow the exit."""

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last = dataframe.iloc[-1] if not dataframe.empty else pd.Series()

        self._log_trade(trade, rate, exit_reason, current_time, last)
        return True

    # ─── custom_stake_amount ───────────────────────────────────────────

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Confidence-scaled position sizing with dynamic ATR stop scaling."""

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]
        confidence = last.get("regime_confidence", 0)
        is_funding = "funding_rate" in (entry_tag or "")
        risk_pct = get_risk_percent(confidence, is_funding)

        if risk_pct == 0.0:
            return 0

        atr = last.get("atr_14", 0)
        atr_sma = last.get("atr_14_sma", atr)

        tag = entry_tag or ""
        if "trend_following" in tag:
            base_mult = TF_STOP_ATR_MULT
        elif "mean_reversion" in tag:
            base_mult = MR_STOP_ATR_MULT
        elif "funding_rate" in tag:
            base_mult = FR_STOP_ATR_MULT
        else:
            base_mult = 2.0

        scaled_mult = scale_atr_stop(base_mult, atr, atr_sma)
        stop_distance = scaled_mult * atr

        if stop_distance <= 0 or current_rate <= 0:
            return proposed_stake

        # stop_distance is in price units; convert to fraction of entry
        stop_frac = stop_distance / current_rate

        # Get wallet balance from proposed_stake (Freqtrade sets proposed = balance / max_trades)
        # We recalculate based on our risk model
        total_balance = proposed_stake * 3  # max_open_trades=3 → proposed ≈ balance/3
        position = calculate_position_size(total_balance, risk_pct, stop_frac)

        # Clamp to Freqtrade limits
        if min_stake and position < min_stake:
            return min_stake
        return min(position, max_stake)

    # ─── leverage ──────────────────────────────────────────────────────

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Max 3x leverage. No exceptions."""
        return min(3.0, max_leverage)

    # ─── Signal Feature Logging ────────────────────────────────────────

    def _log_signal(
        self,
        row: pd.Series,
        entry_tag: Optional[str],
        side: str,
        price: float,
        taken: bool,
        current_time: datetime,
    ) -> None:
        """Write signal features to daily CSV for future ML training."""
        try:
            SIGNAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
            date_str = current_time.strftime("%Y-%m-%d")
            filepath = SIGNAL_LOG_DIR / f"signals_{date_str}.csv"

            write_header = not filepath.exists()

            bb_middle = row.get("mr_bb_middle", None)
            bb_upper = row.get("mr_bb_upper", None)
            bb_lower = row.get("mr_bb_lower", None)
            close = row.get("close", None)
            bb_pct_b = None
            if bb_upper and bb_lower and close and bb_upper != bb_lower:
                bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)

            volume = row.get("volume", None)
            vol_sma = row.get("volume_sma_20", None)
            vol_ratio = volume / vol_sma if volume and vol_sma and vol_sma > 0 else None

            atr = row.get("atr_14", None)
            atr_ratio = atr / close if atr and close and close > 0 else None

            record = {
                "timestamp": current_time.isoformat(),
                "schema_version": SCHEMA_VERSION,
                "rsi_14": row.get("rsi_14", None),
                "macd_histogram": row.get("mr_macd_hist", None),
                "bb_percent_b": bb_pct_b,
                "adx_14": row.get("adx_14", None),
                "supertrend_direction": row.get("supertrend_direction", None),
                "stochrsi_k": row.get("stochrsi_k", None),
                "atr_ratio": atr_ratio,
                "volume_ratio": vol_ratio,
                "regime": row.get("regime", None),
                "regime_confidence": row.get("regime_confidence", None),
                "funding_rate": row.get("funding_rate", None),
                "long_short_ratio": row.get("long_short_ratio", None),
                "return_5": row.get("return_5", None),
                "return_15": row.get("return_15", None),
                "return_60": row.get("return_60", None),
                "volatility_20": row.get("volatility_20", None),
                "hour_of_day": current_time.hour,
                "day_of_week": current_time.weekday(),
                "strategy_name": entry_tag,
                "entry_side": side,
                "entry_price": price,
                "signal_taken": 1 if taken else 0,
            }

            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SIGNAL_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

        except Exception as e:
            logger.warning("Failed to log signal: %s", e)

    def _log_trade(
        self,
        trade: Trade,
        close_rate: float,
        exit_reason: str,
        current_time: datetime,
        last_row: pd.Series,
    ) -> None:
        """Write completed trade to the trade journal CSV.

        Captures full context at entry AND exit so we can analyze what
        worked, what didn't, and why.  This is the feedback loop that
        makes the system learn over time.
        """
        try:
            TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
            month_str = current_time.strftime("%Y-%m")
            filepath = TRADE_LOG_DIR / f"trades_{month_str}.csv"

            write_header = not filepath.exists()

            duration = (current_time - trade.open_date).total_seconds() / 60
            profit_ratio = (close_rate - trade.open_rate) / trade.open_rate
            if trade.enter_tag and "short" in trade.enter_tag:
                profit_ratio = -profit_ratio

            record = {
                "trade_id": trade.id,
                "pair": trade.pair,
                "strategy": trade.enter_tag or "unknown",
                "side": "short" if (trade.enter_tag and "short" in trade.enter_tag) else "long",
                "open_date": trade.open_date.isoformat(),
                "close_date": current_time.isoformat(),
                "duration_minutes": round(duration, 1),
                "open_rate": trade.open_rate,
                "close_rate": close_rate,
                "profit_ratio": round(profit_ratio, 6),
                "profit_abs": round(profit_ratio * trade.stake_amount, 2),
                "stake_amount": trade.stake_amount,
                "leverage": trade.leverage,
                "exit_reason": exit_reason,
                "regime_at_entry": getattr(trade, "_regime_at_entry", None),
                "regime_confidence_at_entry": getattr(trade, "_conf_at_entry", None),
                "regime_at_exit": last_row.get("regime", None),
                "regime_confidence_at_exit": last_row.get("regime_confidence", None),
                "atr_at_entry": getattr(trade, "_atr_at_entry", None),
                "atr_at_exit": last_row.get("atr_14", None),
                "rsi_at_entry": getattr(trade, "_rsi_at_entry", None),
                "rsi_at_exit": last_row.get("rsi_14", None),
                "adx_at_entry": getattr(trade, "_adx_at_entry", None),
                "adx_at_exit": last_row.get("adx_14", None),
                "schema_version": SCHEMA_VERSION,
            }

            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)

            logger.info(
                "Trade closed: %s %s | %.2f%% | %s | %s",
                record["strategy"], record["side"],
                profit_ratio * 100, exit_reason, record["duration_minutes"],
            )

        except Exception as e:
            logger.warning("Failed to log trade: %s", e)
