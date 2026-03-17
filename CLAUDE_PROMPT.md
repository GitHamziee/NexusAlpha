# Instructions for Claude

You are working on **Nexus Alpha** — a crypto trading bot built on Freqtrade. The complete engineering specification is in `prompt.md` in this project root. Read it fully before writing any code.

## Current State

Phase 1 (project skeleton) is DONE. The following files exist and are configured:
- `Dockerfile`, `docker-compose.yml`, `Makefile`, `requirements.txt`
- `config/config.json`, `config/config-dry-run.json`, `config/config-live.json`
- `strategies/NexusAlpha.py` (skeleton — needs full implementation)
- All `__init__.py` files for `strategies/core/`, `strategies/data/`, `strategies/risk/`, `strategies/ml/`
- `.env.example`, `.gitignore`, `scripts/download_data.sh`

## Your Task: Build Phase 2 through Phase 8

Work through the phases in order. For each phase, write the full implementation AND unit tests. Do NOT skip ahead — each phase depends on the previous one.

### Phase 2: Regime Detector (`strategies/core/regime_detector.py`)
- Implement `classify_regime()` with **adaptive thresholds** (ADX baseline via 100-period SMA, not fixed 18/28)
- 5 regimes: TRENDING_BULL, TRENDING_BEAR, RANGING, VOLATILE, TRANSITION
- Multi-timeframe confirmation (1H must agree with 15m)
- TRANSITION zone produces confidence = 0.0 (no trades)
- Write `tests/test_regime_detector.py` with synthetic data for each regime

### Phase 3: Risk Manager (`strategies/risk/risk_manager.py`)
- `calculate_position_size(balance, risk_pct, stop_distance)`
- `get_risk_percent(regime_confidence)` — 1.0% for high, 0.5% for medium, 0 for low
- `check_daily_drawdown()`, `check_max_open_trades()`
- **Dynamic ATR stop scaling**: when ATR > 1.5x its 100-period SMA, widen stops proportionally and shrink position to keep dollar risk constant
- All non-negotiable rules from `prompt.md` Section VIII
- Write `tests/test_risk_manager.py`

### Phase 4: Trend Following (`strategies/core/trend_following.py`)
- Indicators: Supertrend(10,3), ADX(14), EMA(200/50/9), StochRSI(14,3,3), ATR(14), Volume SMA(20), RSI(14)
- 9 entry conditions per `prompt.md` Section V (all must be true)
- Exit rules: 2x ATR stop, 1.5x/3x ATR TPs, Supertrend trail, ADX death, 20-candle time stop
- Write `tests/test_trend_following.py`

### Phase 5: Mean Reversion (`strategies/core/mean_reversion.py`)
- Indicators: BB(20,2), RSI(14), MACD(12,26,9), Volume SMA(20), ATR(14), EMA(200)
- 9 entry conditions per `prompt.md` Section VI
- Exit rules: 1.5x ATR stop, BB middle TP, 12-candle time stop, regime change exit
- Write `tests/test_mean_reversion.py`

### Phase 6: Funding Rate (`strategies/core/funding_rate.py` + `strategies/data/funding_provider.py`)
- **Data provider first**: Pre-fetch on 2-minute timer, 2-min cache TTL, 10-min stale threshold, exponential backoff
- Entry thresholds are EXTREME: funding > 0.08% or < -0.05%
- Always half-size (0.5% risk), 3x ATR stops (wider than other strategies)
- Graceful fallback: if data unavailable, strategy simply doesn't fire
- Write `tests/test_funding_rate.py` (mock the API responses)

### Phase 7: Wire NexusAlpha.py
- Implement all methods in the main strategy class
- `populate_indicators()`: compute ALL indicators, run regime detector, merge 1H data, fetch funding
- `populate_entry_trend()`: Layer 1-4 filtering, tag entries with strategy name
- `populate_exit_trend()`: strategy-specific exits based on `enter_tag`
- `custom_stoploss()`: ATR-based per strategy + dynamic scaling
- `confirm_trade_entry()`: Layer 5 risk checks + signal feature logging (CSV)
- `custom_stake_amount()`: confidence-scaled position sizing
- `leverage()`: always cap at 3x
- Signal feature logging: write `logs/signals/signals_YYYY-MM-DD.csv` per `prompt.md` Section XV

### Phase 8: Tests
- Complete unit tests for any gaps
- Write `tests/test_integration.py`:
  - Feed synthetic data through ALL 5 layers end-to-end
  - Verify TRANSITION regime = zero trades
  - Verify cooldown blocks next signal after a stop loss
  - Verify stale/missing funding data doesn't crash the pipeline
  - Verify confidence scaling produces correct position sizes

## Code Standards

- Use `pandas-ta` for indicator computation (already in requirements.txt)
- Type hints on all functions
- Docstrings explaining the "why", not the "what"
- Each strategy module exports simple functions (not classes) that take a DataFrame and return a DataFrame with signal columns
- Handle edge cases: empty DataFrames, NaN values, missing columns
- Use `Decimal` for ATR/volume ratio chains where precision matters
- Log at appropriate levels (INFO for trades, WARNING for API issues, ERROR for failures)

## What NOT to do

- Do NOT add ML anything — that's Phase 2 (months away)
- Do NOT add extra indicators, strategies, or "improvements" beyond what's in `prompt.md`
- Do NOT change the Docker/config files unless absolutely necessary
- Do NOT add ETH or any other pair — BTC/USDT only
- Do NOT use classes where simple functions work
- Do NOT over-engineer. If `prompt.md` says 2x ATR stop, use 2x ATR stop.
