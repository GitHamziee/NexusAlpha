# NEXUS ALPHA — Production Crypto Trading System

> **Complete engineering spec. Give this entire file to Claude as a single prompt. Every line is deliberate.**

---

## I. MANDATE

You are building **Nexus Alpha** — a high-selectivity, regime-aware crypto trading system on 15-minute candles. This is a production system designed to achieve **70-80% win rate on executed trades** through extreme signal filtering.

**The core principle: A sniper, not a machine gun.**

We do NOT take 60 trades a month at 50% accuracy. We take 10-15 trades a month at 70-80% accuracy. We generate hundreds of candidate signals, filter ruthlessly through multiple layers of confluence, and only execute when conditions overwhelmingly stack in our favor. We sacrifice frequency for precision.

**Targets:**
- **Win rate:** 70-80% on executed trades (through extreme selectivity)
- **Risk-reward ratio:** 1.5:1 to 2:1 per trade
- **Trade frequency:** 10-15 trades per month (2-4 per week)
- **Monthly target:** 20-30% in favorable conditions, 5-10% in choppy markets
- **Max drawdown tolerance:** 15% of account

**How 70-80% win rate is achievable (it's about filtering, not prediction):**
```
Step 1: Strategy generates ~200 raw signals per month
Step 2: Regime filter removes ~60% (wrong market state)     → ~80 remain
Step 3: Multi-timeframe confirmation removes ~40%            → ~48 remain
Step 4: Confluence filter (3+ conditions must align) removes ~50% → ~24 remain
Step 5: Volume + momentum confirmation removes ~30%          → ~17 remain
Step 6: Risk gate (drawdown limits, cooldowns) removes ~20%  → ~13 remain

13 trades/month. Each one passed 6 layers of filtering.
Each one has regime + multi-TF + indicator confluence + volume agreement.
70-80% of these hitting their take profit is realistic.

The trades we DON'T take are the edge.
```

**Every number accounts for real costs:**
- 0.04% maker / 0.06% taker fees (Binance Futures)
- 0.01-0.05% slippage per trade (varies with volatility)
- Funding rate costs (+/-0.01% every 8 hours on open positions)
- API latency and order fill delays (100-500ms)
- Exchange downtime and maintenance windows
- Backtest-to-live degradation (~30-50% with our selective approach, less than typical because we take fewer trades)

**Philosophy:**
- We don't predict. We react to confirmed setups with statistical edge.
- We don't gamble. Every trade has defined risk, target, and invalidation.
- We don't spray. The best trade is often NO trade.
- We don't fight the regime. Wrong strategy in wrong market = death.
- We respect production reality. Every API can fail, every exchange goes down.
- We accept drawdowns. They're the cost of doing business.
- We never override the bot during a drawdown.

**SURVIVE FIRST, PROFIT SECOND.** Position sizing and discipline matter more than signal generation.

---

## II. SYSTEM ARCHITECTURE

```
╔═══════════════════════════════════════════════════════════════╗
║              NEXUS ALPHA — SYSTEM ARCHITECTURE                ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  ┌─────────────────┐    ┌──────────────────┐                 ║
║  │  MARKET DATA    │───▶│ REGIME DETECTOR  │                 ║
║  │  (Binance API)  │    │ (ADX + BB Width) │                 ║
║  └─────────────────┘    └────────┬─────────┘                 ║
║                                  │                            ║
║            LAYER 1: REGIME GATE (kills ~60% of signals)      ║
║              ┌───────────────────┼───────────────┐           ║
║              ▼                   ▼               ▼           ║
║   ┌──────────────────┐ ┌────────────────┐ ┌───────────┐     ║
║   │ STRATEGY 1:      │ │ STRATEGY 2:    │ │STRATEGY 3:│     ║
║   │ Trend Following  │ │ Mean Reversion │ │Funding    │     ║
║   │ (TRENDING only)  │ │ (RANGING only) │ │Rate Arb   │     ║
║   └────────┬─────────┘ └───────┬────────┘ └─────┬─────┘     ║
║            │                   │                 │            ║
║            └───────────────────┼─────────────────┘           ║
║                                ▼                              ║
║   LAYER 2: MULTI-TIMEFRAME CONFIRMATION (kills ~40%)         ║
║                    ┌───────────────────────┐                  ║
║                    │  1H must agree with   │                  ║
║                    │  15m regime + trend    │                  ║
║                    └───────────┬───────────┘                  ║
║                                ▼                              ║
║   LAYER 3: CONFLUENCE FILTER (kills ~50%)                    ║
║                    ┌───────────────────────┐                  ║
║                    │  3+ indicators must   │                  ║
║                    │  align simultaneously │                  ║
║                    └───────────┬───────────┘                  ║
║                                ▼                              ║
║   LAYER 4: VOLUME + MOMENTUM (kills ~30%)                    ║
║                    ┌───────────────────────┐                  ║
║                    │  Above-avg volume +   │                  ║
║                    │  momentum confirming  │                  ║
║                    └───────────┬───────────┘                  ║
║                                ▼                              ║
║   LAYER 5: RISK GATE (kills ~20%)                            ║
║                    ┌───────────────────────┐                  ║
║                    │  Position size + ATR  │                  ║
║                    │  stop + drawdown chk  │                  ║
║                    └───────────┬───────────┘                  ║
║                                ▼                              ║
║                    ┌───────────────────────┐                  ║
║                    │   ORDER EXECUTION     │                  ║
║                    │  (~13 trades/month)   │                  ║
║                    └───────────────────────┘                  ║
║                                                               ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │            PHASE 2 (OPTIONAL, LATER):                   │ ║
║  │  ML Confidence Filter — XGBoost meta-labeler            │ ║
║  │  Adds Layer 6. Only after 3+ months live data.          │ ║
║  └─────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════╝
```

**Why 3 strategies, not 5:**
- Each strategy requires 200+ hours of production tuning
- Solo developer with a day job can maintain 2-3 well
- Better to have 2 battle-tested strategies than 5 half-baked ones

**Why ML is Phase 2, not Phase 1:**
- ML overfits catastrophically without 6+ months of LIVE data
- Our 6-layer filtering already achieves 70-80% without ML
- ML adds 10x complexity to deployment and debugging
- We collect features from day 1, train models later IF needed

**Strategy correlation reality:**
```
Normal markets:
  Trend Following: +5%    Mean Reversion: -2%    Funding: +1%  = +4%
  (Strategies appear uncorrelated)

During a crash (LUNA-style event):
  Trend Following: -8%    Mean Reversion: -15%   Funding: -10% = -33%
  (Correlation spikes to 1.0 in a crisis)

Our defense: With 1% risk per trade, max 3 open trades, 3x leverage:
  Worst single-event loss = ~5-7% (survivable)
  This is why position sizing > signal quality
```

---

## III. TECHNICAL STACK

```
Framework:       Freqtrade (stable, battle-tested, Python)
Exchange:        Binance Futures (most liquid, lowest fees)
Pair:            BTC/USDT perpetual (ONE pair — master it first)
Timeframe:       15m primary, 1H confirmation
Mode:            Futures (long + short), isolated margin
Leverage:        3x maximum (NEVER higher)
Language:        Python 3.11+
Deployment:      Docker on VPS (~$10/mo, 4-core, 8GB RAM)
Monitoring:      FreqUI + Telegram alerts
Data:            OHLCV + funding rates from Binance API
```

**Why BTC/USDT only:**
- Most liquid = lowest slippage, tightest spreads
- Most data = best backtesting coverage
- One pair = one set of parameters to master
- **Known limitation:** Zero diversification. If BTC enters a multi-month bear, we rely on short-side strategies. We accept this tradeoff — crypto pairs are 90%+ correlated anyway, "diversifying" into ETH/SOL is fake diversification.
- Add ETH/USDT ONLY after 6+ months profitable on BTC

---

## IV. REGIME DETECTION ENGINE

**File:** `strategies/core/regime_detector.py`

The regime detector is Layer 1 of our filtering. It decides WHAT market we're in so we only run strategies that match. This single filter eliminates ~60% of bad trades.

### The Lag Problem (Acknowledged)

**Every regime detector is a lagging indicator.** By definition:
- "TRENDING" confirms AFTER the trend started (30-40% of move is over)
- "RANGING" confirms AFTER the trend died
- Transitions are where the detector is WRONG most often

**We handle this by:**
1. Adding a TRANSITION regime (ADX 18-28) where we DON'T TRADE
2. Requiring 1H timeframe confirmation (reduces noise-driven false classifications)
3. Accepting we miss early entries — we catch the reliable middle of moves, not tops/bottoms
4. The time stop on every strategy limits damage from late regime detection

### Classification Logic

```
Input: OHLCV data (15m candles) + 1H confirmation
Output: regime (string) + confidence (0.0 - 0.9)

REGIMES:
├── TRENDING_BULL  → Trend Following (long bias)
├── TRENDING_BEAR  → Trend Following (short bias)
├── RANGING        → Mean Reversion
├── VOLATILE       → Funding Rate only (or sit out)
└── TRANSITION     → NO TRADING (ambiguous, wait for clarity)
```

### Indicators

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| ADX(14) | Period 14 | Trend strength (>28 = trending, <18 = ranging) |
| +DI / -DI | Period 14 | Trend direction |
| BB Width | BB(20, 2.0) | Volatility (narrow = ranging, wide = volatile) |
| EMA(50) slope | 10-period rate of change | Trend direction confirmation |
| ATR(14) ratio | ATR/Close | Normalized volatility |

### Classification Rules

```python
def classify_regime(adx, plus_di, minus_di, bb_width, bb_width_sma,
                    ema_slope, atr_ratio, atr_ratio_sma):

    # VOLATILE — check first (overrides everything)
    if atr_ratio > atr_ratio_sma * 2.0 or bb_width > bb_width_sma * 2.5:
        return "VOLATILE", 0.3

    # STRONG TREND
    if adx > 28:
        if plus_di > minus_di and ema_slope > 0:
            return "TRENDING_BULL", min(0.9, adx / 45)
        elif minus_di > plus_di and ema_slope < 0:
            return "TRENDING_BEAR", min(0.9, adx / 45)

    # CLEAR RANGING
    if adx < 18 and bb_width < bb_width_sma * 0.8:
        return "RANGING", min(0.8, (18 - adx) / 12)

    # TRANSITION — DO NOT TRADE
    if 18 <= adx <= 28:
        return "TRANSITION", 0.0

    return "RANGING", 0.5
```

**Design decisions:**
- Confidence NEVER reaches 1.0 — we're never fully certain
- TRANSITION zone (ADX 18-28) = NO trades. This prevents the most common whipsaw losses.
- VOLATILE checked first — liquidation cascades spike ADX but it's not a tradeable trend
- Wider thresholds (18/28 vs typical 20/25) to reduce flip-flopping

### Multi-Timeframe Confirmation (Layer 2)

1H timeframe must agree with 15m:
- Both agree → confidence as-is
- Disagree → cut confidence by 50% (likely drops below 0.6 threshold, blocking trade)
- 1H says VOLATILE → override to VOLATILE regardless of 15m (higher TF wins for danger)

**Implementation:** Freqtrade's `informative_pairs()` + `merge_informative_pair()`.

---

## V. STRATEGY 1: TREND FOLLOWING

**File:** `strategies/core/trend_following.py`
**Active when:** Regime = TRENDING_BULL or TRENDING_BEAR, confidence >= 0.6
**Expected stats:** 65-75% win rate (after all filtering), 2:1 to 3:1 R:R, ~5-7 trades/month

This strategy catches the middle of trends. We enter on confirmed pullbacks within established trends. By requiring so many conditions to align, we only enter the highest-probability setups.

### Indicators

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| Supertrend | Period 10, Multiplier 3.0 | Trend direction + trailing stop |
| ADX(14) | Period 14 | Trend strength filter |
| EMA(200) | Period 200 | Major trend direction |
| EMA(50) | Period 50 | Intermediate trend |
| EMA(9) | Period 9 | Entry timing |
| StochRSI | RSI(14), Stoch(3,3) | Pullback entry timing |
| ATR(14) | Period 14 | Stop loss / take profit |
| Volume SMA(20) | Period 20 | Volume confirmation |
| RSI(14) | Period 14 | Overbought/oversold guard |

### Entry Conditions — LONG (ALL must be true)

**Layer 1 (Regime):**
1. Regime = TRENDING_BULL, confidence >= 0.6

**Layer 2 (Multi-TF):**
2. 1H regime also TRENDING_BULL or at least not RANGING/VOLATILE

**Layer 3 (Confluence — 3+ indicators must agree):**
3. Supertrend: Bullish (price above Supertrend line)
4. ADX > 28 AND rising (ADX > ADX[3])
5. Price structure: Close > EMA(200) AND Close > EMA(50)

**Layer 4 (Entry timing + Volume):**
6. StochRSI K crosses above D from below 30 (pullback within trend)
7. Volume > Volume SMA(20) x 1.0 (at least average volume)

**Layer 5 (Safety):**
8. RSI(14) < 75 (NOT overbought)
9. Not within 4 candles of a previous stop loss (cooldown)

That's **9 conditions** that must simultaneously be true. This is why 65-75% of these trades win — by the time all 9 align, the probability is heavily stacked.

### Entry Conditions — SHORT (mirror)

1. Regime: TRENDING_BEAR, confidence >= 0.6
2. 1H confirms bearish
3. Supertrend: Bearish
4. ADX > 28 AND rising
5. Close < EMA(200) AND Close < EMA(50)
6. StochRSI K crosses below D from above 70
7. Volume > SMA(20)
8. RSI(14) > 25
9. Cooldown respected

### Exit Rules

| Exit Type | Condition | Priority |
|-----------|-----------|----------|
| **Stop Loss** | 2.0 x ATR(14) from entry | Highest — non-negotiable |
| **Take Profit 1** | 1.5 x ATR(14) — close 50% position | — |
| **Take Profit 2** | 3.0 x ATR(14) — close remaining | — |
| **Trailing Stop** | Supertrend line (activates after TP1) | — |
| **Trend Death** | ADX drops below 18 | Exit all |
| **Time Stop** | 20 candles (5h) without TP1 | Weak setup, cut |

### What Will Go Wrong

- **Whipsaws:** ~25-35% of entries will be false breakouts that stop out. Normal. The 2:1+ R:R compensates.
- **Late entries:** We miss first 30-40% of every trend. Accepted.
- **Black swans:** Blow through stops. Position sizing (1% risk) is the only real defense.
- **Regime lag:** Sometimes enter a "trend" as it's ending. Time stop limits damage.

---

## VI. STRATEGY 2: MEAN REVERSION

**File:** `strategies/core/mean_reversion.py`
**Active when:** Regime = RANGING, confidence >= 0.6
**Expected stats:** 70-80% win rate (highest of all strategies), 1.5:1 R:R, ~4-6 trades/month

Mean reversion in confirmed ranges is the highest win-rate strategy because the BB middle acts as a statistical magnet. Price reverts to the mean ~68% of the time within 2 standard deviations — and we add multiple confirmation layers on top.

### Indicators

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| Bollinger Bands | Period 20, StdDev 2.0 | Range boundaries |
| RSI(14) | Period 14 | Overbought/oversold |
| MACD | (12, 26, 9) | Momentum shift confirmation |
| Volume SMA(20) | Period 20 | Volume confirmation |
| ATR(14) | Period 14 | Stop/TP calculation |
| EMA(200) | Period 200 | Major trend guard |

### Entry Conditions — LONG (ALL must be true)

**Layer 1 (Regime):**
1. Regime = RANGING, confidence >= 0.6

**Layer 2 (Multi-TF):**
2. 1H regime is also RANGING or TRANSITION (not actively trending against us)

**Layer 3 (Confluence):**
3. Close <= BB Lower x 1.001 (at or below lower band)
4. RSI(14) < 32 (oversold)
5. MACD histogram turning positive (histogram > histogram[1] AND histogram[1] < 0)

**Layer 4 (Volume + Momentum):**
6. Volume > Volume SMA(20) x 1.1 (above-average volume on the bounce)
7. Current candle's close > current candle's open (bullish candle — momentum shifting)

**Layer 5 (Safety):**
8. Close > EMA(200) OR EMA(200) slope is flat (not fighting a major downtrend)
9. Cooldown respected

### Entry Conditions — SHORT (mirror)

1. Regime: RANGING, confidence >= 0.6
2. 1H confirms ranging
3. Close >= BB Upper x 0.999
4. RSI(14) > 68
5. MACD histogram turning negative
6. Volume > SMA(20) x 1.1
7. Bearish candle (close < open)
8. Close < EMA(200) OR flat EMA(200)
9. Cooldown respected

### Exit Rules

| Exit Type | Condition |
|-----------|-----------|
| **Stop Loss** | 1.5 x ATR(14) from entry |
| **Take Profit** | BB Middle (SMA 20) — the mean |
| **Extended TP** | Opposite BB, trail stop to middle BB |
| **Time Stop** | 12 candles (3 hours) |
| **Regime Change** | Regime shifts to TRENDING → immediate exit |

### What Will Go Wrong

- **Range breakout:** The #1 killer. Range becomes a trend, our long gets crushed. Regime change exit + stop loss limit damage, but lag exists.
- **False bounces:** Price touches BB, bounces, then continues falling. The MACD turn + bullish candle filter reduces this.

---

## VII. STRATEGY 3: FUNDING RATE ARBITRAGE

**File:** `strategies/core/funding_rate.py`
**File:** `strategies/data/funding_provider.py`
**Active when:** ALL regimes (supplemental income stream)
**Expected stats:** 60-70% win rate, 1:1 R:R, ~2-4 trades/month, 10-20% APY contribution

### How It Works

Binance Futures funding rate: every 8 hours, one side pays the other. Extreme funding = crowded positioning = mean reversion pressure.

### The Death Traps (Built Into Design)

**Trap 1: API Rate Limits**
- During volatility, everyone polls funding endpoints. You WILL get 429 errors when you need data most.
- **Mitigation:** Pre-fetch on 5-minute timer (NOT on-demand). 15-minute stale threshold. If stale, strategy doesn't fire.

**Trap 2: Extreme Funding = Extreme Volatility**
- When funding hits 0.05%, price is going parabolic. Your stop WILL get hit.
- **Mitigation:** Only enter at TRULY extreme levels (>0.08% or <-0.05%). 3x ATR stops (wider). Half-size positions (0.5% risk).

**Trap 3: Crowded Trade**
- Every quant fund runs this. The edge is partially arbitraged by the time you see it.
- **Mitigation:** Supplemental strategy only, not primary. If unprofitable after 3 months, disable.

### Data Provider (`strategies/data/funding_provider.py`)

```python
class FundingDataProvider:
    """
    Binance Futures API — public data, no API key needed.

    Endpoints:
    - GET /fapi/v1/premiumIndex — Current funding rate
    - GET /futures/data/topLongShortPositionRatio — L/S ratio

    Pre-fetch: 5-minute timer (NOT per candle)
    Cache TTL: 5 minutes
    Stale threshold: 15 minutes → return None
    Rate limit: Exponential backoff (1s, 2s, 4s), max 3 retries
    """
```

### Entry Conditions — LONG

1. Funding rate < -0.05% (truly extreme short crowding)
2. RSI(14) < 35 (price confirms oversold)
3. Long/Short ratio < 0.7 (crowd very short)
4. Volume > Volume SMA(20)
5. Regime != VOLATILE (don't fight chaos)

### Entry Conditions — SHORT

1. Funding rate > 0.08% (truly extreme long crowding)
2. RSI(14) > 65 (price confirms overbought)
3. Long/Short ratio > 1.8 (crowd very long)
4. Volume > Volume SMA(20)
5. Regime != VOLATILE

### Exit Rules

| Exit Type | Condition |
|-----------|-----------|
| **Stop Loss** | 3.0 x ATR(14) — wider (funding trades are slower) |
| **Take Profit** | Funding normalizes (-0.01% to +0.01%) |
| **Min Hold** | 1 funding period (8 hours = 32 candles) |
| **Max Hold** | 48 hours |

---

## VIII. RISK MANAGEMENT

**File:** `strategies/risk/risk_manager.py`

This matters more than all strategies combined.

### Position Sizing

```python
def calculate_position_size(account_balance, risk_per_trade, stop_distance):
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / stop_distance
    return min(position_size, account_balance * 0.33)
```

### Risk Rules (NON-NEGOTIABLE)

| Rule | Value | Why |
|------|-------|-----|
| Max risk per trade | 1% of account | Survive 15+ consecutive losses |
| Max open trades | 3 | Combined worst case ~5-7% |
| Max leverage | 3x | Higher = eventual death |
| Max daily drawdown | 3% | Stop trading today |
| Max total drawdown | 15% | Full system review |
| Cooldown after loss | 4 candles (1 hour) | No revenge trading |
| Pair lockout | 3 consecutive stops | 16 candles (4h) lockout |
| Funding trades | 0.5% risk (half size) | Fighting momentum = extra risk |

### Confidence-Scaled Sizing

```
Regime confidence >= 0.8:   1.0% risk per trade
Regime confidence 0.6-0.8:  0.5% risk per trade
Regime confidence < 0.6:    NO TRADE
Funding rate trades:        Always 0.5% max
```

### Circuit Breakers

```python
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
```

### Fee Budget

```
Round-trip cost: ~0.10 - 0.20% per trade
During high volatility: costs can double (slippage alone hits 0.10%)
Minimum edge needed: 0.20% per trade to break even
With our selective approach (13 trades/month), each trade needs to
carry its weight — no "spray and pray" to make up for losses in volume.
```

---

## IX. PRODUCTION HARDENING

### Error Handling

```python
# Every external API call:
# 1. Timeout: 10s data, 30s orders
# 2. Retry: exponential backoff (1s, 2s, 4s), max 3 attempts
# 3. Graceful degradation: fetch fails → skip signal, don't crash
# 4. Logging: every failure with timestamp, error, context
```

### Exchange Issues

| Issue | Mitigation |
|-------|------------|
| Binance maintenance | Detect via status endpoint, pause trading |
| API rate limits | `enableRateLimit: true` + backoff |
| Order rejection | Log, retry once with adjusted price, skip |
| Partial fills | Track actual qty, don't assume full |
| Price precision | Use exchange-provided precision from ccxt |
| Funding API throttle | Pre-fetch timer, 5-min cache, 15-min stale |

### Telegram Notifications

- Every trade entry/exit (with P&L)
- Daily P&L summary
- Circuit breaker activation
- API errors (grouped, not spammy)
- Regime changes
- Bot startup/shutdown
- Weekly summary: win rate, profit factor, drawdown, trade count by strategy

### Logging

```
INFO:    Trades, regime changes, daily summary
WARNING: API retries, partial fills, circuit breakers
ERROR:   API failures, order rejections, exceptions
DEBUG:   Indicator values, signal generation (dev only)
Rotation: Daily files, 30-day retention
```

---

## X. PROJECT STRUCTURE

```
NexusAlpha/
├── docker-compose.yml         # Freqtrade + FreqUI
├── Dockerfile                 # Base image + deps
├── requirements.txt           # Lean deps (no ML in Phase 1)
├── Makefile                   # make backtest, make trade, etc.
├── .env.example
├── .gitignore
├── config/
│   ├── config.json            # Base (dry-run)
│   ├── config-dry-run.json    # Paper trading
│   └── config-live.json       # Live (env vars for secrets)
├── strategies/
│   ├── NexusAlpha.py          # Main IStrategy — orchestrates everything
│   ├── core/
│   │   ├── __init__.py
│   │   ├── regime_detector.py # Regime classification
│   │   ├── trend_following.py # Strategy 1
│   │   ├── mean_reversion.py  # Strategy 2
│   │   └── funding_rate.py    # Strategy 3
│   ├── data/
│   │   ├── __init__.py
│   │   └── funding_provider.py # Binance funding API
│   ├── risk/
│   │   ├── __init__.py
│   │   └── risk_manager.py    # Position sizing + breakers
│   └── ml/                    # Phase 2 — empty for now
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── test_regime_detector.py
│   ├── test_trend_following.py
│   ├── test_mean_reversion.py
│   ├── test_funding_rate.py
│   └── test_risk_manager.py
├── scripts/
│   ├── download_data.sh
│   └── run_backtest.sh
├── notebooks/                 # .gitkeep
├── models/                    # .gitkeep (Phase 2)
└── logs/                      # .gitkeep + logs/signals/ for feature logging
```

---

## XI. FREQTRADE CONFIGURATION

### Base Config (`config/config.json`)

```json
{
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.95,
    "dry_run": true,
    "dry_run_wallet": 10000,
    "exchange": {
        "name": "binance",
        "ccxt_config": {
            "enableRateLimit": true,
            "options": {"defaultType": "future"}
        },
        "pair_whitelist": ["BTC/USDT:USDT"],
        "pair_blacklist": []
    },
    "entry_pricing": {
        "price_side": "other",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exit_pricing": {
        "price_side": "other",
        "use_order_book": true,
        "order_book_top": 1
    },
    "order_types": {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": true
    }
}
```

### NexusAlpha.py — Main Strategy

```python
class NexusAlpha(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '15m'
    informative_timeframes = ['1h']
    can_short = True
    minimal_roi = {"0": 100}        # Disabled — custom exits only
    stoploss = -0.05                # Fallback — custom_stoploss overrides
    trailing_stop = False
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 210

    def populate_indicators(self, dataframe, metadata):
        # 1. Compute ALL indicators
        # 2. Run regime detector
        # 3. Merge 1H informative data
        # 4. Fetch funding rate (from pre-fetched cache)

    def populate_entry_trend(self, dataframe, metadata):
        # Layer 1: Check regime + confidence >= 0.6
        # Layer 2: Verify 1H confirmation
        # Layer 3-4: Run strategy entry conditions (confluence + volume)
        # Tag entry with strategy name

    def populate_exit_trend(self, dataframe, metadata):
        # Strategy-specific exits based on enter_tag

    def custom_stoploss(self, ...):
        # Trend: 2x ATR | Mean Reversion: 1.5x ATR | Funding: 3x ATR

    def confirm_trade_entry(self, ...):
        # Layer 5: Risk checks (drawdown, position count, cooldown)
        # Log signal features for Phase 2 ML training data
        # PHASE 2: ML confidence filter added here

    def custom_stake_amount(self, ...):
        # Fixed fractional + confidence scaling
        # Funding trades: always half size

    def leverage(self, ...):
        return min(3.0, max_leverage)
```

---

## XII. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1)
- [x] Project structure, Docker, configs
- [ ] Regime detector with unit tests (including TRANSITION zone)
- [ ] Risk manager with unit tests
- [ ] Trend following with unit tests
- [ ] Mean reversion with unit tests

### Phase 2: Complete Core (Week 2)
- [ ] Funding rate data provider (with caching + rate limit handling)
- [ ] Funding rate strategy with unit tests
- [ ] Wire everything in NexusAlpha.py (all 5 filtering layers)
- [ ] Signal feature logging (for future ML)
- [ ] Integration tests

### Phase 3: Validate (Week 3)
- [ ] Download 18+ months BTC/USDT 15m data
- [ ] Backtest each strategy individually
- [ ] Backtest combined system
- [ ] Record baseline: win rate, Sharpe, drawdown, trades/month
- [ ] Verify win rate is 65%+ (if not, tighten filters)
- [ ] Hyperopt (SharpeHyperOptLoss, 500 epochs)
- [ ] Walk-forward validation (train 12mo, test 3mo, slide)
- [ ] Run again with 2x slippage to stress-test

### Phase 4: Paper Trade (Weeks 4-8)
- [ ] Deploy on VPS with Docker
- [ ] Paper trade minimum 4 weeks
- [ ] Track: actual win rate, slippage, fill rates, API reliability
- [ ] Compare to backtest (expect some degradation)
- [ ] DO NOT tweak during drawdowns — log observations only, review weekly

### Phase 5: Live (After 4+ weeks profitable paper)
- [ ] Start with $100-500 (money you can lose)
- [ ] Run paper + live in parallel for 2 weeks
- [ ] Scale up gradually over months

### Phase 6: ML Enhancement (After 3+ months live, 200+ trades — OPTIONAL)
- [ ] Review collected signal logs
- [ ] Train XGBoost on LIVE data only
- [ ] A/B test: ML-filtered vs rule-based
- [ ] Deploy ONLY if win rate improves by > 5%
- [ ] If ML doesn't help, delete it

---

## XIII. BACKTESTING PROTOCOL

### Rules

1. **No peeking:** Never optimize on validation data
2. **Walk-forward:** Train 12mo, test 3mo, slide
3. **Realistic fills:** Limit orders only
4. **Fees:** Always on (0.04% maker, 0.06% taker)
5. **Slippage:** 0.02% normal, also test with 0.05%
6. **Startup:** 210 candle warmup excluded
7. **Conditions:** Must include bull + bear + ranging periods

### Metrics

| Metric | Minimum | Target | Red Flag |
|--------|---------|--------|----------|
| Win rate | > 60% | 70-75% | > 85% (overfitting) |
| Profit factor | > 1.5 | > 2.0 | > 4.0 |
| Sharpe ratio | > 1.0 | > 1.5 | > 3.0 |
| Max drawdown | < 20% | < 12% | < 3% (look-ahead) |
| Trades/month | > 8 | 10-15 | > 30 (not selective enough) |
| Avg profit/trade | > 0.3% | 0.5-1.0% | > 2% |

**Red Flags:**
- Win rate > 85% → Overfitting or look-ahead bias
- Sharpe > 3.0 → Too good to be true
- No losing months → Definitely overfitting
- > 30 trades/month → Filters aren't selective enough (tighten them)
- Results change wildly with small param tweaks → Curve fitting

---

## XIV. REALISTIC EXPECTATIONS

### The Timeline

| Period | What Happens |
|--------|-------------|
| Month 1 | Build + backtest. Backtests look great. |
| Month 2-3 | Paper trade. Real fills worse than backtest. Regime detector lags. Some strategies underperform. You adjust. |
| Month 4 | Go live small. First week up, second week down. The urge to override is intense. DON'T. |
| Month 5-8 | System stabilizes. Win rate settles at 65-75%. Monthly returns fluctuate: some months +15-25%, some -5%. |
| Month 9-12 | If edge holds: averaging 10-20% monthly on good months, flat or small loss on bad months. Compounding starts to feel real. |
| Year 2+ | Mature system. You've survived drawdowns, fixed production bugs, and the compounding starts working. |

### Capital Math (Conservative)

```
Starting capital: $1,000 — Average 10% monthly net (mix of good and bad months)

Month 3:   $1,331
Month 6:   $1,772
Month 12:  $3,138
Month 24:  $9,850
Month 36:  $30,913

Starting capital: $5,000 — Same 10% average monthly:

Month 12:  $15,692
Month 24:  $49,248
Month 36:  $154,567
```

**Caveats:**
- "Average 10% monthly" means some months +25%, some months -8%. It's NOT consistent.
- You'll have 2-4 losing months per year. Don't panic.
- Drawdowns of 10-15% will happen. That's 2-3 months of gains wiped out. Recovery takes time.
- The bot won't replace your income in Year 1 unless you start with significant capital.
- Compounding only works if you DON'T withdraw profits early.

### What Will NOT Happen

- 70-80% win rate on day 1 (expect 50-60% initially, improving as you tune)
- Consistent returns every month (expect high variance)
- "Set and forget" (weekly monitoring minimum)
- Working perfectly from deployment (3-6 months of tuning expected)

### The Solo Developer Reality

- You have a day job. The bot runs 24/7 but you don't.
- 3 AM API failures: you'll find out at 7 AM. That's OK — stops are on-exchange.
- After a losing streak, you'll want to tinker. Write observations, review weekly ONLY.
- The psychological challenge of watching money evaporate is real. Having a written rule ("I will not override the bot") helps.

---

## XV. PHASE 2: ML META-LABELING (FUTURE)

> **DO NOT BUILD IN PHASE 1.**

### Feature Logging (Built Into Phase 1)

Log for every signal (taken AND rejected):

```python
FEATURES_TO_LOG = [
    'timestamp', 'schema_version',
    'rsi_14', 'macd_histogram', 'bb_percent_b', 'adx_14',
    'supertrend_direction', 'stochrsi_k', 'atr_ratio',
    'volume_ratio',
    'regime', 'regime_confidence',
    'funding_rate', 'long_short_ratio',
    'return_5', 'return_15', 'return_60',
    'volatility_20',
    'hour_of_day', 'day_of_week',
    'strategy_name', 'entry_side', 'entry_price',
    'signal_taken',
    # After trade closes:
    'profit_ratio', 'trade_duration_minutes', 'exit_reason'
]
SCHEMA_VERSION = 1
```

- Daily CSV: `logs/signals/signals_YYYY-MM-DD.csv`
- Include schema_version in every row
- Null for unavailable features (don't skip rows)
- Log BOTH taken and rejected signals

### ML (After 200+ Trades)

Single XGBoost model. Binary classification. Purged walk-forward CV.
Deploy only if win rate improves >5%. Kill it if it doesn't.

---

## XVI. DEPLOYMENT CHECKLIST

### VPS ($10/mo)
- [ ] 4 cores, 8GB RAM, 200GB SSD
- [ ] Ubuntu 22.04 LTS, Docker, Fail2ban, UFW

### Pre-Live
- [ ] Paper traded 4+ weeks profitably
- [ ] Telegram alerts working
- [ ] Stoploss-on-exchange enabled
- [ ] API key: trade + read ONLY (no withdrawal)
- [ ] Capital you can lose completely
- [ ] Written rule: "I will not override the bot"

---

## XVII. THE EDGE

This system's edge is NOT being smarter than the market. The edge is:

1. **Extreme selectivity** — 200 signals → 13 trades. Each one passed 6 filters.
2. **Regime awareness** — never running the wrong strategy in the wrong market.
3. **The TRANSITION zone** — sitting out when the market is ambiguous. Most bots trade through this and lose.
4. **Discipline** — 1% risk, 3x max leverage, circuit breakers. The bot doesn't panic, doesn't FOMO, doesn't revenge trade.
5. **24/7 compounding** — while you sleep and work your day job.

**The trades we don't take are the edge.**

---

*Built for accuracy. Built for survival. Built to compound.*
