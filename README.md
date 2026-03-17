# Nexus Alpha

High-selectivity crypto trading bot targeting 70-80% win rate through extreme signal filtering. Built on Freqtrade, trading BTC/USDT perpetual futures on 15-minute candles.

## Quick Start

```bash
# Build
docker-compose build

# Download historical data
make download-data

# Run backtest
make backtest

# Paper trade
make dry-run

# Live trade (after paper trading profitably for 4+ weeks)
make trade
```

## Architecture

```
200 raw signals/month → 6 filtering layers → ~13 high-conviction trades/month

Layer 1: Regime Gate (ADX + BB Width)        → kills ~60%
Layer 2: Multi-TF Confirmation (1H agrees)   → kills ~40%
Layer 3: Confluence (3+ indicators align)     → kills ~50%
Layer 4: Volume + Momentum                   → kills ~30%
Layer 5: Risk Gate (drawdown, cooldown)       → kills ~20%
Layer 6: ML Confidence (Phase 2, optional)    → future
```

## Strategies

| # | Strategy | Active When | Win Rate Target | Trades/Month |
|---|----------|-------------|-----------------|--------------|
| 1 | Trend Following | TRENDING regime | 65-75% | 5-7 |
| 2 | Mean Reversion | RANGING regime | 70-80% | 4-6 |
| 3 | Funding Rate Arb | All regimes | 60-70% | 2-4 |

## Regime Detection

5 regimes with **adaptive thresholds** (ADX baseline tracks via 100-period SMA):

- **TRENDING_BULL** (ADX > adaptive trend threshold, +DI > -DI) → Trend Following long
- **TRENDING_BEAR** (ADX > adaptive trend threshold, -DI > +DI) → Trend Following short
- **RANGING** (ADX < adaptive range threshold, narrow BB) → Mean Reversion
- **VOLATILE** (ATR spike > 2x SMA, BB explosion > 2.5x SMA) → Funding Rate only
- **TRANSITION** (between range and trend thresholds) → NO TRADING (sit out)

Thresholds adapt to current market volatility instead of being fixed at 18/28.

## Risk Rules (Non-Negotiable)

- Max 1% risk per trade (0.5% for funding trades)
- Max 3 open trades
- Max 3x leverage
- 3% daily drawdown circuit breaker
- 15% total drawdown → full system review
- 4-candle cooldown after every loss
- Stoploss always on exchange

## Tech Stack

- **Framework:** Freqtrade (Python)
- **Exchange:** Binance Futures
- **Pair:** BTC/USDT:USDT (one pair only)
- **Timeframes:** 15m primary, 1H confirmation
- **Deployment:** Docker on VPS (~$10/mo)
- **Monitoring:** FreqUI (port 8081) + Telegram

## Project Structure

```
NexusAlpha/
├── config/
│   ├── config.json            # Base (dry-run)
│   ├── config-dry-run.json    # Paper trading
│   └── config-live.json       # Live (env vars)
├── strategies/
│   ├── NexusAlpha.py          # Main IStrategy
│   ├── core/
│   │   ├── regime_detector.py # Regime classification
│   │   ├── trend_following.py # Strategy 1
│   │   ├── mean_reversion.py  # Strategy 2
│   │   └── funding_rate.py    # Strategy 3
│   ├── data/
│   │   └── funding_provider.py # Binance funding API
│   ├── risk/
│   │   └── risk_manager.py    # Position sizing + breakers
│   └── ml/                    # Phase 2
├── tests/
├── scripts/
├── logs/                      # + logs/signals/ for ML feature data
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Implementation Status

- [x] Phase 1: Project skeleton, Docker, configs
- [ ] Phase 2: Regime detector
- [ ] Phase 3: Risk manager
- [ ] Phase 4: Trend following strategy
- [ ] Phase 5: Mean reversion strategy
- [ ] Phase 6: Funding rate data provider + strategy
- [ ] Phase 7: Wire NexusAlpha.py (all 5 filtering layers)
- [ ] Phase 8: Unit + integration tests
- [ ] Phase 9: Backtest + tune
- [ ] Phase 10: Paper trade (4+ weeks)
- [ ] Phase 11: Live (small capital)
- [ ] Phase 12: ML enhancement (optional, after 3+ months live)

## Key Design Decisions

1. **70-80% win rate via selectivity, not prediction.** We filter hard, not predict better.
2. **Adaptive regime thresholds.** ADX thresholds track a 100-period baseline, not fixed values.
3. **TRANSITION regime = no trading.** Ambiguous ADX zone produces zero trades.
4. **3 strategies, not 5.** A solo dev can maintain 2-3 well.
5. **ML is Phase 2.** Rule-based filtering achieves 70-80% without ML.
6. **Funding rate is supplemental.** Half-size, wider stops, extreme thresholds only.
7. **Dynamic ATR stops.** Stops widen in high-vol, position shrinks proportionally — same dollar risk.
8. **Docker health checks + auto-restart.** Container recovers from crashes; stops live on-exchange.
9. **Integration tests.** Full pipeline tested end-to-end, not just individual strategies.
10. **Feature logging from day 1.** Versioned schema for future ML training data.

## Full Specification

See `prompt.md` in the project root — the complete engineering spec for the entire system.

## Make Targets

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make up` | Start all services |
| `make down` | Stop all services |
| `make dry-run` | Paper trading |
| `make trade` | Live trading |
| `make backtest` | Run backtest |
| `make hyperopt` | Parameter optimization |
| `make download-data` | Download historical data |
| `make test` | Run tests |
| `make logs` | View live logs |
| `make shell` | Container shell |
| `make status` | Check services |
