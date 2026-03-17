#!/bin/bash
# Download historical data for backtesting
# Usage: ./scripts/download_data.sh

echo "=== Nexus Alpha — Downloading Historical Data ==="
echo "Exchange: Binance Futures"
echo "Pairs: BTC/USDT"
echo "Timeframes: 15m, 30m, 1h, 4h"
echo "Range: 2023-09-01 to present"
echo ""

docker-compose run --rm nexus-alpha download-data \
    --exchange binance \
    --pairs BTC/USDT:USDT \
    --timeframe 15m 30m 1h 4h \
    --timerange 20230901- \
    --trading-mode futures

echo ""
echo "=== Download Complete ==="
echo "Data stored in user_data/data/binance/"
