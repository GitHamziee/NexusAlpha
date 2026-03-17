.PHONY: build up down trade dry-run backtest hyperopt download-data test logs shell status

# Build Docker image
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Stop all services
down:
	docker-compose down

# Run paper trading (dry-run)
dry-run:
	docker-compose run --rm nexus-alpha trade \
		--strategy NexusAlpha \
		--strategy-path /freqtrade/strategies \
		--config /freqtrade/config/config-dry-run.json

# Run live trading
trade:
	docker-compose run --rm nexus-alpha trade \
		--strategy NexusAlpha \
		--strategy-path /freqtrade/strategies \
		--config /freqtrade/config/config-live.json

# Run backtest
backtest:
	docker-compose run --rm nexus-alpha backtesting \
		--strategy NexusAlpha \
		--strategy-path /freqtrade/strategies \
		--config /freqtrade/config/config.json \
		--timeframe 15m \
		--timerange 20240601-20250301 \
		--export trades

# Run hyperopt optimization
hyperopt:
	docker-compose run --rm nexus-alpha hyperopt \
		--strategy NexusAlpha \
		--strategy-path /freqtrade/strategies \
		--config /freqtrade/config/config.json \
		--hyperopt-loss SharpeHyperOptLoss \
		--timeframe 15m \
		--timerange 20240601-20250101 \
		--epochs 500 \
		--spaces buy sell roi stoploss \
		--random-state 42

# Download historical data
download-data:
	docker-compose run --rm nexus-alpha download-data \
		--exchange binance \
		--pairs BTC/USDT:USDT \
		--timeframe 15m 30m 1h 4h \
		--timerange 20230901- \
		--trading-mode futures

# Run tests
test:
	docker-compose run --rm nexus-alpha \
		bash -c "pip install pytest pytest-cov && pytest /freqtrade/strategies/tests/ -v"

# View logs
logs:
	docker-compose logs -f nexus-alpha

# Open shell in container
shell:
	docker-compose run --rm nexus-alpha bash

# Check bot status
status:
	docker-compose ps
