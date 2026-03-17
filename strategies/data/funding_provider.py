"""
Funding Rate Data Provider — Binance Futures public API.

Pre-fetches funding rate and long/short ratio on a timer, NOT per-candle.
Caches results with a 2-minute TTL.  If data is staler than 10 minutes,
returns None so the funding strategy doesn't fire on stale data.

All endpoints are public (no API key needed).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── API endpoints ────────────────────────────────────────────────────────
PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
LS_RATIO_URL = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"

# ── timing ───────────────────────────────────────────────────────────────
FETCH_INTERVAL_SECONDS = 120   # 2 minutes
CACHE_TTL_SECONDS = 120        # 2 minutes
STALE_THRESHOLD_SECONDS = 600  # 10 minutes
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 1       # exponential: 1s, 2s, 4s


class FundingData:
    """Immutable snapshot of funding data."""
    __slots__ = ("funding_rate", "long_short_ratio", "timestamp")

    def __init__(self, funding_rate: float, long_short_ratio: float, timestamp: float):
        self.funding_rate = funding_rate
        self.long_short_ratio = long_short_ratio
        self.timestamp = timestamp


class FundingDataProvider:
    """Thread-safe, timer-based funding data fetcher with caching.

    Usage:
        provider = FundingDataProvider("BTCUSDT")
        provider.start()      # begins background fetch loop
        ...
        data = provider.get() # returns FundingData or None
        ...
        provider.stop()
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self._symbol = symbol
        self._data: Optional[FundingData] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background fetch thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="funding-fetcher")
        self._thread.start()
        logger.info("Funding data provider started for %s", self._symbol)

    def stop(self) -> None:
        """Stop background fetch thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Funding data provider stopped")

    def get(self) -> Optional[FundingData]:
        """Return cached funding data, or None if stale/unavailable."""
        with self._lock:
            if self._data is None:
                return None
            age = time.time() - self._data.timestamp
            if age > STALE_THRESHOLD_SECONDS:
                logger.warning("Funding data stale (%.0fs old), returning None", age)
                return None
            return self._data

    def fetch_once(self) -> Optional[FundingData]:
        """Single fetch attempt (useful for testing without threading)."""
        return self._fetch()

    # ── internals ────────────────────────────────────────────────────

    def _run(self) -> None:
        """Background loop: fetch, sleep, repeat."""
        while not self._stop_event.is_set():
            data = self._fetch()
            if data:
                with self._lock:
                    self._data = data
            self._stop_event.wait(timeout=FETCH_INTERVAL_SECONDS)

    def _fetch(self) -> Optional[FundingData]:
        """Fetch funding rate + L/S ratio with exponential backoff."""
        funding_rate = self._fetch_funding_rate()
        ls_ratio = self._fetch_ls_ratio()

        if funding_rate is None:
            return None

        return FundingData(
            funding_rate=funding_rate,
            long_short_ratio=ls_ratio if ls_ratio is not None else 1.0,
            timestamp=time.time(),
        )

    def _fetch_funding_rate(self) -> Optional[float]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(
                    PREMIUM_INDEX_URL,
                    params={"symbol": self._symbol},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data["lastFundingRate"])
            except Exception as e:
                wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
                logger.warning("Funding rate fetch attempt %d failed: %s. Retrying in %ds",
                               attempt + 1, e, wait)
                time.sleep(wait)
        logger.error("Failed to fetch funding rate after %d attempts", MAX_RETRIES)
        return None

    def _fetch_ls_ratio(self) -> Optional[float]:
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(
                    LS_RATIO_URL,
                    params={"symbol": self._symbol, "period": "5m", "limit": 1},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
                if data and len(data) > 0:
                    return float(data[0]["longShortRatio"])
                return None
            except Exception as e:
                wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
                logger.warning("L/S ratio fetch attempt %d failed: %s. Retrying in %ds",
                               attempt + 1, e, wait)
                time.sleep(wait)
        logger.error("Failed to fetch L/S ratio after %d attempts", MAX_RETRIES)
        return None
