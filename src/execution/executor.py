"""
execution/executor.py — Abstract executor interface.

Every exchange adapter subclasses BaseExecutor so the Engine stays
exchange-agnostic. The engine only ever touches this interface — it
never imports anything exchange-specific.

To add a new exchange:
  1. Subclass BaseExecutor → execution/{exchange}/executor.py
  2. Subclass BaseFeed     → data/feeds/{exchange}.py
  3. Register both in execution/factory.py
"""

from __future__ import annotations

import abc

import pandas as pd

from core.models import FillResult, FundingSnapshot, Position, Side


class BaseExecutor(abc.ABC):
    """Exchange order-execution layer. Every adapter must implement these methods."""

    @property
    @abc.abstractmethod
    def exchange_name(self) -> str:
        """Lowercase identifier: 'hyperliquid', 'binance', 'alpaca', …"""
        ...

    # ── Account state ─────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_equity(self) -> float:
        """Total account equity (margin + unrealized PnL)."""
        ...

    @abc.abstractmethod
    def get_position(self, symbol: str) -> Position:
        """Current position on *symbol* (FLAT if none)."""
        ...

    @abc.abstractmethod
    def get_mid_price(self, symbol: str) -> float:
        """Current mid price."""
        ...

    @abc.abstractmethod
    def get_open_orders(self, symbol: str) -> list[dict]:
        """All open orders on *symbol*."""
        ...

    # ── Order placement ───────────────────────────────────────────────────

    @abc.abstractmethod
    def market_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        reduce_only: bool = False,
    ) -> FillResult:
        ...

    @abc.abstractmethod
    def limit_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> FillResult:
        ...

    @abc.abstractmethod
    def cancel_all(self, symbol: str) -> int:
        """Cancel all open orders on *symbol*. Return count cancelled."""
        ...

    @abc.abstractmethod
    def close_position(self, symbol: str) -> FillResult:
        """Flatten any open position on *symbol*."""
        ...

    # ── Optional ──────────────────────────────────────────────────────────

    def close_all_positions(self, cancel_orders: bool = True) -> int:
        """
        Flatten every open position. Returns count of positions closed.
        Default returns 0 — override if the exchange has a native close-all endpoint.
        """
        return 0

    def set_leverage(self, symbol: str, leverage: int, cross: bool = True):
        """Set leverage (no-op for exchanges that don't support it)."""
        pass

    def fetch_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        """
        Fetch historical OHLCV candles for engine warm-up.
        Returns list of dicts with keys: timestamp, open, high, low, close, volume.
        """
        raise NotImplementedError(
            f"{self.exchange_name} does not implement fetch_historical_candles"
        )

    def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """Current funding rate for a perpetual, or None if unsupported."""
        return None
