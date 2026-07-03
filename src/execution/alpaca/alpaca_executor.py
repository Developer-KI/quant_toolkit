"""
execution/alpaca/alpaca_executor.py — Alpaca Markets paper-trading executor.

Implements BaseExecutor using alpaca-py TradingClient.
Targets US equities on the Alpaca paper trading endpoint.

Funding rates and leverage are not applicable — fetch_funding_rate returns
None and set_leverage is a no-op.

Install:
    pip install alpaca-py
"""

from __future__ import annotations

import logging

import pandas as pd

from core.models import FillResult, FundingSnapshot, Position, Side
from execution.executor import BaseExecutor

logger = logging.getLogger(__name__)


class AlpacaExecutor(BaseExecutor):
    """Alpaca Markets executor (paper trading, US equities)."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: alpaca-py. Install with: pip install alpaca-py"
            ) from exc

        self._client = TradingClient(api_key=api_key, secret_key=api_secret, paper=paper)
        self._data = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
        logger.info("AlpacaExecutor ready (paper=%s)", paper)

    @property
    def exchange_name(self) -> str:
        return "alpaca"

    # ── Account state ────────────────────────────────────────────────────

    def get_equity(self) -> float:
        account = self._client.get_account()
        return float(account.portfolio_value)

    def get_position(self, symbol: str) -> Position:
        try:
            pos = self._client.get_open_position(symbol)
            side = Side.LONG if pos.side.value == "long" else Side.SHORT
            return Position(
                side=side,
                size=float(pos.qty),
                entry_price=float(pos.avg_entry_price),
            )
        except Exception:
            return Position()

    def get_mid_price(self, symbol: str) -> float:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._data.get_stock_latest_quote(req)
            q = quotes[symbol]
            return (float(q.bid_price) + float(q.ask_price)) / 2.0
        except Exception as exc:
            logger.warning("get_mid_price(%s) failed: %s", symbol, exc)
            return 0.0

    def get_open_orders(self, symbol: str) -> list[dict]:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = self._client.get_orders(filter=req)
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "qty": float(o.qty or 0),
                    "side": o.side.value,
                }
                for o in orders
            ]
        except Exception as exc:
            logger.warning("get_open_orders(%s) failed: %s", symbol, exc)
            return []

    # ── Order placement ──────────────────────────────────────────────────

    def market_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        reduce_only: bool = False,
    ) -> FillResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        alpaca_side = OrderSide.BUY if side == Side.LONG else OrderSide.SELL
        # Alpaca only supports fractional shares for BUY market orders;
        # SELL (close or short) orders must use whole share quantities.
        qty = size if alpaca_side == OrderSide.BUY else int(size)
        if qty <= 0:
            return FillResult(success=False, status="size rounded to zero", exchange=self.exchange_name)
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            return FillResult(
                success=True,
                fill_price=float(order.filled_avg_price or 0),
                filled_size=float(order.filled_qty or size),
                order_id=str(order.id),
                status=order.status.value,
                exchange=self.exchange_name,
            )
        except Exception as exc:
            logger.error("Alpaca market_order failed: %s", exc)
            return FillResult(success=False, status=f"error: {exc}", exchange=self.exchange_name)

    def limit_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        price: float,
        reduce_only: bool = False,
    ) -> FillResult:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        alpaca_side = OrderSide.BUY if side == Side.LONG else OrderSide.SELL
        qty = size if alpaca_side == OrderSide.BUY else int(size)
        if qty <= 0:
            return FillResult(success=False, status="size rounded to zero", exchange=self.exchange_name)
        try:
            req = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=price,
            )
            order = self._client.submit_order(req)
            return FillResult(
                success=True,
                fill_price=price,
                filled_size=float(order.filled_qty or 0),
                order_id=str(order.id),
                status=order.status.value,
                exchange=self.exchange_name,
            )
        except Exception as exc:
            logger.error("Alpaca limit_order failed: %s", exc)
            return FillResult(success=False, status=f"error: {exc}", exchange=self.exchange_name)

    def cancel_all(self, symbol: str) -> int:
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = self._client.get_orders(filter=req)
            for o in orders:
                self._client.cancel_order_by_id(str(o.id))
            return len(orders)
        except Exception as exc:
            logger.warning("cancel_all(%s) failed: %s", symbol, exc)
            return 0

    def close_position(self, symbol: str) -> FillResult:
        try:
            self._client.close_position(symbol)
            return FillResult(success=True, status="closed", exchange=self.exchange_name)
        except Exception as exc:
            logger.warning("close_position(%s) failed: %s", symbol, exc)
            return FillResult(success=False, status=f"error: {exc}", exchange=self.exchange_name)

    def close_all_positions(self, cancel_orders: bool = True) -> int:
        try:
            responses = self._client.close_all_positions(cancel_orders=cancel_orders)
            count = len(responses) if responses else 0
            logger.info("close_all_positions: submitted close orders for %d position(s)", count)
            return count
        except Exception as exc:
            logger.error("close_all_positions failed: %s", exc)
            return 0

    # ── Optional ─────────────────────────────────────────────────────────

    def fetch_historical_candles(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[dict]:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import DataFeed
            # End 5 min before now so recently unindexed bars don't cause a miss
            end_ts = pd.Timestamp(end_ms, unit="ms", tz="UTC") - pd.Timedelta(minutes=5)
            requested_start = pd.Timestamp(start_ms, unit="ms", tz="UTC")

            # US equities trade ~390 min/day. The requested window may fall entirely
            # outside market hours (pre-open, overnight, weekend). Retry with a wider
            # window (5x, then 14x) to ensure we cross into the previous session.
            df = pd.DataFrame()
            for factor in (1, 5, 14):
                start_ts = end_ts - (end_ts - requested_start) * factor
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=start_ts,
                    end=end_ts,
                    feed=DataFeed.IEX,
                )
                bars = self._data.get_stock_bars(req)
                df = bars.df
                if not df.empty:
                    break

            if df.empty:
                return []
            if isinstance(df.index, pd.MultiIndex):
                if symbol not in df.index.get_level_values("symbol"):
                    return []
                df = df.xs(symbol, level="symbol")
            df = df.sort_index()
            return [
                {
                    "timestamp": pd.Timestamp(row.Index),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
                for row in df.itertuples()
            ]
        except Exception as exc:
            logger.error("fetch_historical_candles(%s) failed: %s", symbol, exc)
            return []

    def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        return None

    def set_leverage(self, symbol: str, leverage: int, cross: bool = True):
        pass
