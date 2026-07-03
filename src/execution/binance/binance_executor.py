"""
execution/binance_executor.py — Binance USD-M Futures executor.

Implements BaseExecutor using the python-binance SDK (async under the hood,
but we expose only sync methods to match the engine contract).

Install:
    pip install python-binance

Usage:
    executor = BinanceExecutor(
        api_key="...",
        api_secret="...",
        testnet=True,
    )
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from core.models import Side, Position, FundingSnapshot
from core.models import FillResult
from ..executor import BaseExecutor

logger = logging.getLogger(__name__)


# ── Symbol info cache ────────────────────────────────────────────────────────

class _SymbolInfo:
    """Cached precision / filter rules for one Binance symbol."""
    __slots__ = ("qty_precision", "price_precision", "min_qty", "step_size", "tick_size")

    def __init__(self, qty_precision=3, price_precision=2, min_qty=0.001,
                 step_size=0.001, tick_size=0.01):
        self.qty_precision = qty_precision
        self.price_precision = price_precision
        self.min_qty = min_qty
        self.step_size = step_size
        self.tick_size = tick_size


class BinanceExecutor(BaseExecutor):
    """
    Binance USD-M Futures executor.

    Talks to Binance Futures via the python-binance UMFutures client.
    Supports both testnet and mainnet.

    Symbol mapping:
        The framework uses short names like "ETH", "BTC".
        Binance uses "ETHUSDT", "BTCUSDT", etc.
        Pass symbol_map={"ETH": "ETHUSDT", "BTC": "BTCUSDT"} to translate,
        or leave it empty and the executor appends "USDT" automatically.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        symbol_map: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._symbol_map = symbol_map or {}

        try:
            from binance.um_futures import UMFutures
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}. Install with: pip install binance-futures-connector"
            ) from e

        if testnet:
            base_url = "https://testnet.binancefuture.com"
        else:
            base_url = "https://fapi.binance.com"

        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url=base_url,
        )

        self._info_cache: dict[str, _SymbolInfo] = {}
        self._load_exchange_info()

        logger.info(
            "BinanceExecutor initialised (testnet=%s)", testnet
        )

    # ── Symbol translation ───────────────────────────────────────────────

    def _to_binance_symbol(self, symbol: str) -> str:
        """Convert framework symbol → Binance symbol (e.g. ETH → ETHUSDT)."""
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]
        # Default: append USDT
        if not symbol.endswith("USDT"):
            return f"{symbol}USDT"
        return symbol

    def _from_binance_symbol(self, bsym: str) -> str:
        """Convert Binance symbol → framework symbol."""
        # Check reverse map first
        for short, full in self._symbol_map.items():
            if full == bsym:
                return short
        # Strip USDT suffix
        if bsym.endswith("USDT"):
            return bsym[:-4]
        return bsym

    # ── Exchange info / precision ────────────────────────────────────────

    def _load_exchange_info(self):
        """Cache symbol precision and filter rules."""
        try:
            info = self.client.exchange_info()
            for s in info.get("symbols", []):
                sym = s["symbol"]
                qty_prec = int(s.get("quantityPrecision", 3))
                price_prec = int(s.get("pricePrecision", 2))

                min_qty = 0.001
                step_size = 0.001
                tick_size = 0.01

                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        min_qty = float(f.get("minQty", min_qty))
                        step_size = float(f.get("stepSize", step_size))
                    elif f["filterType"] == "PRICE_FILTER":
                        tick_size = float(f.get("tickSize", tick_size))

                self._info_cache[sym] = _SymbolInfo(
                    qty_precision=qty_prec,
                    price_precision=price_prec,
                    min_qty=min_qty,
                    step_size=step_size,
                    tick_size=tick_size,
                )
        except Exception as e:
            logger.warning("Failed to load Binance exchange info: %s", e)

    def _get_info(self, symbol: str) -> _SymbolInfo:
        bsym = self._to_binance_symbol(symbol)
        return self._info_cache.get(bsym, _SymbolInfo())

    def _round_size(self, size: float, symbol: str) -> float:
        info = self._get_info(symbol)
        # Round down to step_size
        if info.step_size > 0:
            size = math.floor(size / info.step_size) * info.step_size
        return round(size, info.qty_precision)

    def _round_price(self, price: float, symbol: str) -> float:
        info = self._get_info(symbol)
        if info.tick_size > 0:
            price = round(price / info.tick_size) * info.tick_size
        return round(price, info.price_precision)

    # ── BaseExecutor interface ───────────────────────────────────────────

    @property
    def exchange_name(self) -> str:
        return "binance"

    def get_equity(self) -> float:
        try:
            account = self.client.account()
            return float(account.get("totalWalletBalance", 0)) + \
                   float(account.get("totalUnrealizedProfit", 0))
        except Exception as e:
            logger.error("get_equity failed: %s", e)
            return 0.0

    def get_position(self, symbol: str) -> Position:
        bsym = self._to_binance_symbol(symbol)
        try:
            positions = self.client.get_position_risk(symbol=bsym)
            for p in positions:
                if p["symbol"] == bsym:
                    amt = float(p.get("positionAmt", 0))
                    entry = float(p.get("entryPrice", 0))
                    if amt > 0:
                        return Position(side=Side.LONG, size=abs(amt), entry_price=entry)
                    elif amt < 0:
                        return Position(side=Side.SHORT, size=abs(amt), entry_price=entry)
                    return Position()
        except Exception as e:
            logger.error("get_position(%s) failed: %s", symbol, e)
        return Position()

    def get_mid_price(self, symbol: str) -> float:
        bsym = self._to_binance_symbol(symbol)
        try:
            book = self.client.book_ticker(symbol=bsym)
            if isinstance(book, list):
                book = book[0]
            bid = float(book.get("bidPrice", 0))
            ask = float(book.get("askPrice", 0))
            return (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
        except Exception as e:
            logger.error("get_mid_price(%s) failed: %s", symbol, e)
            return 0.0

    def get_open_orders(self, symbol: str) -> list[dict]:
        bsym = self._to_binance_symbol(symbol)
        try:
            return self.client.get_orders(symbol=bsym)
        except Exception as e:
            logger.error("get_open_orders(%s) failed: %s", symbol, e)
            return []

    def market_order(self, symbol, side, size, reduce_only=False) -> FillResult:
        bsym = self._to_binance_symbol(symbol)
        size = self._round_size(size, symbol)
        info = self._get_info(symbol)

        if size < info.min_qty:
            return FillResult(
                success=False, status="below_min_qty",
                exchange=self.exchange_name,
            )

        bn_side = "BUY" if side == Side.LONG else "SELL"

        try:
            params = dict(
                symbol=bsym,
                side=bn_side,
                type="MARKET",
                quantity=size,
            )
            if reduce_only:
                params["reduceOnly"] = "true"

            result = self.client.new_order(**params)

            status = result.get("status", "")
            avg_price = float(result.get("avgPrice", 0))
            filled_qty = float(result.get("executedQty", 0))
            order_id = str(result.get("orderId", ""))

            if status == "FILLED":
                return FillResult(
                    success=True,
                    fill_price=avg_price,
                    filled_size=filled_qty,
                    order_id=order_id,
                    status="filled",
                    exchange=self.exchange_name,
                    raw=result,
                )
            elif status in ("NEW", "PARTIALLY_FILLED"):
                return FillResult(
                    success=True,
                    fill_price=avg_price,
                    filled_size=filled_qty,
                    order_id=order_id,
                    status="partial",
                    exchange=self.exchange_name,
                    raw=result,
                )
            return FillResult(
                success=False, status=status,
                exchange=self.exchange_name, raw=result,
            )

        except Exception as e:
            logger.error("Binance market order failed: %s", e)
            return FillResult(
                success=False, status=f"error: {e}",
                exchange=self.exchange_name,
            )

    def limit_order(self, symbol, side, size, price, reduce_only=False) -> FillResult:
        bsym = self._to_binance_symbol(symbol)
        size = self._round_size(size, symbol)
        price = self._round_price(price, symbol)
        info = self._get_info(symbol)

        if size < info.min_qty:
            return FillResult(
                success=False, status="below_min_qty",
                exchange=self.exchange_name,
            )

        bn_side = "BUY" if side == Side.LONG else "SELL"

        try:
            params = dict(
                symbol=bsym,
                side=bn_side,
                type="LIMIT",
                quantity=size,
                price=price,
                timeInForce="GTC",
            )
            if reduce_only:
                params["reduceOnly"] = "true"

            result = self.client.new_order(**params)
            status = result.get("status", "")
            order_id = str(result.get("orderId", ""))
            filled_qty = float(result.get("executedQty", 0))
            avg_price = float(result.get("avgPrice", 0))

            if status == "FILLED":
                return FillResult(
                    success=True,
                    fill_price=avg_price,
                    filled_size=filled_qty,
                    order_id=order_id,
                    status="filled",
                    exchange=self.exchange_name,
                    raw=result,
                )
            elif status in ("NEW", "PARTIALLY_FILLED"):
                return FillResult(
                    success=True,
                    fill_price=price,
                    filled_size=filled_qty,
                    order_id=order_id,
                    status="resting",
                    exchange=self.exchange_name,
                    raw=result,
                )
            return FillResult(
                success=False, status=status,
                exchange=self.exchange_name, raw=result,
            )
        except Exception as e:
            logger.error("Binance limit order failed: %s", e)
            return FillResult(
                success=False, status=f"error: {e}",
                exchange=self.exchange_name,
            )

    def cancel_all(self, symbol: str) -> int:
        bsym = self._to_binance_symbol(symbol)
        try:
            result = self.client.cancel_open_orders(symbol=bsym)
            cancelled = len(result) if isinstance(result, list) else 1
            logger.info("Cancelled %d orders on %s", cancelled, symbol)
            return cancelled
        except Exception as e:
            # Binance returns error if no open orders — that's fine
            if "Unknown order" not in str(e) and "-2011" not in str(e):
                logger.error("cancel_all(%s) failed: %s", symbol, e)
            return 0

    def close_position(self, symbol: str) -> FillResult:
        pos = self.get_position(symbol)
        if pos.side == Side.FLAT or pos.size == 0:
            return FillResult(
                success=True, status="already_flat",
                exchange=self.exchange_name,
            )
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        return self.market_order(symbol, close_side, pos.size, reduce_only=True)

    def set_leverage(self, symbol: str, leverage: int, cross: bool = True):
        bsym = self._to_binance_symbol(symbol)
        try:
            self.client.change_leverage(symbol=bsym, leverage=leverage)
            margin_type = "CROSSED" if cross else "ISOLATED"
            try:
                self.client.change_margin_type(symbol=bsym, marginType=margin_type)
            except Exception:
                pass  # Already set — Binance returns error if unchanged
            logger.info(
                "Leverage set to %dx (%s) for %s",
                leverage, margin_type.lower(), symbol,
            )
        except Exception as e:
            logger.warning("Failed to set leverage on %s: %s", symbol, e)

    def fetch_historical_candles(self, symbol, interval, start_ms, end_ms) -> list[dict]:
        """
        Fetch klines from Binance Futures.

        The interval parameter is ignored — we map bar_interval_s to
        a Binance interval string internally.
        """
        bsym = self._to_binance_symbol(symbol)
        try:
            # Binance klines: [[open_time, o, h, l, c, vol, close_time, ...], ...]
            klines = self.client.klines(
                symbol=bsym,
                interval="1m",
                startTime=start_ms,
                endTime=end_ms,
                limit=1500,
            )
            rows = []
            for k in klines:
                rows.append({
                    "timestamp": pd.Timestamp(int(k[0]), unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return rows
        except Exception as e:
            logger.error("fetch_historical_candles(%s) failed: %s", symbol, e)
            return []

    def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """
        Fetch funding rate, mark price, and index (oracle) price from Binance.

        Uses the premiumIndex endpoint which returns all three in one call.
        """
        bsym = self._to_binance_symbol(symbol)
        try:
            result = self.client.mark_price(symbol=bsym)
            if isinstance(result, list):
                result = result[0] if result else {}

            rate = float(result.get("lastFundingRate", 0))
            ts_ms = int(result.get("time", 0))
            mark = float(result.get("markPrice", 0))
            index = float(result.get("indexPrice", 0))
            # Binance reports per-8h rate; annualise to bps
            rate_ann_bps = rate * 3 * 365 * 1e4

            return FundingSnapshot(
                timestamp=pd.Timestamp(ts_ms, unit="ms") if ts_ms else pd.Timestamp.utcnow(),
                rate=rate,
                rate_annualized=rate_ann_bps,
                oracle_price=index,
                mark_price=mark,
            )
        except Exception as e:
            logger.warning("fetch_funding_rate(%s) failed: %s", symbol, e)
        return None