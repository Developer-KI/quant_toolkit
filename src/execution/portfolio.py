"""
execution/portfolio.py — Multi-exchange portfolio aggregator.

Thread-safe: all reads go through a lock because multiple engine threads
(one per exchange) may query simultaneously.
"""

from __future__ import annotations

import logging
import threading

from core.models import AggregatedPosition, ExchangePosition, FillResult, FundingSnapshot, Position, Side
from .executor import BaseExecutor

logger = logging.getLogger(__name__)


class MultiExchangePortfolio:
    """Aggregates positions and equity across registered exchanges."""

    def __init__(self):
        self._executors: dict[str, BaseExecutor] = {}
        self._lock = threading.Lock()
        self._equity: dict[str, float] = {}

    def register(self, executor: BaseExecutor):
        name = executor.exchange_name
        with self._lock:
            self._executors[name] = executor
            self._equity[name] = 0.0
        logger.info("Portfolio registered exchange: %s", name)

    def unregister(self, exchange_name: str):
        with self._lock:
            self._executors.pop(exchange_name, None)
            self._equity.pop(exchange_name, None)

    @property
    def exchanges(self) -> list[str]:
        with self._lock:
            return list(self._executors.keys())

    def get_executor(self, exchange: str) -> BaseExecutor:
        with self._lock:
            if exchange not in self._executors:
                raise KeyError(
                    f"Exchange '{exchange}' not registered. "
                    f"Available: {list(self._executors.keys())}"
                )
            return self._executors[exchange]

    # ── Equity ────────────────────────────────────────────────────────────

    def refresh_equity(self):
        """Query equity from all exchanges."""
        with self._lock:
            executors = dict(self._executors)
        for name, ex in executors.items():
            try:
                eq = ex.get_equity()
                with self._lock:
                    self._equity[name] = eq
            except Exception as e:
                logger.warning("Equity refresh failed for %s: %s", name, e)

    def total_equity(self) -> float:
        with self._lock:
            return sum(self._equity.values())

    def equity_breakdown(self) -> dict[str, float]:
        with self._lock:
            return dict(self._equity)

    # ── Positions ─────────────────────────────────────────────────────────

    def get_position(self, symbol: str, exchange: str) -> Position:
        return self.get_executor(exchange).get_position(symbol)

    def net_position(self, symbol: str) -> AggregatedPosition:
        """Net position for a symbol aggregated across all exchanges."""
        with self._lock:
            executors = dict(self._executors)

        agg = AggregatedPosition(symbol=symbol)
        net = 0.0
        for name, ex in executors.items():
            try:
                pos = ex.get_position(symbol)
                ep = ExchangePosition(
                    exchange=name, symbol=symbol,
                    side=pos.side, size=pos.size,
                    entry_price=pos.entry_price,
                    unrealized_pnl=getattr(pos, "unrealized_pnl", 0.0),
                )
                agg.per_exchange.append(ep)
                if pos.side == Side.LONG:
                    net += pos.size
                    agg.gross_long += pos.size
                elif pos.side == Side.SHORT:
                    net -= pos.size
                    agg.gross_short += pos.size
            except Exception as e:
                logger.warning("Position query failed for %s on %s: %s", symbol, name, e)

        agg.net_size = net
        if net > 1e-10:
            agg.net_side = Side.LONG
        elif net < -1e-10:
            agg.net_side = Side.SHORT
        else:
            agg.net_side = Side.FLAT
        return agg

    def all_positions(self, symbols: list[str] | None = None) -> dict[str, AggregatedPosition]:
        if symbols:
            return {s: self.net_position(s) for s in symbols}

        with self._lock:
            executors = dict(self._executors)

        seen: set[str] = set()
        for name, ex in executors.items():
            try:
                if hasattr(ex, "get_all_positions"):
                    for pos in ex.get_all_positions():
                        if pos.side != Side.FLAT:
                            seen.add(pos.symbol if hasattr(pos, "symbol") else "")
            except Exception:
                pass
        return {s: self.net_position(s) for s in seen if s}

    # ── Exposure ──────────────────────────────────────────────────────────

    def net_exposure(self, symbols: list[str], prices: dict[str, float]) -> float:
        total = 0.0
        for sym in symbols:
            agg = self.net_position(sym)
            total += agg.net_size * prices.get(sym, 0.0)
        return total

    def net_exposure_pct(self, symbols: list[str], prices: dict[str, float]) -> float:
        equity = self.total_equity()
        if equity <= 0:
            return 0.0
        return self.net_exposure(symbols, prices) / equity

    def gross_exposure(self, symbols: list[str], prices: dict[str, float]) -> float:
        total = 0.0
        for sym in symbols:
            agg = self.net_position(sym)
            total += agg.gross_exposure * prices.get(sym, 0.0)
        return total

    # ── Convenience ───────────────────────────────────────────────────────

    def flatten_all(self, symbols: list[str]):
        """Close all positions on all exchanges for the given symbols."""
        with self._lock:
            executors = dict(self._executors)
        for name, ex in executors.items():
            for sym in symbols:
                try:
                    pos = ex.get_position(sym)
                    if pos.side != Side.FLAT and pos.size > 0:
                        logger.info("Flattening %s on %s", sym, name)
                        ex.close_position(sym)
                        ex.cancel_all(sym)
                except Exception as e:
                    logger.error("Flatten %s on %s failed: %s", sym, name, e)

    def summary(self, symbols: list[str], prices: dict[str, float] | None = None) -> str:
        lines = [
            "═══ Portfolio Summary ═══",
            f"Total equity: ${self.total_equity():,.2f}",
        ]
        for name, eq in self.equity_breakdown().items():
            lines.append(f"  {name}: ${eq:,.2f}")
        lines.append("")
        for sym in symbols:
            agg = self.net_position(sym)
            px = prices.get(sym, 0.0) if prices else 0.0
            notional = abs(agg.net_size) * px
            lines.append(f"{sym}:")
            lines.append(
                f"  Net: {agg.net_side.name} {abs(agg.net_size):.4f} (${notional:,.2f})"
            )
            if agg.is_hedged:
                lines.append(
                    f"  HEDGED — long {agg.gross_long:.4f}, short {agg.gross_short:.4f}"
                )
            for ep in agg.per_exchange:
                if ep.side != Side.FLAT:
                    lines.append(
                        f"    {ep.exchange}: {ep.side.name} {ep.size:.4f} @ {ep.entry_price:.2f}"
                    )
        return "\n".join(lines)
