"""
strategy/base.py — Unified Strategy base, PortfolioTarget, and StrategyContext.

Single-exchange usage (same as before):
    strategy.setup(universe)
    strategy.generate(ctx)  →  PortfolioTarget with symbol-keyed allocations

Multi-exchange usage:
    strategy.setup({"binance": u1, "kraken": u2})
    strategy.generate(ctx)  →  PortfolioTarget with (exchange, symbol)-keyed
                                exchange_allocations

The context (StrategyContext) carries both single-exchange convenience fields
(universe, equity, positions) and their multi-exchange counterparts
(universes, equity_by_exchange, all_positions).  __post_init__ keeps them in
sync so existing strategy code that reads ctx.universe / ctx.positions still
works unchanged.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models import Side, OrderBookSnapshot, Position, FundingSnapshot, Allocation
from core.universe import Universe


# ── Portfolio target ─────────────────────────────────────────────────────────


@dataclass
class PortfolioTarget:
    """
    Desired portfolio state returned by Strategy.generate().

    Single-exchange: populate ``allocations`` (symbol → Allocation).
    Multi-exchange:  populate ``exchange_allocations`` ((exchange, symbol) → Allocation).
    Both dicts may coexist; is_multi_exchange is True when exchange_allocations
    is non-empty.

    Assets / legs absent from both dicts are treated as FLAT.
    """
    allocations: dict[str, Allocation] = field(default_factory=dict)
    timestamp: pd.Timestamp | None = None
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    exchange_allocations: dict[tuple[str, str], Allocation] = field(default_factory=dict)

    # ── key dispatch ────────────────────────────────────────────────────

    def __getitem__(self, key: str | tuple[str, str]) -> Allocation:
        if isinstance(key, tuple):
            return self.exchange_allocations.get(key, Allocation())
        return self.allocations.get(key, Allocation())

    def __setitem__(self, key: str | tuple[str, str], alloc: Allocation):
        if isinstance(key, tuple):
            self.exchange_allocations[key] = alloc
        else:
            self.allocations[key] = alloc

    def __contains__(self, key: str | tuple[str, str]) -> bool:
        if isinstance(key, tuple):
            return key in self.exchange_allocations
        return key in self.allocations

    # ── introspection ────────────────────────────────────────────────────

    @property
    def is_multi_exchange(self) -> bool:
        return bool(self.exchange_allocations)

    @property
    def exchanges(self) -> list[str]:
        """Exchange names referenced in exchange_allocations."""
        return list({ex for ex, _ in self.exchange_allocations.keys()})

    def active_symbols(self, exchange: str | None = None) -> list[str]:
        """Non-FLAT symbols (optionally filtered to one exchange)."""
        if self.is_multi_exchange:
            if exchange is not None:
                return [
                    sym for (ex, sym), a in self.exchange_allocations.items()
                    if ex == exchange and a.side != Side.FLAT
                ]
            return list({
                sym for (ex, sym), a in self.exchange_allocations.items()
                if a.side != Side.FLAT
            })
        return [s for s, a in self.allocations.items() if a.side != Side.FLAT]

    def active_legs(self) -> list[tuple[str, str, Allocation]]:
        """Non-FLAT multi-exchange legs as (exchange, symbol, alloc)."""
        return [
            (ex, sym, a)
            for (ex, sym), a in self.exchange_allocations.items()
            if a.side != Side.FLAT
        ]

    def symbols_on(self, exchange: str) -> list[str]:
        """Non-FLAT symbols on a specific exchange."""
        return [
            sym for (ex, sym), a in self.exchange_allocations.items()
            if ex == exchange and a.side != Side.FLAT
        ]

    # ── single-exchange view of exchange_allocations ─────────────────────

    def for_exchange(self, exchange: str) -> dict[str, Allocation]:
        """
        Extract a single-exchange allocation dict for sizer / stop compat.
        Falls back to the plain allocations dict when exchange_allocations is empty.
        """
        if self.is_multi_exchange:
            return {
                sym: alloc
                for (ex, sym), alloc in self.exchange_allocations.items()
                if ex == exchange
            }
        return dict(self.allocations)

    # ── aggregate helpers ────────────────────────────────────────────────

    @property
    def total_weight(self) -> float:
        w1 = sum(a.weight for a in self.allocations.values() if a.side != Side.FLAT)
        w2 = sum(a.weight for a in self.exchange_allocations.values() if a.side != Side.FLAT)
        return w1 + w2

    def normalize(self, max_total: float = 1.0):
        """Scale all weights down proportionally if total exceeds max_total."""
        total = self.total_weight
        if total > max_total and total > 0:
            scale = max_total / total
            for alloc in self.allocations.values():
                alloc.weight *= scale
            for alloc in self.exchange_allocations.values():
                alloc.weight *= scale

    # ── factory ──────────────────────────────────────────────────────────

    @staticmethod
    def from_exchange_targets(targets: dict[str, "PortfolioTarget"]) -> "PortfolioTarget":
        """Merge per-exchange PortfolioTargets into one multi-exchange PortfolioTarget."""
        merged = PortfolioTarget()
        for exchange, pt in targets.items():
            merged.timestamp = merged.timestamp or pt.timestamp
            for sym, alloc in pt.allocations.items():
                merged[(exchange, sym)] = alloc
        return merged


# ── Strategy context (passed to generate each bar) ──────────────────────────


@dataclass
class StrategyContext:
    """
    Everything a Strategy sees when generating targets for one bar.

    Single-exchange strategies use the classic fields:
        universe, equity, positions

    Multi-exchange strategies use the extended fields:
        universes, equity_by_exchange, all_positions

    __post_init__ keeps both sets in sync, so a strategy only needs to read
    the subset it cares about.
    """
    # ── single-exchange fields (existing API, unchanged) ─────────────────
    universe: Universe | None = None
    bar_idx: int = 0
    timestamp: pd.Timestamp | None = None
    equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trade_history: list = field(default_factory=list)
    # ── multi-exchange extensions ────────────────────────────────────────
    universes: dict[str, Universe] = field(default_factory=dict)
    equity_by_exchange: dict[str, float] = field(default_factory=dict)
    all_positions: dict[str, dict[str, Position]] = field(default_factory=dict)

    def __post_init__(self):
        # universe ↔ universes
        if self.universe is not None and not self.universes:
            self.universes = {"default": self.universe}
        elif self.universes and self.universe is None and len(self.universes) == 1:
            self.universe = next(iter(self.universes.values()))

        # equity ↔ equity_by_exchange
        if not self.equity_by_exchange and self.universes:
            keys = list(self.universes.keys())
            if len(keys) == 1:
                self.equity_by_exchange = {keys[0]: self.equity}
        elif self.equity_by_exchange and self.equity == 0.0:
            self.equity = sum(self.equity_by_exchange.values())

        # positions ↔ all_positions
        if self.positions and not self.all_positions:
            default_ex = next(iter(self.universes.keys()), "default")
            self.all_positions = {default_ex: self.positions}
        elif self.all_positions and not self.positions and len(self.all_positions) == 1:
            self.positions = next(iter(self.all_positions.values()))

    # ── single-exchange convenience (unchanged API) ──────────────────────

    def price(self, symbol: str, exchange: str | None = None) -> float:
        """Current close price of a symbol (optionally on a specific exchange)."""
        uni = self._universe_for(exchange)
        if uni is None:
            return float("nan")
        ohlcv = uni.ohlcv(symbol)
        if self.bar_idx < len(ohlcv):
            return ohlcv["close"].iat[self.bar_idx]
        return float("nan")

    def prices(self, exchange: str | None = None) -> dict[str, float]:
        """Current close prices for all symbols (optionally on one exchange)."""
        if exchange is not None:
            uni = self.universes.get(exchange)
            if uni is None:
                return {}
            return {s: self.price(s, exchange) for s in uni.symbols}
        # single-exchange shortcut
        if self.universe is not None:
            return {s: self.price(s) for s in self.universe.symbols}
        # multi-exchange: gather all
        all_syms: set[str] = set()
        for u in self.universes.values():
            all_syms.update(u.symbols)
        return {s: self.price(s) for s in all_syms}

    def ohlcv(self, symbol: str, exchange: str | None = None) -> pd.DataFrame:
        """Full OHLCV up to current bar (inclusive)."""
        uni = self._universe_for(exchange)
        if uni is None:
            return pd.DataFrame()
        return uni.ohlcv(symbol).iloc[: self.bar_idx + 1]

    def aux(self, source_name: str, exchange: str | None = None) -> pd.DataFrame:
        """Auxiliary data up to current bar."""
        uni = self._universe_for(exchange)
        if uni is None:
            return pd.DataFrame()
        df = uni.aux(source_name)
        return df.iloc[: self.bar_idx + 1]

    def l2(self, symbol: str, exchange: str | None = None) -> OrderBookSnapshot | None:
        """Current L2 snapshot for a symbol."""
        uni = self._universe_for(exchange)
        if uni is None:
            return None
        l2_list = uni.l2(symbol)
        if l2_list and self.bar_idx < len(l2_list):
            return l2_list[self.bar_idx]
        return None

    def funding(self, symbol: str, exchange: str | None = None) -> FundingSnapshot | None:
        """Current funding rate snapshot for a symbol."""
        uni = self._universe_for(exchange)
        if uni is None:
            return None
        return uni.funding_at(symbol, self.bar_idx)

    def is_positioned(self, symbol: str, exchange: str | None = None) -> bool:
        if exchange is not None:
            pos = self.position_on(exchange, symbol)
            return pos.side != Side.FLAT
        pos = self.positions.get(symbol)
        return pos is not None and pos.side != Side.FLAT

    def net_exposure(self, exchange: str | None = None) -> float:
        """Net dollar exposure as fraction of equity."""
        positions = (
            self.all_positions.get(exchange, {}) if exchange is not None
            else self.positions
        )
        eq = (
            self.equity_by_exchange.get(exchange, 1.0) if exchange is not None
            else self.equity
        )
        total = 0.0
        for sym, pos in positions.items():
            if pos.side != Side.FLAT:
                px = self.price(sym, exchange)
                direction = 1 if pos.side == Side.LONG else -1
                total += direction * pos.size * px
        return total / eq if eq > 0 else 0.0

    # ── multi-exchange convenience ────────────────────────────────────────

    def position_on(self, exchange: str, symbol: str) -> Position:
        """Position on a specific exchange for a symbol."""
        return self.all_positions.get(exchange, {}).get(symbol, Position())

    def net_exposure_pct(self, exchange: str | None = None) -> float:
        """Net exposure as % of total equity (or single exchange equity)."""
        if exchange is not None:
            return self.net_exposure(exchange)
        total_eq = sum(self.equity_by_exchange.values()) if self.equity_by_exchange else self.equity
        if total_eq <= 0:
            return 0.0
        total_exp = 0.0
        for ex, positions in self.all_positions.items():
            for sym, pos in positions.items():
                if pos.side != Side.FLAT:
                    px = self.price(sym, ex)
                    direction = 1 if pos.side == Side.LONG else -1
                    total_exp += direction * pos.size * px
        return total_exp / total_eq

    # ── internal ─────────────────────────────────────────────────────────

    def _universe_for(self, exchange: str | None) -> Universe | None:
        if exchange is None:
            return self.universe
        return self.universes.get(exchange)


# ── Abstract Strategy base ──────────────────────────────────────────────────


class Strategy(abc.ABC):
    """
    Unified trading strategy base.

    Works as a single-exchange strategy (return symbol-keyed PortfolioTarget)
    or a multi-exchange strategy (return (exchange, symbol)-keyed
    PortfolioTarget.exchange_allocations), depending on how generate() is
    implemented.

    Subclass and implement:
      • setup(universes)  — pre-compute indicators; accepts a single Universe
                            or a dict of exchange → Universe
      • generate(ctx)     — return PortfolioTarget for the current bar
      • params            — tunable parameters for stress tests
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup(self, universes: Universe | dict[str, Universe]):
        """
        Pre-compute indicators on all assets / exchanges.
        Called once before the backtest loop (or on live engine start).
        Default is a no-op; override as needed.
        """
        pass

    @abc.abstractmethod
    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        """
        Return the desired portfolio allocation for the current bar.

        The engine diffs this against current positions and executes
        the necessary trades (entries, exits, rebalances) on each exchange.
        """
        ...

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]:
        """Dict of tunable parameters (for optimization / stress tests)."""
        ...

    def set_params(self, new: dict[str, Any]):
        for k, v in new.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def generate_all(
        self, universes: Universe | dict[str, Universe]
    ) -> tuple[dict[str, np.ndarray], ...] | None:
        """
        Optional batch generation for all bars at once (vectorised fast path).

        Returns (sides, weights, reasons, metas, confidences) where each value
        is a dict keyed by symbol.  The engine activates the vectorised path
        when this returns non-None **and** all stops are NopStopLoss **and**
        all sizers are vectorizable.

        Multi-exchange strategies should return None (default) to use the
        per-bar generate() path.
        """
        return None

    def _batch_generate(
        self, universe: Universe
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, list],
        dict[str, list],
        dict[str, np.ndarray],
    ]:
        """
        Generic batch helper: calls generate() for every bar with a minimal
        context (equity=0, no open positions).  Safe for strategies that only
        read bar_idx and pre-computed universe data.
        Subclasses may call this from generate_all() to opt into vectorisation.
        """
        from core.models import Position as _Position

        symbols = universe.symbols
        index = (
            universe.common_index()
            if len(symbols) > 1
            else universe.ohlcv(symbols[0]).index
        )
        if len(index) == 0:
            index = universe.ohlcv(symbols[0]).index
        n = len(index)

        sides_all       = {sym: np.zeros(n, dtype=np.int8)    for sym in symbols}
        weights_all     = {sym: np.zeros(n, dtype=np.float64) for sym in symbols}
        confidences_all = {sym: np.zeros(n, dtype=np.float64) for sym in symbols}
        reasons_all     = {sym: [""] * n                       for sym in symbols}
        metas_all       = {sym: [{}   for _ in range(n)]       for sym in symbols}

        flat_positions = {sym: _Position() for sym in symbols}
        for i in range(n):
            ctx = StrategyContext(
                universe=universe,
                bar_idx=i,
                timestamp=index[i],
                equity=0.0,
                positions=flat_positions,
                trade_history=[],
            )
            target = self.generate(ctx)
            for sym in symbols:
                alloc = target[sym]
                sides_all[sym][i]       = np.int8(alloc.side.value)
                weights_all[sym][i]     = alloc.weight
                confidences_all[sym][i] = alloc.confidence
                reasons_all[sym][i]     = alloc.reason
                metas_all[sym][i]       = dict(alloc.meta)

        return sides_all, weights_all, reasons_all, metas_all, confidences_all

    def on_fill(
        self,
        symbol: str,
        side: Side,
        size: float,
        price: float,
        exchange: str = "default",
    ):
        """Optional callback when a fill occurs (for bookkeeping)."""
        pass


# ── Strategy registry ────────────────────────────────────────────────────────


_STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register_strategy(name: str):
    """Class decorator: @register_strategy("my_strategy")."""
    def _wrap(cls):
        _STRATEGY_REGISTRY[name] = cls
        cls._registry_name = name
        return cls
    return _wrap


def get_strategy(name: str) -> type[Strategy]:
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy '{name}' not registered. Available: {list(_STRATEGY_REGISTRY)}"
        )
    return _STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    return list(_STRATEGY_REGISTRY.keys())


def __getattr__(name: str):
    if name == "SingleAssetStrategy":
        from strategy.built_in import SingleAssetStrategy
        return SingleAssetStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
