from __future__ import annotations

import abc

from core.models import Side, Allocation
from .base import StrategyContext, PortfolioTarget


class PortfolioOverlay(abc.ABC):
    """
    Post-strategy risk filter for cross-exchange constraints.

    After each bar:
      1. Strategy generates a PortfolioTarget (with exchange_allocations populated
         for multi-exchange strategies)
      2. The overlay adjusts the target (scale, veto, hedge)
      3. Engine executes the adjusted target

    Use cases:
      • Cap net exposure across exchanges
      • Force delta-neutral (auto-generate hedge legs)
      • Block new entries when drawdown exceeds threshold
      • Enforce per-exchange position limits
    """

    @abc.abstractmethod
    def adjust(
        self,
        target: PortfolioTarget,
        ctx: StrategyContext,
    ) -> PortfolioTarget:
        """
        Adjust allocations for cross-exchange constraints.
        Return the (possibly modified) target.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in overlays
# ═══════════════════════════════════════════════════════════════════════════


class NetExposureOverlay(PortfolioOverlay):
    """
    Cap net directional exposure across all exchanges.

    If long 10 ETH on HL and short 4 ETH on Binance, net = 6 ETH long.
    If that exceeds max_net_weight * equity, scale the larger side down.
    """

    def __init__(self, max_net_weight: float = 0.5):
        self.max_net_weight = max_net_weight

    def adjust(self, target: PortfolioTarget, ctx: StrategyContext) -> PortfolioTarget:
        net_weight_by_symbol: dict[str, float] = {}
        for (ex, sym), alloc in target.exchange_allocations.items():
            direction = 1 if alloc.side == Side.LONG else (
                -1 if alloc.side == Side.SHORT else 0
            )
            net_weight_by_symbol[sym] = (
                net_weight_by_symbol.get(sym, 0.0) + direction * alloc.weight
            )

        for sym, net_w in net_weight_by_symbol.items():
            if abs(net_w) > self.max_net_weight and abs(net_w) > 0:
                scale = self.max_net_weight / abs(net_w)
                for (ex, s), alloc in target.exchange_allocations.items():
                    if s == sym and alloc.side != Side.FLAT:
                        alloc.weight *= scale

        return target


class DeltaNeutralOverlay(PortfolioOverlay):
    """
    Enforce delta-neutral positioning per symbol.

    If after all strategies run a symbol has net directional exposure,
    this overlay adds a counter-position on a specified hedge exchange.
    """

    def __init__(self, hedge_exchange: str, max_residual_weight: float = 0.02):
        self.hedge_exchange = hedge_exchange
        self.max_residual_weight = max_residual_weight

    def adjust(self, target: PortfolioTarget, ctx: StrategyContext) -> PortfolioTarget:
        net_by_symbol: dict[str, float] = {}
        for (ex, sym), alloc in target.exchange_allocations.items():
            if alloc.side == Side.FLAT:
                continue
            direction = 1 if alloc.side == Side.LONG else -1
            net_by_symbol[sym] = (
                net_by_symbol.get(sym, 0.0) + direction * alloc.weight
            )

        for sym, net_w in net_by_symbol.items():
            if abs(net_w) > self.max_residual_weight:
                hedge_side = Side.SHORT if net_w > 0 else Side.LONG
                target[(self.hedge_exchange, sym)] = Allocation(
                    side=hedge_side,
                    weight=abs(net_w),
                    confidence=1.0,
                    reason=f"DeltaNeutral hedge: net={net_w:.4f}",
                    meta={"auto_hedge": True},
                )

        return target
