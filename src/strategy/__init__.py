"""
strategy/ — Unified trading strategy framework.

Modules:
    base.py        — Allocation, PortfolioTarget, Strategy, StrategyContext
    built_in.py    — SingleAssetStrategy, CompositeStrategy, PerAssetStrategy, and multi-asset built-ins
    indicators.py  — Stateless indicator functions (EMA, RSI, ATR, …)
    sizing.py      — Position sizing (Sizer hierarchy)
    stops.py       — Stop-loss / take-profit (StopLoss hierarchy)
    core/universe.py — Universe + auxiliary data sources (moved to core)
"""

# ── Universe & data ──────────────────────────────────────────────────────────

from core.universe import (
    DataSource,
    StaticDataSource,
    CallableDataSource,
    AssetData,
    Universe,
)

# ── Strategy base ────────────────────────────────────────────────────────────

from .base import (
    PortfolioTarget,
    StrategyContext,
    Strategy,
    register_strategy,
    get_strategy,
    list_strategies,
)

# ── Built-in strategies ──────────────────────────────────────────────────────

from .built_in import (
    SingleAssetStrategy,
    CompositeStrategy,
    PerAssetStrategy,
    MeanReversionBasketStrategy,
)

# ── Allocation (re-exported from core for convenience) ───────────────────────

from core.models import Allocation

# ── Sizing ──────────────────────────────────────────────────────────────────

from .sizing import (
    Sizer,
    SizingContext,
    FixedFractionalSizer,
    FixedNotionalSizer,
    VolatilityTargetSizer,
    KellySizer,
    AntiMartingaleSizer,
    DrawdownScalingSizer,
    L2LiquiditySizer,
    CompositeSizer,
    default_sizer,
)

# ── Stops ────────────────────────────────────────────────────────────────────

from .stops import (
    StopLoss,
    StopResult,
    StopContext,
    FixedPercentStop,
    ATRStop,
    TrailingStop,
    TrailingATRStop,
    BreakevenStop,
    TimeStop,
    RiskRewardStop,
    CompositeStopLoss,
    EmbeddedStop,
    NopStopLoss,
    default_stop_loss,
)

# ── Indicators ──────────────────────────────────────────────────────────────

from .indicators import (
    ema,
    sma,
    rsi,
    atr,
    bollinger,
    vwap_rolling,
    order_flow_imbalance,
    book_imbalance,
    compute_atr_column,
)

# ── Overlays ────────────────────────────────────────────────────────────────

from .overlay import (
    PortfolioOverlay,
    NetExposureOverlay,
    DeltaNeutralOverlay,
)

# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    # Universe & data
    "DataSource", "StaticDataSource", "CallableDataSource",
    "AssetData", "Universe",
    # Strategy base
    "Allocation", "PortfolioTarget", "StrategyContext",
    "Strategy", "SingleAssetStrategy",
    "register_strategy", "get_strategy", "list_strategies",
    # Built-in strategies
    "CompositeStrategy", "PerAssetStrategy",
    "MeanReversionBasketStrategy",
    # Sizing
    "Sizer", "SizingContext",
    "FixedFractionalSizer", "FixedNotionalSizer", "VolatilityTargetSizer",
    "KellySizer", "AntiMartingaleSizer", "DrawdownScalingSizer",
    "L2LiquiditySizer", "CompositeSizer", "default_sizer",
    # Stops
    "StopLoss", "StopResult", "StopContext",
    "FixedPercentStop", "ATRStop", "TrailingStop", "TrailingATRStop",
    "BreakevenStop", "TimeStop", "RiskRewardStop",
    "CompositeStopLoss", "EmbeddedStop", "NopStopLoss", "default_stop_loss",
    # Indicators
    "ema", "sma", "rsi", "atr", "bollinger",
    "vwap_rolling", "order_flow_imbalance", "book_imbalance",
    "compute_atr_column",
    # Overlays
    "PortfolioOverlay", "NetExposureOverlay", "DeltaNeutralOverlay",
]
