"""
execution/ — Exchange-agnostic live trading framework.

Add new exchanges by implementing BaseExecutor (execution/executor.py) +
BaseFeed (data/feeds/) and registering both in factory.py.
"""

from .executor import BaseExecutor
from .portfolio import MultiExchangePortfolio
from .state import LiveState, _AssetLiveState
from .engine import Engine
from .factory import create_executor, create_feed, create_bar_builder, register_exchange
from core.feeds import BaseBarBuilder

# Backward-compat aliases
LiveEngine = Engine
MultiExchangeEngine = Engine

# Exchange adapters
from .hyperliquid.hyperliquid_executor import HyperliquidExecutor
from .binance.binance_executor import BinanceExecutor
from .alpaca.alpaca_executor import AlpacaExecutor

# Feeds (canonical: data/feeds/)
from data.feeds.hyperliquid import HyperliquidFeed
from data.feeds.binance import BinanceFeed
from data.feeds.alpaca import AlpacaFeed

from core.models import FillResult

__all__ = [
    # Core interface
    "BaseExecutor", "BaseBarBuilder",
    # Portfolio aggregator
    "MultiExchangePortfolio",
    # State
    "LiveState", "_AssetLiveState",
    # Factory
    "create_executor", "create_feed", "create_bar_builder", "register_exchange",
    # Engine
    "Engine", "LiveEngine", "MultiExchangeEngine",
    # Models
    "FillResult",
    # Exchange adapters
    "HyperliquidExecutor", "HyperliquidFeed",
    "BinanceExecutor", "BinanceFeed",
    "AlpacaExecutor", "AlpacaFeed",
]
