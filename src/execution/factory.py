"""
execution/factory.py — Create exchange-specific components by name.

Uses a Registry dict so adding a new exchange requires only:
  1. Write the executor + feed modules
  2. Add one entry to EXECUTOR_REGISTRY and FEED_REGISTRY

No if/elif chains to maintain.
"""

from __future__ import annotations

import logging
from dotenv import dotenv_values
from typing import Callable

_env = dotenv_values()

from .executor import BaseExecutor
from core.feeds import BaseBarBuilder
from core.feeds import BaseFeed
from core.models import ExchangeCredentials

logger = logging.getLogger(__name__)


# ── Registry tables ───────────────────────────────────────────────────────────
# Each entry is a factory callable: (cred) → executor  or  (kwargs) → feed

def _make_hyperliquid_executor(cred: ExchangeCredentials) -> BaseExecutor:
    from .hyperliquid.hyperliquid_executor import HyperliquidExecutor
    api_url = (
        "https://api.hyperliquid-testnet.xyz" if cred.testnet
        else "https://api.hyperliquid.xyz"
    )
    return HyperliquidExecutor(
        account_address=cred.account_address,
        secret_key=cred.secret_key,
        api_url=api_url,
    )


def _make_binance_executor(cred: ExchangeCredentials) -> BaseExecutor:
    from .binance.binance_executor import BinanceExecutor
    return BinanceExecutor(
        api_key=cred.api_key,
        api_secret=cred.api_secret,
        testnet=cred.testnet,
        symbol_map=cred.symbol_map or None,
    )


def _make_hyperliquid_feed(symbol: str, testnet: bool, **_) -> BaseFeed:
    from data.feeds.hyperliquid import HyperliquidFeed
    ws_url = (
        "wss://api.hyperliquid-testnet.xyz/ws" if testnet
        else "wss://api.hyperliquid.xyz/ws"
    )
    return HyperliquidFeed(ws_url=ws_url, symbol=symbol)


def _make_binance_feed(symbol: str, testnet: bool, symbol_map=None, **_) -> BaseFeed:
    from data.feeds.binance import BinanceFeed
    binance_symbol = symbol_map.get(symbol) if symbol_map else None
    return BinanceFeed(symbol=symbol, testnet=testnet, binance_symbol=binance_symbol)


def _make_alpaca_executor(cred: ExchangeCredentials) -> BaseExecutor:
    from execution.alpaca.alpaca_executor import AlpacaExecutor
    api_key = cred.api_key or _env.get("ALP_PAPER_KEY", "")
    api_secret = cred.api_secret or _env.get("ALP_PAPER_SECRET", "")
    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca credentials missing — set ALP_PAPER_KEY and ALP_PAPER_SECRET in .env"
        )
    return AlpacaExecutor(api_key=api_key, api_secret=api_secret, paper=cred.testnet)


def _make_alpaca_feed(symbol: str, testnet: bool = True, **_) -> BaseFeed:
    from data.feeds.alpaca import AlpacaFeed
    api_key = _env.get("ALP_PAPER_KEY", "")
    api_secret = _env.get("ALP_PAPER_SECRET", "")
    return AlpacaFeed(symbol=symbol, api_key=api_key, api_secret=api_secret, paper=testnet)


EXECUTOR_REGISTRY: dict[str, Callable[[ExchangeCredentials], BaseExecutor]] = {
    "hyperliquid": _make_hyperliquid_executor,
    "binance": _make_binance_executor,
    "alpaca": _make_alpaca_executor,
}

FEED_REGISTRY: dict[str, Callable[..., BaseFeed]] = {
    "hyperliquid": _make_hyperliquid_feed,
    "binance": _make_binance_feed,
    "alpaca": _make_alpaca_feed,
}


# ── Public factory functions ───────────────────────────────────────────────────


def create_executor(cred: ExchangeCredentials) -> BaseExecutor:
    """Instantiate the right executor for an exchange."""
    name = cred.exchange.lower()
    if name not in EXECUTOR_REGISTRY:
        raise ValueError(
            f"Unknown exchange '{name}'. Supported: {sorted(EXECUTOR_REGISTRY)}"
        )
    return EXECUTOR_REGISTRY[name](cred)


def create_feed(
    exchange: str,
    symbol: str,
    testnet: bool = True,
    symbol_map: dict[str, str] | None = None,
    **kwargs,
) -> BaseFeed:
    """Instantiate the right WebSocket feed for an exchange."""
    name = exchange.lower()
    if name not in FEED_REGISTRY:
        raise ValueError(
            f"Unknown exchange '{name}'. Supported: {sorted(FEED_REGISTRY)}"
        )
    return FEED_REGISTRY[name](symbol=symbol, testnet=testnet, symbol_map=symbol_map, **kwargs)


def create_bar_builder(
    interval_s: int = 60,
    max_bars: int = 2000,
    on_bar_close=None,
) -> BaseBarBuilder:
    """Create an exchange-agnostic bar builder."""
    return BaseBarBuilder(
        interval_s=interval_s,
        max_bars=max_bars,
        on_bar_close=on_bar_close,
    )


def register_exchange(
    name: str,
    executor_factory: Callable[[ExchangeCredentials], BaseExecutor],
    feed_factory: Callable[..., BaseFeed],
):
    """
    Register a new exchange at runtime.

    Usage:
        from execution.factory import register_exchange
        register_exchange("bybit", make_bybit_executor, make_bybit_feed)
    """
    EXECUTOR_REGISTRY[name.lower()] = executor_factory
    FEED_REGISTRY[name.lower()] = feed_factory
    logger.info("Registered exchange: %s", name)
