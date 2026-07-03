"""
execution/hyperliquid_executor.py — Hyperliquid executor (refactored).

Now inherits from BaseExecutor so the engine is exchange-agnostic.
All Hyperliquid-specific SDK calls stay here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from core.models import Side, Position, FundingSnapshot
from core.models import FillResult
from ..executor import BaseExecutor

logger = logging.getLogger(__name__)


class HyperliquidExecutor(BaseExecutor):

    def __init__(
        self,
        account_address: str,
        secret_key: str,
        api_url: str,
    ):
        self.account_address = account_address
        self.api_url = api_url

        try:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from eth_account import Account
        except ImportError as e:
            raise ImportError(
                f"Missing dependency for HyperliquidExecutor: {e}. "
                "Install with: pip install hyperliquid-python-sdk eth-account"
            ) from e

        is_testnet = "testnet" in api_url
        spot_meta = {"universe": [], "tokens": []} if is_testnet else None

        self.info = Info(api_url, skip_ws=True, spot_meta=spot_meta)
        wallet = Account.from_key(secret_key)
        self.exchange = Exchange(
            wallet, api_url,
            account_address=account_address,
            spot_meta=spot_meta,
        )
        self._meta: list[dict] | None = None
        logger.info("HyperliquidExecutor initialised for %s on %s", account_address, api_url)

    # ── BaseExecutor interface ───────────────────────────────────────────

    @property
    def exchange_name(self) -> str:
        return "hyperliquid"

    def get_equity(self) -> float:
        state = self.info.user_state(self.account_address)
        mv = state.get("marginSummary", {})
        return float(mv.get("accountValue", 0))

    def get_position(self, symbol: str) -> Position:
        state = self.info.user_state(self.account_address)
        for pos in state.get("assetPositions", []):
            item = pos.get("position", {})
            if item.get("coin") == symbol:
                sz = float(item.get("szi", 0))
                entry = float(item.get("entryPx", 0))
                side = Side.LONG if sz > 0 else (Side.SHORT if sz < 0 else Side.FLAT)
                return Position(side=side, size=abs(sz), entry_price=entry)
        return Position()

    def get_mid_price(self, symbol: str) -> float:
        mids = self.info.all_mids()
        return float(mids.get(symbol, 0))

    def get_open_orders(self, symbol: str) -> list[dict]:
        all_orders = self.info.open_orders(self.account_address)
        return [o for o in all_orders if o.get("coin") == symbol]

    def market_order(self, symbol, side, size, reduce_only=False) -> FillResult:
        is_buy = side == Side.LONG
        size = self._round_size(size, symbol)
        if size <= 0:
            return FillResult(success=False, status="zero_size", exchange=self.exchange_name)

        try:
            mid = self.get_mid_price(symbol)
            slippage_mult = 1.005 if is_buy else 0.995
            limit_px = round(mid * slippage_mult, 6)

            result = self.exchange.order(
                symbol, is_buy, size, limit_px,
                {"limit": {"tif": "Ioc"}},
                reduce_only=reduce_only,
            )
            status = result.get("status", "unknown")
            if status == "ok":
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                if statuses and "filled" in statuses[0]:
                    fill = statuses[0]["filled"]
                    return FillResult(
                        success=True,
                        fill_price=float(fill.get("avgPx", limit_px)),
                        filled_size=float(fill.get("totalSz", size)),
                        order_id=str(fill.get("oid", "")),
                        status="filled",
                        exchange=self.exchange_name,
                        raw=result,
                    )
                elif statuses and "resting" in statuses[0]:
                    rest = statuses[0]["resting"]
                    return FillResult(
                        success=True,
                        fill_price=limit_px,
                        filled_size=0.0,
                        order_id=str(rest.get("oid", "")),
                        status="resting",
                        exchange=self.exchange_name,
                        raw=result,
                    )
            logger.warning("Order response: %s", result)
            return FillResult(success=False, status=str(status), exchange=self.exchange_name, raw=result)
        except Exception as e:
            logger.error("Market order failed: %s", e)
            return FillResult(success=False, status=f"error: {e}", exchange=self.exchange_name)

    def limit_order(self, symbol, side, size, price, reduce_only=False) -> FillResult:
        is_buy = side == Side.LONG
        size = self._round_size(size, symbol)
        if size <= 0:
            return FillResult(success=False, status="zero_size", exchange=self.exchange_name)

        try:
            result = self.exchange.order(
                symbol, is_buy, size, round(price, 6),
                {"limit": {"tif": "Gtc"}},
                reduce_only=reduce_only,
            )
            status = result.get("status", "unknown")
            if status == "ok":
                statuses = result.get("response", {}).get("data", {}).get("statuses", [])
                if statuses:
                    entry = statuses[0]
                    if "filled" in entry:
                        fill = entry["filled"]
                        return FillResult(
                            success=True,
                            fill_price=float(fill.get("avgPx", price)),
                            filled_size=float(fill.get("totalSz", size)),
                            order_id=str(fill.get("oid", "")),
                            status="filled",
                            exchange=self.exchange_name,
                            raw=result,
                        )
                    elif "resting" in entry:
                        rest = entry["resting"]
                        return FillResult(
                            success=True,
                            fill_price=price,
                            filled_size=0.0,
                            order_id=str(rest.get("oid", "")),
                            status="resting",
                            exchange=self.exchange_name,
                            raw=result,
                        )
            return FillResult(success=False, status=str(status), exchange=self.exchange_name, raw=result)
        except Exception as e:
            logger.error("Limit order failed: %s", e)
            return FillResult(success=False, status=f"error: {e}", exchange=self.exchange_name)

    def cancel_all(self, symbol: str) -> int:
        open_orders = self.get_open_orders(symbol)
        if not open_orders:
            return 0
        cancelled = 0
        for o in open_orders:
            try:
                self.exchange.cancel(symbol, o["oid"])
                cancelled += 1
            except Exception as e:
                logger.error("Cancel oid=%s failed: %s", o["oid"], e)
        logger.info("Cancelled %d/%d orders on %s", cancelled, len(open_orders), symbol)
        return cancelled

    def close_position(self, symbol: str) -> FillResult:
        pos = self.get_position(symbol)
        if pos.side == Side.FLAT or pos.size == 0:
            return FillResult(success=True, status="already_flat", exchange=self.exchange_name)
        close_side = Side.SHORT if pos.side == Side.LONG else Side.LONG
        return self.market_order(symbol, close_side, pos.size, reduce_only=True)

    def set_leverage(self, symbol: str, leverage: int, cross: bool = True):
        try:
            self.exchange.update_leverage(leverage, symbol, is_cross=cross)
            logger.info("Leverage set to %dx (%s) for %s", leverage, "cross" if cross else "isolated", symbol)
        except Exception as e:
            logger.warning("Failed to set leverage: %s", e)

    def fetch_historical_candles(self, symbol, interval, start_ms, end_ms) -> list[dict]:
        candles = self.info.candles_snapshot(symbol, interval, start_ms, end_ms)
        rows = []
        for c in candles:
            rows.append({
                "timestamp": pd.Timestamp(int(c["t"]), unit="ms"),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        return rows

    def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """
        Fetch funding rate, mark price, and oracle price from Hyperliquid.

        Uses the metaAndAssetCtxs endpoint which returns per-asset contexts
        with funding, markPx, and oraclePx.
        """
        try:
            resp = requests.post(
                f"{self.api_url}/info",
                json={"type": "metaAndAssetCtxs"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            meta_universe = data[0].get("universe", [])
            asset_ctxs = data[1]

            for asset_meta, ctx in zip(meta_universe, asset_ctxs):
                if asset_meta.get("name") == symbol:
                    rate = float(ctx.get("funding", 0))
                    rate_ann_bps = rate * 24 * 365 * 1e4
                    return FundingSnapshot(
                        timestamp=pd.Timestamp.utcnow(),
                        rate=rate,
                        rate_annualized=rate_ann_bps,
                        oracle_price=float(ctx.get("oraclePx", 0)),
                        mark_price=float(ctx.get("markPx", 0)),
                    )
        except Exception as e:
            logger.warning("fetch_funding_rate(%s) failed: %s", symbol, e)
        return None

    # ── Internal ─────────────────────────────────────────────────────────

    def _get_meta(self) -> list[dict]:
        if self._meta is None:
            self._meta = self.info.meta()["universe"]
        return self._meta

    def _sz_decimals(self, symbol: str) -> int:
        for asset in self._get_meta():
            if asset["name"] == symbol:
                return asset.get("szDecimals", 4)
        return 4

    def _round_size(self, size: float, symbol: str) -> float:
        return round(size, self._sz_decimals(symbol))