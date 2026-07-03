"""Centralized Alpaca historical bar fetching for the UI."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1H", "4H", "1D"]

# Maps Alpaca-style labels → core parser labels (used by backtester / timeframe_to_seconds)
_ALPACA_TO_CORE: dict[str, str] = {
    "1Min": "1m",
    "5Min": "5m",
    "15Min": "15m",
    "30Min": "30m",
    "1H":   "1h",
    "4H":   "4h",
    "1D":   "1d",
}


def to_core_timeframe(alpaca_tf: str) -> str:
    """Convert an Alpaca-style timeframe label to the core parser label.

    e.g. "1Min" → "1m", "1H" → "1h", "1D" → "1d"
    """
    try:
        return _ALPACA_TO_CORE[alpaca_tf]
    except KeyError:
        raise ValueError(f"Unknown Alpaca timeframe {alpaca_tf!r}. Valid: {TIMEFRAMES}")


def _build_timeframe(tf: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    _map = {
        "1Min":  TimeFrame(1,  TimeFrameUnit.Minute),
        "5Min":  TimeFrame(5,  TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1H":    TimeFrame(1,  TimeFrameUnit.Hour),
        "4H":    TimeFrame(4,  TimeFrameUnit.Hour),
        "1D":    TimeFrame(1,  TimeFrameUnit.Day),
    }
    if tf not in _map:
        raise ValueError(f"Unknown timeframe {tf!r}. Valid: {TIMEFRAMES}")
    return _map[tf]


def get_credentials(api_key: str = "", api_secret: str = "") -> tuple[str, str]:
    """Returns (key, secret): .env values if UI fields blank, else UI values."""
    try:
        from dotenv import dotenv_values
        env = dotenv_values(_ROOT / ".env")
    except Exception:
        env = {}
    key = api_key or env.get("ALP_PAPER_KEY", "")
    secret = api_secret or env.get("ALP_PAPER_SECRET", "")
    return key, secret


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    api_key: str = "",
    api_secret: str = "",
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca API.
    Returns DataFrame with UTC-aware DatetimeIndex, columns: open high low close volume.
    Raises ValueError on missing credentials or empty response.
    """
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    key, secret = get_credentials(api_key, api_secret)
    if not key or not secret:
        raise ValueError(
            "Alpaca credentials not set. Add ALP_PAPER_KEY and ALP_PAPER_SECRET to .env "
            "or enter them in the sidebar."
        )

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    tf = _build_timeframe(timeframe)
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start_dt,
        end=end_dt,
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if df.empty:
        raise ValueError(f"No data returned for {symbol} {timeframe} {start} → {end}.")

    # Multi-index (symbol, timestamp) → timestamp-only index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level="symbol", drop=True)

    df.index.name = "timestamp"

    wanted = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in wanted if c in df.columns]]
    return df


def load_bars_cached(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    api_key: str = "",
    api_secret: str = "",
    cache_key_prefix: str = "viz",
) -> pd.DataFrame | None:
    """
    Wraps fetch_ohlcv with st.session_state caching keyed on all fetch params.
    Returns None on error (caller is expected to show st.error via the raised exception
    being caught inside this function).
    """
    cache_key = f"_bars_{cache_key_prefix}_{symbol}_{timeframe}_{start}_{end}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        df = fetch_ohlcv(symbol, timeframe, start, end, api_key, api_secret)
        st.session_state[cache_key] = df
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None


def build_universe(symbol: str, df: pd.DataFrame):
    """Wrap an OHLCV DataFrame in a Universe for backtester/strategy consumption."""
    from core.universe import Universe
    uni = Universe(symbols=[symbol])
    uni.add_asset(symbol, df)
    return uni
