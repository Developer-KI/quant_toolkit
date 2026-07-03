"""Data Visualizer — load Alpaca bars and overlay indicators or strategy signals."""
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from components.alpaca_data import TIMEFRAMES, build_universe, get_credentials, load_bars_cached
from components.charts import (
    atr_chart, bollinger_traces, candlestick_chart,
    macd_chart, rsi_chart, trade_markers, volume_bars,
)
from components.forms import signal_form
from components.style import inject

st.set_page_config(page_title="Visualizer", page_icon="📊", layout="wide")
inject()
st.title("Data Visualizer")


def _to_naive(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return dti.tz_convert("UTC").tz_localize(None) if dti.tz is not None else dti


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    with st.expander("Alpaca API", expanded=False):
        env_key, env_secret = get_credentials()
        api_key = st.text_input("API Key", value=env_key, type="password", key="viz_api_key")
        api_secret = st.text_input("API Secret", value=env_secret, type="password", key="viz_api_secret")
        st.caption("Leave blank to use .env values (ALP_PAPER_KEY / ALP_PAPER_SECRET)")

    st.divider()
    st.header("Data")
    symbol = st.text_input("Symbol", value="SPY", key="viz_sym").upper()
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=6, key="viz_tf")  # default 1D
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("From", value=date.today() - timedelta(days=365), key="viz_start")
    end_date = col_e.date_input("To", value=date.today(), key="viz_end")
    load_btn = st.button("Load Data", type="primary", use_container_width=True, key="viz_load")

    # Indicator controls — only shown when data is loaded
    df_loaded: pd.DataFrame | None = st.session_state.get("viz_ohlcv")

    if df_loaded is not None:
        st.divider()
        st.header("Indicators")
        overlay_ema = st.checkbox("EMA", value=True, key="viz_ema")
        if overlay_ema:
            ema_fast = st.number_input("EMA fast", value=12, step=1, min_value=2, key="viz_ef")
            ema_slow = st.number_input("EMA slow", value=26, step=1, min_value=2, key="viz_es")
        else:
            ema_fast, ema_slow = 12, 26

        overlay_sma = st.checkbox("SMA", value=False, key="viz_sma")
        sma_period = st.number_input("SMA period", value=50, step=1, min_value=2, key="viz_sp") if overlay_sma else 50

        overlay_bb = st.checkbox("Bollinger Bands", value=False, key="viz_bb")
        if overlay_bb:
            bb_window = st.number_input("BB window", value=20, step=1, min_value=5, key="viz_bbw")
            bb_std = st.number_input("BB std devs", value=2.0, step=0.5, min_value=0.5, key="viz_bbs")
        else:
            bb_window, bb_std = 20, 2.0

        show_rsi = st.checkbox("RSI", value=True, key="viz_rsi")
        rsi_period = st.number_input("RSI period", value=14, step=1, min_value=2, key="viz_rp") if show_rsi else 14

        show_atr = st.checkbox("ATR", value=False, key="viz_atr")
        atr_period = st.number_input("ATR period", value=14, step=1, min_value=2, key="viz_ap") if show_atr else 14

        show_macd = st.checkbox("MACD", value=False, key="viz_macd")
        if show_macd:
            macd_fast = st.number_input("MACD fast", value=12, step=1, min_value=2, key="viz_mf")
            macd_slow = st.number_input("MACD slow", value=26, step=1, min_value=2, key="viz_ms")
            macd_sig  = st.number_input("MACD signal", value=9, step=1, min_value=2, key="viz_mg")
        else:
            macd_fast, macd_slow, macd_sig = 12, 26, 9

        st.divider()
        with st.expander("Strategy Signal Overlay", expanded=False):
            enable_overlay = st.checkbox("Show signals on chart", value=False, key="viz_overlay")
            if enable_overlay:
                sig_cls, sig_params = signal_form(st.sidebar, key_prefix="viz_sig")
                run_overlay = st.button("Compute Signals", key="viz_run_overlay")
            else:
                sig_cls = sig_params = None
                run_overlay = False

# ── Data loading ──────────────────────────────────────────────────────────────

if load_btn:
    with st.spinner(f"Fetching {symbol} {timeframe} bars from Alpaca…"):
        df = load_bars_cached(
            symbol, timeframe,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            api_key, api_secret,
            cache_key_prefix="viz",
        )
        if df is not None:
            st.session_state["viz_ohlcv"] = df
            st.session_state["viz_symbol"] = symbol
            st.session_state["viz_timeframe"] = timeframe
            st.session_state.pop("viz_trades_df", None)  # clear stale overlay
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

df: pd.DataFrame | None = st.session_state.get("viz_ohlcv")
viz_symbol: str = st.session_state.get("viz_symbol", symbol)
viz_timeframe: str = st.session_state.get("viz_timeframe", timeframe)

if df is None:
    st.info("Enter a symbol and date range in the sidebar, then click **Load Data**.")
    st.stop()

# ── Strategy signal overlay computation ───────────────────────────────────────

if enable_overlay and run_overlay and sig_cls is not None:
    with st.spinner("Running strategy on loaded data…"):
        try:
            from strategy.built_in import SingleAssetStrategy
            from backtester.engine import Backtester
            from core.models import BacktestConfig

            if issubclass(sig_cls, SingleAssetStrategy):
                strategy = sig_cls(symbol=viz_symbol, **sig_params)
            else:
                strategy = sig_cls(**sig_params)

            uni = build_universe(viz_symbol, df)
            result = Backtester(strategy=strategy, config=BacktestConfig()).run(universe=uni)
            st.session_state["viz_trades_df"] = result.trades_df()
            st.success(f"Signal computed — {len(st.session_state['viz_trades_df'])} trades.")
        except Exception as e:
            st.error(f"Signal overlay failed: {e}")

trades_df_overlay: pd.DataFrame | None = st.session_state.get("viz_trades_df")

# ── View range filter (no re-fetch) ──────────────────────────────────────────

idx_naive = _to_naive(df.index)
min_date = idx_naive.min().date()
max_date = idx_naive.max().date()

vc1, vc2, vc3 = st.columns([2, 2, 6])
view_start = vc1.date_input("View from", value=min_date, min_value=min_date, max_value=max_date, key="viz_vs")
view_end   = vc2.date_input("View to",   value=max_date, min_value=min_date, max_value=max_date, key="viz_ve")

start_ts = pd.Timestamp(view_start)
end_ts   = pd.Timestamp(view_end) + pd.Timedelta(days=1)
mask = (idx_naive >= start_ts) & (idx_naive < end_ts)
df_view = df[mask]

if df_view.empty:
    st.warning("No data in the selected view range.")
    st.stop()

# ── Metrics strip ─────────────────────────────────────────────────────────────

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last Close", f"${df_view['close'].iloc[-1]:,.4f}")
m2.metric("Volume (last bar)", f"{df_view['volume'].iloc[-1]:,.0f}")
m3.metric("Bars loaded", f"{len(df_view):,}")
m4.metric("Range", f"{df_view.index.min().strftime('%Y-%m-%d')} → {df_view.index.max().strftime('%Y-%m-%d')}")

# ── Price chart ───────────────────────────────────────────────────────────────

from strategy.indicators import ema, sma, bollinger, rsi as _rsi, atr as _atr

overlays: dict = {}
if overlay_ema:
    overlays[f"EMA {ema_fast}"] = ema(df_view["close"], ema_fast)
    overlays[f"EMA {ema_slow}"] = ema(df_view["close"], ema_slow)
if overlay_sma:
    overlays[f"SMA {sma_period}"] = sma(df_view["close"], sma_period)

fig = candlestick_chart(df_view, overlays=overlays, title=f"{viz_symbol} — {viz_timeframe}")

if overlay_bb:
    mid, upper, lower = bollinger(df_view["close"], window=bb_window, num_std=bb_std)
    for trace in bollinger_traces(mid, upper, lower):
        fig.add_trace(trace)

if trades_df_overlay is not None and not trades_df_overlay.empty:
    # Filter overlay trades to view range
    if "timestamp" in trades_df_overlay.columns:
        ts_col = pd.to_datetime(trades_df_overlay["timestamp"])
        ts_naive = ts_col.dt.tz_convert("UTC").dt.tz_localize(None) if ts_col.dt.tz is not None else ts_col
        view_mask = (ts_naive >= start_ts) & (ts_naive < end_ts)
        fig = trade_markers(fig, trades_df_overlay[view_mask])
    else:
        fig = trade_markers(fig, trades_df_overlay)

st.plotly_chart(fig, use_container_width=True)

# ── Sub-charts ────────────────────────────────────────────────────────────────

st.plotly_chart(volume_bars(df_view), use_container_width=True)

if show_rsi:
    rsi_series = _rsi(df_view["close"], period=rsi_period)
    st.plotly_chart(rsi_chart(rsi_series, period=rsi_period), use_container_width=True)

if show_atr:
    atr_series = _atr(df_view["high"], df_view["low"], df_view["close"], period=atr_period)
    st.plotly_chart(atr_chart(atr_series, period=atr_period), use_container_width=True)

if show_macd:
    st.plotly_chart(macd_chart(df_view["close"], fast=macd_fast, slow=macd_slow, signal=macd_sig),
                    use_container_width=True)

st.caption(f"{len(df_view):,} bars  |  {df_view.index[0]}  →  {df_view.index[-1]}")
