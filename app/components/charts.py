"""Reusable Plotly chart builders for the trading dashboard."""
from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

_DARK = "plotly_dark"
_GREEN = "#26a69a"
_RED = "#ef5350"
_BLUE = "#2196F3"
_ORANGE = "#FF9800"
_PURPLE = "#9C27B0"
_CYAN = "#00BCD4"

_OVERLAY_COLORS = [_BLUE, _ORANGE, _PURPLE, _CYAN, "#4CAF50", "#F44336"]

# Shared style constants matching the dashboard terminal aesthetic
_GRID = dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)")
_LEGEND_H = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
_MARGIN_MAIN = dict(l=50, r=20, t=50, b=20)
_MARGIN_MINI = dict(l=50, r=20, t=10, b=20)


def candlestick_chart(
    df: pd.DataFrame,
    overlays: dict[str, pd.Series] | None = None,
    title: str = "",
    height: int = 500,
) -> go.Figure:
    """
    OHLCV candlestick with optional indicator overlays.
    overlays: {label: pd.Series} added as line traces on the same y-axis.
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
    ))
    for i, (name, series) in enumerate((overlays or {}).items()):
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            name=name,
            line=dict(color=_OVERLAY_COLORS[i % len(_OVERLAY_COLORS)], width=1.5),
        ))
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template=_DARK,
        height=height,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def bollinger_traces(mid: pd.Series, upper: pd.Series, lower: pd.Series) -> list[go.Scatter]:
    """Return three Scatter traces for Bollinger Bands (add to an existing figure)."""
    return [
        go.Scatter(x=upper.index, y=upper.values, name="BB Upper",
                   line=dict(color=_ORANGE, width=1, dash="dot"), showlegend=True),
        go.Scatter(x=lower.index, y=lower.values, name="BB Lower",
                   line=dict(color=_ORANGE, width=1, dash="dot"),
                   fill="tonexty", fillcolor="rgba(255,152,0,0.05)", showlegend=True),
        go.Scatter(x=mid.index, y=mid.values, name="BB Mid",
                   line=dict(color=_ORANGE, width=1), showlegend=True),
    ]


def volume_bars(df: pd.DataFrame) -> go.Figure:
    """Color-coded volume bar chart."""
    colors = [_GREEN if c >= o else _RED for c, o in zip(df["close"], df["open"])]
    fig = go.Figure(go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volume"))
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        showlegend=False,
        yaxis_title="Volume",
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def rsi_chart(series: pd.Series, period: int = 14) -> go.Figure:
    """RSI line chart with overbought/oversold reference lines."""
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.08)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.08)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=_RED, opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color=_GREEN, opacity=0.5)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3)
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name=f"RSI({period})", line=dict(color=_ORANGE, width=1.5),
    ))
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        yaxis=dict(range=[0, 100]),
        margin=_MARGIN_MINI,
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def atr_chart(series: pd.Series, period: int = 14) -> go.Figure:
    """ATR area chart."""
    fig = go.Figure(go.Scatter(
        x=series.index, y=series.values,
        name=f"ATR({period})", line=dict(color=_PURPLE, width=1.5),
        fill="tozeroy", fillcolor="rgba(156,39,176,0.12)",
    ))
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def equity_chart(equity_curve: pd.Series, drawdown: pd.Series) -> go.Figure:
    """Equity curve (top) + drawdown % (bottom), shared x-axis."""
    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.02,
    )
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        name="Equity", line=dict(color=_BLUE, width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=(drawdown.values * 100),
        name="Drawdown %", line=dict(color=_RED, width=1),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.25)",
    ), row=2, col=1)
    fig.update_layout(
        template=_DARK, height=460,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def trade_markers(fig: go.Figure, trades_df: pd.DataFrame) -> go.Figure:
    """Overlay entry (triangle) and exit (x) markers on an existing price figure."""
    if trades_df is None or trades_df.empty:
        return fig

    def _to_str(v):
        s = str(v)
        return s.split(".")[-1] if "." in s else s

    side_col = trades_df["side"].map(_to_str) if "side" in trades_df.columns else pd.Series(["LONG"] * len(trades_df), index=trades_df.index)
    ts_col = "timestamp"
    ep_col = "entry_price"
    xp_col = "exit_price"

    for side, color, sym in [("LONG", _GREEN, "triangle-up"), ("SHORT", _RED, "triangle-down")]:
        sub = trades_df[side_col == side]
        if sub.empty:
            continue
        reason_entry = sub["reason_entry"] if "reason_entry" in sub.columns else pd.Series(["—"] * len(sub), index=sub.index)
        custom = pd.DataFrame({
            "entry_price": sub[ep_col],
            "side": side,
            "reason": reason_entry.fillna("—"),
        }).values
        fig.add_trace(go.Scatter(
            x=sub[ts_col], y=sub[ep_col],
            mode="markers", name=f"{side} Entry",
            legendgroup=f"{side} Entry",
            marker=dict(color=color, size=14, symbol=sym, line=dict(color="white", width=1.5)),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[1]} ENTRY</b><br>"
                "Price: %{y:.4f}<br>"
                "Reason: %{customdata[2]}<extra></extra>"
            ),
        ))

    if xp_col not in trades_df.columns:
        return fig
    exits = trades_df.dropna(subset=[xp_col])
    if exits.empty:
        return fig

    pnl_series = exits["pnl"] if "pnl" in exits.columns else pd.Series([0.0] * len(exits), index=exits.index)
    pnl_pct_series = exits["pnl_pct"] if "pnl_pct" in exits.columns else pd.Series([0.0] * len(exits), index=exits.index)
    exit_ts = exits["exit_timestamp"] if "exit_timestamp" in exits.columns else exits[ts_col]

    for label, color, sym, mask in [
        ("Exit WIN",  _GREEN, "star", pnl_series > 0),
        ("Exit LOSS", _RED,   "x",   pnl_series <= 0),
    ]:
        grp = exits[mask]
        if grp.empty:
            continue
        custom = pd.DataFrame({
            "exit_price": grp[xp_col],
            "pnl": pnl_series[mask].fillna(0),
            "pnl_pct": pnl_pct_series[mask].fillna(0),
        }).values
        fig.add_trace(go.Scatter(
            x=exit_ts[mask], y=grp[xp_col],
            mode="markers", name=label,
            legendgroup="Exits",
            marker=dict(color=color, size=11, symbol=sym, line=dict(color="white", width=1.2)),
            customdata=custom,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Price: %{y:.4f}<br>"
                "PnL: %{customdata[1]:.2f} (%{customdata[2]:.2f}%)<extra></extra>"
            ),
        ))
    return fig


def signal_log_chart(signal_log: pd.DataFrame, height: int = 320) -> go.Figure:
    """Two-panel signal log: direction bars (top) + confidence/weight lines (bottom)."""
    ts = pd.to_datetime(signal_log["timestamp"], unit="ms", errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(signal_log["timestamp"], errors="coerce")

    direction = signal_log["side"].map({"LONG": 1, "FLAT": 0, "SHORT": -1}).fillna(0)
    bar_colors = [
        _GREEN if d == 1 else _RED if d == -1 else "rgba(100,100,100,0.35)"
        for d in direction
    ]
    reason = signal_log["reason"] if "reason" in signal_log.columns else pd.Series([""] * len(signal_log))
    side_str = signal_log["side"] if "side" in signal_log.columns else pd.Series([""] * len(signal_log))

    fig = sp.make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.60],
        vertical_spacing=0.04,
        subplot_titles=["Direction", "Confidence / Weight"],
    )

    # Row 1 — direction bars
    fig.add_trace(go.Bar(
        x=ts, y=direction,
        marker_color=bar_colors,
        name="Direction",
        showlegend=False,
        customdata=pd.concat([side_str.rename("side"), reason.rename("reason")], axis=1).values,
        hovertemplate="<b>%{customdata[0]}</b><br>Reason: %{customdata[1]}<extra></extra>",
    ), row=1, col=1)

    # Row 2 — confidence + weight
    if "confidence" in signal_log.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=signal_log["confidence"],
            name="Confidence", line=dict(color=_BLUE, width=1.5),
            hovertemplate="Confidence: %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    if "weight" in signal_log.columns:
        fig.add_trace(go.Scatter(
            x=ts, y=signal_log["weight"],
            name="Weight", line=dict(color=_ORANGE, width=1.5),
            hovertemplate="Weight: %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        template=_DARK,
        height=height,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
        bargap=0,
    )
    fig.update_yaxes(
        tickvals=[-1, 0, 1],
        ticktext=["SHORT", "FLAT", "LONG"],
        range=[-1.5, 1.5],
        row=1, col=1,
        **_GRID,
    )
    fig.update_yaxes(range=[0, 1.05], row=2, col=1, **_GRID)
    fig.update_xaxes(**_GRID)
    return fig


def depth_chart(snapshot) -> go.Figure:
    """Mirrored bid/ask depth bars from an OrderBookSnapshot."""
    if snapshot is None:
        return go.Figure()

    bids = snapshot.bids[:20]
    asks = snapshot.asks[:20]

    fig = go.Figure()
    if bids:
        fig.add_trace(go.Bar(
            x=[lvl.price for lvl in bids],
            y=[lvl.size for lvl in bids],
            name="Bids", marker_color=_GREEN,
        ))
    if asks:
        fig.add_trace(go.Bar(
            x=[lvl.price for lvl in asks],
            y=[lvl.size for lvl in asks],
            name="Asks", marker_color=_RED,
        ))
    fig.update_layout(
        template=_DARK, height=350, barmode="overlay",
        hovermode="closest",
        margin=_MARGIN_MAIN,
        xaxis_title="Price", yaxis_title="Size",
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def funding_chart(df: pd.DataFrame) -> go.Figure:
    """Funding rate (bps) bar + mark/oracle price lines."""
    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.4, 0.6], vertical_spacing=0.02,
    )
    if "funding_rate" in df.columns:
        vals = df["funding_rate"] * 1e4
        colors = [_GREEN if v >= 0 else _RED for v in vals]
        fig.add_trace(go.Bar(
            x=df.index, y=vals, name="Funding (bps)", marker_color=colors,
        ), row=1, col=1)

    for col, color, label in [("mark_price", _BLUE, "Mark"), ("oracle_price", _ORANGE, "Oracle")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=label,
                line=dict(color=color, width=1.5),
            ), row=2, col=1)

    fig.update_layout(
        template=_DARK, height=420,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        legend=_LEGEND_H,
    )
    fig.update_yaxes(title_text="Funding (bps)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def sentiment_scatter(df: pd.DataFrame) -> go.Figure:
    """Sentiment score scatter plot colored by source."""
    if df.empty:
        return go.Figure()

    ts_col = next((c for c in ["created_at", "timestamp", "date"] if c in df.columns), None)
    score_col = next((c for c in ["score", "sentiment_score", "compound"] if c in df.columns), None)
    if not ts_col or not score_col:
        return go.Figure()

    src_colors = {
        "x": "#1DA1F2", "reddit": "#FF4500",
        "telegram": "#0088CC", "chan": "#7A6A4F",
    }
    source_col = "source" if "source" in df.columns else None
    sources = df[source_col].unique() if source_col else ["all"]

    fig = go.Figure()
    for src in sources:
        sub = df[df[source_col] == src] if source_col else df
        fig.add_trace(go.Scatter(
            x=sub[ts_col], y=sub[score_col],
            mode="markers", name=str(src),
            marker=dict(color=src_colors.get(str(src), _PURPLE), size=5, opacity=0.7),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template=_DARK, height=300,
        hovermode="closest",
        margin=_MARGIN_MAIN,
        xaxis_title="Time", yaxis_title="Sentiment Score",
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def spread_chart(
    timestamps: list,
    spread_bps: list,
    title: str = "",
    height: int = 150,
) -> go.Figure:
    """Bid-ask spread (bps) time series."""
    fig = go.Figure(go.Scatter(
        x=timestamps, y=spread_bps,
        name="Spread (bps)", line=dict(color=_CYAN, width=1),
        fill="tozeroy", fillcolor="rgba(0,188,212,0.10)",
    ))
    fig.update_layout(
        template=_DARK, height=height,
        hovermode="x unified",
        title=title,
        margin=_MARGIN_MAIN if title else _MARGIN_MINI,
        yaxis_title="Spread (bps)",
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def funding_rate_mini(df: pd.DataFrame) -> go.Figure:
    """Compact funding rate (bps) bar chart — used as a subplot below the price chart."""
    if "funding_rate" not in df.columns:
        return go.Figure()
    vals = df["funding_rate"] * 1e4
    colors = [_GREEN if v >= 0 else _RED for v in vals]
    fig = go.Figure(go.Bar(
        x=df.index, y=vals, name="Funding (bps)", marker_color=colors,
    ))
    fig.add_hline(y=0, line_color="gray", line_width=0.5, opacity=0.4)
    fig.update_layout(
        template=_DARK, height=150,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        yaxis_title="Funding (bps)",
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def macd_chart(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> go.Figure:
    """MACD line + signal line (top) and histogram (bottom), shared x-axis."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.02,
    )
    fig.add_trace(go.Scatter(
        x=macd_line.index, y=macd_line.values,
        name=f"MACD({fast},{slow})", line=dict(color=_BLUE, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=signal_line.index, y=signal_line.values,
        name=f"Signal({signal})", line=dict(color=_ORANGE, width=1.5),
    ), row=1, col=1)
    bar_colors = [_GREEN if v >= 0 else _RED for v in histogram]
    fig.add_trace(go.Bar(
        x=histogram.index, y=histogram.values,
        name="Histogram", marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_width=0.5, opacity=0.4, row=2, col=1)
    fig.update_layout(
        template=_DARK, height=250,
        hovermode="x unified",
        margin=_MARGIN_MINI,
        legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


def macro_chart(df: pd.DataFrame, col: str, label: str, color: str = _BLUE) -> go.Figure:
    """Generic macro time-series area chart."""
    if df.empty or col not in df.columns:
        return go.Figure()
    ts_col = next((c for c in ["timestamp", "date", "datetime"] if c in df.columns), df.index.name or None)
    x = df[ts_col] if ts_col and ts_col in df.columns else df.index
    fig = go.Figure(go.Scatter(
        x=x, y=df[col], name=label,
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.10)",
    ))
    fig.update_layout(
        template=_DARK, height=260, title=label,
        hovermode="x unified",
        margin=_MARGIN_MAIN,
        showlegend=False,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


# ---------------------------------------------------------------------------
# Orderflow & feature charts (research / analysis)
# ---------------------------------------------------------------------------

def plot_orderflow(
    tick_df: pd.DataFrame,
    l2_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    start_time=None,
    end_time=None,
    title: str = "Orderflow & Tick Dynamics",
    height: int = 900,
) -> go.Figure:
    """4-panel orderflow chart: price/BBO/trades, volume, spread %, L2 imbalance."""
    if start_time and end_time:
        tick_sub = tick_df.loc[start_time:end_time]
        l2_sub = l2_df.loc[start_time:end_time]
        ohlcv_sub = ohlcv_df.loc[start_time:end_time]
    else:
        tick_sub = tick_df.copy()
        l2_sub = l2_df.copy()
        ohlcv_sub = ohlcv_df.copy()

    fig = sp.make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2, 0.2],
        subplot_titles=(
            "Price, BBO & Trades",
            "Traded Volume",
            "Bid-Ask Spread",
            "L2 Depth Imbalance (Bid vs Ask Size)",
        ),
    )

    fig.add_trace(go.Candlestick(
        x=ohlcv_sub.index,
        open=ohlcv_sub["open"], high=ohlcv_sub["high"],
        low=ohlcv_sub["low"], close=ohlcv_sub["close"],
        name="1m OHLC",
        increasing_line_color=_GREEN,
        decreasing_line_color=_RED,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=l2_sub["bid_px"],
        mode="lines", line=dict(color="rgba(38,166,154,0.6)", width=1, shape="hv"),
        name="Best Bid",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=l2_sub["ask_px"],
        mode="lines", line=dict(color="rgba(239,83,80,0.6)", width=1, shape="hv"),
        name="Best Ask",
    ), row=1, col=1)

    buys = tick_sub[tick_sub["side"].str.lower() == "buy"]
    sells = tick_sub[tick_sub["side"].str.lower() == "sell"]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["price"], mode="markers",
        marker=dict(symbol="triangle-up", color="#00ff00", size=6,
                    line=dict(width=0.5, color="black")),
        name="Buy Trade", text=buys["size"],
        hovertemplate="Time: %{x}<br>Price: %{y}<br>Size: %{text}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["price"], mode="markers",
        marker=dict(symbol="triangle-down", color=_RED, size=6,
                    line=dict(width=0.5, color="black")),
        name="Sell Trade", text=sells["size"],
        hovertemplate="Time: %{x}<br>Price: %{y}<br>Size: %{text}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ohlcv_sub.index, y=ohlcv_sub["volume"],
        mode="lines", fill="tozeroy",
        line=dict(color="#82b1ff", width=1, shape="hv"),
        name="1m Volume",
    ), row=2, col=1)

    mid_price = (l2_sub["ask_px"] + l2_sub["bid_px"]) / 2
    pct_spread = ((l2_sub["ask_px"] - l2_sub["bid_px"]) / mid_price) * 100
    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=pct_spread,
        mode="lines", line=dict(color=_ORANGE, width=1.5, shape="hv"),
        fill="tozeroy", name="Spread %",
        hovertemplate="Time: %{x}<br>Spread: %{y:.4f}%<extra></extra>",
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=l2_sub["bid_sz"],
        mode="lines", line=dict(color=_GREEN, width=1, shape="hv"),
        fill="tozeroy", name="Bid Size (Support)",
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=l2_sub.index, y=-l2_sub["ask_sz"],
        mode="lines", line=dict(color=_RED, width=1, shape="hv"),
        fill="tozeroy", name="Ask Size (Resistance)",
    ), row=4, col=1)

    fig.update_layout(
        template=_DARK, title_text=title,
        xaxis_rangeslider_visible=False,
        hovermode="x unified", height=height,
        margin=_MARGIN_MAIN, legend=_LEGEND_H,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Spread %", ticksuffix="%", tickformat=".4f", row=3, col=1)
    fig.update_yaxes(title_text="Size (+Bid / -Ask)", row=4, col=1)
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig


Transformable = (
    str
    | dict[str, pd.Series | pd.DataFrame]
    | Callable[[pd.DataFrame], pd.Series]
)


def plot_features(
    df: pd.DataFrame,
    columns: Transformable | list[Transformable],
    title: str = "Feature Analysis",
    mode: str = "lines",
    separate_subplots: bool = False,
    height: int = 600,
    line_shape: str = "linear",
) -> go.Figure:
    """Generic multi-series chart for feature analysis."""
    if not isinstance(columns, list):
        columns = [columns]

    plot_data = []
    for item in columns:
        if isinstance(item, str):
            plot_data.append((item, df[item]))
        elif isinstance(item, dict):
            for label, series in item.items():
                plot_data.append((label, series))
        elif callable(item):
            series = item(df)
            name = getattr(series, "name", f"Transform_{len(plot_data)}")
            plot_data.append((name, series))

    num_plots = len(plot_data)
    if separate_subplots:
        fig = sp.make_subplots(
            rows=num_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        )
        for i, (name, series) in enumerate(plot_data, start=1):
            fig.add_trace(go.Scattergl(
                x=series.index, y=series, mode=mode, name=name, line_shape=line_shape,
            ), row=i, col=1)
            fig.update_yaxes(title_text=name, row=i, col=1)
    else:
        fig = go.Figure()
        for name, series in plot_data:
            fig.add_trace(go.Scattergl(
                x=series.index, y=series, mode=mode, name=name, line_shape=line_shape,
            ))

    fig.update_layout(
        template=_DARK, title_text=title, height=height,
        hovermode="x unified", margin=_MARGIN_MAIN, legend=_LEGEND_H,
    )
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig
