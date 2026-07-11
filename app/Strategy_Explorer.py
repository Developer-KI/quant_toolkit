"""
Strategy Explorer & Backtester — unified EDA and backtest dashboard.

Launch from project root:
    streamlit run app/Strategy_Explorer.py
"""

import inspect
import sys
from datetime import date, timedelta
from pathlib import Path

_APP = Path(__file__).resolve().parent
_ROOT = _APP.parent
_SRC = _ROOT / "src"
for _p in [str(_SRC), str(_ROOT), str(_APP)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.lse_data import TIMEFRAMES, BACKTEST_TIMEFRAMES, build_universe, get_api_key, load_bars_cached
from components.charts import (
    atr_chart, bollinger_traces, candlestick_chart,
    equity_chart, macd_chart, rsi_chart, signal_log_chart,
    trade_markers, volume_bars,
)
from components.forms import backtest_config_form, signal_form, sizer_form, stop_form
from components.style import inject

st.set_page_config(
    page_title="Strategy Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()
st.title("Strategy Explorer")


def _to_naive(dti: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return dti.tz_convert("UTC").tz_localize(None) if dti.tz is not None else dti


# ── Indicator defaults (used even before data loads) ──────────────────────────

overlay_ema = False
overlay_sma = False
overlay_bb = False
show_rsi = True
show_atr = False
show_macd = False
show_vol_regime = False
ema_fast, ema_slow = 12, 26
sma_period = 50
bb_window, bb_std = 20, 2.0
rsi_period = 14
atr_period = 14
macd_fast, macd_slow, macd_sig = 12, 26, 9
vol_window = 20

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    with st.expander("LSE API Key", expanded=False):
        env_key = get_api_key()
        api_key = st.text_input(
            "API Key", value=env_key, type="password", key="api_key",
            help="Leave blank to use LSE_DATA from .env",
        )

    st.divider()
    st.header("Data")
    symbol = st.text_input("Symbol", value="AAPL", key="sym").upper()
    timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=6, key="tf")
    col_s, col_e = st.columns(2)
    start_date = col_s.date_input("From", value=date.today() - timedelta(days=730), key="start")
    end_date   = col_e.date_input("To",   value=date.today(), key="end")
    load_btn = st.button("Load Data", type="primary", use_container_width=True, key="load")

    df_loaded: pd.DataFrame | None = st.session_state.get("main_ohlcv")

    if df_loaded is not None:
        st.divider()
        st.header("Indicators")

        overlay_ema = st.checkbox("EMA", value=True, key="ema")
        if overlay_ema:
            ema_fast = st.number_input("EMA fast", value=12, step=1, min_value=2, key="ef")
            ema_slow = st.number_input("EMA slow", value=26, step=1, min_value=2, key="es")

        overlay_sma = st.checkbox("SMA", value=False, key="sma")
        if overlay_sma:
            sma_period = st.number_input("SMA period", value=50, step=1, min_value=2, key="sp")

        overlay_bb = st.checkbox("Bollinger Bands", value=False, key="bb")
        if overlay_bb:
            bb_window = st.number_input("BB window",   value=20,  step=1,   min_value=5,   key="bbw")
            bb_std    = st.number_input("BB std devs", value=2.0, step=0.5, min_value=0.5, key="bbs")

        show_rsi = st.checkbox("RSI", value=True, key="rsi")
        if show_rsi:
            rsi_period = st.number_input("RSI period", value=14, step=1, min_value=2, key="rp")

        show_atr = st.checkbox("ATR", value=False, key="atr")
        if show_atr:
            atr_period = st.number_input("ATR period", value=14, step=1, min_value=2, key="ap")

        show_macd = st.checkbox("MACD", value=False, key="macd")
        if show_macd:
            macd_fast = st.number_input("MACD fast",   value=12, step=1, min_value=2, key="mf")
            macd_slow = st.number_input("MACD slow",   value=26, step=1, min_value=2, key="ms")
            macd_sig  = st.number_input("MACD signal", value=9,  step=1, min_value=2, key="mg")

        show_vol_regime = st.checkbox("Volatility Regime", value=False, key="vreg")
        if show_vol_regime:
            vol_window = st.number_input("Vol window", value=20, step=1, min_value=5, key="vw")

    st.divider()
    st.header("Signal")
    signal_cls, sig_params = signal_form(st.sidebar, key_prefix="sig")

    st.divider()
    st.header("Config")
    config = backtest_config_form(st.sidebar, key_prefix="cfg")

    st.divider()
    st.header("Sizer")
    sizer = sizer_form(st.sidebar, key_prefix="sizer")

    st.divider()
    st.header("Stop Loss")
    stop = stop_form(st.sidebar, key_prefix="stop")

    st.divider()
    can_backtest = timeframe in BACKTEST_TIMEFRAMES
    run_bt = st.button(
        "Run Backtest", type="primary", use_container_width=True,
        disabled=not can_backtest,
        help=None if can_backtest else f"'{timeframe}' is not supported for backtesting.",
    )

# ── Data loading ──────────────────────────────────────────────────────────────

if load_btn:
    with st.spinner(f"Fetching {symbol} {timeframe} bars from LSE…"):
        df = load_bars_cached(
            symbol, timeframe,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            api_key,
            cache_key_prefix="main",
        )
        if df is not None:
            st.session_state["main_ohlcv"]     = df
            st.session_state["main_symbol"]    = symbol
            st.session_state["main_timeframe"] = timeframe
            for k in ["main_bt_result", "main_ht_results",
                      "main_sweep_result", "main_regime_result", "main_mc_result"]:
                st.session_state.pop(k, None)
            st.rerun()

# ── Run backtest ──────────────────────────────────────────────────────────────

if run_bt:
    stored_ohlcv = st.session_state.get("main_ohlcv")
    if signal_cls is None:
        st.error("No signal selected.")
    elif stored_ohlcv is None:
        st.error("Load data first.")
    else:
        bt_sym = st.session_state.get("main_symbol", symbol)
        bt_tf  = st.session_state.get("main_timeframe", timeframe)
        with st.spinner("Running backtest…"):
            try:
                from strategy.built_in import SingleAssetStrategy
                from testing.backtester.engine import Backtester
                from testing.backtester.costs import CompositeCostModel, aggressive_cost_stack

                strategy = signal_cls(symbol=bt_sym, **sig_params) \
                    if issubclass(signal_cls, SingleAssetStrategy) \
                    else signal_cls(**sig_params)

                uni  = build_universe(bt_sym, stored_ohlcv)
                cost = CompositeCostModel(models=aggressive_cost_stack())
                bt   = Backtester(strategy=strategy, config=config, cost_model=cost,
                                  sizer=sizer, stop_loss=stop)
                result_new = bt.run(universe=uni, timeframe=bt_tf)

                st.session_state["main_bt_result"] = result_new
                for k in ["main_ht_results", "main_sweep_result",
                          "main_regime_result", "main_mc_result"]:
                    st.session_state.pop(k, None)
                st.success("Backtest complete.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)

# ── Guard: nothing loaded yet ─────────────────────────────────────────────────

df: pd.DataFrame | None = st.session_state.get("main_ohlcv")
main_symbol:    str = st.session_state.get("main_symbol", symbol)
main_timeframe: str = st.session_state.get("main_timeframe", timeframe)

if df is None:
    st.info("Enter a symbol and date range in the sidebar, then click **Load Data**.")
    st.stop()

result = st.session_state.get("main_bt_result")

# ── Tabs ──────────────────────────────────────────────────────────────────────

(tab_explorer, tab_results,
 tab_hypothesis, tab_sweep, tab_regime, tab_mc) = st.tabs([
    "Explorer", "Results", "Hypothesis Tests", "Param Sweep", "Regime Test", "Monte Carlo",
])

# ══════════════════════════════════════════════════════════════════ Explorer tab

with tab_explorer:
    idx_naive = _to_naive(df.index)
    min_date = idx_naive.min().date()
    max_date = idx_naive.max().date()

    vc1, vc2, _ = st.columns([2, 2, 6])
    view_start = vc1.date_input("View from", value=min_date, min_value=min_date, max_value=max_date, key="vs")
    view_end   = vc2.date_input("View to",   value=max_date, min_value=min_date, max_value=max_date, key="ve")

    mask = (
        (idx_naive >= pd.Timestamp(view_start)) &
        (idx_naive < pd.Timestamp(view_end) + pd.Timedelta(days=1))
    )
    df_view = df[mask]

    if df_view.empty:
        st.warning("No data in the selected view range.")
        st.stop()

    returns = df_view["close"].pct_change().dropna()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Last Close",  f"${df_view['close'].iloc[-1]:,.4f}")
    m2.metric("Bars",         f"{len(df_view):,}")
    m3.metric("Ann. Vol",     f"{returns.std() * (252**0.5) * 100:.1f}%" if len(returns) > 1 else "—")
    m4.metric("Total Return", f"{(df_view['close'].iloc[-1] / df_view['close'].iloc[0] - 1)*100:.2f}%")
    m5.metric("Range",        f"{min_date} → {max_date}")

    from strategy.indicators import ema, sma, bollinger, rsi as _rsi, atr as _atr

    overlays: dict = {}
    if overlay_ema:
        overlays[f"EMA {ema_fast}"] = ema(df_view["close"], ema_fast)
        overlays[f"EMA {ema_slow}"] = ema(df_view["close"], ema_slow)
    if overlay_sma:
        overlays[f"SMA {sma_period}"] = sma(df_view["close"], sma_period)

    fig = candlestick_chart(df_view, overlays=overlays, title=f"{main_symbol} — {main_timeframe}")

    if overlay_bb:
        mid, upper, lower = bollinger(df_view["close"], window=bb_window, num_std=bb_std)
        for trace in bollinger_traces(mid, upper, lower):
            fig.add_trace(trace)

    if result is not None:
        trades_ov = result.trades_df()
        if not trades_ov.empty and "timestamp" in trades_ov.columns:
            ts = pd.to_datetime(trades_ov["timestamp"])
            ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts
            v_mask = (
                (ts_naive >= pd.Timestamp(view_start)) &
                (ts_naive < pd.Timestamp(view_end) + pd.Timedelta(days=1))
            )
            fig = trade_markers(fig, trades_ov[v_mask])

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(volume_bars(df_view), use_container_width=True)

    if show_rsi:
        st.plotly_chart(
            rsi_chart(_rsi(df_view["close"], period=rsi_period), period=rsi_period),
            use_container_width=True,
        )
    if show_atr:
        st.plotly_chart(
            atr_chart(_atr(df_view["high"], df_view["low"], df_view["close"], period=atr_period), period=atr_period),
            use_container_width=True,
        )
    if show_macd:
        st.plotly_chart(
            macd_chart(df_view["close"], fast=macd_fast, slow=macd_slow, signal=macd_sig),
            use_container_width=True,
        )
    if show_vol_regime:
        rv   = df_view["close"].pct_change().rolling(vol_window).std() * (252**0.5) * 100
        q_lo = rv.expanding(min_periods=vol_window).quantile(0.33)
        q_hi = rv.expanding(min_periods=vol_window).quantile(0.66)
        fig_rv = go.Figure()
        fig_rv.add_trace(go.Scatter(x=rv.index,   y=rv,   name=f"Ann. Vol ({vol_window})", line=dict(color="#2196F3", width=1.5)))
        fig_rv.add_trace(go.Scatter(x=q_lo.index, y=q_lo, name="33rd pctl", line=dict(color="#26a69a", dash="dot", width=1)))
        fig_rv.add_trace(go.Scatter(x=q_hi.index, y=q_hi, name="66th pctl", line=dict(color="#ef5350", dash="dot", width=1)))
        fig_rv.update_layout(
            template="plotly_dark", height=260,
            title=f"Volatility Regime — {vol_window}-bar rolling ann. vol",
            yaxis_title="Ann. vol %", hovermode="x unified",
            margin=dict(l=40, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_rv, use_container_width=True)

    with st.expander("Returns Analysis", expanded=False):
        from scipy import stats as sp_stats

        col_dist, col_stats_ex = st.columns([3, 2])
        with col_dist:
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(
                x=returns * 100, nbinsx=80, name="Returns",
                marker_color="#2196F3", opacity=0.75,
            ))
            fig_ret.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)
            fig_ret.update_layout(
                template="plotly_dark", height=300,
                title="Return Distribution (%)",
                xaxis_title="Return %", yaxis_title="Count",
                showlegend=False, margin=dict(l=40, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with col_stats_ex:
            st.markdown("**Descriptive Statistics**")
            r = returns.dropna()
            ann = 252**0.5
            sharpe = (r.mean() / r.std() * ann) if r.std() > 0 else 0
            skew = float(sp_stats.skew(r))
            kurt = float(sp_stats.kurtosis(r))
            _, pval_norm = sp_stats.normaltest(r)
            st.table(pd.DataFrame({
                "Metric": [
                    "Mean daily ret %", "Std daily %", "Ann. vol %",
                    "Ann. return %", "Sharpe (ann.)",
                    "Skewness", "Excess kurtosis", "Normality p-val",
                ],
                "Value": [
                    f"{r.mean()*100:.4f}", f"{r.std()*100:.4f}", f"{r.std()*ann*100:.2f}",
                    f"{((1+r.mean())**252 - 1)*100:.2f}", f"{sharpe:.3f}",
                    f"{skew:.3f}", f"{kurt:.3f}", f"{pval_norm:.4f}",
                ],
            }))

    with st.expander("Raw OHLCV Data", expanded=False):
        st.dataframe(
            df_view.style.format({c: "{:,.4f}" for c in ["open", "high", "low", "close"]}),
            use_container_width=True,
        )
        csv = df_view.reset_index().to_csv(index=False).encode()
        st.download_button(
            "Download CSV", csv,
            file_name=f"{main_symbol}_{main_timeframe}.csv", mime="text/csv",
        )

    st.caption(f"{len(df_view):,} bars  ·  {df_view.index[0]}  →  {df_view.index[-1]}")

# ═══════════════════════════════════════════════════════════════════ Results tab

with tab_results:
    if result is None:
        st.info("Configure a signal in the sidebar and click **Run Backtest**.")
    else:
        summary = result.summary()

        st.subheader("Summary")
        c1 = st.columns(4)
        c1[0].metric("Total Return",  f"{summary.get('total_return_pct', 0):.2f}%")
        c1[1].metric("Sharpe Ratio",  f"{summary.get('sharpe_ratio', 0):.3f}")
        c1[2].metric("Max Drawdown",  f"{summary.get('max_drawdown_pct', 0):.2f}%")
        c1[3].metric("Win Rate",      f"{summary.get('win_rate_pct', 0):.1f}%")

        c2 = st.columns(5)
        c2[0].metric("Calmar",        f"{summary.get('calmar_ratio', 0):.3f}")
        c2[1].metric("Sortino",       f"{summary.get('sortino_ratio', 0):.3f}")
        c2[2].metric("Profit Factor", f"{summary.get('profit_factor', 0):.3f}")
        c2[3].metric("Trades",        f"{summary.get('num_trades', 0)}")
        c2[4].metric("Total Fees",    f"${summary.get('total_fees', 0):,.2f}")

        st.divider()

        st.subheader("Equity Curve")
        eq = result.equity_curve
        dd = (eq - eq.cummax()) / eq.cummax()
        st.plotly_chart(equity_chart(eq, dd), use_container_width=True)

        st.subheader("Trades on Price Chart")
        price_fig = candlestick_chart(df, title=f"{main_symbol} — {main_timeframe}")
        trades_df = result.trades_df()
        if not trades_df.empty:
            price_fig = trade_markers(price_fig, trades_df)
        st.plotly_chart(price_fig, use_container_width=True)

        sig_log = getattr(result, "signal_log", None)
        if sig_log is not None and not sig_log.empty:
            st.subheader("Signal Log")
            st.plotly_chart(signal_log_chart(sig_log, height=320), use_container_width=True)
            with st.expander("Signal Log Table", expanded=False):
                display_log = sig_log.copy()
                if "timestamp" in display_log.columns:
                    ts_conv = pd.to_datetime(display_log["timestamp"], unit="ms", errors="coerce")
                    if not ts_conv.isna().all():
                        display_log["timestamp"] = ts_conv

                def _color_side(val):
                    return {"LONG": "color: #26a69a", "SHORT": "color: #ef5350"}.get(val, "color: #9ba3b8")

                st.dataframe(
                    display_log.style.map(
                        _color_side,
                        subset=["side"] if "side" in display_log.columns else [],
                    ),
                    use_container_width=True,
                )

        st.subheader("Trade Log")
        if trades_df.empty:
            st.info("No completed trades.")
        else:
            def _color_pnl(val):
                if isinstance(val, (int, float)):
                    return f"color: {'#26a69a' if val > 0 else '#ef5350' if val < 0 else 'inherit'}"
                return ""

            display_cols = [c for c in [
                "timestamp", "side", "size", "entry_price", "exit_price",
                "pnl", "pnl_pct", "fees", "reason_entry", "reason_exit",
            ] if c in trades_df.columns]
            st.dataframe(
                trades_df[display_cols].style.map(
                    _color_pnl, subset=["pnl"] if "pnl" in display_cols else [],
                ),
                use_container_width=True,
            )
            csv = trades_df[display_cols].to_csv(index=False).encode()
            st.download_button(
                "Download trades CSV", csv,
                file_name=f"{main_symbol}_trades.csv", mime="text/csv",
            )

# ══════════════════════════════════════════════════════════ Hypothesis Tests tab

with tab_hypothesis:
    if result is None:
        st.info("Run a backtest first (Results tab).")
    else:
        st.info(
            "Run statistical tests on the current backtest result. "
            "For a proper TTV workflow, use the Param Sweep and Regime tabs on the **test** split only, "
            "then run hypothesis tests on the held-out **validate** split."
        )

        ht_col1, ht_col2 = st.columns(2)
        n_permutations = ht_col1.number_input("Permutation test samples", value=2_000, step=500, min_value=200, key="ht_perms")
        n_bootstrap    = ht_col2.number_input("Bootstrap CI samples",     value=2_000, step=500, min_value=200, key="ht_boot")
        n_trials_input = ht_col1.number_input(
            "Number of param combos tried (for DSR)", value=1, step=1, min_value=1, key="ht_ntrials",
            help="Total parameter combinations tested on the test split — used by Deflated Sharpe Ratio.",
        )
        ci_level = ht_col2.slider("Bootstrap CI level", min_value=0.80, max_value=0.99, value=0.95, step=0.01, key="ht_ci")

        if st.button("Run Hypothesis Tests", type="primary", key="run_ht"):
            try:
                from testing.hypothesis import HypothesisTests, PermutationTest, BootstrapCI, DeflatedSharpeRatio, report
                with st.spinner("Running hypothesis tests…"):
                    tests = HypothesisTests.run_all(result)
                    pt    = PermutationTest(metric="sharpe_ratio", n_permutations=int(n_permutations)).run(result)
                    ci    = BootstrapCI(n_bootstrap=int(n_bootstrap), ci=ci_level).run(result)
                    dsr   = DeflatedSharpeRatio().compute(result, n_trials=int(n_trials_input))
                st.session_state["main_ht_results"] = (tests, pt, ci, dsr)
                st.success("Done.")
            except Exception as e:
                st.error(f"Hypothesis tests failed: {e}")
                st.exception(e)

        ht_data = st.session_state.get("main_ht_results")
        if ht_data:
            tests, pt, ci, dsr = ht_data
            from testing.hypothesis import report

            st.subheader("Statistical Test Battery")
            st.code(report(tests))

            st.subheader("Permutation Test  (Sharpe ratio)")
            verdict = "Significant" if pt.reject_null else "Not significant"
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Observed Sharpe",     f"{pt.statistic:.4f}")
            col_b.metric("Null median",          f"{pt.meta.get('null_mean', 0):.4f}")
            col_c.metric(f"p-value → {verdict}", f"{pt.p_value:.4f}")

            st.subheader(f"Bootstrap {int(ci_level*100)}% Confidence Intervals")
            ci_rows = [
                {"Metric": m, "Observed": v["observed"], "Lower": v["lower"], "Upper": v["upper"]}
                for m, v in ci.items()
            ]
            st.dataframe(pd.DataFrame(ci_rows), use_container_width=True)

            st.subheader(f"Deflated Sharpe Ratio  (n_trials={int(n_trials_input)})")
            dsr_verdict = "Genuine edge" if dsr.reject_null else "Likely overfit"
            da, db, dc, dd_ = st.columns(4)
            da.metric("Observed Sharpe", f"{dsr.observed_sharpe:.4f}")
            db.metric("Deflated Sharpe", f"{dsr.deflated_sharpe:.4f}")
            dc.metric("p-value",         f"{dsr.p_value:.4f}")
            dd_.metric("Verdict",        dsr_verdict)

# ══════════════════════════════════════════════════════════════ Param Sweep tab

with tab_sweep:
    if result is None:
        st.info("Run a backtest first (Results tab).")
    elif signal_cls is None:
        st.info("Select a signal in the sidebar first.")
    else:
        sweep_params = [
            k for k, v in inspect.signature(signal_cls.__init__).parameters.items()
            if k != "self"
            and v.default is not inspect.Parameter.empty
            and isinstance(v.default, (int, float))
        ]

        if not sweep_params:
            st.info("This strategy has no numeric parameters to sweep.")
        else:
            sc1, sc2 = st.columns(2)
            p1      = sc1.selectbox("Parameter 1", sweep_params, key="sweep_p1")
            p1_vals = sc1.text_input("Values (comma-sep)", placeholder="e.g. 5,10,15,20", key="sweep_p1v")

            p2_opts = ["(none)"] + [p for p in sweep_params if p != p1]
            p2      = sc2.selectbox("Parameter 2 (optional)", p2_opts, key="sweep_p2")
            p2_vals = sc2.text_input("Values (comma-sep)", placeholder="e.g. 20,30,40", key="sweep_p2v") \
                if p2 != "(none)" else ""

            metric_pick = st.selectbox("Optimise metric", [
                "sharpe_ratio", "total_return_pct", "max_drawdown_pct",
                "calmar_ratio", "win_rate_pct", "profit_factor",
            ], key="sweep_metric")

            if st.button("Run Sweep", type="primary", key="run_sweep"):
                try:
                    vals1 = [float(v.strip()) for v in p1_vals.split(",") if v.strip()]
                    vals2 = [float(v.strip()) for v in p2_vals.split(",") if v.strip()] if p2_vals else []
                    if not vals1:
                        st.warning("Enter at least one value for Parameter 1.")
                    else:
                        from testing.backtester.costs import CompositeCostModel, aggressive_cost_stack
                        from testing.backtester.stress import ParamSweep

                        param_grid = {p1: vals1}
                        if vals2:
                            param_grid[p2] = vals2
                        n_combos = len(vals1) * max(len(vals2), 1)
                        uni  = build_universe(main_symbol, df)
                        cost = CompositeCostModel(models=aggressive_cost_stack())

                        with st.spinner(f"Running {n_combos} backtest(s)…"):
                            sweep_result = ParamSweep(
                                strategy_cls=signal_cls,
                                param_grid=param_grid,
                                config=config,
                                cost_model=cost,
                                sizer=sizer,
                                stop_loss=stop,
                            ).run(universe=uni, timeframe=main_timeframe)

                        st.session_state["main_sweep_result"] = (sweep_result, p1, p2, metric_pick, vals1, vals2)
                        st.info(f"Tracked {n_combos} trial(s) — enter this in Hypothesis Tests for DSR correction.")
                        st.success("Sweep complete.")
                except Exception as e:
                    st.error(f"Sweep failed: {e}")
                    st.exception(e)

    sweep_data = st.session_state.get("main_sweep_result")
    if sweep_data:
        sweep, p1, p2, metric, vals1, vals2 = sweep_data
        summary_df: pd.DataFrame = sweep.summary

        if (p2 and p2 != "(none)" and vals2
                and all(c in summary_df.columns for c in [metric, p1, p2])):
            pivot = summary_df.pivot(index=p2, columns=p1, values=metric)
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale="RdYlGn",
                text=[[f"{v:.3f}" for v in row] for row in pivot.values],
                texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                template="plotly_dark", height=420,
                title=f"{metric} — {p1} vs {p2}",
                xaxis_title=p1, yaxis_title=p2,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        elif metric in summary_df.columns and p1 in summary_df.columns:
            fig_bar = go.Figure(go.Bar(
                x=summary_df[p1].astype(str), y=summary_df[metric], marker_color="#2196F3",
            ))
            fig_bar.update_layout(
                template="plotly_dark", height=300,
                title=f"{metric} vs {p1}",
                xaxis_title=p1, yaxis_title=metric,
                margin=dict(l=40, r=40, t=50, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if metric in summary_df.columns:
            best  = sweep.best(metric)
            worst = sweep.worst(metric)
            b1, b2 = st.columns(2)
            b1.success(f"Best  {metric}: {best[metric]:.4f} @ {best.drop(metric).to_dict()}")
            b2.error(  f"Worst {metric}: {worst[metric]:.4f} @ {worst.drop(metric).to_dict()}")

        st.dataframe(summary_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════ Regime Test tab

with tab_regime:
    if result is None:
        st.info("Run a backtest first (Results tab).")
    elif signal_cls is None:
        st.info("Select a signal in the sidebar first.")
    else:
        regime_choice = st.radio(
            "Regime classifier",
            ["Volatility (default)", "Trend (SMA)", "Volume"],
            key="regime_choice",
        )
        sma_win = (
            st.number_input("SMA window", value=50, step=5, min_value=10, key="regime_sma")
            if regime_choice == "Trend (SMA)" else 50
        )

        if st.button("Run Regime Test", type="primary", key="run_regime"):
            try:
                from strategy.built_in import SingleAssetStrategy
                from testing.backtester.stress import RegimeStressTest
                from testing.backtester.costs import CompositeCostModel, aggressive_cost_stack

                strategy_reg = signal_cls(symbol=main_symbol, **sig_params) \
                    if issubclass(signal_cls, SingleAssetStrategy) \
                    else signal_cls(**sig_params)

                if regime_choice == "Volatility (default)":
                    regime_fn = None
                elif regime_choice == "Trend (SMA)":
                    _w = sma_win
                    regime_fn = lambda df_: RegimeStressTest.trend_regime(df_, sma_window=_w)
                else:
                    regime_fn = RegimeStressTest.volume_regime

                cost = CompositeCostModel(models=aggressive_cost_stack())
                uni  = build_universe(main_symbol, df)

                with st.spinner("Running regime stress test…"):
                    regime_new = RegimeStressTest(
                        regime_fn=regime_fn, config=config, cost_model=cost,
                    ).run(strategy=strategy_reg, universe=uni)

                st.session_state["main_regime_result"] = regime_new
                st.success("Regime test complete.")
            except Exception as e:
                st.error(f"Regime test failed: {e}")
                st.exception(e)

    regime_result = st.session_state.get("main_regime_result")
    if regime_result is not None:
        st.dataframe(regime_result.summary, use_container_width=True)
        for regime_name, res in regime_result.results.items():
            with st.expander(f"Regime: {regime_name}", expanded=False):
                eq_r = res.equity_curve
                dd_r = (eq_r - eq_r.cummax()) / eq_r.cummax()
                st.plotly_chart(equity_chart(eq_r, dd_r), use_container_width=True)
                rs = res.summary()
                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Return", f"{rs.get('total_return_pct', 0):.2f}%")
                rc2.metric("Sharpe", f"{rs.get('sharpe_ratio', 0):.3f}")
                rc3.metric("Max DD", f"{rs.get('max_drawdown_pct', 0):.2f}%")
                rc4.metric("Trades", f"{rs.get('num_trades', 0)}")

# ══════════════════════════════════════════════════════════════ Monte Carlo tab

with tab_mc:
    if result is None:
        st.info("Run a backtest first (Results tab).")
    else:
        trades_df_mc = result.trades_df()
        if trades_df_mc.empty or "pnl" not in trades_df_mc.columns:
            st.warning("No trades with PnL data available for Monte Carlo simulation.")
        else:
            mc1, mc2, mc3 = st.columns(3)
            n_sims    = mc1.number_input("Simulations", value=1000, step=100, min_value=100, max_value=10000, key="mc_n")
            mc_method = mc2.radio("Method", ["bootstrap", "shuffle", "block_bootstrap"], key="mc_method")
            mc_seed   = mc3.number_input("Seed", value=42, step=1, key="mc_seed")

            if st.button("Run Monte Carlo", type="primary", key="run_mc"):
                try:
                    from testing.backtester.stress import MonteCarloStress
                    with st.spinner(f"Running {n_sims} simulations…"):
                        mc_new = MonteCarloStress(
                            n_simulations=int(n_sims), seed=int(mc_seed), method=mc_method,
                        ).run(result)
                    st.session_state["main_mc_result"] = mc_new
                    st.success("Monte Carlo complete.")
                except Exception as e:
                    st.error(f"Monte Carlo failed: {e}")
                    st.exception(e)

    mc_result = st.session_state.get("main_mc_result")
    if mc_result is not None and not mc_result.summary.empty:
        meta = mc_result.meta
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Median Return",    f"{meta.get('median_return', 0):.2f}%")
        mm2.metric("5th Pctl Return",  f"{meta.get('5th_pctl_return', 0):.2f}%")
        mm3.metric("95th Pctl Return", f"{meta.get('95th_pctl_return', 0):.2f}%")
        mm4.metric("Median Max DD",    f"{meta.get('median_max_dd', 0):.2f}%")

        mc_df = mc_result.summary

        def _mc_hist(col: str, title: str) -> go.Figure:
            vals = mc_df[col]
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=vals, nbinsx=60, marker_color="#2196F3", opacity=0.75))
            fig_mc.add_vline(x=vals.median(), line_dash="dash", line_color="#ef5350",
                             annotation_text=f"Median {vals.median():.2f}", annotation_position="top right")
            fig_mc.add_vline(x=vals.quantile(0.05), line_dash="dot", line_color="#FF9800",
                             annotation_text=f"5th {vals.quantile(0.05):.2f}", annotation_position="top left")
            fig_mc.add_vline(x=vals.quantile(0.95), line_dash="dot", line_color="#26a69a",
                             annotation_text=f"95th {vals.quantile(0.95):.2f}", annotation_position="top right")
            fig_mc.update_layout(template="plotly_dark", height=300, title=title,
                                 showlegend=False, margin=dict(l=40, r=20, t=50, b=20))
            return fig_mc

        st.plotly_chart(_mc_hist("total_return_pct", "Return % Distribution"), use_container_width=True)
        st.plotly_chart(_mc_hist("max_drawdown_pct", "Max Drawdown % Distribution"), use_container_width=True)

        with st.expander("Full statistics"):
            st.dataframe(mc_df.describe(), use_container_width=True)

        st.download_button(
            "Download MC results CSV",
            mc_df.to_csv(index=False).encode(),
            file_name="monte_carlo_results.csv",
            mime="text/csv",
        )
