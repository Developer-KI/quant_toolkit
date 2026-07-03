"""Live Trading — configure, launch, and monitor the Alpaca live engine."""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_ROOT / "src"), str(_ROOT), str(_ROOT / "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from components.alpaca_data import TIMEFRAMES, get_credentials, to_core_timeframe
from components.charts import candlestick_chart
from components.forms import signal_form, sizer_form, stop_form
from components.style import inject

st.set_page_config(page_title="Live Trading", page_icon="🚀", layout="wide")
inject()
st.title("Live Trading")

# ── Session state guards ──────────────────────────────────────────────────────

if "runner" not in st.session_state:
    from components.engine_runner import EngineRunner
    st.session_state["runner"] = EngineRunner()

runner = st.session_state["runner"]

# ── Layout ────────────────────────────────────────────────────────────────────

col_cfg, col_dash = st.columns([4, 6])

# ═══════════════════════════════════════════════════════════ Configuration col

with col_cfg:
    st.subheader("Configuration")
    disabled = runner.is_alive

    # Credentials
    with st.expander("Alpaca API", expanded=not disabled):
        env_key, env_secret = get_credentials()
        api_key = st.text_input("API Key",    value=env_key,    type="password", key="live_alp_key",    disabled=disabled)
        api_secret = st.text_input("API Secret", value=env_secret, type="password", key="live_alp_secret", disabled=disabled)
        paper = st.checkbox("Paper trading", value=True, key="live_paper", disabled=disabled)
        st.caption("Paper trading — no leverage; market hours 9:30–16:00 ET")

    # Market
    from core.parser import timeframe_to_seconds
    symbol = st.text_input("Symbol", value="SPY", key="live_sym", disabled=disabled).upper()
    live_timeframe = st.selectbox("Bar timeframe", TIMEFRAMES, index=0, key="live_tf", disabled=disabled)
    warmup_bars = st.number_input("Warm-up bars", value=200, step=10, min_value=10, key="live_warmup", disabled=disabled)

    # Risk
    st.markdown("**Risk**")
    max_pos_pct = st.number_input("Max position %", value=0.25, step=0.05,
                                   min_value=0.01, max_value=1.0, key="live_mpp", disabled=disabled)
    leverage = st.number_input("Leverage", value=1.0, step=0.5,
                                min_value=1.0, max_value=20.0, key="live_lev", disabled=disabled)

    # Daily limits
    st.markdown("**Daily limits**")
    max_daily_trades = st.number_input("Max daily trades", value=50, step=5,
                                        min_value=1, key="live_mdt", disabled=disabled)
    max_daily_loss = st.number_input("Max daily loss %", value=5.0, step=0.5,
                                      min_value=0.5, key="live_mdl", disabled=disabled)

    # Signal / sizer / stop
    st.divider()
    st.markdown("**Signal**")
    live_signal_cls, live_sig_params = signal_form(col_cfg, key_prefix="live_sig")
    st.markdown("**Sizer**")
    live_sizer = sizer_form(col_cfg, key_prefix="live_sizer")
    st.markdown("**Stop loss**")
    live_stop = stop_form(col_cfg, key_prefix="live_stop")

    st.divider()

    # Control buttons
    btn_start, btn_stop, btn_flat = st.columns(3)
    with btn_start:
        start_pressed = st.button(
            "Start Engine", type="primary",
            disabled=runner.is_alive or live_signal_cls is None,
            use_container_width=True,
        )
    with btn_stop:
        stop_pressed = st.button(
            "Stop Engine",
            disabled=not runner.is_alive,
            use_container_width=True,
        )
    with btn_flat:
        flat_pressed = st.button(
            "Emergency Flatten",
            disabled=not runner.is_alive,
            use_container_width=True,
            type="secondary",
        )

# ── Handle button actions ─────────────────────────────────────────────────────

if start_pressed and live_signal_cls is not None:
    try:
        from core.models import LiveConfig, ExchangeCredentials
        from execution import Engine
        from strategy.built_in import SingleAssetStrategy

        if issubclass(live_signal_cls, SingleAssetStrategy):
            live_strategy = live_signal_cls(symbol=symbol, **live_sig_params)
        else:
            live_strategy = live_signal_cls(**live_sig_params)

        config = LiveConfig(
            exchange="alpaca",
            use_testnet=paper,
            exchanges=[ExchangeCredentials(
                exchange="alpaca",
                api_key=api_key,
                api_secret=api_secret,
                testnet=paper,
            )],
            symbol=symbol,
            bar_interval_s=timeframe_to_seconds(to_core_timeframe(live_timeframe)),
            warmup_bars=int(warmup_bars),
            max_position_pct=max_pos_pct,
            leverage=leverage,
            max_daily_trades=int(max_daily_trades),
            max_daily_loss_pct=max_daily_loss,
        )
        engine = Engine(
            strategy=live_strategy,
            config=config,
            sizer=live_sizer,
            stop_loss=live_stop,
        )
        runner.start(engine)
        st.success("Engine starting…")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        st.error(f"Failed to start engine: {e}")
        st.exception(e)

if stop_pressed:
    runner.stop()
    st.info("Stop signal sent.")
    time.sleep(0.5)
    st.rerun()

if flat_pressed:
    runner.emergency_flatten()
    st.warning("Emergency flatten triggered — all positions will be closed.")
    time.sleep(0.5)
    st.rerun()

# ═══════════════════════════════════════════════════════════ Dashboard col

with col_dash:
    st.subheader("Status")

    status = runner.status
    if status == "running":
        st.success("Engine is **running**")
    elif status == "starting":
        st.info("Engine is **starting** (warming up)…")
    elif status == "error":
        st.error("Engine stopped with error")
        if runner.error:
            with st.expander("Error details"):
                st.code(runner.error)
    else:
        st.warning("Engine is **stopped**")

    state = runner.state
    if state is not None:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Equity", f"${state.equity:,.2f}")

        daily_pnl_pct = (state.daily_pnl / state.starting_equity * 100) if state.starting_equity else 0
        mc2.metric("Daily PnL", f"${state.daily_pnl:,.2f}", delta=f"{daily_pnl_pct:+.2f}%")

        pos = state.position
        mc3.metric("Position", f"{pos.side.name} {pos.size:.4f}" if pos.size else "FLAT")

        peak_dd = ((state.equity - state.peak_equity) / state.peak_equity * 100) if state.peak_equity else 0
        mc4.metric("Peak DD", f"{peak_dd:.2f}%")

        # Daily limits
        st.markdown("**Daily limits**")
        if state.starting_equity and runner.engine:
            max_loss = runner.engine.config.max_daily_loss_pct / 100
            loss_used = min(abs(min(state.daily_pnl, 0)) / (state.starting_equity * max_loss + 1e-9), 1.0)
            st.progress(loss_used, text=f"Daily loss: ${state.daily_pnl:+,.2f} / {runner.engine.config.max_daily_loss_pct:.1f}% limit")

            max_trades = runner.engine.config.max_daily_trades
            trades_used = min(state.daily_trades / max(max_trades, 1), 1.0)
            st.progress(trades_used, text=f"Daily trades: {state.daily_trades} / {max_trades}")

        st.divider()

        # Live price chart
        assets = runner.assets
        if assets:
            first_key = next(iter(assets))
            ast = assets[first_key]
            display_sym = first_key.split(":", 1)[-1]
            if ast.bar_builder is not None:
                try:
                    live_df = ast.bar_builder.to_dataframe()
                    if not live_df.empty:
                        live_view = live_df.tail(100)
                        st.plotly_chart(
                            candlestick_chart(live_view, title=f"{display_sym} — last {len(live_view)} bars", height=350),
                            use_container_width=True,
                        )
                except Exception:
                    pass

        # Open trade
        if state.trades:
            open_trade = state.trades[-1]
            if open_trade.exit_price is None:
                with st.expander("Open trade", expanded=True):
                    oc1, oc2, oc3 = st.columns(3)
                    oc1.metric("Side", open_trade.side.name)
                    oc2.metric("Size", f"{open_trade.size:.4f}")
                    oc3.metric("Entry", f"${open_trade.entry_price:,.4f}")
                    if state.positions:
                        unr = state.position.unrealized_pnl
                        st.metric("Unrealised PnL", f"${unr:+,.2f}" if unr else "—")

        # Closed trades
        closed = state.closed_trades
        if closed:
            st.markdown(f"**Closed trades ({len(closed)})**")
            rows = []
            for t in reversed(closed[-50:]):
                rows.append({
                    "Time":  str(t.timestamp)[:19],
                    "Side":  t.side.name,
                    "Size":  f"{t.size:.4f}",
                    "Entry": f"${t.entry_price:,.4f}",
                    "Exit":  f"${t.exit_price:,.4f}" if t.exit_price else "—",
                    "PnL":   f"${t.pnl:+,.2f}",
                    "Reason": t.reason_exit or "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Start the engine to see live metrics.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────

if runner.is_alive:
    time.sleep(3)
    st.rerun()
