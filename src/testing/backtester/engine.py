"""
backtester/engine.py — Unified backtester engine.

  Single-asset:
      strategy = EMACrossStrategy(symbol="ETH", fast=12, slow=26)
      bt = Backtester(strategy=strategy)
      result = bt.run(data=eth_df)

  Multi-asset:
      bt = Backtester(strategy=my_strategy)
      result = bt.run(universe=universe)

The returned BacktestResult has the same .summary(), .trades_df(),
.plot_equity(), .to_csv() interface for both cases.  Multi-asset runs
add positions_log, allocation_log, and trades_by_symbol.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models import (
    BacktestConfig,
    OrderBookSnapshot,
    Position,
    Side,
    Trade,
)
from core.parser import timeframe_to_seconds
from testing.backtester.costs import CostModel, CompositeCostModel, NullCostModel
from strategy.sizing import Sizer, SizingContext, default_sizer
from strategy.stops import (
    StopLoss,
    StopContext,
    EmbeddedStop,
    default_stop_loss,
)

from strategy.base import Strategy, StrategyContext, PortfolioTarget
from core.universe import Universe


def _cost_model_label(m: CostModel) -> "str | list[str]":
    if isinstance(m, CompositeCostModel):
        return [type(c).__name__ for c in m.models]
    return type(m).__name__


# ── Result container (superset of old BacktestResult) ────────────────────────


@dataclass
class BacktestResult:
    """
    Backward-compatible result container.

    Has every field the old BacktestResult had, plus optional multi-asset
    extras.  Code that used the old result object works unchanged.
    """

    trades: list[Trade]
    equity_curve: pd.Series
    positions: pd.Series                            # side per bar (single-asset compat)
    config: BacktestConfig
    run_time_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    # ── multi-asset extras (None when single-asset) ──────────────────────
    positions_log: pd.DataFrame | None = None       # per-bar, per-asset positions
    allocation_log: pd.DataFrame | None = None      # per-bar, per-asset allocations
    # ── multi-exchange extras (None when single-exchange) ─────────────────
    equity_curves_by_exchange: dict[str, pd.Series] | None = None

    # ── export helpers ───────────────────────────────────────────────────

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def trades_by_symbol(self, symbol: str) -> pd.DataFrame:
        """Filter trades to one symbol (multi-asset runs)."""
        df = self.trades_df()
        if df.empty or "meta_symbol" not in df.columns:
            return df
        return df[df["meta_symbol"] == symbol]

    def to_csv(self, path: str = "trades.csv"):
        df = self.trades_df()
        df.to_csv(path, index=False)
        return path

    # ── analytics (identical to old engine) ───────────────────────────────

    def summary(self) -> dict[str, Any]:
        eq = self.equity_curve
        returns = eq.pct_change().dropna()
        tdf = self.trades_df()

        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1 if len(eq) > 1 else 0
        n_bars = len(eq)

        ann_factor = 365 * 24 if n_bars > 1 else 1
        if "timeframe" in self.meta:
            ann_factor = int(365 * 24 * 3600 / timeframe_to_seconds(self.meta["timeframe"]))
        elif isinstance(eq.index, pd.DatetimeIndex) and len(eq.index) > 2:
            secs = eq.index.to_series().diff().median().total_seconds()
            if secs > 0:
                ann_factor = int(365 * 24 * 3600 / secs)

        ann_return = (1 + total_return) ** (ann_factor / max(n_bars, 1)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor) if len(returns) > 1 else 0
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = dd.min()

        underwater = dd < 0
        pct_time_underwater = float(underwater.mean() * 100)
        uw_arr = underwater.values
        if uw_arr.any():
            transitions = np.diff(uw_arr.astype(np.int8), prepend=0, append=0)
            lengths = np.where(transitions == -1)[0] - np.where(transitions == 1)[0]
            max_time_underwater_bars = int(lengths.max())
            avg_time_underwater_bars = float(lengths.mean())
        else:
            max_time_underwater_bars = 0
            avg_time_underwater_bars = 0.0

        if len(tdf) > 0 and "pnl" in tdf.columns:
            wins = (tdf["pnl"] > 0).sum()
            win_rate = wins / len(tdf)
            avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if wins > 0 else 0
            avg_loss = (
                tdf.loc[tdf["pnl"] <= 0, "pnl"].mean()
                if (len(tdf) - wins) > 0
                else 0
            )
            profit_factor = (
                abs(avg_win * wins / (avg_loss * (len(tdf) - wins)))
                if avg_loss != 0
                else np.inf
            )
            total_fees = tdf["fees"].sum()
        else:
            win_rate = avg_win = avg_loss = profit_factor = total_fees = 0

        calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
        sortino_vol = (
            returns[returns < 0].std() * np.sqrt(ann_factor)
            if (returns < 0).any()
            else 0
        )
        sortino = ann_return / sortino_vol if sortino_vol > 0 else 0

        result = {
            "total_return_pct": round(total_return * 100, 4),
            "annualised_return_pct": round(ann_return * 100, 4),
            "annualised_volatility_pct": round(ann_vol * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "pct_time_underwater": round(pct_time_underwater, 2),
            "max_time_underwater_bars": max_time_underwater_bars,
            "avg_time_underwater_bars": round(avg_time_underwater_bars, 1),
            "num_trades": len(tdf),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "total_fees": round(total_fees, 4),
            "run_time_s": round(self.run_time_s, 3),
        }

        # Extra field for multi-asset
        symbols = self.meta.get("symbols")
        if symbols and len(symbols) > 1:
            result["symbols_traded"] = list(
                tdf["meta_symbol"].unique()
            ) if "meta_symbol" in tdf.columns else symbols

        # Extra field for multi-exchange
        exchanges = self.meta.get("exchanges")
        if exchanges:
            result["exchanges"] = exchanges

        return result

    def plot_equity(self, save_path: str | None = None, show: bool = False):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        eq = self.equity_curve
        # Normalise to 1.0 at start so curves are comparable regardless of capital
        norm = eq / eq.iloc[0] if eq.iloc[0] != 0 else eq
        peak = norm.cummax()
        dd = (norm - peak) / peak

        ax1.plot(norm.index, norm.values, linewidth=1.2, color="#2563eb", label="Equity")
        ax1.fill_between(norm.index, norm.values, 1.0, where=(norm.values >= 1.0),
                         alpha=0.08, color="#2563eb")
        ax1.fill_between(norm.index, norm.values, 1.0, where=(norm.values < 1.0),
                         alpha=0.08, color="#dc2626")
        ax1.axhline(1.0, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.2f}×")
        )
        summary = self.summary()
        title_parts = [
            f"Total: {summary['total_return_pct']:+.1f}%",
            f"Sharpe: {summary['sharpe_ratio']:.2f}",
            f"Max DD: {summary['max_drawdown_pct']:.1f}%",
        ]
        symbols = self.meta.get("symbols")
        label = ", ".join(symbols) if symbols else "Equity"
        ax1.set_ylabel("Normalised equity")
        ax1.set_title(f"{label}  —  " + "  |  ".join(title_parts))
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(dd.index, dd.values * 100, 0, color="#dc2626", alpha=0.4)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = save_path or "equity_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return path

    def save(self, run_name: str, base_dir: str = "logs/test") -> str:
        """Save log.json, trades.csv, and equity_curve.png to logs/test/<run_name>/<timestamp>/."""
        import dataclasses
        from datetime import datetime, timezone

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        run_dir = Path(base_dir) / run_name / ts_str
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = self.summary()
        config_dict = dataclasses.asdict(self.config)

        eq = self.equity_curve
        data_range: dict[str, Any] = {"bars": len(eq)}
        if isinstance(eq.index, pd.DatetimeIndex) and len(eq) > 0:
            data_range["start"] = str(eq.index[0])
            data_range["end"] = str(eq.index[-1])

        meta_out: dict[str, Any] = {}
        for k, v in self.meta.items():
            if isinstance(v, (list, tuple)):
                meta_out[k] = list(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                meta_out[k] = v
            else:
                meta_out[k] = str(v)

        log_data = {
            "run_name": run_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "data_range": data_range,
            "summary": summary,
            "config": config_dict,
            "meta": meta_out,
        }
        with open(run_dir / "log.json", "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        self.to_csv(run_dir / "trades.csv")
        self.plot_equity(save_path=run_dir / "equity_curve.png")

        if self.equity_curves_by_exchange:
            ex_df = pd.DataFrame(self.equity_curves_by_exchange)
            ex_df.index.name = "timestamp"
            ex_df.to_csv(run_dir / "equity_curves_by_exchange.csv")

        return str(run_dir)


# ── Per-asset state ──────────────────────────────────────────────────────────


@dataclass
class _AssetState:
    position: Position = field(default_factory=Position)
    stop_loss: StopLoss | None = None
    open_trade: Trade | None = None


# ── Unified Backtester ───────────────────────────────────────────────────────


class Backtester:
    """
    Backtester engine.

    Accepts a Strategy and runs it over a Universe (or single DataFrame).

    The run() method accepts EITHER:
      • data + l2   — single DataFrame (auto-wrapped in a Universe)
      • universe    — multi-asset Universe

    Components (sizer, stop_loss, cost_model) can be:
      • A single instance           — shared across all assets
      • A dict[symbol, instance]    — per-asset overrides
    """

    def __init__(
        self,
        strategy: Strategy,
        config: BacktestConfig | None = None,
        cost_model: CostModel | dict[str, CostModel] | None = None,
        sizer: Sizer | dict[str, Sizer] | None = None,
        stop_loss: StopLoss | dict[str, StopLoss] | None = None,
        symbol: str | None = None,
        exchange_costs: dict[str, CostModel] | None = None,
        capital_by_exchange: dict[str, float] | None = None,
    ):
        self.config = config or BacktestConfig()
        self._cost_model_spec = cost_model
        self._sizer_spec = sizer
        self._stop_loss_spec = stop_loss
        self._strategy = strategy
        self._default_symbol = symbol or "ASSET"
        self._exchange_costs = exchange_costs or {}
        self._capital_by_exchange = capital_by_exchange

    # ── Component resolution ─────────────────────────────────────────────

    def _resolve(self, spec, symbol, default_fn):
        if isinstance(spec, dict):
            return copy.deepcopy(spec.get(symbol, default_fn()))
        elif spec is not None:
            return copy.deepcopy(spec)
        return default_fn()

    # ── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        data: pd.DataFrame | None = None,
        l2: list[OrderBookSnapshot] | None = None,
        universe: Universe | None = None,
        timeframe: str | None = None,
        universes: dict[str, Universe] | None = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Old API:  result = bt.run(data=df, l2=snapshots)
        New API:  result = bt.run(universe=universe, timeframe="1h")
        Multi-exchange: result = bt.run(universes={"binance": u1, "kraken": u2})

        timeframe — bar size label (e.g. "1m", "5m", "1h", "1d").
                    Used to compute the annualisation factor for Sharpe/vol metrics.
                    When omitted the factor is inferred from the bar index spacing.
        """
        n_sources = sum(x is not None for x in (data, universe, universes))
        if n_sources > 1:
            raise ValueError("Provide exactly one of data=, universe=, or universes=")

        if universes is not None:
            if len(universes) == 1:
                # Single-exchange dict: unwrap and use the normal path
                universe = next(iter(universes.values()))
            else:
                return self._run_loop_multi_exchange(
                    strategy=self._strategy,
                    universes=universes,
                    timeframe=timeframe,
                )

        if data is not None:
            sym = self._default_symbol
            universe = Universe(symbols=[sym])
            universe.add_asset(sym, data, l2=l2)
        elif universe is None:
            raise ValueError("Provide either data=, universe=, or universes=")

        symbols = universe.symbols
        is_single_asset = len(symbols) == 1

        return self._run_loop(
            strategy=self._strategy,
            universe=universe,
            symbols=symbols,
            is_single_asset=is_single_asset,
            timeframe=timeframe,
        )

    # ── Vectorised fast-path helpers ──────────────────────────────────────

    def _can_vectorize(
        self,
        symbols: list[str],
        states: dict[str, _AssetState],
        sizers: dict[str, Sizer],
    ) -> bool:
        """True when stops are all NopStopLoss and every sizer is vectorizable."""
        from strategy.stops import NopStopLoss
        return (
            all(isinstance(states[s].stop_loss, NopStopLoss) for s in symbols)
            and all(sizers[s].vectorizable for s in symbols)
        )

    def _run_vectorized(
        self,
        t0: float,
        sides_all: dict[str, np.ndarray],
        weights_all: dict[str, np.ndarray],
        universe: Universe,
        symbols: list[str],
        is_single_asset: bool,
        timeframe: str | None,
        index,
        n_bars: int,
        sizers: dict[str, Sizer],
        cost_models: dict[str, CostModel],
        reasons_all: dict[str, list[str]] | None = None,
        metas_all: dict[str, list[dict]] | None = None,
        confidences_all: dict[str, np.ndarray] | None = None,
    ) -> BacktestResult:
        """
        Fully vectorised backtest — replaces the O(N) Python bar loop with
        NumPy array operations when stops are NopStopLoss and sizers are
        vectorizable.

        Algorithm per symbol
        --------------------
        1. Detect entry / exit bars as transitions in the side array.
        2. Compute entry prices, sizes, and fees for each trade (O(M) where
           M = number of trades).
        3. Build the equity curve without any bar iteration:
           - Scatter entry-fee deductions and gross-PnL events onto their bars
             with np.add.at, then cumsum to get realised equity.
           - Forward-fill entry prices and sizes to every in-position bar using
             np.maximum.accumulate, then compute unrealised PnL elementwise.
           - equity[i] = initial_capital + cumsum_realised[i] + unrealised[i]
        """
        initial_capital = self.config.initial_capital

        # Pre-extract close prices per symbol
        _closes: dict[str, np.ndarray] = {}
        for sym in symbols:
            df = universe.ohlcv(sym)
            locs = df.index.get_indexer(index)
            vi = np.where(locs >= 0)[0]
            arr = np.full(n_bars, np.nan)
            arr[vi] = df["close"].values[locs[vi]]
            _closes[sym] = arr

        # Accumulate equity deltas across all symbols
        equity_changes = np.zeros(n_bars)
        all_trades: list[Trade] = []

        if is_single_asset:
            sym0 = symbols[0]
            _pos_sides_arr = sides_all[sym0].astype(np.int8)

        for sym in symbols:
            sides   = sides_all[sym]      # (n_bars,) int8: 1, -1, 0
            weights = weights_all[sym]    # (n_bars,) float64

            closes = _closes[sym]
            valid  = ~np.isnan(closes)

            # Detect transitions: side changed from previous bar
            prev = np.empty_like(sides)
            prev[0] = 0
            prev[1:] = sides[:-1]
            changed = sides != prev

            # Entry: side becomes non-zero at a bar with a valid price
            entry_mask = changed & (sides != 0) & valid
            # Close: side transitions FROM non-zero (includes flips)
            close_mask = changed & (prev != 0)

            entry_bars = np.where(entry_mask)[0]
            close_bars_natural = np.where(close_mask)[0]

            n_natural = len(close_bars_natural)
            was_force_closed = len(entry_bars) > n_natural
            close_bars = (
                np.append(close_bars_natural, n_bars - 1)
                if was_force_closed
                else close_bars_natural
            )

            if len(entry_bars) == 0:
                continue

            entry_prices  = closes[entry_bars]
            exit_prices   = closes[close_bars]
            directions    = sides[entry_bars].astype(np.float64)
            entry_weights = weights[entry_bars]

            # ── Sizes ────────────────────────────────────────────────────
            sizes = sizers[sym].compute_vectorized(
                entry_prices, entry_weights, self.config
            )
            if is_single_asset:
                max_notional = (
                    initial_capital
                    * self.config.max_position_pct
                    * self.config.leverage
                    * np.ones(len(entry_bars))
                )
            else:
                max_notional = initial_capital * entry_weights * self.config.leverage
            sizes = np.minimum(sizes, np.where(entry_prices > 0, max_notional / entry_prices, 0.0))
            sizes = np.maximum(sizes, 0.0)

            # ── Fees per trade (O(M), M << N) ────────────────────────────
            # Pass the position's side for both entry and exit (matches sequential path)
            entry_fees = np.array([
                cost_models[sym].compute(ep, sz, Side(int(d)), self.config, None, {})
                for ep, sz, d in zip(entry_prices, sizes, directions)
            ])
            exit_fees = np.array([
                cost_models[sym].compute(xp, sz, Side(int(d)), self.config, None, {})
                for xp, sz, d in zip(exit_prices, sizes, directions)
            ])

            gross_pnl = (exit_prices - entry_prices) * sizes * directions
            net_pnl   = gross_pnl - entry_fees - exit_fees

            # ── Equity curve (fully vectorised, no trade loop) ────────────
            # Entry bars: deduct entry fee immediately
            entry_fee_arr = np.zeros(n_bars)
            np.add.at(entry_fee_arr, entry_bars, entry_fees)

            # Close bars: add gross PnL minus exit fee
            close_event_arr = np.zeros(n_bars)
            np.add.at(close_event_arr, close_bars, gross_pnl - exit_fees)

            cum_realized = np.cumsum(close_event_arr - entry_fee_arr)

            # Unrealised PnL: forward-fill entry price and size to every in-position bar
            # using np.maximum.accumulate on the "last entry bar seen so far" index.
            last_entry_idx = np.where(entry_mask, np.arange(n_bars), 0)
            np.maximum.accumulate(last_entry_idx, out=last_entry_idx)

            ep_sparse   = np.zeros(n_bars)
            sz_sparse   = np.zeros(n_bars)
            np.add.at(ep_sparse, entry_bars, entry_prices)
            np.add.at(sz_sparse, entry_bars, sizes)

            in_pos = (sides != 0) & valid
            # At force-close bars the strategy output hasn't transitioned to 0, so sides[bar]
            # is still non-zero — but the position is exited at this bar's close.
            # Exclude these from unrealized to prevent double-counting with the
            # close event already scattered into close_event_arr.
            if was_force_closed:
                in_pos[close_bars[n_natural:]] = False

            act_entry = np.where(in_pos, ep_sparse[last_entry_idx], 0.0)
            act_size  = np.where(in_pos, sz_sparse[last_entry_idx], 0.0)
            act_dir   = np.where(in_pos, sides.astype(np.float64), 0.0)

            unrealized = (closes - act_entry) * act_size * act_dir
            np.nan_to_num(unrealized, nan=0.0, copy=False)

            equity_changes += cum_realized + unrealized

            # ── Build Trade objects (O(M)) ────────────────────────────────
            sym_reasons     = reasons_all.get(sym)     if reasons_all     else None
            sym_metas       = metas_all.get(sym)       if metas_all       else None
            sym_confidences = confidences_all.get(sym) if confidences_all else None
            for k in range(len(entry_bars)):
                is_forced = was_force_closed and k == len(entry_bars) - 1
                notional  = float(entry_prices[k]) * float(sizes[k])
                trade_meta = {"symbol": sym} if not is_single_asset else {}

                entry_i = int(entry_bars[k])
                close_i = int(close_bars[k])

                reason_entry = sym_reasons[entry_i]       if sym_reasons     else ""
                bar_vals     = sym_metas[entry_i]         if sym_metas       else {}
                confidence   = float(sym_confidences[entry_i]) if sym_confidences is not None else 0.0

                if is_forced:
                    reason_exit = "End of data"
                elif sym_reasons:
                    reason_exit = sym_reasons[close_i] or "strategy"
                else:
                    reason_exit = "strategy"

                all_trades.append(Trade(
                    timestamp=index[entry_i],
                    side=Side(int(directions[k])),
                    size=float(sizes[k]),
                    entry_price=float(entry_prices[k]),
                    exit_price=float(exit_prices[k]),
                    exit_timestamp=index[close_i],
                    pnl=float(net_pnl[k]),
                    pnl_pct=float(net_pnl[k] / notional) if notional > 0 else 0.0,
                    fees=float(entry_fees[k] + exit_fees[k]),
                    confidence=confidence,
                    reason_entry=reason_entry,
                    reason_exit=reason_exit,
                    bar_values=bar_vals,
                    meta=trade_meta,
                ))

        # ── Assemble result ───────────────────────────────────────────────
        equity_arr = initial_capital + equity_changes
        eq_series  = pd.Series(equity_arr, index=index, name="equity")

        if is_single_asset:
            pos_series = pd.Series(_pos_sides_arr, index=index, name="position")
        else:
            all_trades.sort(key=lambda t: t.timestamp)
            pos_series = pd.Series(np.zeros(n_bars, dtype=np.int8), index=index, name="position")

        sym0 = symbols[0]
        meta: dict[str, Any] = {
            "symbols": symbols,
            "vectorized": True,
            "sizer": type(sizers[sym0]).__name__,
            "stop_loss": "NopStopLoss",
            "cost_model": _cost_model_label(cost_models[sym0]),
        }
        if timeframe is not None:
            meta["timeframe"] = timeframe

        return BacktestResult(
            trades=all_trades,
            equity_curve=eq_series,
            positions=pos_series,
            config=self.config,
            run_time_s=time.perf_counter() - t0,
            meta=meta,
            positions_log=None,
            allocation_log=None,
        )

    # ── Core loop ────────────────────────────────────────────────────────

    def _run_loop(
        self,
        strategy: Strategy,
        universe: Universe,
        symbols: list[str],
        is_single_asset: bool,
        timeframe: str | None = None,
    ) -> BacktestResult:
        t0 = time.perf_counter()
        n_bars = universe.bar_count()
        if n_bars == 0:
            raise ValueError("No data in universe")

        if is_single_asset:
            index = universe.ohlcv(symbols[0]).index
        else:
            index = universe.common_index()
            if len(index) == 0:
                index = universe.ohlcv(symbols[0]).index
        n_bars = len(index)

        strategy.setup(universe)

        states: dict[str, _AssetState] = {}
        sizers: dict[str, Sizer] = {}
        cost_models: dict[str, CostModel] = {}
        for sym in symbols:
            states[sym] = _AssetState(
                stop_loss=self._resolve(self._stop_loss_spec, sym, default_stop_loss),
            )
            sizers[sym] = self._resolve(self._sizer_spec, sym, default_sizer)
            cost_models[sym] = self._resolve(self._cost_model_spec, sym, NullCostModel)

        # ── Try vectorised fast path ──────────────────────────────────────
        if self._can_vectorize(symbols, states, sizers):
            batch = strategy.generate_all(universe)
            if batch is not None:
                sides_all, weights_all = batch[0], batch[1]
                reasons_all     = batch[2] if len(batch) > 2 else None
                metas_all       = batch[3] if len(batch) > 3 else None
                confidences_all = batch[4] if len(batch) > 4 else None
                return self._run_vectorized(
                    t0, sides_all, weights_all, universe, symbols,
                    is_single_asset, timeframe,
                    index, n_bars, sizers, cost_models,
                    reasons_all=reasons_all, metas_all=metas_all,
                    confidences_all=confidences_all,
                )

        # ── Pre-extract OHLCV as numpy arrays (O(n) bulk alignment) ──────
        # Replaces per-bar ohlcv.index.get_loc(ts) + ohlcv[col].iat[loc]
        # with direct array indexing, eliminating O(n log n) overhead.
        ohlcv_dfs: dict[str, pd.DataFrame] = {sym: universe.ohlcv(sym) for sym in symbols}

        # _local_idx[sym][global_i] = local row index, -1 if symbol has no bar here
        _local_idx: dict[str, np.ndarray] = {}
        _has_bar: dict[str, np.ndarray] = {}
        _closes: dict[str, np.ndarray] = {}
        _opens: dict[str, np.ndarray] = {}
        _highs: dict[str, np.ndarray] = {}
        _lows: dict[str, np.ndarray] = {}
        # All numeric columns (OHLCV + indicator columns from strategy.setup())
        _col_arrays: dict[str, dict[str, np.ndarray]] = {}

        for sym in symbols:
            df = ohlcv_dfs[sym]
            locs = df.index.get_indexer(index)   # single O(n) pass
            valid = locs >= 0
            vi = np.where(valid)[0]   # global bar positions with data
            vl = locs[vi]             # corresponding local row positions

            _local_idx[sym] = locs
            _has_bar[sym] = valid

            def _mk(col: str, _df=df, _vi=vi, _vl=vl) -> np.ndarray:
                a = np.full(n_bars, np.nan)
                a[_vi] = _df[col].values[_vl]
                return a

            _closes[sym] = _mk("close")
            _opens[sym] = _mk("open")
            _highs[sym] = _mk("high")
            _lows[sym] = _mk("low")

            col_dict: dict[str, np.ndarray] = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_dict[col] = _mk(col)
            _col_arrays[sym] = col_dict

        # ── Pre-allocate output arrays ────────────────────────────────────
        equity_arr = np.full(n_bars, np.nan)
        equity = self.config.initial_capital
        equity_arr[0] = equity

        pos_side_arr = np.zeros(n_bars, dtype=np.int8) if is_single_asset else None

        all_trades: list[Trade] = []
        closed_trades: list[Trade] = []

        alloc_log_rows: list[dict] = []
        pos_log_rows: list[dict] = []

        # ── Bar loop ──────────────────────────────────────────────────────

        for i in range(n_bars):
            ts = index[i]

            # Build prices/locs/bar_dicts from pre-extracted arrays (O(1) per bar)
            prices: dict[str, float] = {}
            bar_locs: dict[str, int] = {}
            bar_dicts: dict[str, dict] = {}

            for sym in symbols:
                if not _has_bar[sym][i]:
                    continue
                loc = int(_local_idx[sym][i])
                prices[sym] = float(_closes[sym][i])
                bar_locs[sym] = loc
                bar_dicts[sym] = {
                    col: float(arr[i]) for col, arr in _col_arrays[sym].items()
                }

                # Inject funding if available
                funding_snap = universe.funding_at(sym, loc)
                if funding_snap is not None:
                    bar_dicts[sym]["funding_rate"] = funding_snap.rate
                    bar_dicts[sym]["funding_rate_ann_bps"] = funding_snap.rate_annualized
                    if funding_snap.oracle_price > 0:
                        bar_dicts[sym]["oracle_price"] = funding_snap.oracle_price
                    if funding_snap.mark_price > 0:
                        bar_dicts[sym]["mark_price"] = funding_snap.mark_price

            # ── Mark-to-market ────────────────────────────────────────────
            for sym in symbols:
                if sym not in prices:
                    continue
                st = states[sym]
                pos = st.position
                if pos.side != Side.FLAT and pos.size > 0:
                    direction = 1 if pos.side == Side.LONG else -1
                    pos.unrealized_pnl = (prices[sym] - pos.entry_price) * pos.size * direction

            # ── Stop-loss checks ──────────────────────────────────────────
            for sym in symbols:
                st = states[sym]
                pos = st.position
                if pos.side == Side.FLAT or sym not in prices:
                    continue

                loc = bar_locs.get(sym)
                if loc is None:
                    continue

                df = ohlcv_dfs[sym]
                l2_list = universe.l2(sym)
                l2_snap = l2_list[loc] if l2_list and loc < len(l2_list) else None

                stop_ctx = StopContext(
                    position=pos,
                    bar_idx=loc,
                    open=float(_opens[sym][i]),
                    high=float(_highs[sym][i]),
                    low=float(_lows[sym][i]),
                    close=prices[sym],
                    data=df,
                    l2=l2_snap,
                    bar_data=bar_dicts.get(sym, {}),
                )
                st.stop_loss.update(stop_ctx)
                stop_result = st.stop_loss.check(stop_ctx)

                # EmbeddedStop: check allocation-embedded SL/TP levels
                if not stop_result.triggered and isinstance(st.stop_loss, EmbeddedStop):
                    stop_result = st.stop_loss.check_with_levels(stop_ctx)

                if stop_result.triggered:
                    exit_p = stop_result.exit_price
                    cost = cost_models[sym].compute(
                        exit_p, pos.size, pos.side, self.config,
                        l2_snap, bar_dicts.get(sym, {}),
                    )
                    raw_pnl = (
                        (exit_p - pos.entry_price) * pos.size
                        if pos.side == Side.LONG
                        else (pos.entry_price - exit_p) * pos.size
                    )
                    entry_fee = st.open_trade.fees if st.open_trade is not None else 0.0
                    pnl = raw_pnl - cost - entry_fee
                    equity += pnl

                    if st.open_trade is not None:
                        st.open_trade.exit_price = exit_p
                        st.open_trade.exit_timestamp = ts
                        st.open_trade.pnl = pnl
                        st.open_trade.pnl_pct = (
                            pnl / (pos.entry_price * pos.size)
                            if pos.entry_price * pos.size > 0 else 0
                        )
                        st.open_trade.fees += cost
                        st.open_trade.reason_exit = stop_result.reason
                        st.open_trade.meta.update(stop_result.meta)
                        if not is_single_asset:
                            st.open_trade.meta["symbol"] = sym
                        closed_trades.append(st.open_trade)

                    st.position = Position()
                    st.open_trade = None
                    st.stop_loss.reset()

            # ── Generate strategy targets ──────────────────────────────────
            ctx = StrategyContext(
                universe=universe,
                bar_idx=i,
                timestamp=ts,
                equity=equity,
                positions={sym: states[sym].position for sym in symbols},
                trade_history=closed_trades,
            )
            target = strategy.generate(ctx)

            if not is_single_asset:
                row = {"timestamp": ts}
                for sym in symbols:
                    alloc = target[sym]
                    row[f"{sym}_side"] = alloc.side.name
                    row[f"{sym}_weight"] = alloc.weight
                    row[f"{sym}_confidence"] = alloc.confidence
                    row[f"{sym}_reason"] = alloc.reason
                alloc_log_rows.append(row)

            # ── Close positions that should be flat or flipped ────────────
            for sym in symbols:
                st = states[sym]
                pos = st.position
                if pos.side == Side.FLAT:
                    continue

                desired = target[sym]
                if desired.side != Side.FLAT and desired.side == pos.side:
                    continue   # holding — no action
                if sym not in prices:
                    continue

                price = prices[sym]
                cost = cost_models[sym].compute(
                    price, pos.size, pos.side, self.config,
                    None, bar_dicts.get(sym, {}),
                )
                entry_fee = st.open_trade.fees if st.open_trade is not None else 0.0
                pnl = pos.unrealized_pnl - cost - entry_fee
                pnl_pct = (
                    pnl / (pos.entry_price * pos.size)
                    if pos.entry_price * pos.size > 0 else 0
                )

                trade_meta = {"symbol": sym} if not is_single_asset else {}
                trade = Trade(
                    timestamp=pos.entry_timestamp,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    exit_timestamp=ts,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    fees=entry_fee + cost,
                    confidence=(st.open_trade.confidence if st.open_trade else 0.0),
                    reason_entry=(st.open_trade.reason_entry if st.open_trade else ""),
                    reason_exit=desired.reason or "target_flat",
                    bar_values=(st.open_trade.bar_values if st.open_trade else {}),
                    meta=trade_meta,
                )
                all_trades.append(trade)
                closed_trades.append(trade)
                equity += pnl

                st.position = Position()
                st.open_trade = None
                st.stop_loss.reset()

            # ── Open new positions ─────────────────────────────────────────
            for sym in symbols:
                st = states[sym]
                if st.position.side != Side.FLAT:
                    continue

                alloc = target[sym]
                if alloc.side == Side.FLAT or alloc.weight <= 0 or sym not in prices:
                    continue

                price = prices[sym]
                loc = bar_locs.get(sym, 0)
                df = ohlcv_dfs[sym]

                l2_list = universe.l2(sym)
                l2_snap = l2_list[loc] if l2_list and loc < len(l2_list) else None

                sizing_ctx = SizingContext(
                    equity=equity,
                    price=price,
                    allocation=alloc,
                    config=self.config,
                    position=st.position,
                    data=df,
                    bar_idx=loc,
                    trade_history=closed_trades,
                    l2=l2_snap,
                    bar_data=bar_dicts.get(sym, {}),
                )
                size = sizers[sym].compute(sizing_ctx)

                if is_single_asset:
                    max_notional = equity * self.config.max_position_pct * self.config.leverage
                else:
                    max_notional = equity * alloc.weight * self.config.leverage
                max_size = max_notional / price if price > 0 else 0
                size = min(size, max_size)

                if size <= 0:
                    continue

                cost = cost_models[sym].compute(
                    price, size, alloc.side, self.config,
                    l2_snap, bar_dicts.get(sym, {}),
                )

                st.position = Position(
                    side=alloc.side,
                    size=size,
                    entry_price=price,
                    entry_timestamp=ts,
                )

                o_i = float(_opens[sym][i])
                h_i = float(_highs[sym][i])
                l_i = float(_lows[sym][i])
                stop_ctx = StopContext(
                    position=st.position,
                    bar_idx=loc,
                    open=o_i if not np.isnan(o_i) else price,
                    high=h_i if not np.isnan(h_i) else price,
                    low=l_i if not np.isnan(l_i) else price,
                    close=price,
                    data=df,
                    l2=l2_snap,
                    bar_data=bar_dicts.get(sym, {}),
                )
                st.stop_loss.on_entry(st.position, stop_ctx)
                if isinstance(st.stop_loss, EmbeddedStop):
                    st.stop_loss.set_levels(alloc.stop_loss, alloc.take_profit)

                trade_meta = {"symbol": sym} if not is_single_asset else {}
                trade = Trade(
                    timestamp=ts,
                    side=alloc.side,
                    size=size,
                    entry_price=price,
                    fees=cost,
                    confidence=alloc.confidence,
                    reason_entry=alloc.reason,
                    bar_values=alloc.meta,
                    meta=trade_meta,
                )
                all_trades.append(trade)
                st.open_trade = trade

            # ── Record equity ──────────────────────────────────────────────
            unrealized = sum(
                st.position.unrealized_pnl
                for st in states.values()
                if st.position.side != Side.FLAT
            )
            equity_arr[i] = equity + unrealized

            if is_single_asset:
                pos_side_arr[i] = states[symbols[0]].position.side.value
            else:
                row = {"timestamp": ts}
                for sym in symbols:
                    st = states[sym]
                    row[f"{sym}_side"] = st.position.side.value
                    row[f"{sym}_size"] = st.position.size
                pos_log_rows.append(row)

        # ── Force-close remaining positions ───────────────────────────────
        last_ts = index[-1]
        for sym in symbols:
            st = states[sym]
            pos = st.position
            if pos.side == Side.FLAT:
                continue

            last_close = float(_closes[sym][-1])
            if np.isnan(last_close):
                last_close = float(ohlcv_dfs[sym]["close"].iloc[-1])

            cost = cost_models[sym].compute(
                last_close, pos.size, pos.side, self.config, None, None,
            )
            raw_pnl = (
                (last_close - pos.entry_price) * pos.size
                if pos.side == Side.LONG
                else (pos.entry_price - last_close) * pos.size
            )
            entry_fee = st.open_trade.fees if st.open_trade is not None else 0.0
            pnl = raw_pnl - cost - entry_fee
            equity += pnl  # track force-close in equity so equity_arr[-1] is correct below

            if st.open_trade is not None:
                st.open_trade.exit_price = last_close
                st.open_trade.exit_timestamp = last_ts
                st.open_trade.pnl = pnl
                st.open_trade.pnl_pct = (
                    pnl / (pos.entry_price * pos.size)
                    if pos.entry_price * pos.size > 0 else 0
                )
                st.open_trade.fees += cost
                st.open_trade.reason_exit = "End of data"
                if not is_single_asset:
                    st.open_trade.meta["symbol"] = sym
                closed_trades.append(st.open_trade)
            else:
                trade_meta = {"symbol": sym} if not is_single_asset else {}
                trade = Trade(
                    timestamp=pos.entry_timestamp,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    exit_price=last_close,
                    exit_timestamp=last_ts,
                    pnl=pnl,
                    pnl_pct=(
                        pnl / (pos.entry_price * pos.size)
                        if pos.entry_price * pos.size > 0 else 0
                    ),
                    fees=cost,
                    reason_exit="End of data",
                    meta=trade_meta,
                )
                all_trades.append(trade)
                closed_trades.append(trade)

        # After force-closes, update equity_arr[-1] to reflect the true realized
        # equity (exit fees now deducted). Without this, equity_arr[-1] would show
        # mark-to-market + unrealized from the bar loop, which omits the exit fees
        # of positions that were force-closed after the loop.
        equity_arr[-1] = equity

        # ── Build result ──────────────────────────────────────────────────
        final_trades = [t for t in all_trades if t.exit_price is not None]
        elapsed = time.perf_counter() - t0

        eq_series = pd.Series(equity_arr, index=index, name="equity")

        if is_single_asset:
            pos_series = pd.Series(pos_side_arr, index=index, name="position")
        else:
            if pos_log_rows:
                pos_df = pd.DataFrame(pos_log_rows)
                side_cols = [f"{s}_side" for s in symbols]
                pos_series = pos_df[side_cols].sum(axis=1)
                pos_series.index = index
                pos_series.name = "position"
            else:
                pos_series = pd.Series(
                    np.zeros(n_bars, dtype=int), index=index, name="position",
                )

        sym0 = symbols[0]
        meta: dict[str, Any] = {
            "symbols": symbols,
            "vectorized": False,
            "sizer": type(sizers[sym0]).__name__,
            "stop_loss": type(states[sym0].stop_loss).__name__,
            "cost_model": _cost_model_label(cost_models[sym0]),
        }
        if timeframe is not None:
            meta["timeframe"] = timeframe

        return BacktestResult(
            trades=final_trades,
            equity_curve=eq_series,
            positions=pos_series,
            config=self.config,
            run_time_s=elapsed,
            meta=meta,
            positions_log=(
                pd.DataFrame(pos_log_rows) if pos_log_rows else None
            ),
            allocation_log=(
                pd.DataFrame(alloc_log_rows) if alloc_log_rows else None
            ),
        )

    # ── Multi-exchange bar loop ───────────────────────────────────────────

    def _run_loop_multi_exchange(
        self,
        strategy: Strategy,
        universes: dict[str, Universe],
        timeframe: str | None = None,
    ) -> BacktestResult:
        """
        Per-bar backtest across multiple exchanges simultaneously.

        Each exchange has its own position book, equity account, and cost model.
        The strategy receives a unified StrategyContext with data from all
        exchanges and returns a PortfolioTarget whose exchange_allocations routes
        each allocation to the correct exchange.

        The aggregate equity curve is the sum of all per-exchange curves.
        """
        t0 = time.perf_counter()
        exchange_names = list(universes.keys())
        n_exchanges = len(exchange_names)

        # ── Bar index: intersection of all exchange indices ───────────────
        common_index: pd.Index | None = None
        for uni in universes.values():
            idx = uni.common_index() if len(uni.symbols) > 1 else uni.ohlcv(uni.symbols[0]).index
            common_index = idx if common_index is None else common_index.intersection(idx)
        if common_index is None or len(common_index) == 0:
            raise ValueError("No common bars across universes")
        index = common_index
        n_bars = len(index)

        # ── Capital split ─────────────────────────────────────────────────
        if self._capital_by_exchange:
            equity_by_exchange: dict[str, float] = dict(self._capital_by_exchange)
        else:
            per_ex = self.config.initial_capital / n_exchanges
            equity_by_exchange = {ex: per_ex for ex in exchange_names}

        # ── Per-exchange component dicts ──────────────────────────────────
        # states[exchange][symbol], sizers[exchange][symbol], cost_models[exchange][symbol]
        ex_states:       dict[str, dict[str, _AssetState]] = {}
        ex_sizers:       dict[str, dict[str, Sizer]]       = {}
        ex_cost_models:  dict[str, dict[str, CostModel]]   = {}

        for ex in exchange_names:
            uni = universes[ex]
            syms = uni.symbols
            ex_states[ex]      = {}
            ex_sizers[ex]      = {}
            ex_cost_models[ex] = {}
            # Per-exchange cost default; falls back to global cost_model spec
            ex_cost_spec = self._exchange_costs.get(ex, self._cost_model_spec)
            for sym in syms:
                ex_states[ex][sym] = _AssetState(
                    stop_loss=self._resolve(self._stop_loss_spec, sym, default_stop_loss),
                )
                ex_sizers[ex][sym]      = self._resolve(self._sizer_spec, sym, default_sizer)
                ex_cost_models[ex][sym] = self._resolve(ex_cost_spec, sym, NullCostModel)

        # ── Strategy setup ────────────────────────────────────────────────
        strategy.setup(universes)

        # ── Pre-extract OHLCV arrays per exchange per symbol ──────────────
        ohlcv_dfs: dict[str, dict[str, pd.DataFrame]] = {
            ex: {sym: universes[ex].ohlcv(sym) for sym in universes[ex].symbols}
            for ex in exchange_names
        }

        ex_local_idx:   dict[str, dict[str, np.ndarray]] = {}
        ex_has_bar:     dict[str, dict[str, np.ndarray]] = {}
        ex_closes:      dict[str, dict[str, np.ndarray]] = {}
        ex_opens:       dict[str, dict[str, np.ndarray]] = {}
        ex_highs:       dict[str, dict[str, np.ndarray]] = {}
        ex_lows:        dict[str, dict[str, np.ndarray]] = {}
        ex_col_arrays:  dict[str, dict[str, dict[str, np.ndarray]]] = {}

        for ex in exchange_names:
            ex_local_idx[ex]  = {}
            ex_has_bar[ex]    = {}
            ex_closes[ex]     = {}
            ex_opens[ex]      = {}
            ex_highs[ex]      = {}
            ex_lows[ex]       = {}
            ex_col_arrays[ex] = {}
            for sym, df in ohlcv_dfs[ex].items():
                locs  = df.index.get_indexer(index)
                valid = locs >= 0
                vi    = np.where(valid)[0]
                vl    = locs[vi]

                ex_local_idx[ex][sym] = locs
                ex_has_bar[ex][sym]   = valid

                def _mk(col: str, _df=df, _vi=vi, _vl=vl) -> np.ndarray:
                    a = np.full(n_bars, np.nan)
                    a[_vi] = _df[col].values[_vl]
                    return a

                ex_closes[ex][sym] = _mk("close")
                ex_opens[ex][sym]  = _mk("open")
                ex_highs[ex][sym]  = _mk("high")
                ex_lows[ex][sym]   = _mk("low")

                col_dict: dict[str, np.ndarray] = {}
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_dict[col] = _mk(col)
                ex_col_arrays[ex][sym] = col_dict

        # ── Output accumulators ───────────────────────────────────────────
        ex_equity_arrs: dict[str, np.ndarray] = {
            ex: np.full(n_bars, np.nan) for ex in exchange_names
        }
        all_trades:    list[Trade] = []
        closed_trades: list[Trade] = []
        pos_log_rows:  list[dict] = []
        alloc_log_rows: list[dict] = []

        # ── Bar loop ──────────────────────────────────────────────────────
        for i in range(n_bars):
            ts = index[i]

            # Per-exchange prices / bar dicts
            ex_prices:    dict[str, dict[str, float]] = {}
            ex_bar_locs:  dict[str, dict[str, int]]   = {}
            ex_bar_dicts: dict[str, dict[str, dict]]  = {}

            for ex in exchange_names:
                ex_prices[ex]    = {}
                ex_bar_locs[ex]  = {}
                ex_bar_dicts[ex] = {}
                for sym in universes[ex].symbols:
                    if not ex_has_bar[ex][sym][i]:
                        continue
                    loc = int(ex_local_idx[ex][sym][i])
                    ex_prices[ex][sym]   = float(ex_closes[ex][sym][i])
                    ex_bar_locs[ex][sym] = loc
                    ex_bar_dicts[ex][sym] = {
                        col: float(arr[i])
                        for col, arr in ex_col_arrays[ex][sym].items()
                    }
                    funding_snap = universes[ex].funding_at(sym, loc)
                    if funding_snap is not None:
                        ex_bar_dicts[ex][sym]["funding_rate"] = funding_snap.rate
                        ex_bar_dicts[ex][sym]["funding_rate_ann_bps"] = funding_snap.rate_annualized

            # ── Mark-to-market ────────────────────────────────────────────
            for ex in exchange_names:
                for sym, st in ex_states[ex].items():
                    pos = st.position
                    if pos.side != Side.FLAT and pos.size > 0 and sym in ex_prices[ex]:
                        direction = 1 if pos.side == Side.LONG else -1
                        pos.unrealized_pnl = (
                            (ex_prices[ex][sym] - pos.entry_price) * pos.size * direction
                        )

            # ── Stop-loss checks ──────────────────────────────────────────
            for ex in exchange_names:
                for sym, st in ex_states[ex].items():
                    pos = st.position
                    if pos.side == Side.FLAT or sym not in ex_prices[ex]:
                        continue
                    loc = ex_bar_locs[ex].get(sym)
                    if loc is None:
                        continue
                    df      = ohlcv_dfs[ex][sym]
                    l2_list = universes[ex].l2(sym)
                    l2_snap = l2_list[loc] if l2_list and loc < len(l2_list) else None
                    stop_ctx = StopContext(
                        position=pos,
                        bar_idx=loc,
                        open=float(ex_opens[ex][sym][i]),
                        high=float(ex_highs[ex][sym][i]),
                        low=float(ex_lows[ex][sym][i]),
                        close=ex_prices[ex][sym],
                        data=df,
                        l2=l2_snap,
                        bar_data=ex_bar_dicts[ex].get(sym, {}),
                    )
                    st.stop_loss.update(stop_ctx)
                    stop_result = st.stop_loss.check(stop_ctx)
                    if not stop_result.triggered and isinstance(st.stop_loss, EmbeddedStop):
                        stop_result = st.stop_loss.check_with_levels(stop_ctx)

                    if stop_result.triggered:
                        exit_p = stop_result.exit_price
                        cost = ex_cost_models[ex][sym].compute(
                            exit_p, pos.size, pos.side, self.config, l2_snap, ex_bar_dicts[ex].get(sym, {}),
                        )
                        raw_pnl = (
                            (exit_p - pos.entry_price) * pos.size
                            if pos.side == Side.LONG
                            else (pos.entry_price - exit_p) * pos.size
                        )
                        entry_fee = st.open_trade.fees if st.open_trade else 0.0
                        pnl = raw_pnl - cost - entry_fee
                        equity_by_exchange[ex] += pnl

                        if st.open_trade is not None:
                            st.open_trade.exit_price     = exit_p
                            st.open_trade.exit_timestamp = ts
                            st.open_trade.pnl            = pnl
                            st.open_trade.pnl_pct        = (
                                pnl / (pos.entry_price * pos.size)
                                if pos.entry_price * pos.size > 0 else 0
                            )
                            st.open_trade.fees          += cost
                            st.open_trade.reason_exit    = stop_result.reason
                            st.open_trade.meta.update(stop_result.meta)
                            st.open_trade.meta["exchange"] = ex
                            closed_trades.append(st.open_trade)

                        st.position   = Position()
                        st.open_trade = None
                        st.stop_loss.reset()
                        strategy.on_fill(sym, pos.side, pos.size, exit_p, exchange=ex)

            # ── Build unified context and generate targets ─────────────────
            all_positions_ctx: dict[str, dict[str, Position]] = {
                ex: {sym: ex_states[ex][sym].position for sym in universes[ex].symbols}
                for ex in exchange_names
            }
            total_equity = sum(equity_by_exchange.values())
            ctx = StrategyContext(
                universes=universes,
                bar_idx=i,
                timestamp=ts,
                equity_by_exchange=dict(equity_by_exchange),
                all_positions=all_positions_ctx,
                trade_history=closed_trades,
            )
            target: PortfolioTarget = strategy.generate(ctx)

            # Log allocations
            row: dict = {"timestamp": ts}
            for ex in exchange_names:
                for sym in universes[ex].symbols:
                    alloc = target[(ex, sym)] if target.is_multi_exchange else target[sym]
                    row[f"{ex}:{sym}_side"]       = alloc.side.name
                    row[f"{ex}:{sym}_weight"]     = alloc.weight
                    row[f"{ex}:{sym}_confidence"] = alloc.confidence
            alloc_log_rows.append(row)

            # ── Close / open per exchange ─────────────────────────────────
            for ex in exchange_names:
                for sym in universes[ex].symbols:
                    alloc = (
                        target[(ex, sym)] if target.is_multi_exchange
                        else target[sym]
                    )
                    st    = ex_states[ex][sym]
                    pos   = st.position
                    price = ex_prices[ex].get(sym)

                    # Close positions that should be flat or flip
                    if pos.side != Side.FLAT and price is not None:
                        if alloc.side == Side.FLAT or alloc.side != pos.side:
                            cost = ex_cost_models[ex][sym].compute(
                                price, pos.size, pos.side, self.config,
                                None, ex_bar_dicts[ex].get(sym, {}),
                            )
                            entry_fee = st.open_trade.fees if st.open_trade else 0.0
                            pnl = pos.unrealized_pnl - cost - entry_fee
                            pnl_pct = (
                                pnl / (pos.entry_price * pos.size)
                                if pos.entry_price * pos.size > 0 else 0
                            )
                            trade = Trade(
                                timestamp=pos.entry_timestamp,
                                side=pos.side,
                                size=pos.size,
                                entry_price=pos.entry_price,
                                exit_price=price,
                                exit_timestamp=ts,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                fees=entry_fee + cost,
                                confidence=(st.open_trade.confidence if st.open_trade else 0.0),
                                reason_entry=(st.open_trade.reason_entry if st.open_trade else ""),
                                reason_exit=alloc.reason or "target_flat",
                                bar_values=(st.open_trade.bar_values if st.open_trade else {}),
                                meta={"symbol": sym, "exchange": ex},
                            )
                            all_trades.append(trade)
                            closed_trades.append(trade)
                            equity_by_exchange[ex] += pnl
                            st.position   = Position()
                            st.open_trade = None
                            st.stop_loss.reset()
                            strategy.on_fill(sym, pos.side, pos.size, price, exchange=ex)

                    # Open new positions
                    st  = ex_states[ex][sym]
                    pos = st.position
                    if (
                        pos.side == Side.FLAT
                        and alloc.side != Side.FLAT
                        and alloc.weight > 0
                        and price is not None
                    ):
                        loc = ex_bar_locs[ex].get(sym, 0)
                        df  = ohlcv_dfs[ex][sym]
                        l2_list = universes[ex].l2(sym)
                        l2_snap = l2_list[loc] if l2_list and loc < len(l2_list) else None

                        sizing_ctx = SizingContext(
                            equity=equity_by_exchange[ex],
                            price=price,
                            allocation=alloc,
                            config=self.config,
                            position=pos,
                            data=df,
                            bar_idx=loc,
                            trade_history=closed_trades,
                            l2=l2_snap,
                            bar_data=ex_bar_dicts[ex].get(sym, {}),
                        )
                        size = ex_sizers[ex][sym].compute(sizing_ctx)

                        max_notional = equity_by_exchange[ex] * alloc.weight * self.config.leverage
                        max_size = max_notional / price if price > 0 else 0
                        size = min(size, max_size)
                        if size <= 0:
                            continue

                        cost = ex_cost_models[ex][sym].compute(
                            price, size, alloc.side, self.config,
                            l2_snap, ex_bar_dicts[ex].get(sym, {}),
                        )

                        st.position = Position(
                            side=alloc.side,
                            size=size,
                            entry_price=price,
                            entry_timestamp=ts,
                        )

                        o_i = float(ex_opens[ex][sym][i])
                        h_i = float(ex_highs[ex][sym][i])
                        l_i = float(ex_lows[ex][sym][i])
                        stop_ctx = StopContext(
                            position=st.position,
                            bar_idx=loc,
                            open=o_i  if not np.isnan(o_i)  else price,
                            high=h_i  if not np.isnan(h_i)  else price,
                            low=l_i   if not np.isnan(l_i)  else price,
                            close=price,
                            data=df,
                            l2=l2_snap,
                            bar_data=ex_bar_dicts[ex].get(sym, {}),
                        )
                        st.stop_loss.on_entry(st.position, stop_ctx)
                        if isinstance(st.stop_loss, EmbeddedStop):
                            st.stop_loss.set_levels(alloc.stop_loss, alloc.take_profit)

                        trade = Trade(
                            timestamp=ts,
                            side=alloc.side,
                            size=size,
                            entry_price=price,
                            fees=cost,
                            confidence=alloc.confidence,
                            reason_entry=alloc.reason,
                            bar_values=alloc.meta,
                            meta={"symbol": sym, "exchange": ex},
                        )
                        all_trades.append(trade)
                        st.open_trade = trade
                        strategy.on_fill(sym, alloc.side, size, price, exchange=ex)

            # ── Record equity ──────────────────────────────────────────────
            for ex in exchange_names:
                unrealized_ex = sum(
                    st.position.unrealized_pnl
                    for st in ex_states[ex].values()
                    if st.position.side != Side.FLAT
                )
                ex_equity_arrs[ex][i] = equity_by_exchange[ex] + unrealized_ex

            # Position log
            row_p: dict = {"timestamp": ts}
            for ex in exchange_names:
                for sym, st in ex_states[ex].items():
                    row_p[f"{ex}:{sym}_side"] = st.position.side.value
                    row_p[f"{ex}:{sym}_size"] = st.position.size
            pos_log_rows.append(row_p)

        # ── Force-close remaining positions ───────────────────────────────
        last_ts = index[-1]
        for ex in exchange_names:
            for sym, st in ex_states[ex].items():
                pos = st.position
                if pos.side == Side.FLAT:
                    continue
                last_close = float(ex_closes[ex][sym][-1])
                if np.isnan(last_close):
                    last_close = float(ohlcv_dfs[ex][sym]["close"].iloc[-1])

                cost = ex_cost_models[ex][sym].compute(
                    last_close, pos.size, pos.side, self.config, None, None,
                )
                raw_pnl = (
                    (last_close - pos.entry_price) * pos.size
                    if pos.side == Side.LONG
                    else (pos.entry_price - last_close) * pos.size
                )
                entry_fee = st.open_trade.fees if st.open_trade else 0.0
                pnl = raw_pnl - cost - entry_fee
                equity_by_exchange[ex] += pnl

                if st.open_trade is not None:
                    st.open_trade.exit_price     = last_close
                    st.open_trade.exit_timestamp = last_ts
                    st.open_trade.pnl            = pnl
                    st.open_trade.pnl_pct        = (
                        pnl / (pos.entry_price * pos.size)
                        if pos.entry_price * pos.size > 0 else 0
                    )
                    st.open_trade.fees          += cost
                    st.open_trade.reason_exit    = "End of data"
                    st.open_trade.meta["exchange"] = ex
                    closed_trades.append(st.open_trade)
                else:
                    trade = Trade(
                        timestamp=pos.entry_timestamp,
                        side=pos.side,
                        size=pos.size,
                        entry_price=pos.entry_price,
                        exit_price=last_close,
                        exit_timestamp=last_ts,
                        pnl=pnl,
                        pnl_pct=(
                            pnl / (pos.entry_price * pos.size)
                            if pos.entry_price * pos.size > 0 else 0
                        ),
                        fees=cost,
                        reason_exit="End of data",
                        meta={"symbol": sym, "exchange": ex},
                    )
                    all_trades.append(trade)
                    closed_trades.append(trade)

                ex_equity_arrs[ex][-1] = equity_by_exchange[ex]

        # ── Assemble result ───────────────────────────────────────────────
        # Total equity = sum across all exchanges
        total_equity_arr = sum(ex_equity_arrs[ex] for ex in exchange_names)
        eq_series = pd.Series(total_equity_arr, index=index, name="equity")

        equity_curves_by_exchange = {
            ex: pd.Series(ex_equity_arrs[ex], index=index, name=f"equity_{ex}")
            for ex in exchange_names
        }

        final_trades = [t for t in all_trades if t.exit_price is not None]

        # Aggregate symbols across all exchanges for meta
        all_symbols: list[str] = []
        for ex in exchange_names:
            all_symbols.extend(universes[ex].symbols)

        sym0 = exchange_names[0]
        first_sym = universes[sym0].symbols[0] if universes[sym0].symbols else "ASSET"
        meta: dict[str, Any] = {
            "symbols":    all_symbols,
            "exchanges":  exchange_names,
            "vectorized": False,
            "sizer":      type(ex_sizers[exchange_names[0]][first_sym]).__name__,
            "stop_loss":  type(ex_states[exchange_names[0]][first_sym].stop_loss).__name__,
            "cost_model": _cost_model_label(ex_cost_models[exchange_names[0]][first_sym]),
        }
        if timeframe is not None:
            meta["timeframe"] = timeframe

        pos_series = pd.Series(
            np.zeros(n_bars, dtype=int), index=index, name="position"
        )

        return BacktestResult(
            trades=final_trades,
            equity_curve=eq_series,
            positions=pos_series,
            config=self.config,
            run_time_s=time.perf_counter() - t0,
            meta=meta,
            positions_log=pd.DataFrame(pos_log_rows) if pos_log_rows else None,
            allocation_log=pd.DataFrame(alloc_log_rows) if alloc_log_rows else None,
            equity_curves_by_exchange=equity_curves_by_exchange,
        )