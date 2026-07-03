"""
walk_forward.py — Walk-forward analysis for out-of-sample strategy validation.

Classes:
    WalkForwardResult   — result container with stitched OOS equity, IS/OOS metrics,
                          consistency score, IS/OOS efficiency ratio, and OOS equity plot
    WalkForwardAnalysis — runs a strategy through rolling/expanding folds;
                          optionally re-optimizes parameters on each IS fold
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

from core.models import BacktestConfig
from backtester.engine import Backtester, BacktestResult
from backtester.costs import CostModel
from strategy.base import Strategy
from strategy.built_in import SingleAssetStrategy
from core.universe import Universe
from .splits import WalkForwardSplits, Split


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    folds: list[dict[str, Any]]     # fold metadata (dates, best params if optimized)
    oos_equity: pd.Series            # stitched, continuously scaled OOS equity curve
    oos_trades: list                 # all OOS Trade objects
    is_summary: pd.DataFrame         # IS metrics per fold (index = fold number)
    oos_summary: pd.DataFrame        # OOS metrics per fold (index = fold number)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def consistency_score(self) -> float:
        """Fraction of OOS folds that produced a positive return."""
        if self.oos_summary.empty or "total_return_pct" not in self.oos_summary.columns:
            return float("nan")
        return float((self.oos_summary["total_return_pct"] > 0).mean())

    @property
    def efficiency_ratio(self) -> float:
        """
        Mean OOS Sharpe / Mean IS Sharpe.

        Values near 1.0 indicate the strategy generalizes well.
        Values near 0 indicate overfitting.
        """
        col = "sharpe_ratio"
        if col not in self.is_summary.columns or col not in self.oos_summary.columns:
            return float("nan")
        is_mean = self.is_summary[col].mean()
        oos_mean = self.oos_summary[col].mean()
        return float(oos_mean / is_mean) if is_mean != 0 else float("nan")

    def summary_table(self) -> pd.DataFrame:
        """Side-by-side IS vs OOS key metrics per fold."""
        key_cols = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "num_trades"]
        is_cols = [c for c in key_cols if c in self.is_summary.columns]
        oos_cols = [c for c in key_cols if c in self.oos_summary.columns]
        is_df = self.is_summary[is_cols].add_prefix("is_")
        oos_df = self.oos_summary[oos_cols].add_prefix("oos_")
        return pd.concat([is_df, oos_df], axis=1)

    def plot_oos_equity(self, save_path: str | None = None) -> str:
        """Plot the stitched OOS equity curve with fold boundaries marked."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self.oos_equity.empty:
            raise ValueError("No OOS equity data to plot")

        fig, ax = plt.subplots(figsize=(13, 5))
        norm = self.oos_equity / self.oos_equity.iloc[0] * 100
        ax.plot(norm.index, norm.values, color="#2563eb", linewidth=1.5, label="OOS equity")

        for row in self.folds:
            ts = pd.Timestamp(row["test_start"])
            ax.axvline(ts, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

        ax.axhline(100, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
        ax.set_title("Walk-Forward Out-of-Sample Equity (normalized to 100)")
        ax.set_ylabel("Equity")
        ax.legend(loc="upper left")
        fig.tight_layout()
        path = save_path or "wfa_oos_equity.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def __repr__(self) -> str:
        n = len(self.folds)
        cs = self.consistency_score
        er = self.efficiency_ratio
        return (
            f"WalkForwardResult(folds={n}, "
            f"consistency={cs:.0%}, "
            f"IS/OOS_efficiency={er:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward analysis runner
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardAnalysis:
    """
    Run a strategy through sequential walk-forward folds and report IS/OOS metrics.

    Optionally performs in-sample parameter optimization on each fold, then applies
    the best parameters to the out-of-sample window — this tests whether parameter
    selection generalizes rather than just overfit to a single backtest.

    Usage (simple, fixed params):
        wfa = WalkForwardAnalysis(
            strategy_cls=EMACross,
            strategy_params={"fast": 20, "slow": 100},
            fixed_params={"symbol": "SPY"},
        )
        result = wfa.run(universe=universe, n_splits=5)
        print(result.consistency_score)     # e.g. 0.8 → 80% of OOS folds profitable
        print(result.efficiency_ratio)      # e.g. 0.7 → OOS Sharpe is 70% of IS Sharpe
        result.plot_oos_equity()

    Usage (per-fold parameter optimization):
        result = wfa.run(
            universe=universe,
            optimize=True,
            param_grid={"fast": [10, 20, 50], "slow": [50, 100, 200]},
            opt_metric="sharpe_ratio",
        )
    """

    def __init__(
        self,
        strategy_cls: type[Strategy],
        strategy_params: dict | None = None,
        fixed_params: dict | None = None,
        config: BacktestConfig | None = None,
        cost_model: CostModel | None = None,
        sizer=None,
        stop_loss=None,
    ):
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params or {}
        self.fixed_params = fixed_params or {}
        self.config = config or BacktestConfig()
        self.cost_model = cost_model
        self.sizer = sizer
        self.stop_loss = stop_loss

    def _build_strategy(self, params: dict) -> Strategy:
        all_params = {**self.fixed_params, **params}
        return self.strategy_cls(**all_params)

    def _run_on(
        self,
        universe: Universe,
        params: dict,
        timeframe: str | None,
    ) -> BacktestResult | None:
        strategy = self._build_strategy(params)
        bt = Backtester(
            strategy=strategy,
            config=self.config,
            cost_model=self.cost_model,
            sizer=self.sizer,
            stop_loss=self.stop_loss,
        )
        try:
            return bt.run(universe=copy.deepcopy(universe), timeframe=timeframe)
        except Exception:
            return None

    def _optimize_on_fold(
        self,
        train_universe: Universe,
        param_grid: dict,
        timeframe: str | None,
        opt_metric: str,
    ) -> dict:
        """Mini param sweep on the IS fold; return the best parameter combo."""
        from backtester.stress import ParamSweep
        sweep = ParamSweep(
            strategy_cls=self.strategy_cls,
            param_grid=param_grid,
            config=self.config,
            cost_model=self.cost_model,
            sizer=self.sizer,
            stop_loss=self.stop_loss,
            fixed_params=self.fixed_params,
        )
        sr = sweep.run(universe=copy.deepcopy(train_universe), timeframe=timeframe)
        if sr.summary.empty or opt_metric not in sr.summary.columns:
            return self.strategy_params
        best_row = sr.summary.loc[sr.summary[opt_metric].idxmax()]
        return {k: best_row[k] for k in param_grid}

    def run(
        self,
        universe: Universe,
        timeframe: str | None = None,
        n_splits: int = 5,
        split_method: str = "expanding",
        train_size: int | None = None,
        test_size: int | None = None,
        embargo_bars: int = 0,
        optimize: bool = False,
        param_grid: dict | None = None,
        opt_metric: str = "sharpe_ratio",
    ) -> WalkForwardResult:
        """
        Args:
            n_splits:       Number of IS/OOS folds.
            split_method:   "expanding" or "rolling".
            train_size:     Train window in bars (required for "rolling").
            test_size:      Test window in bars (auto-divided if None).
            embargo_bars:   Bars to remove between train end and test start.
            optimize:       Re-run param sweep on each IS fold.
            param_grid:     Required when optimize=True.
            opt_metric:     Metric to maximize during IS optimization.
        """
        if optimize and not param_grid:
            raise ValueError("param_grid is required when optimize=True")

        splitter = WalkForwardSplits(
            n_splits=n_splits,
            method=split_method,
            train_size=train_size,
            test_size=test_size,
            embargo_bars=embargo_bars,
        )

        is_rows: list[dict] = []
        oos_rows: list[dict] = []
        fold_meta: list[dict] = []
        all_oos_trades: list = []
        oos_equity_parts: list[pd.Series] = []
        running_equity = float(self.config.initial_capital)

        for split in splitter.split(universe):
            params = self.strategy_params
            if optimize:
                params = self._optimize_on_fold(split.train, param_grid, timeframe, opt_metric)

            res_is = self._run_on(split.train, params, timeframe)
            res_oos = self._run_on(split.test, params, timeframe)

            fold_meta.append({
                "fold": split.fold,
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "best_params": params if optimize else None,
            })

            if res_is is not None:
                is_rows.append({"fold": split.fold, **res_is.summary()})

            if res_oos is not None:
                oos_rows.append({"fold": split.fold, **res_oos.summary()})
                all_oos_trades.extend(res_oos.trades)

                eq = res_oos.equity_curve
                if not eq.empty:
                    # Scale so stitched curve starts from running_equity
                    scaled = eq * (running_equity / eq.iloc[0])
                    oos_equity_parts.append(scaled)
                    running_equity = float(scaled.iloc[-1])

        oos_equity = (
            pd.concat(oos_equity_parts) if oos_equity_parts else pd.Series(dtype=float)
        )

        def _to_df(rows: list[dict]) -> pd.DataFrame:
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows).set_index("fold")

        return WalkForwardResult(
            folds=fold_meta,
            oos_equity=oos_equity,
            oos_trades=all_oos_trades,
            is_summary=_to_df(is_rows),
            oos_summary=_to_df(oos_rows),
            meta={
                "n_folds": len(fold_meta),
                "split_method": split_method,
                "optimized": optimize,
                "opt_metric": opt_metric if optimize else None,
            },
        )
