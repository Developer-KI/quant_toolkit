"""
SPY Trend + Mean Reversion — Literature-Based Dual-Leg Strategy
===============================================================
Implements the blueprint from research/trend+mean_rev.md:
  Leg 1 — Long-only monthly trend (Faber 2007 / Zakamulin):
           Long when last month-end close > 10-month (≈200-day) SMA.
           Signal evaluated at month-end only; held until next month-end.

  Leg 2 — Long mean reversion (Connors RSI(2) + Pagonidis IBS):
           Long when above 200-day SMA AND (RSI(2) < threshold OR IBS < threshold).
           Exit when close > 5-day SMA OR RSI(2) > exit threshold.
           No hard stop (Connors: stops hurt index MR).

  Combined — blends both legs; trend leg weight ~65%, MR leg fills to 100%.
             Only MR above the 200-day MA (regime gate from the report).

Execution: all signals shifted by 1 bar — trade on next bar's open/close.
           This enforces the Zakamulin rule (never trade the signal bar).

Usage:
    python trading/trend_meanrev_strategy.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, default_cost_stack
from backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest
from core.models import Allocation, BacktestConfig, Side
from core.universe import Universe
from hypothesis import (
    HypothesisTests,
    PermutationTest,
    BootstrapCI,
    WalkForwardAnalysis,
    DeflatedSharpeRatio,
    TrainTestValidateSplit,
    report as hypothesis_report,
)
from strategy.built_in import SingleAssetStrategy
from strategy.indicators import sma, rsi
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nan(*vals) -> bool:
    return any(math.isnan(float(v)) for v in vals)


def _ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Internal Bar Strength = (close - low) / (high - low). Range [0, 1]."""
    rng = (high - low).replace(0, np.nan)
    return (close - low) / rng


def _monthly_trend_signal(close: pd.Series, sma_period: int) -> pd.Series:
    """
    Faber-style monthly trend signal: 1 if month-end close > SMA, else 0.
    Evaluated at each month-end then forward-filled to daily.
    Returns a float Series aligned to close.index.
    """
    sma_vals = close.rolling(sma_period).mean()
    above = (close > sma_vals).astype(float)
    # Resample to month-end: take last value of each month
    monthly = above.resample("ME").last()
    # Re-align to daily index, forward-fill until next month-end
    daily = monthly.reindex(close.index, method="ffill")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_spy(start: str = "2003-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    env = dotenv_values()
    key    = env.get("ALP_PAPER_KEY", "")
    secret = env.get("ALP_PAPER_SECRET", "")
    if not key or not secret:
        raise ValueError("Set ALP_PAPER_KEY and ALP_PAPER_SECRET in .env")
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise ImportError("pip install alpaca-py") from exc

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)
    req = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start, tz="US/Eastern"),
        end=pd.Timestamp(end, tz="US/Eastern"),
        adjustment="all",
    )
    bars = client.get_stock_bars(req)
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("SPY", level="symbol")
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open", "high", "low", "close", "volume"]].sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

class BuyAndHoldStrategy(SingleAssetStrategy):
    @property
    def params(self) -> dict: return {}

    def bar(self, _d, _i) -> Allocation:
        return Allocation(side=Side.LONG, weight=1.0, reason="buy_and_hold")


# ─────────────────────────────────────────────────────────────────────────────
# Leg 1 — Monthly Trend
# ─────────────────────────────────────────────────────────────────────────────

class TrendLegStrategy(SingleAssetStrategy):
    """
    Faber 2007 monthly SMA timing rule.
    Long if last month-end close > sma_period-day SMA; else flat.
    Signal shifted by 1 bar (next-bar execution per Zakamulin).
    """

    def __init__(self, symbol: str, sma_period: int = 200, **kw):
        super().__init__(symbol=symbol, **kw)
        self.sma_period = sma_period

    @property
    def params(self) -> dict:
        return {"sma_period": self.sma_period}

    def setup_data(self, data: pd.DataFrame, l2=None):
        sig = _monthly_trend_signal(data["close"], self.sma_period)
        # Shift by 1: signal from yesterday's close → execute today
        data["_trend_sig"] = sig.shift(1)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.sma_period + 22:   # needs at least one month-end after SMA warmup
            return Allocation()

        sig = data["_trend_sig"].iat[idx]
        if _nan(sig):
            return Allocation()

        if sig > 0.5:
            return Allocation(
                side=Side.LONG, weight=1.0, confidence=1.0,
                reason=f"trend_long sma={self.sma_period}",
            )
        return Allocation(reason="trend_flat below SMA")


# ─────────────────────────────────────────────────────────────────────────────
# Leg 2 — Mean Reversion (RSI(2) + IBS)
# ─────────────────────────────────────────────────────────────────────────────

class MeanRevLegStrategy(SingleAssetStrategy):
    """
    Long-only index mean reversion per Connors/Alvarez RSI(2) + Pagonidis IBS.

    Entry (above 200-SMA regime gate):
        RSI(2) < rsi_entry  OR  IBS < ibs_entry

    Exit:
        close > 5-day SMA  OR  RSI(2) > rsi_exit

    No hard stop (Connors: stops hurt index MR).
    All signals shifted 1 bar for next-bar execution.
    """

    def __init__(
        self,
        symbol: str,
        sma_regime: int = 200,
        rsi_entry: float = 10.0,
        ibs_entry: float = 0.20,
        rsi_exit: float = 65.0,
        sma_exit: int = 5,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.sma_regime = sma_regime
        self.rsi_entry  = rsi_entry
        self.ibs_entry  = ibs_entry
        self.rsi_exit   = rsi_exit
        self.sma_exit   = sma_exit

    @property
    def params(self) -> dict:
        return {
            "sma_regime": self.sma_regime,
            "rsi_entry":  self.rsi_entry,
            "ibs_entry":  self.ibs_entry,
            "rsi_exit":   self.rsi_exit,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        close = data["close"]
        high  = data["high"]
        low   = data["low"]

        regime_sma = sma(close, self.sma_regime)
        rsi2       = rsi(close, 2)
        ibs_vals   = _ibs(high, low, close)
        exit_sma   = sma(close, self.sma_exit)

        # Shift all signals by 1 — next-bar execution
        data["_mr_above_regime"] = (close > regime_sma).astype(float).shift(1)
        data["_mr_rsi2"]         = rsi2.shift(1)
        data["_mr_ibs"]          = ibs_vals.shift(1)
        data["_mr_exit_sma"]     = exit_sma.shift(1)
        data["_mr_close_prev"]   = close.shift(1)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        warmup = self.sma_regime + 5
        if idx < warmup:
            return Allocation()

        above_regime = data["_mr_above_regime"].iat[idx]
        rsi2_val     = data["_mr_rsi2"].iat[idx]
        ibs_val      = data["_mr_ibs"].iat[idx]
        exit_sma_val = data["_mr_exit_sma"].iat[idx]
        close_prev   = data["_mr_close_prev"].iat[idx]

        if _nan(above_regime, rsi2_val, ibs_val, exit_sma_val):
            return Allocation()

        # Regime gate: only trade long side above 200-SMA
        if above_regime < 0.5:
            return Allocation(reason="mr_flat below_regime")

        # Exit condition: above 5-SMA or RSI(2) mean-reverted
        above_exit = close_prev > exit_sma_val
        rsi_exited = rsi2_val > self.rsi_exit

        if above_exit or rsi_exited:
            return Allocation(
                reason=f"mr_flat exit | above_exit={above_exit} RSI2={rsi2_val:.1f}"
            )

        # Entry: oversold on RSI(2) or IBS
        rsi_entry = rsi2_val < self.rsi_entry
        ibs_entry = ibs_val  < self.ibs_entry

        if rsi_entry or ibs_entry:
            return Allocation(
                side=Side.LONG, weight=1.0, confidence=1.0,
                reason=(
                    f"mr_long | RSI2={rsi2_val:.1f}<{self.rsi_entry}"
                    f" IBS={ibs_val:.2f}<{self.ibs_entry}"
                ),
            )

        return Allocation(reason=f"mr_flat | RSI2={rsi2_val:.1f} IBS={ibs_val:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Combined — Dual-Leg Strategy
# ─────────────────────────────────────────────────────────────────────────────

class TrendMeanRevStrategy(SingleAssetStrategy):
    """
    Dual-leg strategy combining Faber monthly trend + Connors/IBS mean reversion.

    Allocation logic (stateless, next-bar execution):
      - Below 200-SMA regime → flat (no MR; trend is off too)
      - Above 200-SMA, trend signal on, no MR signal → trend_weight (partial long)
      - Above 200-SMA, trend signal on, MR entry active → 1.0 (both legs)
      - Above 200-SMA, trend signal off, MR entry active → mr_weight (MR-only)
      - Above 200-SMA, trend off, no MR → flat

    This directly models 60-70% trend / 30-40% MR risk-budget split from report.
    """

    def __init__(
        self,
        symbol: str,
        sma_trend: int = 200,
        trend_weight: float = 0.65,
        rsi_entry: float = 10.0,
        ibs_entry: float = 0.20,
        rsi_exit: float = 65.0,
        sma_exit: int = 5,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.sma_trend    = sma_trend
        self.trend_weight = trend_weight
        self.mr_weight    = round(1.0 - trend_weight, 4)
        self.rsi_entry    = rsi_entry
        self.ibs_entry    = ibs_entry
        self.rsi_exit     = rsi_exit
        self.sma_exit     = sma_exit

    @property
    def params(self) -> dict:
        return {
            "sma_trend":    self.sma_trend,
            "trend_weight": self.trend_weight,
            "rsi_entry":    self.rsi_entry,
            "ibs_entry":    self.ibs_entry,
            "rsi_exit":     self.rsi_exit,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        close = data["close"]
        high  = data["high"]
        low   = data["low"]

        # ── Trend leg ────────────────────────────────────────────────────────
        trend_sig = _monthly_trend_signal(close, self.sma_trend)
        data["_trend_sig"] = trend_sig.shift(1)

        # ── Regime gate (daily, same SMA) ────────────────────────────────────
        regime_sma = sma(close, self.sma_trend)
        data["_above_regime"] = (close > regime_sma).astype(float).shift(1)

        # ── Mean-reversion signals ───────────────────────────────────────────
        rsi2      = rsi(close, 2)
        ibs_vals  = _ibs(high, low, close)
        exit_sma  = sma(close, self.sma_exit)

        data["_rsi2"]      = rsi2.shift(1)
        data["_ibs"]       = ibs_vals.shift(1)
        data["_exit_sma"]  = exit_sma.shift(1)
        data["_close_lag"] = close.shift(1)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        warmup = self.sma_trend + 25
        if idx < warmup:
            return Allocation()

        trend_sig    = data["_trend_sig"].iat[idx]
        above_regime = data["_above_regime"].iat[idx]
        rsi2_val     = data["_rsi2"].iat[idx]
        ibs_val      = data["_ibs"].iat[idx]
        exit_sma_val = data["_exit_sma"].iat[idx]
        close_lag    = data["_close_lag"].iat[idx]

        if _nan(trend_sig, above_regime, rsi2_val, ibs_val, exit_sma_val):
            return Allocation()

        # Below 200-SMA regime → flat (MR only works above it)
        if above_regime < 0.5:
            return Allocation(reason="flat: below 200-SMA regime")

        trend_on = trend_sig > 0.5

        # MR exit: above 5-SMA or RSI(2) recovered
        mr_exited = (close_lag > exit_sma_val) or (rsi2_val > self.rsi_exit)
        mr_entry  = (rsi2_val < self.rsi_entry) or (ibs_val < self.ibs_entry)
        mr_active = mr_entry and not mr_exited

        if trend_on and mr_active:
            # Both legs agree → full allocation
            return Allocation(
                side=Side.LONG, weight=1.0, confidence=1.0,
                reason=(
                    f"trend+MR | RSI2={rsi2_val:.1f} IBS={ibs_val:.2f}"
                ),
            )
        elif trend_on:
            # Trend only → partial allocation (60-70% of capital)
            return Allocation(
                side=Side.LONG, weight=self.trend_weight, confidence=0.8,
                reason=f"trend_only | RSI2={rsi2_val:.1f}",
            )
        elif mr_active:
            # MR only above 200-SMA (trend monthly signal is off but regime gate ok)
            return Allocation(
                side=Side.LONG, weight=self.mr_weight, confidence=0.6,
                reason=f"MR_only | RSI2={rsi2_val:.1f} IBS={ibs_val:.2f}",
            )

        return Allocation(reason=f"flat: trend_on={trend_on} mr_active={mr_active}")


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_METRICS = [
    ("Total return %",    "total_return_pct"),
    ("Ann. return %",     "annualised_return_pct"),
    ("Ann. vol %",        "annualised_volatility_pct"),
    ("Sharpe",            "sharpe_ratio"),
    ("Sortino",           "sortino_ratio"),
    ("Max DD %",          "max_drawdown_pct"),
    ("% time underwater", "pct_time_underwater"),
    ("# trades",          "num_trades"),
    ("Win rate %",        "win_rate_pct"),
]


def print_table(summaries: list[tuple[str, dict]]) -> None:
    col = 22
    headers = "".join(f"{name:>15}" for name, _ in summaries)
    print(f"\n  {'Metric':<{col}} {headers}")
    print("  " + "-" * (col + 15 * len(summaries) + 1))
    for label, key in _METRICS:
        row = []
        for _, s in summaries:
            v = s.get(key, float("nan"))
            row.append(f"{float(v):>15.2f}" if isinstance(v, (int, float)) else f"{v:>15}")
        print(f"  {label:<{col}} {''.join(row)}")


def run_bt(strategy, universe, tf, config, cost_model, sizer, stop_loss):
    return Backtester(
        strategy=strategy, config=config,
        cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
    ).run(universe=universe, timeframe=tf)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

SYMBOL = "SPY"
TF     = "1d"
START  = "2003-01-01"
END    = "2025-12-31"


def main() -> None:
    print(f"\nFetching {SYMBOL} daily bars from Alpaca ({START} → {END})...")
    data = fetch_spy(START, END)
    print(f"  {len(data)} bars  |  {data.index[0].date()} → {data.index[-1].date()}")

    universe = Universe(symbols=[SYMBOL])
    universe.add_asset(SYMBOL, data)

    ttv = TrainTestValidateSplit.by_fractions(
        universe, train_frac=0.60, test_frac=0.20, embargo_bars=10,
    )
    print(f"\n{ttv}")

    config     = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    cost_model = CompositeCostModel(default_cost_stack())
    sizer      = FixedNotionalSizer(notional=100_000)
    nop        = NopStopLoss()

    def run(strat, univ=None):
        return run_bt(strat, univ or ttv.train, TF, config, cost_model, sizer, nop)

    bah    = BuyAndHoldStrategy(symbol=SYMBOL)
    trend  = TrendLegStrategy(symbol=SYMBOL, sma_period=200)
    mr     = MeanRevLegStrategy(symbol=SYMBOL)
    combo  = TrendMeanRevStrategy(symbol=SYMBOL)

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 1 — TRAIN
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  PHASE 1 — TRAIN")
    print(f"  {ttv.train_start.date()} → {ttv.train_end.date()}")
    print("═" * 72)

    r_bah    = run(bah)
    r_trend  = run(trend)
    r_mr     = run(mr)
    r_combo  = run(combo)

    print("\n  Individual legs vs benchmark on TRAIN data:")
    print_table([
        ("Trend only",   r_trend.summary()),
        ("MR only",      r_mr.summary()),
        ("Trend+MR",     r_combo.summary()),
        ("Buy & Hold",   r_bah.summary()),
    ])

    # Observe: trend leg primarily reduces drawdown; MR adds return in ranging mkt
    print("\n  Note: trend leg is a drawdown-reduction overlay (Zakamulin/Faber).")
    print("  MR provides return in range-bound regimes; combo raises Sharpe.")

    # Walk-Forward Analysis on train set
    print("\n  Walk-Forward Analysis — TrendMeanRevStrategy, 5 expanding folds...")
    wfa = WalkForwardAnalysis(
        strategy_cls=TrendMeanRevStrategy,
        strategy_params={},
        fixed_params={"symbol": SYMBOL},
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=nop,
    )
    wf = wfa.run(universe=ttv.train, timeframe=TF, n_splits=5, split_method="expanding")
    print(f"  Consistency: {wf.consistency_score:.0%}  |  IS/OOS efficiency: {wf.efficiency_ratio:.2f}")
    for fold, row in wf.summary_table().iterrows():
        isr  = row.get("is_sharpe_ratio",      float("nan"))
        oosr = row.get("oos_sharpe_ratio",     float("nan"))
        iret = row.get("is_total_return_pct",  float("nan"))
        oret = row.get("oos_total_return_pct", float("nan"))
        print(f"    Fold {fold}  IS ret={iret:>7.2f}%  SR={isr:.3f}  │  OOS ret={oret:>7.2f}%  SR={oosr:.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 2 — TEST (parameter sweep)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  PHASE 2 — TEST (parameter sweep)")
    print(f"  {ttv.test_start.date()} → {ttv.test_end.date()}")
    print("═" * 72)

    # Sweep trend leg SMA period
    trend_grid = {"sma_period": [150, 200, 252]}
    n_trend = len(trend_grid["sma_period"])
    print(f"\n  Sweeping TrendLegStrategy ({n_trend} combos)...")
    trend_sweep = ParamSweep(
        strategy_cls=TrendLegStrategy,
        param_grid=trend_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=nop,
    ).run(universe=ttv.test, timeframe=TF)
    best_trend_row    = trend_sweep.best("sharpe_ratio")
    best_sma_period   = int(best_trend_row["sma_period"])
    print(f"  Best TrendLeg: sma_period={best_sma_period}  SR={best_trend_row['sharpe_ratio']:.3f}")

    # Sweep MR leg thresholds
    mr_grid = {
        "rsi_entry": [5.0, 10.0, 15.0],
        "ibs_entry": [0.15, 0.20, 0.25],
        "rsi_exit":  [60.0, 70.0],
    }
    n_mr = 3 * 3 * 2  # 18
    print(f"\n  Sweeping MeanRevLegStrategy ({n_mr} combos)...")
    mr_sweep = ParamSweep(
        strategy_cls=MeanRevLegStrategy,
        param_grid=mr_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=nop,
    ).run(universe=ttv.test, timeframe=TF)
    best_mr_row    = mr_sweep.best("sharpe_ratio")
    best_mr_params = {k: best_mr_row[k] for k in mr_grid}
    print(f"  Best MR params: {best_mr_params}  SR={best_mr_row['sharpe_ratio']:.3f}")

    # Sweep combined strategy
    combo_grid = {
        "trend_weight": [0.50, 0.65, 0.70],
        "rsi_entry":    [5.0, 10.0],
        "ibs_entry":    [0.15, 0.20],
        "rsi_exit":     [60.0, 70.0],
    }
    n_combo = 3 * 2 * 2 * 2  # 24
    n_total_trials = n_trend + n_mr + n_combo  # used for DSR
    print(f"\n  Sweeping TrendMeanRevStrategy ({n_combo} combos)...")
    combo_sweep = ParamSweep(
        strategy_cls=TrendMeanRevStrategy,
        param_grid=combo_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=nop,
    ).run(universe=ttv.test, timeframe=TF)
    best_combo_row    = combo_sweep.best("sharpe_ratio")
    best_combo_params = {k: best_combo_row[k] for k in combo_grid}
    print(f"  Best combo params: {best_combo_params}  SR={best_combo_row['sharpe_ratio']:.3f}")
    print(f"\n  Total trials tracked for DSR: {n_total_trials}")

    # ═════════════════════════════════════════════════════════════════════════
    #  PHASE 3 — VALIDATE (blind final evaluation)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  PHASE 3 — VALIDATE (blind final evaluation)")
    print(f"  {ttv.validate_start.date()} → {ttv.validate_end.date()}")
    print("═" * 72)

    final_trend = TrendLegStrategy(symbol=SYMBOL, sma_period=best_sma_period)
    final_mr    = MeanRevLegStrategy(symbol=SYMBOL, **best_mr_params)
    final_combo = TrendMeanRevStrategy(symbol=SYMBOL, **best_combo_params)

    r_val_trend = run(final_trend,  ttv.validate)
    r_val_mr    = run(final_mr,     ttv.validate)
    r_val_combo = run(final_combo,  ttv.validate)
    r_val_bah   = run(bah,          ttv.validate)

    print("\n  Final strategies on VALIDATE (blind):")
    print_table([
        (f"Trend (sma={best_sma_period})", r_val_trend.summary()),
        ("MR (tuned)",                      r_val_mr.summary()),
        ("Trend+MR (tuned)",               r_val_combo.summary()),
        ("Buy & Hold",                      r_val_bah.summary()),
    ])

    save_dir = r_val_combo.save("trend_meanrev_validate")
    print(f"\n  Result saved → {save_dir}")

    # ── Hypothesis test battery ───────────────────────────────────────────────
    print("\n\n=== Hypothesis Tests — VALIDATE (Trend+MR combined) ===")
    tests = HypothesisTests.run_all(r_val_combo)
    print(hypothesis_report(tests))

    print("\n=== Trend+MR vs Buy & Hold (VALIDATE) ===")
    for metric in ("sharpe_ratio", "total_return_pct"):
        t = HypothesisTests.compare(r_val_combo, r_val_bah, metric=metric)
        verdict = "✓ edge" if t.reject_null else "✗ no significant edge"
        print(f"  {metric:<28} p={t.p_value:.4f}  {verdict}")

    # ── Permutation test ──────────────────────────────────────────────────────
    print("\n=== Permutation Test — VALIDATE (2000 perms, sharpe_ratio) ===")
    pt  = PermutationTest(metric="sharpe_ratio", n_permutations=2_000)
    ptr = pt.run(r_val_combo)
    print(
        f"  Observed SR={ptr.statistic:.3f}  "
        f"null median={ptr.meta['null_mean']:.3f}  "
        f"p={ptr.p_value:.4f}  "
        f"{'✓ significant' if ptr.reject_null else '✗ not significant'}"
    )

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    print("\n=== Bootstrap 95% CIs — VALIDATE (2000 samples) ===")
    ci  = BootstrapCI(n_bootstrap=2_000, ci=0.95)
    cis = ci.run(r_val_combo)
    w   = 22
    print(f"  {'Metric':<{w}} {'Observed':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
    print("  " + "-" * (w + 32))
    for m, v in cis.items():
        print(f"  {m:<{w}} {v['observed']:>10.3f} {v['lower']:>10.3f} {v['upper']:>10.3f}")

    # ── Deflated Sharpe Ratio ─────────────────────────────────────────────────
    print(f"\n=== Deflated Sharpe Ratio (n_trials={n_total_trials}) ===")
    dsr = DeflatedSharpeRatio()
    d   = dsr.compute(r_val_combo, n_trials=n_total_trials)
    print(
        f"  SR={d.observed_sharpe:.4f}  deflated_SR={d.deflated_sharpe:.4f}  "
        f"p={d.p_value:.4f}  "
        f"{'✓ genuine edge' if d.reject_null else '✗ likely overfit'}"
    )

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print("\n=== Monte Carlo — VALIDATE (1000 bootstrap runs) ===")
    mc     = MonteCarloStress(n_simulations=1_000, method="bootstrap")
    mc_res = mc.run(r_val_combo)
    mm     = mc_res.meta
    print(
        f"  Median return: {mm['median_return']:.2f}%  "
        f"5th pct: {mm['5th_pctl_return']:.2f}%  "
        f"95th pct: {mm['95th_pctl_return']:.2f}%  "
        f"Median DD: {mm['median_max_dd']:.2f}%"
    )

    # ── Regime stress tests ───────────────────────────────────────────────────
    print("\n=== Regime Stress Tests — VALIDATE ===")
    cols = ["regime", "n_bars", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
    for label, fn in [("Volatility", None), ("Trend", RegimeStressTest.trend_regime)]:
        fresh = TrendMeanRevStrategy(symbol=SYMBOL, **best_combo_params)
        rst   = RegimeStressTest(regime_fn=fn, config=config, cost_model=cost_model)
        sr    = rst.run(strategy=fresh, universe=ttv.validate)
        df    = sr.summary.sort_values("regime")[cols].reset_index(drop=True)
        print(f"\n  {label} regimes:")
        print(df.to_string(index=False))

    # ── Report card ───────────────────────────────────────────────────────────
    print("\n\n" + "═" * 72)
    print("  REPORT CARD")
    print("═" * 72)
    combo_s = r_val_combo.summary()
    bah_s   = r_val_bah.summary()
    sharpe_delta = combo_s["sharpe_ratio"] - bah_s["sharpe_ratio"]
    dd_delta     = combo_s["max_drawdown_pct"] - bah_s["max_drawdown_pct"]
    print(f"\n  Sharpe vs B&H:       {combo_s['sharpe_ratio']:.3f} vs {bah_s['sharpe_ratio']:.3f}"
          f"  (Δ {sharpe_delta:+.3f})")
    print(f"  Max DD vs B&H:       {combo_s['max_drawdown_pct']:.1f}% vs {bah_s['max_drawdown_pct']:.1f}%"
          f"  (Δ {dd_delta:+.1f}pp)")
    print(f"  Ann. return vs B&H:  {combo_s['annualised_return_pct']:.1f}% vs"
          f" {bah_s['annualised_return_pct']:.1f}%")
    if sharpe_delta > 0.20:
        verdict = "✓ PASS — Sharpe edge >0.20 vs B&H (report threshold)"
    elif sharpe_delta > 0.0:
        verdict = "△ MARGINAL — Sharpe edge present but below 0.20 report threshold"
    else:
        verdict = "✗ FAIL — No Sharpe edge vs B&H; simplify to trend leg alone"
    print(f"\n  Verdict: {verdict}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
