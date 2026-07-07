from __future__ import annotations

import pandas as pd
from dotenv import load_dotenv, dotenv_values

from core.models import Allocation, BacktestConfig, Side
from core.universe import Universe
from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, default_cost_stack
from backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest
from hypothesis import (
    HypothesisTests,
    PermutationTest,
    BootstrapCI,
    WalkForwardAnalysis,
    DeflatedSharpeRatio,
    TrainTestValidateSplit,
    report as hypothesis_report,
)
from strategy.built_in import CompositeStrategy, SingleAssetStrategy
from strategy.indicators import bollinger, ema, rsi
from strategy.sizing import FixedNotionalSizer, VolatilityTargetSizer
from strategy.stops import NopStopLoss


# ═══════════════════════════════════════════════════════════════════════════
#  Data fetching
# ═══════════════════════════════════════════════════════════════════════════
def load_credentials() -> dict:
    load_dotenv()
    _env = dotenv_values()
    return {
        "key": _env.get("ALP_PAPER_KEY", ""),
        "secret": _env.get("ALP_PAPER_SECRET", ""),
    }


def fetch_alpaca_bars(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1d",
    api_key: str | None = None,
    api_secret: str | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca for a single symbol.

    Parameters
    ----------
    symbol    : ticker, e.g. "AAPL", "SPY"
    start     : ISO date string, e.g. "2023-01-01"
    end       : ISO date string, e.g. "2024-01-01"
    timeframe : one of "1d", "1h", "30m", "15m", "5m", "1m"
    api_key   : Alpaca key; falls back to ALPACA_KEY env var
    api_secret: Alpaca secret; falls back to ALPACA_SECRET env var

    Returns
    -------
    DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: alpaca-py. Install with: pip install alpaca-py"
        ) from exc

    _env = dotenv_values()
    key = api_key or _env.get("ALPACA_KEY", "")
    secret = api_secret or _env.get("ALPACA_SECRET", "")
    if not key or not secret:
        raise ValueError(
            "Alpaca credentials required. Set ALPACA_KEY and ALPACA_SECRET "
            "environment variables, or pass api_key/api_secret directly."
        )

    client = StockHistoricalDataClient(api_key=key, secret_key=secret)

    tf_map = {
        "1d":  TimeFrame.Day,
        "1h":  TimeFrame.Hour,
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "5m":  TimeFrame(5, TimeFrameUnit.Minute),
        "1m":  TimeFrame.Minute,
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(tf_map)}")

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=pd.Timestamp(start, tz="US/Eastern"),
        end=pd.Timestamp(end, tz="US/Eastern"),
        adjustment="all", 
    )
    bars = client.get_stock_bars(req)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    return df

class EmaRsiStrategy(SingleAssetStrategy):
    """
    Long-only trend strategy filtered by RSI and volatility regime.

    Entry:  close > slow EMA AND RSI < rsi_overbought AND vol regime = medium
    Exit:   close <= slow EMA OR RSI >= rsi_overbought OR vol regime != medium

    Vol regime uses the same expanding-quantile approach as RegimeStressTest:
    20-bar rolling realised vol is classified as low (<=q_low), high (>=q_high),
    or medium (between). Only the medium band allows open positions.
    """

    def __init__(
        self,
        symbol: str,
        slow: int = 200,
        rsi_period: int = 14,
        rsi_overbought: float = 80.0,
        vol_window: int = 20,
        vol_q_low: float = 0.33,
        vol_q_high: float = 0.66,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.vol_window = vol_window
        self.vol_q_low = vol_q_low
        self.vol_q_high = vol_q_high

    @property
    def params(self) -> dict:
        return {
            "slow": self.slow,
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
            "vol_q_low": self.vol_q_low,
            "vol_q_high": self.vol_q_high,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        data["ema_slow"] = ema(data["close"], self.slow)
        data["rsi"] = rsi(data["close"], self.rsi_period)
        # Expanding quantiles: at bar t only bars 0..t are used — no look-ahead.
        # Mirrors RegimeStressTest._default_vol_regime exactly.
        rv = data["close"].pct_change().rolling(self.vol_window).std()
        data["_rv"]     = rv
        data["_rv_q_lo"] = rv.expanding(min_periods=self.vol_window).quantile(self.vol_q_low)
        data["_rv_q_hi"] = rv.expanding(min_periods=self.vol_window).quantile(self.vol_q_high)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation()

        close   = data["close"].iat[idx]
        es      = data["ema_slow"].iat[idx]
        rsi_val = data["rsi"].iat[idx]
        rv      = data["_rv"].iat[idx]
        q_lo    = data["_rv_q_lo"].iat[idx]
        q_hi    = data["_rv_q_hi"].iat[idx]

        if any(v != v for v in (close, es, rsi_val)):
            return Allocation()

        if rv != rv or q_lo != q_lo or q_hi != q_hi:
            return Allocation(reason="vol_warmup")

        if rv <= q_lo:
            return Allocation(reason=f"vol_low rv={rv:.4f} q_lo={q_lo:.4f}")
        if rv >= q_hi:
            return Allocation(reason=f"vol_high rv={rv:.4f} q_hi={q_hi:.4f}")

        above_ema = close > es
        rsi_ok    = rsi_val < self.rsi_overbought

        if above_ema and rsi_ok:
            return Allocation(
                side=Side.LONG,
                weight=1.0,
                confidence=1.0,
                reason=f"above EMA{self.slow} | RSI={rsi_val:.0f} | vol=medium",
            )

        return Allocation(reason=f"no signal | above_ema={above_ema} | RSI={rsi_val:.0f} | vol=medium")


class BollingerMeanReversionStrategy(SingleAssetStrategy):
    """
    Long/short Bollinger Band mean reversion.

    Long entry:  close crosses below the lower band (oversold)
    Long exit:   close crosses back above the midline
    Short entry: close crosses above the upper band (overbought)
    Short exit:  close crosses back below the midline
    """

    def __init__(
        self,
        symbol: str,
        window: int = 20,
        num_std: float = 2.0,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.window = window
        self.num_std = num_std

    @property
    def params(self) -> dict:
        return {"window": self.window, "num_std": self.num_std}

    def setup_data(self, data: pd.DataFrame, l2=None):
        data["bb_mid"], data["bb_upper"], data["bb_lower"] = bollinger(
            data["close"], self.window, self.num_std
        )

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.window:
            return Allocation()

        close = data["close"].iat[idx]
        mid   = data["bb_mid"].iat[idx]
        upper = data["bb_upper"].iat[idx]
        lower = data["bb_lower"].iat[idx]

        if any(v != v for v in (close, mid, upper, lower)):
            return Allocation()

        if close < lower:
            return Allocation(
                side=Side.LONG,
                weight=1.0,
                confidence=1.0,
                reason=f"BB oversold | close={close:.2f} < lower={lower:.2f}",
            )

        if close > upper:
            return Allocation(
                side=Side.SHORT,
                weight=1.0,
                confidence=1.0,
                reason=f"BB overbought | close={close:.2f} > upper={upper:.2f}",
            )

        return Allocation(reason=f"BB no signal | close={close:.2f} mid={mid:.2f}")


class VolFilteredCompositeStrategy(CompositeStrategy):
    """
    CompositeStrategy that sits out when volatility is elevated.

    Computes a rolling vol (std of returns over `vol_window` bars) and
    compares it to a longer rolling median of that vol.  If the current
    vol exceeds `vol_multiplier` × median, the bar is skipped.
    This avoids look-ahead bias because the median is purely backward-looking.
    """

    def __init__(
        self,
        vol_window: int = 20,
        vol_multiplier: float = 1.5,
        **kw,
    ):
        super().__init__(**kw)
        self.vol_window = vol_window
        self.vol_multiplier = vol_multiplier

    @property
    def params(self) -> dict:
        return {
            **super().params,
            "vol_window": self.vol_window,
            "vol_multiplier": self.vol_multiplier,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        super().setup_data(data, l2)
        rv = data["close"].pct_change().rolling(self.vol_window).std()
        data["_rv"] = rv
        data["_rv_median"] = rv.rolling(self.vol_window * 3).median()

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.vol_window * 4:
            return Allocation()

        rv     = data["_rv"].iat[idx]
        rv_med = data["_rv_median"].iat[idx]

        if rv != rv or rv_med != rv_med or rv_med == 0:
            return Allocation()

        if rv > self.vol_multiplier * rv_med:
            return Allocation(reason=f"vol filter | rv={rv:.4f} > {self.vol_multiplier}x med={rv_med:.4f}")

        return super().bar(data, idx)


class BuyAndHoldStrategy(SingleAssetStrategy):
    """Always long at full weight from bar 0."""

    def bar(self, _data: pd.DataFrame, _idx: int) -> Allocation:
        return Allocation(side=Side.LONG, weight=1.0, reason="buy and hold")


# ═══════════════════════════════════════════════════════════════════════════
#  Demo runner
# ═══════════════════════════════════════════════════════════════════════════

_METRICS = [
    ("Total return %",    "total_return_pct"),
    ("Ann. return %",     "annualised_return_pct"),
    ("Ann. volatility %", "annualised_volatility_pct"),
    ("Sharpe",            "sharpe_ratio"),
    ("Sortino",           "sortino_ratio"),
    ("Max drawdown %",    "max_drawdown_pct"),
    ("Num trades",        "num_trades"),
    ("Win rate %",        "win_rate_pct"),
]


def _print_metrics_table(summaries: list[tuple[str, dict]]) -> None:
    col = 24
    headers = "".join(f"{label:>12}" for label, _ in summaries)
    print(f"\n{'Metric':<{col}} {headers}")
    print("-" * (col + 12 * len(summaries) + 1))
    for label, key in _METRICS:
        row = "".join(f"{s[key]:>12}" for _, s in summaries)
        print(f"{label:<{col}} {row}")


def demo(
    symbol: str = "SPY",
    start: str = "2005-01-01",
    end: str = "2026-01-01",
    timeframe: str = "1d",
):
    creds = load_credentials()
    print(f"\nFetching {symbol} {timeframe} bars from Alpaca ({start} → {end})...")
    data = fetch_alpaca_bars(
        symbol, start=start, end=end, timeframe=timeframe,
        api_key=creds["key"], api_secret=creds["secret"],
    )
    print(f"  {len(data)} bars loaded  |  {data.index[0].date()} → {data.index[-1].date()}")

    universe = Universe(symbols=[symbol])
    universe.add_asset(symbol, data)

    # ── Train / Test / Validate split (60 / 20 / 20, 10-bar embargo) ──────
    ttv = TrainTestValidateSplit.by_fractions(
        universe, train_frac=0.60, test_frac=0.20, embargo_bars=10
    )
    print(f"\n{ttv}")

    config = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    cost_model = CompositeCostModel(default_cost_stack())
    sizer = FixedNotionalSizer(notional=100_000)
    stoploss = NopStopLoss()

    def run_on(strategy, univ):
        return Backtester(
            strategy=strategy, config=config,
            sizer=sizer, stop_loss=stoploss, cost_model=cost_model,
        ).run(universe=univ, timeframe=timeframe)

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 1 — TRAIN  (strategy design, IS exploration)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 1 — TRAIN  (strategy design / IS exploration)")
    print(f"  {ttv.train_start.date()} → {ttv.train_end.date()}")
    print("═" * 70)

    print("\nRunning candidate strategies on TRAIN data...")
    train_ema  = run_on(EmaRsiStrategy(symbol=symbol, slow=200), ttv.train)
    train_bb   = run_on(BollingerMeanReversionStrategy(symbol=symbol), ttv.train)
    train_comp = run_on(
        VolFilteredCompositeStrategy(
            symbol=symbol,
            strategies=[
                EmaRsiStrategy(symbol=symbol, slow=200),
                BollingerMeanReversionStrategy(symbol=symbol, window=20, num_std=2.0),
            ],
            weights=[1, 0], threshold=0.4,
        ),
        ttv.train,
    )
    train_bah  = run_on(BuyAndHoldStrategy(symbol=symbol), ttv.train)

    _print_metrics_table([
        ("EMA/RSI",    train_ema.summary()),
        ("Mean Rev",   train_bb.summary()),
        ("Composite",  train_comp.summary()),
        ("Buy & Hold", train_bah.summary()),
    ])

    print("\n--- Walk-Forward Analysis on TRAIN data (5 expanding folds) ---")
    wfa = WalkForwardAnalysis(
        strategy_cls=EmaRsiStrategy,
        strategy_params={"slow": 200},
        fixed_params={"symbol": symbol},
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=stoploss,
    )
    wf = wfa.run(universe=ttv.train, timeframe=timeframe, n_splits=5, split_method="expanding")
    print(f"  Consistency score : {wf.consistency_score:.0%}  (fraction of IS sub-folds profitable)")
    print(f"  IS/OOS efficiency : {wf.efficiency_ratio:.2f}  (sub-OOS Sharpe / IS Sharpe)")
    tbl = wf.summary_table()
    for fold, row in tbl.iterrows():
        is_sr   = row.get("is_sharpe_ratio",       float("nan"))
        oos_sr  = row.get("oos_sharpe_ratio",      float("nan"))
        is_ret  = row.get("is_total_return_pct",   float("nan"))
        oos_ret = row.get("oos_total_return_pct",  float("nan"))
        print(
            f"    Fold {fold}  "
            f"IS  ret={is_ret:>7.2f}%  SR={is_sr:>6.3f}  │  "
            f"OOS ret={oos_ret:>7.2f}%  SR={oos_sr:>6.3f}"
        )

    print("\n  Decision: proceeding with EMA/RSI + Bollinger composite candidate.")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — TEST  (parameter optimisation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 2 — TEST  (parameter optimisation)")
    print(f"  {ttv.test_start.date()} → {ttv.test_end.date()}")
    print("═" * 70)

    ema_grid = {"slow": [100, 150, 200], "rsi_overbought": [65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 100.0]}
    bb_grid  = {"window": [10, 15, 20, 30], "num_std": [1.5, 2.0, 2.5]}
    n_ema_trials = 3 * 3        # 9
    n_bb_trials  = 4 * 3       # 12
    n_trials     = n_ema_trials + n_bb_trials  # used for DSR in validate

    print(f"\nSweeping EmaRsiStrategy ({n_ema_trials} combos) on TEST data...")
    ema_sweep = ParamSweep(
        strategy_cls=EmaRsiStrategy,
        param_grid=ema_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=stoploss,
    ).run(universe=ttv.test, timeframe=timeframe)
    best_ema_row  = ema_sweep.best("sharpe_ratio")
    best_ema_params = {k: best_ema_row[k] for k in ema_grid}
    best_ema_params["slow"] = int(best_ema_params["slow"])
    print(f"  Best EMA params : {best_ema_params}  →  SR={best_ema_row['sharpe_ratio']:.3f}")

    print(f"\nSweeping BollingerMeanReversionStrategy ({n_bb_trials} combos) on TEST data...")
    bb_sweep = ParamSweep(
        strategy_cls=BollingerMeanReversionStrategy,
        param_grid=bb_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=stoploss,
    ).run(universe=ttv.test, timeframe=timeframe)
    best_bb_row    = bb_sweep.best("sharpe_ratio")
    best_bb_params = {k: best_bb_row[k] for k in bb_grid}
    best_bb_params["window"] = int(best_bb_params["window"])
    print(f"  Best BB params  : {best_bb_params}  →  SR={best_bb_row['sharpe_ratio']:.3f}")

    print(f"\n  Total trials tracked for DSR: {n_trials}")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 3 — VALIDATE  (blind final evaluation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 70)
    print("  PHASE 3 — VALIDATE  (blind final evaluation)")
    print(f"  {ttv.validate_start.date()} → {ttv.validate_end.date()}")
    print("═" * 70)

    best_ema_strat = EmaRsiStrategy(symbol=symbol, **best_ema_params)
    best_bb_strat  = BollingerMeanReversionStrategy(symbol=symbol, **best_bb_params)
    final_composite = VolFilteredCompositeStrategy(
        symbol=symbol,
        strategies=[best_ema_strat, best_bb_strat],
        weights=[0.5, 0.5], threshold=0.4, vol_window=20, vol_multiplier=1.5,
    )

    print("\nRunning final strategies on VALIDATE data...")
    val_comp = run_on(final_composite, ttv.validate)
    val_bah  = run_on(BuyAndHoldStrategy(symbol=symbol), ttv.validate)

    run_dir = val_comp.save("ttv_validate")
    print(f"  Result saved to: {run_dir}")

    _print_metrics_table([
        ("Composite (tuned)", val_comp.summary()),
        ("Buy & Hold",        val_bah.summary()),
    ])

    # ── Hypothesis test battery ───────────────────────────────────────────
    print("\n\n=== Hypothesis Tests — VALIDATE (final composite) ===")
    tests = HypothesisTests.run_all(val_comp)
    print(hypothesis_report(tests))

    # ── Strategy comparison ───────────────────────────────────────────────
    print("\n=== Strategy Comparison: Composite vs Buy & Hold (VALIDATE) ===")
    for metric in ("sharpe_ratio", "total_return_pct"):
        t = HypothesisTests.compare(val_comp, val_bah, metric=metric)
        verdict = "✓ Composite wins" if t.reject_null else "✗ No significant edge"
        print(f"  {metric:<22} p={t.p_value:.4f}  {verdict}")

    # ── Permutation test ──────────────────────────────────────────────────
    print("\n=== Permutation Test — VALIDATE composite (sharpe_ratio, 2000 perms) ===")
    pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000)
    pt_result = pt.run(val_comp)
    print(
        f"  Observed SR={pt_result.statistic:.3f}  "
        f"null median={pt_result.meta['null_mean']:.3f}  "
        f"p={pt_result.p_value:.4f}  "
        f"{'✓ Significant' if pt_result.reject_null else '✗ Not significant'}"
    )

    # ── Bootstrap confidence intervals ────────────────────────────────────
    print("\n=== Bootstrap 95% CIs — VALIDATE composite (2000 samples) ===")
    ci = BootstrapCI(n_bootstrap=2_000, ci=0.95)
    cis = ci.run(val_comp)
    ci_col = 22
    print(f"  {'Metric':<{ci_col}} {'Observed':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
    print("  " + "-" * (ci_col + 32))
    for metric, vals in cis.items():
        print(
            f"  {metric:<{ci_col}}"
            f" {vals['observed']:>10.3f}"
            f" {vals['lower']:>10.3f}"
            f" {vals['upper']:>10.3f}"
        )

    # ── Deflated Sharpe Ratio (accounts for all test-phase trials) ────────
    print(f"\n=== Deflated Sharpe Ratio — VALIDATE (n_trials={n_trials}) ===")
    dsr = DeflatedSharpeRatio()
    d = dsr.compute(val_comp, n_trials=n_trials)
    verdict = "✓ Genuine edge" if d.reject_null else "✗ Likely overfit"
    print(
        f"  Composite (tuned)  SR={d.observed_sharpe:.4f}  "
        f"deflated_SR={d.deflated_sharpe:.4f}  "
        f"p={d.p_value:.4f}  {verdict}"
    )

    # ── Monte Carlo simulations ───────────────────────────────────────────
    print("\n=== Monte Carlo Simulations — VALIDATE (1000 bootstrap runs) ===")
    mc = MonteCarloStress(n_simulations=1_000, method="bootstrap")
    mc_res = mc.run(val_comp)
    m = mc_res.meta
    print(
        f"  Median ret={m['median_return']:.2f}%  "
        f"5th={m['5th_pctl_return']:.2f}%  "
        f"95th={m['95th_pctl_return']:.2f}%  "
        f"Median DD={m['median_max_dd']:.2f}%"
    )

    # ── Regime stress tests on validate universe ──────────────────────────
    print("\n=== Regime Stress Tests — VALIDATE ===")
    stress_cfg  = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    stress_cols = ["regime", "n_bars", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]
    for regime_label, regime_fn in [
        ("Volatility", None),
        ("Trend",      RegimeStressTest.trend_regime),
    ]:
        rst = RegimeStressTest(regime_fn=regime_fn, config=stress_cfg, cost_model=cost_model)
        sr  = rst.run(strategy=final_composite, universe=ttv.validate)
        df  = sr.summary.sort_values("regime")[stress_cols].reset_index(drop=True)
        print(f"\n  {regime_label} regimes:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    demo()
