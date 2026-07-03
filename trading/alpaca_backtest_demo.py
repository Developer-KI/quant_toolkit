from __future__ import annotations

import pandas as pd
from dotenv import load_dotenv, dotenv_values

from core.models import Allocation, BacktestConfig, Side
from core.universe import Universe
from backtester.engine import Backtester
from backtester.costs import CompositeCostModel, default_cost_stack
from backtester.stress import MonteCarloStress, RegimeStressTest
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
    Long-only EMA crossover filtered by RSI.

    Entry:  fast EMA crosses above slow EMA AND RSI < rsi_overbought
    Exit:   fast EMA crosses below slow EMA OR RSI >= rsi_overbought
    Weight: proportional to ATR-normalised trend strength, capped at 1.0
    """

    def __init__(
        self,
        symbol: str,
        fast: int = 50,
        slow: int = 200,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought

    @property
    def params(self) -> dict:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "rsi_period": self.rsi_period,
            "rsi_overbought": self.rsi_overbought,
        }

    def setup_data(self, data: pd.DataFrame, l2=None):
        data["ema_fast"] = ema(data["close"], self.fast)
        data["ema_slow"] = ema(data["close"], self.slow)
        data["rsi"] = rsi(data["close"], self.rsi_period)

    def bar(self, data: pd.DataFrame, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation()

        ef = data["ema_fast"].iat[idx]
        es = data["ema_slow"].iat[idx]
        rsi_val = data["rsi"].iat[idx]

        if any(v != v for v in (ef, es, rsi_val)):  # NaN check
            return Allocation()

        bull_cross = ef > es
        rsi_ok = rsi_val < self.rsi_overbought

        if bull_cross and rsi_ok:
            return Allocation(
                side=Side.LONG,
                weight=1.0,
                confidence=1.0,
                reason=f"EMA bull | RSI={rsi_val:.0f}",
            )

        return Allocation(reason=f"no signal | EMA bull={bull_cross} | RSI={rsi_val:.0f}")


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


def demo(
    symbol: str = "SPY",
    start: str = "2010-01-01",
    end: str = "2026-06-01",
    timeframe: str = "1d",
):
    creds = load_credentials()
    print(f"\nFetching {symbol} {timeframe} bars from Alpaca ({start} → {end})...")
    data = fetch_alpaca_bars(symbol, start=start, end=end, timeframe=timeframe, api_key=creds["key"], api_secret=creds["secret"])
    print(f"  {len(data)} bars loaded  |  {data.index[0].date()} → {data.index[-1].date()}")

    universe = Universe(symbols=[symbol])
    universe.add_asset(symbol, data)

    config = BacktestConfig(
        initial_capital=100_000.0,
        max_position_pct=1.0,
        leverage=2.0,
    )

    cost_model = CompositeCostModel(default_cost_stack())
    sizer = FixedNotionalSizer(notional=100_000)
    stoploss = NopStopLoss()

    def run_bt(strategy):
        return Backtester(
            strategy=strategy,
            config=config,
            sizer=sizer,
            stop_loss=stoploss,
            cost_model=cost_model,
        ).run(universe=universe, timeframe=timeframe)

    ema_rsi = EmaRsiStrategy(symbol=symbol, fast=50, slow=200)
    mean_rev = BollingerMeanReversionStrategy(symbol=symbol)

    print("\nRunning EMA/RSI strategy...")
    result = run_bt(ema_rsi)

    print("Running Bollinger mean-reversion strategy...")
    mr = run_bt(mean_rev)

    print("Running composite (EMA/RSI + mean reversion, vol-filtered)...")
    composite = VolFilteredCompositeStrategy(
        symbol=symbol,
        strategies=[
            EmaRsiStrategy(symbol=symbol, fast=50, slow=200),
            BollingerMeanReversionStrategy(symbol=symbol, window=20, num_std=2),
        ],
        weights=[0.5, 0.5],
        threshold=0.4,
        vol_window=20,
        vol_multiplier=1.5,
    )
    comp = run_bt(composite)

    print("Running buy-and-hold benchmark...")
    bah = run_bt(BuyAndHoldStrategy(symbol=symbol))

    run_dir = comp.save("rev + trend")
    print(f"Backtest saved to: {run_dir}")

    

    s1 = result.summary()
    s2 = mr.summary()
    s3 = comp.summary()
    s4 = bah.summary()
    metrics = [
        ("Total return %",      "total_return_pct"),
        ("Ann. return %",       "annualised_return_pct"),
        ("Ann. volatility %",   "annualised_volatility_pct"),
        ("Sharpe",              "sharpe_ratio"),
        ("Sortino",             "sortino_ratio"),
        ("Max drawdown %",      "max_drawdown_pct"),
        ("Num trades",          "num_trades"),
        ("Win rate %",          "win_rate_pct"),
    ]
    col = 24
    print(f"\n{'Metric':<{col}} {'EMA/RSI':>12} {'Mean Rev':>12} {'Composite':>12} {'Buy & Hold':>12}")
    print("-" * (col + 51))
    for label, key in metrics:
        print(f"{label:<{col}} {s1[key]:>12} {s2[key]:>12} {s3[key]:>12} {s4[key]:>12}")

    # ── Regime stress tests ───────────────────────────────────────────────
    print("\n\n=== Regime Stress Tests (Composite) ===")
    stress_cfg = BacktestConfig(initial_capital=100_000.0, max_position_pct=1.0, leverage=1.0)
    stress_cols = ["regime", "n_bars", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"]

    for regime_label, regime_fn in [
        ("Volatility", None),
        ("Trend",      RegimeStressTest.trend_regime),
    ]:
        rst = RegimeStressTest(regime_fn=regime_fn, config=stress_cfg, cost_model=cost_model)
        sr = rst.run(strategy=composite, universe=universe)
        df = sr.summary.sort_values("regime")[stress_cols].reset_index(drop=True)
        print(f"\n{regime_label} regimes:")
        print(df.to_string(index=False))

    # ── Monte Carlo simulations ───────────────────────────────────────────
    print("\n\n=== Monte Carlo Simulations (1 000 runs, bootstrap) ===")
    mc = MonteCarloStress(n_simulations=1_000, method="bootstrap")

    mc_col = 14
    print(f"\n{'Strategy':<{mc_col}} {'Median ret%':>12} {'5th pctl%':>10} {'95th pctl%':>11} {'Median DD%':>11}")
    print("-" * (mc_col + 46))
    for strat_label, bt_res in [("EMA/RSI", result), ("Mean Rev", mr), ("Composite", comp)]:
        mc_res = mc.run(bt_res)
        m = mc_res.meta
        print(
            f"{strat_label:<{mc_col}}"
            f" {m['median_return']:>12.2f}"
            f" {m['5th_pctl_return']:>10.2f}"
            f" {m['95th_pctl_return']:>11.2f}"
            f" {m['median_max_dd']:>11.2f}"
        )


if __name__ == "__main__":
    demo()
