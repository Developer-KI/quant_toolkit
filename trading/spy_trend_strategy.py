"""
SPY Trend-Following Research — Iterative Strategy Development
=============================================================
Explores multiple trend-following ideas on SPY daily bars, comparing
each against buy-and-hold. Refines through backtesting, parameter
sweeps, stress testing, and hypothesis testing.

Usage:
    python trading/spy_trend_strategy.py
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
from strategy.indicators import ema, sma, rsi, atr
from strategy.sizing import FixedNotionalSizer
from strategy.stops import ATRStop, NopStopLoss


# ─────────────────────────────────────────────────────────────────────────────
# Custom indicators
# ─────────────────────────────────────────────────────────────────────────────

def donchian(
    high: pd.Series, low: pd.Series, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(upper, lower, mid) over rolling window."""
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    return upper, lower, (upper + lower) / 2


def adx_indicators(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (adx, plus_di, minus_di) using Wilder smoothing."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up = high.diff()
    dn = -low.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)

    alpha = 1.0 / period
    smooth_tr  = tr.ewm(alpha=alpha, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=alpha, adjust=False).mean() / smooth_tr
    ndi = 100 * ndm.ewm(alpha=alpha, adjust=False).mean() / smooth_tr

    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100
    adx_line = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx_line, pdi, ndi


def macd_indicators(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ml = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=signal, adjust=False).mean()
    return ml, sl, ml - sl


def _nan(*vals) -> bool:
    return any(math.isnan(float(v)) for v in vals)


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
# Strategies
# ─────────────────────────────────────────────────────────────────────────────

class BuyAndHoldStrategy(SingleAssetStrategy):
    @property
    def params(self) -> dict: return {}

    def bar(self, _d, _i) -> Allocation:
        return Allocation(side=Side.LONG, weight=1.0, reason="buy_and_hold")


# ── Iteration 1: Dual EMA Crossover ─────────────────────────────────────────

class EmaCrossStrategy(SingleAssetStrategy):
    """Long only. Fast EMA above slow EMA = long, else flat."""

    def __init__(self, symbol: str, fast: int = 20, slow: int = 50, **kw):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.slow = slow

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow}

    def setup_data(self, data, l2=None):
        data["ef"] = ema(data["close"], self.fast)
        data["es"] = ema(data["close"], self.slow)

    def bar(self, data, idx) -> Allocation:
        if idx < self.slow + 5:
            return Allocation()
        f, s = data["ef"].iat[idx], data["es"].iat[idx]
        if _nan(f, s):
            return Allocation()
        if f > s:
            return Allocation(side=Side.LONG, weight=1.0,
                              reason=f"EMA{self.fast}>{self.slow}")
        return Allocation(reason="bearish_cross")


# ── Iteration 2: Triple EMA Stack + RSI ──────────────────────────────────────

class EmaTrendRsiStrategy(SingleAssetStrategy):
    """Triple EMA stack aligned (fast>mid>slow) + RSI overbought guard."""

    def __init__(self, symbol: str, fast: int = 20, mid: int = 50, slow: int = 200,
                 rsi_period: int = 14, rsi_ob: float = 75.0, **kw):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.mid  = mid
        self.slow = slow
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "mid": self.mid, "slow": self.slow, "rsi_ob": self.rsi_ob}

    def setup_data(self, data, l2=None):
        data["ef"]    = ema(data["close"], self.fast)
        data["em"]    = ema(data["close"], self.mid)
        data["es"]    = ema(data["close"], self.slow)
        data["rsi_v"] = rsi(data["close"], self.rsi_period)

    def bar(self, data, idx) -> Allocation:
        if idx < self.slow + 5:
            return Allocation()
        f = data["ef"].iat[idx]
        m = data["em"].iat[idx]
        s = data["es"].iat[idx]
        r = data["rsi_v"].iat[idx]
        if _nan(f, m, s, r):
            return Allocation()
        if f > m > s and r < self.rsi_ob:
            return Allocation(side=Side.LONG, weight=1.0, confidence=(f - s) / s,
                              reason=f"stack RSI={r:.0f}")
        return Allocation(reason=f"no_signal RSI={r:.0f}")


# ── Iteration 3: Donchian Channel Breakout ────────────────────────────────────

class DonchianTrendStrategy(SingleAssetStrategy):
    """
    Turtle-style Donchian: long above entry_window-bar high (prior bar), flat
    below exit_window-bar low (prior bar). Trend filter: only above trend SMA.
    Between channels → hold long (simplified stateless interpretation).
    """

    def __init__(self, symbol: str, entry_window: int = 20, exit_window: int = 10,
                 trend_period: int = 200, **kw):
        super().__init__(symbol=symbol, **kw)
        self.entry_window = entry_window
        self.exit_window  = exit_window
        self.trend_period = trend_period

    @property
    def params(self) -> dict:
        return {"entry_window": self.entry_window, "exit_window": self.exit_window,
                "trend_period": self.trend_period}

    def setup_data(self, data, l2=None):
        up, lo, _ = donchian(data["high"], data["low"], self.entry_window)
        _, ex, _  = donchian(data["high"], data["low"], self.exit_window)
        data["don_up"]   = up.shift(1)   # previous bar to avoid look-ahead
        data["don_exit"] = ex.shift(1)
        data["sma_t"]    = sma(data["close"], self.trend_period)

    def bar(self, data, idx) -> Allocation:
        warmup = max(self.entry_window, self.trend_period) + 2
        if idx < warmup:
            return Allocation()
        close = data["close"].iat[idx]
        up    = data["don_up"].iat[idx]
        ex    = data["don_exit"].iat[idx]
        tr    = data["sma_t"].iat[idx]
        if _nan(close, up, ex, tr):
            return Allocation()

        above_trend = close > tr
        if not above_trend or close < ex:
            return Allocation(reason="exit")
        if close >= up:
            return Allocation(side=Side.LONG, weight=1.0, confidence=0.8,
                              reason=f"breakout {close:.2f}>{up:.2f}")
        # Between exit and entry channels: hold long
        return Allocation(side=Side.LONG, weight=1.0, confidence=0.5, reason="hold")


# ── Iteration 4: ADX-Confirmed EMA Trend ─────────────────────────────────────

class AdxEmaStrategy(SingleAssetStrategy):
    """
    EMA crossover (fast>slow) confirmed by ADX strength and +DI>-DI direction.
    RSI filter avoids overbought entries.
    """

    def __init__(self, symbol: str, fast: int = 20, slow: int = 100,
                 adx_period: int = 14, adx_threshold: float = 25.0,
                 rsi_period: int = 14, rsi_ob: float = 75.0, **kw):
        super().__init__(symbol=symbol, **kw)
        self.fast          = fast
        self.slow          = slow
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period    = rsi_period
        self.rsi_ob        = rsi_ob

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow,
                "adx_threshold": self.adx_threshold, "rsi_ob": self.rsi_ob}

    def setup_data(self, data, l2=None):
        data["ef"]    = ema(data["close"], self.fast)
        data["es"]    = ema(data["close"], self.slow)
        av, pdi, ndi  = adx_indicators(data["high"], data["low"], data["close"], self.adx_period)
        data["adx_v"] = av
        data["pdi"]   = pdi
        data["ndi"]   = ndi
        data["rsi_v"] = rsi(data["close"], self.rsi_period)

    def bar(self, data, idx) -> Allocation:
        warmup = max(self.slow, self.adx_period * 3) + 5
        if idx < warmup:
            return Allocation()
        f   = data["ef"].iat[idx]
        s   = data["es"].iat[idx]
        av  = data["adx_v"].iat[idx]
        pdi = data["pdi"].iat[idx]
        ndi = data["ndi"].iat[idx]
        r   = data["rsi_v"].iat[idx]
        if _nan(f, s, av, r):
            return Allocation()
        if f > s and pdi > ndi and av > self.adx_threshold and r < self.rsi_ob:
            return Allocation(side=Side.LONG, weight=1.0,
                              confidence=min(av / 50.0, 1.0),
                              reason=f"ADX={av:.1f} RSI={r:.0f}")
        return Allocation(reason=f"no_signal ADX={av:.1f}")


# ── Iteration 5: MACD + EMA Trend Filter ─────────────────────────────────────

class MacdTrendStrategy(SingleAssetStrategy):
    """MACD histogram positive + above 200 EMA + ADX confirmation."""

    def __init__(self, symbol: str, macd_fast: int = 12, macd_slow: int = 26,
                 macd_sig: int = 9, trend_ema: int = 200,
                 adx_period: int = 14, adx_threshold: float = 20.0, **kw):
        super().__init__(symbol=symbol, **kw)
        self.macd_fast     = macd_fast
        self.macd_slow     = macd_slow
        self.macd_sig      = macd_sig
        self.trend_ema     = trend_ema
        self.adx_period    = adx_period
        self.adx_threshold = adx_threshold

    @property
    def params(self) -> dict:
        return {"trend_ema": self.trend_ema, "adx_threshold": self.adx_threshold}

    def setup_data(self, data, l2=None):
        _, _, hist          = macd_indicators(data["close"], self.macd_fast,
                                              self.macd_slow, self.macd_sig)
        data["macd_hist"]   = hist
        data["ema_t"]       = ema(data["close"], self.trend_ema)
        av, _, _            = adx_indicators(data["high"], data["low"], data["close"],
                                             self.adx_period)
        data["adx_v"]       = av

    def bar(self, data, idx) -> Allocation:
        warmup = max(self.trend_ema, self.macd_slow + self.macd_sig + self.adx_period) + 5
        if idx < warmup:
            return Allocation()
        close = data["close"].iat[idx]
        hist  = data["macd_hist"].iat[idx]
        trend = data["ema_t"].iat[idx]
        av    = data["adx_v"].iat[idx]
        if _nan(close, hist, trend, av):
            return Allocation()
        if close > trend and hist > 0 and av > self.adx_threshold:
            return Allocation(side=Side.LONG, weight=1.0,
                              confidence=min(av / 50.0, 1.0),
                              reason=f"MACD+ ADX={av:.1f}")
        return Allocation(reason=f"exit hist={hist:.3f} trend={'above' if close > trend else 'below'}")


# ── Iteration 6: Adaptive Multi-Signal Synthesis ─────────────────────────────

class AdaptiveTrendStrategy(SingleAssetStrategy):
    """
    Combines the best signals from prior iterations:
      - Triple EMA stack (fast > mid > slow)
      - ADX strength threshold + +DI > -DI direction
      - MACD histogram positive (momentum aligned)
      - RSI overbought guard
      - Volatility regime filter (sit out during vol spikes)
    """

    def __init__(
        self, symbol: str,
        fast: int = 20, mid: int = 50, slow: int = 200,
        adx_period: int = 14, adx_min: float = 22.0,
        rsi_period: int = 14, rsi_ob: float = 72.0,
        vol_window: int = 20, vol_mult: float = 2.0,
        **kw,
    ):
        super().__init__(symbol=symbol, **kw)
        self.fast       = fast
        self.mid        = mid
        self.slow       = slow
        self.adx_period = adx_period
        self.adx_min    = adx_min
        self.rsi_period = rsi_period
        self.rsi_ob     = rsi_ob
        self.vol_window = vol_window
        self.vol_mult   = vol_mult

    @property
    def params(self) -> dict:
        return {
            "fast": self.fast, "mid": self.mid, "slow": self.slow,
            "adx_min": self.adx_min, "rsi_ob": self.rsi_ob, "vol_mult": self.vol_mult,
        }

    def setup_data(self, data, l2=None):
        data["ef"]       = ema(data["close"], self.fast)
        data["em"]       = ema(data["close"], self.mid)
        data["es"]       = ema(data["close"], self.slow)
        av, pdi, ndi     = adx_indicators(data["high"], data["low"], data["close"], self.adx_period)
        data["adx_v"]    = av
        data["pdi"]      = pdi
        data["ndi"]      = ndi
        data["rsi_v"]    = rsi(data["close"], self.rsi_period)
        _, _, hist       = macd_indicators(data["close"])
        data["macd_h"]   = hist
        rv               = data["close"].pct_change().rolling(self.vol_window).std()
        data["rv"]       = rv
        data["rv_med"]   = rv.rolling(self.vol_window * 3).median()

    def bar(self, data, idx) -> Allocation:
        warmup = max(self.slow, self.adx_period * 3, self.vol_window * 4) + 5
        if idx < warmup:
            return Allocation()

        ef     = data["ef"].iat[idx]
        em     = data["em"].iat[idx]
        es     = data["es"].iat[idx]
        av     = data["adx_v"].iat[idx]
        pdi    = data["pdi"].iat[idx]
        ndi    = data["ndi"].iat[idx]
        r      = data["rsi_v"].iat[idx]
        hist   = data["macd_h"].iat[idx]
        rv     = data["rv"].iat[idx]
        rv_med = data["rv_med"].iat[idx]

        if _nan(ef, em, es, av, r):
            return Allocation()

        # Volatility regime filter
        if not _nan(rv, rv_med) and rv_med > 0 and rv > self.vol_mult * rv_med:
            return Allocation(reason=f"vol_spike rv={rv:.4f}")

        ema_stack  = ef > em > es
        di_bull    = pdi > ndi
        adx_ok     = av > self.adx_min
        rsi_ok     = r < self.rsi_ob
        macd_ok    = not _nan(hist) and hist > 0

        if ema_stack and di_bull and adx_ok and rsi_ok and macd_ok:
            conf = min((av - self.adx_min) / 28.0, 1.0)
            return Allocation(
                side=Side.LONG, weight=1.0, confidence=conf,
                reason=f"FULL ADX={av:.1f} RSI={r:.0f}",
            )
        return Allocation(reason=f"no_signal stack={ema_stack} ADX={av:.1f}")


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
    ("Max UW bars",       "max_time_underwater_bars"),
    ("Avg UW bars",       "avg_time_underwater_bars"),
    ("# trades",          "num_trades"),
    ("Win rate %",        "win_rate_pct"),
]


def print_table(summaries: list[tuple[str, dict]]) -> None:
    col = 20
    headers = "".join(f"{name:>15}" for name, _ in summaries)
    print(f"\n  {'Metric':<{col}} {headers}")
    print("  " + "-" * (col + 15 * len(summaries) + 1))
    for label, key in _METRICS:
        row = []
        for _, s in summaries:
            v = s.get(key, float("nan"))
            row.append(f"{float(v):>15.2f}" if isinstance(v, (int, float)) else f"{v:>15}")
        print(f"  {label:<{col}} {''.join(row)}")


def backtest(strategy, universe, timeframe, config, cost_model, sizer, stop_loss):
    return Backtester(
        strategy=strategy, config=config,
        cost_model=cost_model, sizer=sizer, stop_loss=stop_loss,
    ).run(universe=universe, timeframe=timeframe)


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

    def run(strat, univ=None, stop=None):
        return backtest(strat, univ or ttv.train, TF, config, cost_model, sizer, stop or nop)

    bah = BuyAndHoldStrategy(symbol=SYMBOL)

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 1 — Dual EMA Crossover
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 1 — Dual EMA Crossover")
    print("═" * 72)

    r1_2050  = run(EmaCrossStrategy(symbol=SYMBOL, fast=20, slow=50))
    r1_50200 = run(EmaCrossStrategy(symbol=SYMBOL, fast=50, slow=200))
    r1_bah   = run(bah)

    print_table([
        ("EMA 20/50",  r1_2050.summary()),
        ("EMA 50/200", r1_50200.summary()),
        ("B&H",        r1_bah.summary()),
    ])
    print("\n  Observation: raw EMA cross is whippy — too many false signals on SPY.")
    print("  Next: add 200-day trend filter + RSI overbought guard.")

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 2 — Triple EMA Stack + RSI
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 2 — Triple EMA Stack (20/50/200) + RSI Overbought Guard")
    print("═" * 72)

    r2_70 = run(EmaTrendRsiStrategy(symbol=SYMBOL, rsi_ob=70.0))
    r2_75 = run(EmaTrendRsiStrategy(symbol=SYMBOL, rsi_ob=75.0))
    r2_80 = run(EmaTrendRsiStrategy(symbol=SYMBOL, rsi_ob=80.0))

    print_table([
        ("EMA+RSI OB=70", r2_70.summary()),
        ("EMA+RSI OB=75", r2_75.summary()),
        ("EMA+RSI OB=80", r2_80.summary()),
        ("B&H",           r1_bah.summary()),
    ])
    print("\n  Observation: trend stack reduces drawdown; RSI threshold matters less.")
    print("  Next: try Donchian channel for cleaner, momentum-based entries.")

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 3 — Donchian Channel (Turtle-style)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 3 — Donchian Channel Breakout")
    print("═" * 72)

    r3_2010  = run(DonchianTrendStrategy(symbol=SYMBOL, entry_window=20, exit_window=10))
    r3_5520  = run(DonchianTrendStrategy(symbol=SYMBOL, entry_window=55, exit_window=20))

    print_table([
        ("Donchian 20/10", r3_2010.summary()),
        ("Donchian 55/20", r3_5520.summary()),
        ("B&H",            r1_bah.summary()),
    ])
    print("\n  Observation: 55-day breakout holds trades longer — fewer but cleaner.")
    print("  Next: add ADX to only trade genuine trending environments.")

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 4 — ADX-Confirmed EMA
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 4 — ADX Trend Strength + EMA Direction")
    print("═" * 72)

    r4_20 = run(AdxEmaStrategy(symbol=SYMBOL, adx_threshold=20.0))
    r4_25 = run(AdxEmaStrategy(symbol=SYMBOL, adx_threshold=25.0))
    r4_30 = run(AdxEmaStrategy(symbol=SYMBOL, adx_threshold=30.0))

    print_table([
        ("ADX>20", r4_20.summary()),
        ("ADX>25", r4_25.summary()),
        ("ADX>30", r4_30.summary()),
        ("B&H",    r1_bah.summary()),
    ])
    print("\n  Observation: ADX filter improves win rate and Sharpe.")
    print("  Next: add MACD histogram for momentum direction confirmation.")

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 5 — MACD + EMA Trend
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 5 — MACD Momentum + 200-EMA Trend + ADX")
    print("═" * 72)

    r5_20 = run(MacdTrendStrategy(symbol=SYMBOL, adx_threshold=20.0))
    r5_25 = run(MacdTrendStrategy(symbol=SYMBOL, adx_threshold=25.0))

    print_table([
        ("MACD+ADX>20", r5_20.summary()),
        ("MACD+ADX>25", r5_25.summary()),
        ("B&H",         r1_bah.summary()),
    ])

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 6 — Parameter Sweep on Best Candidates (TRAIN)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 6 — Parameter Sweep (TRAIN set)")
    print("═" * 72)

    adx_grid = {
        "fast":          [10, 20, 30],
        "slow":          [50, 100, 150],
        "adx_threshold": [18.0, 22.0, 27.0, 32.0],
        "rsi_ob":        [70.0, 75.0, 80.0],
    }
    n_adx = 3 * 3 * 4 * 3  # 108
    print(f"\n  Sweeping AdxEmaStrategy ({n_adx} combos)...")
    adx_sweep = ParamSweep(
        strategy_cls=AdxEmaStrategy,
        param_grid=adx_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=nop,
        n_jobs=-1,
    ).run(universe=ttv.train, timeframe=TF)

    best_adx_row    = adx_sweep.best("sharpe_ratio")
    best_adx_params = {k: best_adx_row[k] for k in adx_grid}
    best_adx_params["fast"] = int(best_adx_params["fast"])
    best_adx_params["slow"] = int(best_adx_params["slow"])
    print(f"  Best ADX params: {best_adx_params}  →  SR={best_adx_row['sharpe_ratio']:.3f}")

    don_grid = {
        "entry_window": [10, 20, 40, 55],
        "exit_window":  [5, 10, 20],
        "trend_period": [100, 200],
    }
    n_don = 4 * 3 * 2  # 24
    print(f"\n  Sweeping DonchianTrendStrategy ({n_don} combos)...")
    don_sweep = ParamSweep(
        strategy_cls=DonchianTrendStrategy,
        param_grid=don_grid,
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=nop,
        n_jobs=-1,
    ).run(universe=ttv.train, timeframe=TF)

    best_don_row    = don_sweep.best("sharpe_ratio")
    best_don_params = {k: best_don_row[k] for k in don_grid}
    best_don_params["entry_window"] = int(best_don_params["entry_window"])
    best_don_params["exit_window"]  = int(best_don_params["exit_window"])
    best_don_params["trend_period"] = int(best_don_params["trend_period"])
    print(f"  Best Donchian params: {best_don_params}  →  SR={best_don_row['sharpe_ratio']:.3f}")

    n_total_trials = n_adx + n_don  # for DSR later

    # ═════════════════════════════════════════════════════════════════════════
    #  ITERATION 7 — Adaptive Multi-Signal + ATR Stop
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print("  ITERATION 7 — Adaptive Multi-Signal Strategy (TRAIN)")
    print("═" * 72)

    atr_stop = ATRStop(atr_mult_sl=2.0, atr_mult_tp=4.0)

    r7_nop  = run(AdaptiveTrendStrategy(symbol=SYMBOL))
    r7_atr  = run(AdaptiveTrendStrategy(symbol=SYMBOL), stop=atr_stop)

    print_table([
        ("Adaptive (no stop)",  r7_nop.summary()),
        ("Adaptive (ATR 2x/4x)", r7_atr.summary()),
        ("B&H",                 r1_bah.summary()),
    ])

    use_atr = r7_atr.summary()["sharpe_ratio"] > r7_nop.summary()["sharpe_ratio"]
    adapt_stop = atr_stop if use_atr else nop
    print(f"\n  Selected stop: {'ATR 2x/4x' if use_atr else 'none'}")

    print("\n  Walk-Forward Analysis — Adaptive strategy on TRAIN (5 expanding folds)...")
    wfa = WalkForwardAnalysis(
        strategy_cls=AdaptiveTrendStrategy,
        strategy_params={},
        fixed_params={"symbol": SYMBOL},
        config=config, cost_model=cost_model, sizer=sizer, stop_loss=adapt_stop,
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
    #  TEST SET — Pick best candidate
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print(f"  TEST SET  ({ttv.test_start.date()} → {ttv.test_end.date()})")
    print("═" * 72)

    r_t_adapt = run(AdaptiveTrendStrategy(symbol=SYMBOL),      ttv.test, adapt_stop)
    r_t_adx   = run(AdxEmaStrategy(symbol=SYMBOL, **best_adx_params), ttv.test, nop)
    r_t_don   = run(DonchianTrendStrategy(symbol=SYMBOL, **best_don_params), ttv.test, nop)
    r_t_bah   = run(bah, ttv.test, nop)

    print_table([
        ("Adaptive",     r_t_adapt.summary()),
        ("ADX (tuned)",  r_t_adx.summary()),
        ("Donchian",     r_t_don.summary()),
        ("B&H",          r_t_bah.summary()),
    ])

    candidates = [
        ("Adaptive",  r_t_adapt, AdaptiveTrendStrategy, {},            {"symbol": SYMBOL}, adapt_stop),
        ("ADX",       r_t_adx,   AdxEmaStrategy,         best_adx_params, {"symbol": SYMBOL}, nop),
        ("Donchian",  r_t_don,   DonchianTrendStrategy,  best_don_params, {"symbol": SYMBOL}, nop),
    ]
    best_name, best_test_r, best_cls, best_kw, best_fixed, best_final_stop = max(
        candidates, key=lambda x: x[1].summary()["sharpe_ratio"]
    )
    print(f"\n  Best on TEST: {best_name}  SR={best_test_r.summary()['sharpe_ratio']:.3f}")

    # ═════════════════════════════════════════════════════════════════════════
    #  VALIDATE SET — Blind final evaluation
    # ═════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 72)
    print(f"  VALIDATE SET — BLIND  ({ttv.validate_start.date()} → {ttv.validate_end.date()})")
    print("═" * 72)

    final_strat = best_cls(**{**best_fixed, **best_kw})
    r_val       = run(final_strat, ttv.validate, best_final_stop)
    r_val_bah   = run(bah, ttv.validate, nop)

    print_table([
        (f"{best_name} (final)", r_val.summary()),
        ("Buy & Hold",           r_val_bah.summary()),
    ])

    save_dir = r_val.save("spy_trend_validate")
    print(f"\n  Result saved → {save_dir}")

    # ── Hypothesis tests ──────────────────────────────────────────────────────
    print("\n\n=== Hypothesis Tests (VALIDATE) ===")
    tests = HypothesisTests.run_all(r_val)
    print(hypothesis_report(tests))

    print("\n=== Strategy vs Buy & Hold (VALIDATE) ===")
    for metric in ("sharpe_ratio", "total_return_pct"):
        t = HypothesisTests.compare(r_val, r_val_bah, metric=metric)
        verdict = "✓ edge" if t.reject_null else "✗ no significant edge"
        print(f"  {metric:<28} p={t.p_value:.4f}  {verdict}")

    # ── Permutation test ──────────────────────────────────────────────────────
    print("\n=== Permutation Test (VALIDATE, 2000 perms) ===")
    pt  = PermutationTest(metric="sharpe_ratio", n_permutations=2_000)
    ptr = pt.run(r_val)
    print(
        f"  Observed SR={ptr.statistic:.3f}  "
        f"null median={ptr.meta['null_mean']:.3f}  "
        f"p={ptr.p_value:.4f}  "
        f"{'✓ significant' if ptr.reject_null else '✗ not significant'}"
    )

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    print("\n=== Bootstrap 95% CIs (VALIDATE, 2000 samples) ===")
    ci  = BootstrapCI(n_bootstrap=2_000, ci=0.95)
    cis = ci.run(r_val)
    w   = 22
    print(f"  {'Metric':<{w}} {'Observed':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
    print("  " + "-" * (w + 32))
    for m, v in cis.items():
        print(f"  {m:<{w}} {v['observed']:>10.3f} {v['lower']:>10.3f} {v['upper']:>10.3f}")

    # ── Deflated Sharpe Ratio ─────────────────────────────────────────────────
    print(f"\n=== Deflated Sharpe Ratio (n_trials={n_total_trials}) ===")
    dsr = DeflatedSharpeRatio()
    d   = dsr.compute(r_val, n_trials=n_total_trials)
    print(
        f"  SR={d.observed_sharpe:.4f}  deflated_SR={d.deflated_sharpe:.4f}  "
        f"p={d.p_value:.4f}  "
        f"{'✓ genuine edge' if d.reject_null else '✗ likely overfit'}"
    )

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print("\n=== Monte Carlo (VALIDATE, 1000 bootstrap runs) ===")
    mc     = MonteCarloStress(n_simulations=1_000, method="bootstrap")
    mc_res = mc.run(r_val)
    mm     = mc_res.meta
    print(
        f"  Median return: {mm['median_return']:.2f}%  "
        f"5th pct: {mm['5th_pctl_return']:.2f}%  "
        f"95th pct: {mm['95th_pctl_return']:.2f}%  "
        f"Median DD: {mm['median_max_dd']:.2f}%"
    )

    # ── Regime stress ─────────────────────────────────────────────────────────
    print("\n=== Regime Stress Tests (VALIDATE) ===")
    cols = ["regime", "n_bars", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
    for label, fn in [("Volatility", None), ("Trend", RegimeStressTest.trend_regime)]:
        fresh  = best_cls(**{**best_fixed, **best_kw})
        rst    = RegimeStressTest(regime_fn=fn, config=config, cost_model=cost_model)
        sr     = rst.run(strategy=fresh, universe=ttv.validate)
        df     = sr.summary.sort_values("regime")[cols].reset_index(drop=True)
        print(f"\n  {label} regimes:")
        print(df.to_string(index=False))

    print("\n  Done.")


if __name__ == "__main__":
    main()
