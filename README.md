# Quantitative Trading Framework

---

## About

My focus is on **doing things right**: clean dependency graphs, protocol-driven interfaces, statistically sound backtest methodology (no look-ahead, proper train/test/validate splits, deflated Sharpe ratio), and code that is easy to extend without being over-engineered.

**Contact:** [ivanov.r.kiril@abv.bg](mailto:ivanov.r.kiril@abv.bg)

---

## Quick Start

```bash
# Python 3.10+
pip install pandas pyarrow numpy streamlit plotly websockets scipy python-dotenv

# Launch the strategy explorer dashboard
streamlit run app/Strategy_Explorer.py

# Run the full backtest demo (single-asset + multi-asset + multi-exchange + hypothesis tests)
python trading/backtest_demo.py

# Run Alpaca paper-trading live demo
python trading/alpaca_livetest_demo.py --symbol SPY

# Collect Hyperliquid ETH data (trades + L2 + funding)
python -m src.data.feeds.hyperliquid --coin ETH --mode all
```

Create a `.env` file in the project root:

```env
# Alpaca paper trading
ALP_PAPER_KEY=your_key
ALP_PAPER_SECRET=your_secret

# Alpaca live trading
ALP_LIVE_KEY=your_key
ALP_LIVE_SECRET=your_secret

# London Strategic Edge historical data (used by the dashboard)
LSE_DATA=your_key
```

---

## Package Layout

```
src/
├── core/                        # Stable contracts — no upward imports
│   ├── models.py                # Side, Allocation, Position, Trade, BacktestConfig, LiveConfig, …
│   ├── protocols.py             # typing.Protocol interfaces (C++ interop seam)
│   ├── events.py                # BarEvent, TradeEvent, L2Event structs
│   ├── universe.py              # Universe — holds OHLCV + L2 + funding per symbol
│   └── parser.py                # OHLCV resampling, L2/funding alignment
│
├── strategy/                    # Pure-Python strategy framework
│   ├── base.py                  # Strategy, StrategyContext, PortfolioTarget, registries
│   ├── built_in.py              # SingleAssetStrategy, CompositeStrategy, PerAssetStrategy, …
│   ├── indicators.py            # Stateless indicator functions
│   ├── sizing.py                # Sizer hierarchy (FixedNotional, FixedFractional, VolTarget, …)
│   ├── stops.py                 # StopLoss hierarchy (NopStop, FixedPct, ATR, Trailing, …)
│   └── overlay.py               # PortfolioOverlay, NetExposureOverlay, DeltaNeutralOverlay
│
├── testing/
│   ├── backtester/
│   │   ├── engine.py            # Backtester + BacktestResult
│   │   ├── costs.py             # Pluggable cost models (fee, slippage, impact, funding)
│   │   └── stress.py            # ParamSweep, MonteCarloStress, RegimeStressTest
│   └── hypothesis/
│       ├── tests.py             # HypothesisTests, PermutationTest, BootstrapCI
│       ├── walk_forward.py      # WalkForwardAnalysis
│       ├── overfitting.py       # DeflatedSharpeRatio, ProbabilityOfBacktestOverfitting
│       └── splits.py            # TrainTestValidateSplit, WalkForwardSplits
│
├── execution/                   # Live trading engine (single- and multi-exchange)
│   ├── engine.py                # Engine — handles single, per-exchange, and cross-exchange modes
│   ├── executor.py              # BaseExecutor ABC
│   ├── portfolio.py             # MultiExchangePortfolio
│   ├── factory.py               # Registry-based executor + feed factory
│   ├── state.py                 # LiveState, _AssetLiveState
│   ├── alpaca/                  # Alpaca executor + bar feed
│   ├── binance/                 # Binance USD-M executor + WebSocket feed
│   └── hyperliquid/             # Hyperliquid executor + WebSocket feed
│
└── data/
    ├── feeds/                   # Live WebSocket scrapers (Hyperliquid, Binance, Alpaca)
    └── auxiliary/macro/         # Macro data helpers

app/
├── Strategy_Explorer.py         # Streamlit dashboard (EDA + backtester + hypothesis + stress)
└── components/
    ├── charts.py                # Plotly chart builders
    ├── forms.py                 # Strategy / sizer / stop / config sidebar widgets
    ├── lse_data.py              # LSE historical data fetching + caching
    ├── alpaca_data.py           # Alpaca historical data fetching + caching
    └── engine_runner.py         # Background thread wrapper for live Engine

trading/
├── backtest_demo.py             # Full demo: single-asset, multi-asset, multi-exchange, TTV workflow
└── alpaca_livetest_demo.py      # Alpaca paper-trading live demo
```

---

## User Guide

### Step 1 — Collect Data

Data is saved as Parquet chunks under `data/<stream>/<exchange>/<symbol>/`.

**Hyperliquid** (crypto perpetuals and spot):

```bash
# All streams: trades, L2 order book, funding rate
python -m src.data.feeds.hyperliquid --coin ETH --mode all

# Trades only
python -m src.data.feeds.hyperliquid --coin ETH --mode trades

# L2 order book, 5 levels of depth
python -m src.data.feeds.hyperliquid --coin ETH --mode l2 --depth 5

# L2 and trades (no funding — for spot symbols)
python -m src.data.feeds.hyperliquid --coin ETH --mode l2/trades

# Monitor a wallet for fills and order updates
python -m src.data.feeds.hyperliquid --wallet 0xYOUR_ADDRESS
```

**Binance** (USD-M futures):

```bash
python -m src.data.feeds.binance --coin ETHUSDT --market futures --streams trades l2 funding
```

**Data directory layout:**

```
data/
├── trades/
│   ├── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet
│   ├── HYPERLIQUID_SPOT/<symbol>/*.parquet
│   └── BINANCE_PERPETUALS/<symbol>/*.parquet
├── l2/
│   └── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet
└── funding/
    └── HYPERLIQUID_PERPETUALS/<symbol>/*.parquet
```

### Step 2 — Load & Parse Data

```python
from core.parser import trades_to_ohlcv, l2_to_orderbook, funding_to_snapshots, align_funding_to_ohlcv

# Resample raw tick trades into any bar size
eth_1h = trades_to_ohlcv("data/trades/HYPERLIQUID_PERPETUALS/ETH", timeframe="1h")
# → DataFrame(DatetimeIndex, columns=[open, high, low, close, volume])

# Load L2 snapshots aligned 1:1 with OHLCV bars
l2_snaps = l2_to_orderbook("data/l2/HYPERLIQUID_PERPETUALS/ETH", ohlcv_data=eth_1h)

# Load funding rates aligned to OHLCV bars
fund_snaps = align_funding_to_ohlcv(
    funding_to_snapshots("data/funding/HYPERLIQUID_PERPETUALS/ETH"),
    eth_1h,
)
```

Supported timeframes: `1s 2s 5s 10s 15s 30s 1m 2m 3m 5m 10m 15m 30m 1h 2h 4h 6h 8h 12h 1d`

### Step 3 — Write a Strategy

All strategies subclass `Strategy`. The base class unifies single- and multi-exchange behaviour: if one exchange is used it behaves like a plain single-exchange strategy; if multiple exchanges are passed in `setup()` it receives all of them in context.

For the common single-asset case, subclass `SingleAssetStrategy` instead — it handles universe wiring automatically and provides a simpler `bar()` interface.

#### Single-asset strategy

```python
from core.models import Allocation, Side
from strategy.built_in import SingleAssetStrategy
from strategy.indicators import ema, rsi

class EmaRsiStrategy(SingleAssetStrategy):
    def __init__(self, symbol: str, fast: int = 50, slow: int = 200, **kw):
        super().__init__(symbol=symbol, **kw)
        self.fast = fast
        self.slow = slow

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow}

    def setup_data(self, data, l2=None):
        data["ema_fast"] = ema(data["close"], self.fast)
        data["ema_slow"] = ema(data["close"], self.slow)
        data["rsi"]      = rsi(data["close"], 14)

    def bar(self, data, idx: int) -> Allocation:
        if idx < self.slow:
            return Allocation()

        if data["ema_fast"].iat[idx] > data["ema_slow"].iat[idx] and data["rsi"].iat[idx] < 80:
            return Allocation(side=Side.LONG, weight=1.0, reason="EMA cross up")

        return Allocation()
```

#### Multi-asset / multi-exchange strategy

For more control — or when operating across multiple exchanges simultaneously — subclass `Strategy` directly and implement `generate()`, which returns a `PortfolioTarget`.

```python
from strategy.base import Strategy, StrategyContext, PortfolioTarget
from core.models import Allocation, Side

class MomentumStrategy(Strategy):
    def __init__(self, symbols: list[str], lookback: int = 20):
        self.symbols = symbols
        self.lookback = lookback

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget()
        for sym in self.symbols:
            close = ctx.ohlcv(sym)["close"]
            if len(close) < self.lookback:
                continue
            ret = close.iloc[-1] / close.iloc[-self.lookback] - 1
            if ret > 0:
                target[sym] = Allocation(side=Side.LONG, weight=1 / len(self.symbols),
                                         reason=f"ret={ret:.2%}")
        return target
```

#### Multi-exchange strategy

The same `Strategy` base class is used. When running on multiple exchanges the context carries `ctx.universes`, `ctx.equity_by_exchange`, and `ctx.all_positions`. Allocations are keyed by `(exchange, symbol)` tuple in `PortfolioTarget.exchange_allocations`.

```python
from strategy.base import Strategy, StrategyContext, PortfolioTarget
from core.models import Allocation, Side
from strategy.indicators import ema, bollinger

class CrossExchangeMomentum(Strategy):
    def __init__(self, symbol: str, trend_exchange: str, reversal_exchange: str):
        self.symbol = symbol
        self.trend_ex = trend_exchange
        self.rev_ex = reversal_exchange

    def generate(self, ctx: StrategyContext) -> PortfolioTarget:
        target = PortfolioTarget()

        # Trend-following leg on trend_exchange
        df_trend = ctx.ohlcv(self.symbol, exchange=self.trend_ex)
        if len(df_trend) >= 200:
            trend_alloc = (
                Allocation(side=Side.LONG, weight=1.0, reason="above EMA200")
                if df_trend["close"].iloc[-1] > ema(df_trend["close"], 200).iloc[-1]
                else Allocation()
            )
            target[(self.trend_ex, self.symbol)] = trend_alloc

        # Mean-reversion leg on reversal_exchange
        df_rev = ctx.ohlcv(self.symbol, exchange=self.rev_ex)
        if len(df_rev) >= 20:
            mid, upper, lower = bollinger(df_rev["close"])
            price = df_rev["close"].iloc[-1]
            if price < lower.iloc[-1]:
                target[(self.rev_ex, self.symbol)] = Allocation(side=Side.LONG, weight=1.0, reason="BB oversold")
            elif price > upper.iloc[-1]:
                target[(self.rev_ex, self.symbol)] = Allocation(side=Side.SHORT, weight=1.0, reason="BB overbought")

        return target
```

**`Allocation` fields:**

| Field | Type | Description |
|---|---|---|
| `side` | `Side` | `LONG`, `SHORT`, or `FLAT` (default) |
| `weight` | `float` | Position size fraction 0–1 (multiplied by sizer) |
| `confidence` | `float` | Optional signal confidence 0–1 |
| `reason` | `str` | Log string for debugging and signal log |
| `stop_loss` | `float \| None` | Absolute stop price |
| `take_profit` | `float \| None` | Absolute take-profit price |

**`StrategyContext` fields:**

| Field | Description |
|---|---|
| `universe` | Active `Universe` (single-exchange; auto-synced from `universes`) |
| `universes` | `dict[str, Universe]` — all exchanges |
| `equity` | Total equity across all exchanges |
| `equity_by_exchange` | `dict[str, float]` — equity per exchange |
| `positions` | `dict[str, Position]` — primary exchange positions |
| `all_positions` | `dict[str, dict[str, Position]]` — positions per exchange |
| `bar_idx` | Current bar index |
| `timestamp` | Current bar timestamp |
| `trade_history` | List of closed `Trade` objects |

Key methods: `ctx.price(sym)`, `ctx.ohlcv(sym)`, `ctx.l2(sym)`, `ctx.funding(sym)`, `ctx.is_positioned(sym)` — all accept an optional `exchange=` keyword for multi-exchange contexts.

**`PortfolioTarget` interface:**

```python
# Single-exchange allocation by symbol
target["AAPL"] = Allocation(side=Side.LONG, weight=0.5)

# Multi-exchange allocation by (exchange, symbol) tuple
target[("nyse", "AAPL")] = Allocation(side=Side.LONG, weight=0.5)
target[("bats", "AAPL")] = Allocation(side=Side.SHORT, weight=0.3)

# Query
alloc = target["AAPL"]             # single-exchange
alloc = target[("nyse", "AAPL")]   # multi-exchange
target.is_multi_exchange            # True when exchange_allocations is populated
target.for_exchange("nyse")         # dict[str, Allocation] for one exchange
target.exchanges                    # list of exchange names present
```

**Available indicators** (`from strategy.indicators import ...`):

| Function | Signature |
|---|---|
| `ema` | `(series, span)` |
| `sma` | `(series, window)` |
| `rsi` | `(series, period=14)` |
| `atr` | `(high, low, close, period=14)` |
| `bollinger` | `(series, window=20, num_std=2.0) → (mid, upper, lower)` |
| `vwap_rolling` | `(price, volume, window)` |
| `order_flow_imbalance` | `(bid_vol, ask_vol, window=20)` |
| `book_imbalance` | `(l2_snapshot)` |

**Built-in strategies** (`from strategy.built_in import ...`):

| Class | Description |
|---|---|
| `SingleAssetStrategy` | Base for single-symbol strategies — implement `bar()` |
| `CompositeStrategy` | Combines multiple strategies with weights and a vote threshold |
| `PerAssetStrategy` | Runs one `SingleAssetStrategy` instance per symbol in the universe |
| `MeanReversionBasketStrategy` | Z-score mean reversion across a basket of symbols |

**Portfolio overlays** (`from strategy.overlay import ...`):

Overlays run after `generate()` and before execution to enforce cross-exchange constraints:

| Class | What it does |
|---|---|
| `NetExposureOverlay` | Caps net directional exposure across all exchanges per symbol |
| `DeltaNeutralOverlay` | Auto-hedges residual exposure on a specified hedge exchange |

```python
from strategy.overlay import NetExposureOverlay, DeltaNeutralOverlay

# Cap net weight to 50% of equity across both exchanges
overlay = NetExposureOverlay(max_net_weight=0.5)

# Auto-hedge residual on "bats" when net > 2%
overlay = DeltaNeutralOverlay(hedge_exchange="bats", max_residual_weight=0.02)
```

### Step 4 — Backtest

#### Single-exchange backtest

```python
from core.models import BacktestConfig
from core.universe import Universe
from testing.backtester.engine import Backtester
from testing.backtester.costs import CompositeCostModel, default_cost_stack
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

universe = Universe(symbols=["ETH"])
universe.add_asset("ETH", eth_1h, l2=l2_snaps, funding=fund_snaps)

config = BacktestConfig(initial_capital=100_000.0, leverage=1.0, max_position_pct=1.0)
cost   = CompositeCostModel(default_cost_stack())

bt = Backtester(
    strategy=EmaRsiStrategy(symbol="ETH", fast=50, slow=200),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
    cost_model=cost,
)
result = bt.run(universe=universe, timeframe="1h")

print(result.summary())
result.save("ema_rsi_eth_1h")   # → backtest_runs/ema_rsi_eth_1h/
```

#### Multi-exchange backtest

Pass `universes` (a dict of exchange name → `Universe`) instead of `universe`. Use `exchange_costs` for per-exchange fee models and `capital_by_exchange` to split the starting capital:

```python
from testing.backtester.costs import ExchangeFeeCost, FixedSlippageCost

u_nyse = Universe(symbols=["AAPL"]); u_nyse.add_asset("AAPL", df_nyse)
u_bats = Universe(symbols=["AAPL"]); u_bats.add_asset("AAPL", df_bats)

nyse_cost = CompositeCostModel([ExchangeFeeCost(maker_bps=2, taker_bps=3), FixedSlippageCost(1)])
bats_cost = CompositeCostModel([ExchangeFeeCost(maker_bps=10, taker_bps=15), FixedSlippageCost(5)])

bt = Backtester(
    strategy=CrossExchangeMomentum("AAPL", trend_exchange="nyse", reversal_exchange="bats"),
    config=BacktestConfig(initial_capital=100_000.0),
    exchange_costs={"nyse": nyse_cost, "bats": bats_cost},
    capital_by_exchange={"nyse": 50_000.0, "bats": 50_000.0},
)
result = bt.run(universes={"nyse": u_nyse, "bats": u_bats}, timeframe="1d")

print(result.summary())
# result.equity_curves_by_exchange → dict[str, pd.Series] — one curve per exchange
result.save("cross_exchange_demo")
```

**`BacktestResult` interface:**

| Member | Description |
|---|---|
| `.summary()` | `dict` — Sharpe, Sortino, Calmar, max DD, win rate, total fees, annualised return/vol; adds `"exchanges"` key for multi-exchange runs |
| `.equity_curve` | `pd.Series` — total equity across all exchanges |
| `.equity_curves_by_exchange` | `dict[str, pd.Series]` — per-exchange equity (multi-exchange only) |
| `.trades_df()` | `DataFrame` of all closed trades; includes `meta["exchange"]` column |
| `.signal_log` | Per-bar signal values |
| `.save(run_name)` | Saves `log.json`, `trades.csv`, `equity_curve.png`, `signal_log.csv`, `equity_curves_by_exchange.csv` to `backtest_runs/<run_name>/` |
| `.meta["vectorized"]` | `True` if the fast NumPy path ran |

**Vectorised fast path** — activates automatically (10–50x faster) when using `NopStopLoss()` and `FixedNotionalSizer(notional=N)`. Only applies to single-exchange runs; multi-exchange always uses the bar loop.

### Step 5 — Position Sizing

```python
from strategy.sizing import FixedNotionalSizer, FixedFractionalSizer, VolatilityTargetSizer, KellySizer

# Fixed dollar notional per trade
sizer = FixedNotionalSizer(notional=10_000)

# Fixed fraction of equity per trade
sizer = FixedNotionalSizer(equity_pct=0.10)

# Risk a fixed fraction of equity per trade (uses stop distance when available)
sizer = FixedFractionalSizer(risk_frac=0.02)

# Target a fixed annualised volatility (scales size to ~15% annual vol)
sizer = VolatilityTargetSizer(target_vol=0.15, lookback=20)

# Kelly criterion (uses trade history win rate and payoff)
sizer = KellySizer(kelly_frac=0.5, min_trades=20)
```

Full list: `FixedNotionalSizer`, `FixedFractionalSizer`, `VolatilityTargetSizer`, `KellySizer`, `AntiMartingaleSizer`, `DrawdownScalingSizer`, `L2LiquiditySizer`, `CompositeSizer`.

### Step 6 — Stop-Loss Modules

```python
from strategy.stops import NopStopLoss, FixedPercentStop, ATRStop, TrailingStop, RiskRewardStop

stop = NopStopLoss()                              # no stop; enables vectorised fast path
stop = FixedPercentStop(sl_pct=2.0, tp_pct=4.0)  # fixed % SL + optional TP
stop = ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0) # ATR-based SL and TP
stop = TrailingStop(trail_pct=1.5)                # trailing stop, locks in profit
stop = RiskRewardStop(sl_pct=1.5, rr_ratio=2.0)  # SL + auto-computed TP from R:R
```

Full list: `NopStopLoss`, `FixedPercentStop`, `ATRStop`, `TrailingStop`, `TrailingATRStop`, `BreakevenStop`, `TimeStop`, `RiskRewardStop`, `CompositeStopLoss`, `EmbeddedStop`.

### Step 7 — Hypothesis Testing & Validation

**Recommended workflow: Train → Test → Validate**

```python
from testing.hypothesis import (
    TrainTestValidateSplit,
    HypothesisTests,
    PermutationTest,
    BootstrapCI,
    WalkForwardAnalysis,
    DeflatedSharpeRatio,
    report as hypothesis_report,
)

# Split data 60% train / 20% test / 20% validate (10-bar embargo between splits)
ttv = TrainTestValidateSplit.by_fractions(universe, train_frac=0.60, test_frac=0.20, embargo_bars=10)
```

**Phase 1 — Train:** Develop the strategy on training data. Check walk-forward consistency.

```python
train_result = Backtester(strategy=my_strategy, ...).run(universe=ttv.train, timeframe="1h")

wfa = WalkForwardAnalysis(
    strategy_cls=EmaRsiStrategy,
    strategy_params={"fast": 50, "slow": 200},
    fixed_params={"symbol": "ETH"},
    config=config, cost_model=cost, sizer=sizer, stop_loss=stop,
)
wf = wfa.run(universe=ttv.train, timeframe="1h", n_splits=5, split_method="expanding")
print(f"Consistency: {wf.consistency_score:.0%}  IS/OOS efficiency: {wf.efficiency_ratio:.2f}")
```

**Phase 2 — Test:** Optimise parameters. Track the number of trials for DSR correction later.

```python
from testing.backtester.stress import ParamSweep

sweep = ParamSweep(
    strategy_cls=EmaRsiStrategy,
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200]},
    config=config, cost_model=cost, sizer=sizer, stop_loss=stop,
).run(universe=ttv.test, timeframe="1h")

best = sweep.best("sharpe_ratio")
n_trials = 3 * 3
```

**Phase 3 — Validate:** Run the tuned strategy once on the held-out set. This is the honest number.

```python
val_result = Backtester(strategy=best_strategy, ...).run(universe=ttv.validate, timeframe="1h")

# Full statistical battery
tests = HypothesisTests.run_all(val_result)
print(hypothesis_report(tests))

# Permutation test
pt = PermutationTest(metric="sharpe_ratio", n_permutations=2_000).run(val_result)
print(f"p={pt.p_value:.4f}  {'Significant' if pt.reject_null else 'Not significant'}")

# Bootstrap CIs
cis = BootstrapCI(n_bootstrap=2_000, ci=0.95).run(val_result)

# Deflated Sharpe — corrects for the number of param combos tried
dsr = DeflatedSharpeRatio().compute(val_result, n_trials=n_trials)
print(f"DSR: {dsr.deflated_sharpe:.3f}  {'Genuine edge' if dsr.reject_null else 'Likely overfit'}")
```

**Hypothesis tools reference:**

| Class | What it checks |
|---|---|
| `HypothesisTests.run_all(result)` | Sharpe > 0, mean return > 0, win rate > 50%, normality, autocorrelation, stationarity |
| `HypothesisTests.compare(r1, r2)` | Is strategy 1 statistically better than strategy 2? |
| `PermutationTest` | Non-parametric: is the metric better than random permutations of the trade sequence? |
| `BootstrapCI` | Bootstrap confidence intervals for any metric |
| `WalkForwardAnalysis` | Expanding or rolling sub-period consistency |
| `DeflatedSharpeRatio` | Sharpe corrected for multiple testing (Minkowski's formula) |
| `MultipleComparisonCorrection` | Bonferroni / Holm correction for family-wise error rate |
| `ProbabilityOfBacktestOverfitting` | Bailey et al. CPCV-based overfit probability |
| `TrainTestValidateSplit` | Three-way holdout with configurable fractions and embargo |

### Step 8 — Stress Testing

```python
from testing.backtester.stress import MonteCarloStress, ParamSweep, RegimeStressTest

# Monte Carlo bootstrap — distribution of outcomes from trade resampling
mc = MonteCarloStress(n_simulations=1_000, method="bootstrap")
mc_res = mc.run(backtest_result)
m = mc_res.meta
print(f"Median return: {m['median_return']:.2f}%  5th: {m['5th_pctl_return']:.2f}%")

# Parameter sweep heatmap
sweep = ParamSweep(
    strategy_cls=EmaRsiStrategy,
    param_grid={"fast": [20, 50, 100], "slow": [100, 150, 200, 250]},
    config=config, cost_model=cost, sizer=sizer, stop_loss=stop,
)
res = sweep.run(universe=universe, timeframe="1h")
res.plot_heatmap("fast", "slow", z="sharpe_ratio")

# Regime stress test — performance across vol / trend / volume regimes
rst = RegimeStressTest(regime_fn=RegimeStressTest.trend_regime, config=config, cost_model=cost)
regime_summary = rst.run(strategy=my_strategy, universe=universe)
print(regime_summary.summary)
```

### Step 9 — Live Trading

The `Engine` handles three modes from a single class:

| Mode | Constructor argument |
|---|---|
| Single exchange | `strategy=my_strategy` |
| Independent strategy per exchange | `per_exchange_strategies={"binance": s1, "hyperliquid": s2}` |
| Cross-exchange strategy (funding arb, stat arb, hedging) | `cross_strategy=my_strategy` |

**Alpaca (US equities / ETFs, paper and live):**

```python
from core.models import LiveConfig, ExchangeCredentials
from execution.engine import Engine
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss

cred = ExchangeCredentials(
    exchange="alpaca",
    api_key="ALP_PAPER_KEY",
    api_secret="ALP_PAPER_SECRET",
    testnet=True,
)
config = LiveConfig(
    exchanges=[cred],
    symbol="SPY",
    bar_interval_s=60,
    warmup_bars=300,
    max_position_pct=0.10,
    leverage=1.0,
    max_daily_trades=20,
    max_daily_loss_pct=3.0,
)
engine = Engine(
    strategy=EmaRsiStrategy(symbol="SPY", fast=50, slow=200),
    config=config,
    sizer=FixedNotionalSizer(notional=10_000),
    stop_loss=NopStopLoss(),
)
engine.start()   # press 'q' + Enter to flatten & stop
```

**Hyperliquid (crypto perpetuals):**

```python
from execution.engine import Engine

config = LiveConfig(
    exchanges=[ExchangeCredentials(
        exchange="hyperliquid",
        account_address="0x...",
        secret_key="0x...",
        testnet=True,
    )],
    symbol="ETH",
    bar_interval_s=300,
    warmup_bars=200,
    max_daily_loss_pct=3.0,
)
engine = Engine(
    strategy=EmaRsiStrategy(symbol="ETH"),
    config=config,
    sizer=VolatilityTargetSizer(target_vol=0.15),
    stop_loss=ATRStop(atr_mult_sl=2.0, atr_mult_tp=3.0),
)
engine.start()
```

**Cross-exchange (funding arbitrage, delta-neutral hedging):**

```python
from execution.engine import Engine

engine = Engine(
    cross_strategy=CrossExchangeMomentum("ETH", "hyperliquid", "binance"),
    config=LiveConfig(exchanges=[hl_cred, binance_cred], symbol="ETH", ...),
)
engine.start()
```

When `cross_strategy=` is used the engine calls `strategy.setup(universes)` with all exchange universes before each bar, builds a unified `StrategyContext` with per-exchange equity and positions, and routes each `(exchange, symbol)` allocation in the returned `PortfolioTarget` to the correct executor. Overlays can be applied at this layer.

### Step 10 — Strategy Registration

Strategies are registered by name for use in the dashboard and demo scripts:

```python
from strategy.base import register_strategy

@register_strategy
class EmaRsiStrategy(SingleAssetStrategy):
    ...

# Then, anywhere:
from strategy.base import get_strategy, list_strategies
cls = get_strategy("EmaRsiStrategy")
print(list_strategies())   # ["EmaRsiStrategy", ...]
```

---

## Streamlit Dashboard

```bash
streamlit run app/Strategy_Explorer.py
```

**Tabs:**

| Tab | Contents |
|---|---|
| **Explorer** | OHLCV candlestick chart with EMA, SMA, Bollinger, RSI, ATR, MACD overlays; volatility regime chart; return distribution + descriptive stats; trade markers from latest backtest; raw data download |
| **Results** | Equity curve + drawdown chart; trade markers on price chart; signal log chart; colour-coded trade log table; trades CSV download |
| **Hypothesis Tests** | Permutation test, bootstrap CIs, Deflated Sharpe Ratio — run against any completed backtest result |
| **Param Sweep** | 1D bar chart or 2D heatmap across any two numeric parameters of the selected strategy |
| **Regime Test** | Backtest split by volatility, trend, or volume regime; per-regime equity curves and stats |
| **Monte Carlo** | Bootstrap trade resampling — distribution of return %, max DD %; full statistics download |

Sidebar controls: symbol + date range, indicator toggles, strategy selector with auto-generated parameter fields, sizer and stop-loss configuration, backtest config (capital, leverage, max position size).

Data is loaded from the LSE API — set `LSE_DATA=your_key` in `.env` or enter it in the sidebar.

---

## Adding an Exchange

1. Create `src/execution/<exchange>/` with an executor (`BaseExecutor`) and optionally a custom feed.
2. Register it in the factory:

```python
from execution.factory import register_exchange
from .my_executor import MyExecutor
from .my_feed import MyFeed

register_exchange(
    "myexchange",
    executor_factory=lambda cred: MyExecutor(cred.api_key, cred.api_secret, cred.testnet),
    feed_factory=lambda symbol, testnet, **_: MyFeed(symbol, testnet),
)
```

3. Pass `ExchangeCredentials(exchange="myexchange", ...)` to `Engine` — the factory resolves the rest.

---
