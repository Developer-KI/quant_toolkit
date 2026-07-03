"""
trading/alpaca_livetest_demo.py — Alpaca paper-trading live demo.

Runs the EMA/RSI strategy on Alpaca paper trading using the LiveEngine.
The feed fires 1-minute bars from Alpaca's IEX stream; the strategy
accumulates them into whichever bar_interval_s is configured.

Usage:
    python trading/alpaca_livetest_demo.py
    python trading/alpaca_livetest_demo.py --symbol AAPL --bar-interval 300

Kill switch: press 'q' + Enter to flatten all positions and stop.

Credentials: set ALP_PAPER_KEY and ALP_PAPER_SECRET in a .env file (or env).
"""

from __future__ import annotations

from dotenv import load_dotenv, dotenv_values

from core.models import LiveConfig, ExchangeCredentials
from execution import Engine as LiveEngine
from strategy.built_in import EmaRsiStrategy
from strategy.sizing import FixedNotionalSizer
from strategy.stops import NopStopLoss


def demo(
    symbol: str = "SPY",
    bar_interval_s: int = 60,
    fast: int = 50,
    slow: int = 200,
    warmup_bars: int = 300,
    max_position_pct: float = 0.10,
    paper: bool = True,
):
    load_dotenv()
    _env = dotenv_values()
    api_key = _env.get("ALP_PAPER_KEY", "")
    api_secret = _env.get("ALP_PAPER_SECRET", "")
    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca credentials missing — set ALP_PAPER_KEY and ALP_PAPER_SECRET in .env"
        )

    print(f"\n{'='*60}")
    print(f"  Alpaca {'PAPER' if paper else 'LIVE'} trading demo")
    print(f"  Symbol        : {symbol}")
    print(f"  Bar interval  : {bar_interval_s}s")
    print(f"  EMA fast/slow : {fast}/{slow}")
    print(f"  Warmup bars   : {warmup_bars}")
    print(f"  Max position  : {max_position_pct*100:.0f}% of equity")
    print(f"{'='*60}\n")

    cred = ExchangeCredentials(
        exchange="alpaca",
        api_key=api_key,
        api_secret=api_secret,
        testnet=paper,
    )

    config = LiveConfig(
        exchange="alpaca",
        use_testnet=paper,
        exchanges=[cred],
        symbol=symbol,
        bar_interval_s=bar_interval_s,
        warmup_bars=warmup_bars,
        max_bars_in_memory=2000,
        max_position_pct=max_position_pct,
        leverage=1.0,
        order_type="market",
        max_daily_trades=20,
        max_daily_loss_pct=3.0,
        log_level="INFO",
        trade_log_csv="trades.csv",
    )

    strategy = EmaRsiStrategy(symbol=symbol, fast=fast, slow=slow)
    sizer = FixedNotionalSizer(notional=10_000)
    stop_loss = NopStopLoss()

    engine = LiveEngine(strategy=strategy, config=config, sizer=sizer, stop_loss=stop_loss)

    print("Starting live engine — press 'q' + Enter to flatten & stop.\n")
    engine.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alpaca paper-trading live demo")
    parser.add_argument("--symbol",       default="SPY",  help="Ticker symbol (default: SPY)")
    parser.add_argument("--bar-interval", default=60,     type=int, help="Bar interval in seconds (default: 60)")
    parser.add_argument("--fast",         default=50,     type=int, help="Fast EMA period (default: 50)")
    parser.add_argument("--slow",         default=200,    type=int, help="Slow EMA period (default: 200)")
    parser.add_argument("--warmup",       default=300,    type=int, help="Warmup bars (default: 300)")
    parser.add_argument("--max-pos-pct",  default=0.10,   type=float, help="Max position as fraction of equity (default: 0.10)")
    parser.add_argument("--live",         action="store_true",        help="Use live account instead of paper")
    args = parser.parse_args()

    demo(
        symbol=args.symbol,
        bar_interval_s=args.bar_interval,
        fast=args.fast,
        slow=args.slow,
        warmup_bars=args.warmup,
        max_position_pct=args.max_pos_pct,
        paper=not args.live,
    )
