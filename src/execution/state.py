"""
execution/state.py — Runtime state containers and engine utilities.

Merges: live_state.py + live_limits.py + _ManualKillSwitch + _sizer_config_shim
"""

from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field

from core.models import BacktestConfig, LiveConfig, Position, Trade
from core.feeds import BaseFeed, BaseBarBuilder
from strategy.stops import StopLoss

logger = logging.getLogger(__name__)

KILL_KEY = "q"


@dataclass
class _AssetLiveState:
    """Mutable per-(exchange, symbol) live state."""
    symbol: str = ""
    exchange: str = ""
    position: Position = field(default_factory=Position)
    open_trade: Trade | None = None
    stop_loss: StopLoss | None = None
    feed: BaseFeed | None = None
    bar_builder: BaseBarBuilder | None = None


@dataclass
class LiveState:
    """Portfolio-level bookkeeping for a running engine."""
    equity: float = 0.0
    peak_equity: float = 0.0
    starting_equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    closed_trades: list[Trade] = field(default_factory=list)
    daily_trades: int = 0
    daily_pnl: float = 0.0
    last_bar_idx: int = 0
    strategy_setup_done: bool = False
    kill_switch: bool = False

    @property
    def position(self) -> Position:
        """Convenience: first position (single-asset engines)."""
        if self.positions:
            return next(iter(self.positions.values()))
        return Position()

    def check_daily_loss_limit(self, config: LiveConfig) -> bool:
        """Return True if the daily loss limit has been breached."""
        if self.starting_equity <= 0:
            return False
        daily_loss_pct = abs(self.daily_pnl) / self.starting_equity * 100
        if self.daily_pnl < 0 and daily_loss_pct >= config.max_daily_loss_pct:
            logger.critical(
                "KILL SWITCH — daily loss %.2f%% exceeds limit %.2f%%",
                daily_loss_pct,
                config.max_daily_loss_pct,
            )
            return True
        return False


class _ManualKillSwitch:
    """
    Background thread that listens for a keypress to trigger emergency shutdown.
    Press KILL_KEY (default: 'q') then Enter.
    Silently disables itself in non-interactive environments (Docker, systemd).
    """

    def __init__(self, callback, key: str = KILL_KEY):
        self._callback = callback
        self._key = key
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        if not sys.stdin.isatty():
            logger.info("Non-interactive terminal — manual kill switch disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True, name="kill-switch")
        self._thread.start()
        logger.info("Manual kill switch active — press '%s' + Enter to flatten & shutdown", self._key)

    def stop(self):
        self._running = False

    def _listen(self):
        try:
            while self._running:
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip().lower() == self._key:
                    logger.critical("MANUAL KILL — '%s' pressed, flattening all positions", self._key)
                    self._callback()
                    return
        except Exception:
            pass


def _sizer_config_shim(config: LiveConfig, equity: float) -> BacktestConfig:
    """Bridge LiveConfig fields into a BacktestConfig for sizer.compute()."""
    return BacktestConfig(
        initial_capital=equity,
        max_position_pct=config.max_position_pct,
        leverage=config.leverage,
    )
