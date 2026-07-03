"""Thread wrapper for Engine — stores engine lifecycle in Streamlit session state."""
from __future__ import annotations

import threading
import traceback


class EngineRunner:
    """
    Runs an Engine in a background daemon thread.
    One instance is stored in st.session_state["runner"] for the app lifetime.
    """

    def __init__(self):
        self.engine = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._status: str = "stopped"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, engine) -> None:
        if self.is_alive:
            return
        self.engine = engine
        self.error = None
        self._status = "starting"
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="live-engine",
        )
        self._thread.start()

    def _run(self) -> None:
        self._status = "running"
        try:
            self.engine.start()
        except Exception:
            self.error = traceback.format_exc()
            self._status = "error"
        else:
            self._status = "stopped"

    def stop(self) -> None:
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
        self._status = "stopped"

    def emergency_flatten(self) -> None:
        """Flatten all positions immediately via the engine's manual kill switch."""
        if self.engine is not None:
            try:
                self.engine._manual_kill()
            except Exception:
                pass

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        if self._thread is not None and not self._thread.is_alive() and self._status == "running":
            self._status = "stopped"
        return self._status

    @property
    def state(self):
        """LiveState of the running engine, or None."""
        return self.engine.state if self.engine is not None else None

    @property
    def assets(self) -> dict:
        """Per-symbol _AssetLiveState dict from the running engine."""
        if self.engine is not None:
            return self.engine.assets
        return {}
