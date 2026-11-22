"""Pipeline Budget record, split verbatim from pipeline.py (code-budget split)."""
from __future__ import annotations
from dataclasses import dataclass, field
import threading
from typing import Callable


@dataclass
class Budget:
    """A shared iteration count, or none at all.

    `remaining=None` means run until told to stop. That is the difference between a
    tool someone runs and a loop that works: the previous shape exited after N
    iterations, so every continuation needed a human to start it again.

    `should_stop` is checked on every draw, so a stop is honoured at the next
    iteration boundary rather than mid-measurement -- a lane holding the device
    finishes what it started, publishes, and only then declines to take more.
    """
    remaining: int | None
    should_stop: Callable[[], bool] | None = None
    drawn: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self) -> bool:
        with self._lock:
            if self.should_stop is not None and self.should_stop():
                return False
            if self.remaining is not None:
                if self.remaining <= 0:
                    return False
                self.remaining -= 1
            self.drawn += 1
            return True
