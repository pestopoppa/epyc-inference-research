"""Run-level ceilings on what an AutoKernel run may consume (AK-CONT-BUDGET).

Per-call budgets (OAB-23) and `--max-iterations` bound one call and one count. Nothing
bounded a whole run: this adds wall-clock, GPU-second and decoded-token ceilings,
enforced by the run process from its own ledger, never by an actor. `run.py` folds
`tripped` into `should_stop`, so a breach parks the run at the next boundary through the
existing drain and ends with the resumable terminal `budget_parked`.

Debits come from two places the actor cannot edit: the `actor_call_metrics.v1` rows the
loop itself writes after each call, and the run's measurement stages. A seat that meters
nothing (codex and claude record no tokens, a local seat can report zero) is an explicit
UNKNOWN: its wall time is debited as always, and because unknown spend cannot be shown
to be under a token cap, any such call fails a configured token cap closed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

SCHEMA = "epyc.autokernel.run_budget.v1"
METRICS_SCHEMA = "epyc.autokernel.actor_call_metrics.v1"
#: Row-scan rate limit: `should_stop` is polled often, the log only grows per actor call.
SCAN_INTERVAL_S = 10.0


@dataclass(frozen=True)
class Ceilings:
    wall_s: float | None = None
    gpu_s: float | None = None
    decoded_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("wall_s", "gpu_s", "decoded_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"run budget {name} must be positive")

    @property
    def any(self) -> bool:
        return any(v is not None for v in (self.wall_s, self.gpu_s, self.decoded_tokens))


def decoded_tokens(row: Mapping[str, Any]) -> int | None:
    """The call's metered decoded tokens, or None when the seat metered nothing."""
    harness = row.get("opencode") if isinstance(row.get("opencode"), Mapping) else row.get("orchestrator")
    totals = harness.get("totals") if isinstance(harness, Mapping) else None
    value = totals.get("decoded_tokens") if isinstance(totals, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value)


class RunBudget:
    """The run's ledger and its latching ceiling check."""

    def __init__(self, ceilings: Ceilings, worker_root: Path, *, started_at: float | None = None,
                 clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.ceilings, self.worker_root, self._clock, self._monotonic = ceilings, Path(worker_root), clock, monotonic
        self.started_at = self._clock() if started_at is None else started_at
        self._lock = threading.Lock()
        self._offsets: dict[Path, int] = {}
        self._lanes: dict[str, dict] = defaultdict(lambda: {"actor_wall_s": 0.0, "stage_wall_s": 0.0,
                                                            "gpu_s": 0.0, "decoded_tokens": 0,
                                                            "unmetered_calls": 0, "calls": 0})
        self._calls: list[dict] = []
        self.gpu_s, self.tokens, self.unmetered = 0.0, 0, 0
        self.breach: str | None = None
        self._next_scan = 0.0

    def debit_stage(self, label: str, wall_s: float, *, gpu: bool, lane: str | None = None) -> None:
        with self._lock:
            row = self._lanes[lane or threading.current_thread().name]
            row["stage_wall_s"] += wall_s
            if gpu:
                row["gpu_s"] += wall_s
                self.gpu_s += wall_s
            self._calls.append({"kind": "stage", "label": label, "wall_s": round(wall_s, 3), "gpu": gpu})

    def _new_rows(self):
        for path in (self.worker_root / "actor-replies" / "actor-calls.jsonl",):
            start = self._offsets.get(path, 0)
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    data = stream.read()
            except OSError:
                continue
            complete = data.rfind(b"\n") + 1
            self._offsets[path] = start + complete
            for line in data[:complete].splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, Mapping) and row.get("schema") == METRICS_SCHEMA:
                    yield row

    def _in_run(self, row: Mapping[str, Any]) -> bool:
        try:
            stamp = time.mktime(time.strptime(str(row.get("ts")), "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
        except (ValueError, OverflowError):
            return True  # unreadable time: debiting it is the fail-closed direction
        return stamp >= int(self.started_at)

    def _scan(self) -> None:
        for row in self._new_rows():
            if not self._in_run(row):
                continue
            wall = float(row.get("wall_s") or 0.0)
            tokens = decoded_tokens(row)
            lane = self._lanes[str(row.get("lane") or "unattributed")]
            lane["actor_wall_s"] += wall
            lane["calls"] += 1
            if tokens is None:
                self.unmetered += 1
                lane["unmetered_calls"] += 1
            else:
                self.tokens += tokens
                lane["decoded_tokens"] += tokens
            self._calls.append({"kind": "actor", "role": row.get("role"), "seat": row.get("backend_kind"),
                                "lane": row.get("lane"), "wall_s": wall, "decoded_tokens": tokens})

    def check(self, *, force: bool = False) -> str | None:
        """Scan new debits and latch the first breached ceiling. Never raises on bad rows."""
        with self._lock:
            now = self._monotonic()
            if force or now >= self._next_scan:
                self._next_scan = now + SCAN_INTERVAL_S
                self._scan()
            if self.breach is not None:
                return self.breach
            limits = self.ceilings
            elapsed = self._clock() - self.started_at
            if limits.wall_s is not None and elapsed >= limits.wall_s:
                self.breach = f"wall_s {elapsed:.0f} >= {limits.wall_s:g}"
            elif limits.gpu_s is not None and self.gpu_s >= limits.gpu_s:
                self.breach = f"gpu_s {self.gpu_s:.0f} >= {limits.gpu_s:g}"
            elif limits.decoded_tokens is not None and self.unmetered:
                self.breach = (f"decoded_tokens unknown: {self.unmetered} unmetered call(s) "
                               f"cannot be shown under {limits.decoded_tokens}")
            elif limits.decoded_tokens is not None and self.tokens >= limits.decoded_tokens:
                self.breach = f"decoded_tokens {self.tokens} >= {limits.decoded_tokens}"
            return self.breach

    @property
    def tripped(self) -> bool:
        return self.ceilings.any and self.check() is not None

    def snapshot(self) -> dict:
        """The whole ledger, for the campaign status body."""
        self.check(force=True)
        with self._lock:
            return {"schema": SCHEMA, "ceilings": {"wall_s": self.ceilings.wall_s,
                                                   "gpu_s": self.ceilings.gpu_s,
                                                   "decoded_tokens": self.ceilings.decoded_tokens},
                    "started_at": self.started_at, "elapsed_s": round(self._clock() - self.started_at, 3),
                    "gpu_s": round(self.gpu_s, 3), "decoded_tokens": self.tokens,
                    "unmetered_calls": self.unmetered, "breach": self.breach,
                    "lanes": {k: dict(v) for k, v in self._lanes.items()}, "calls": list(self._calls)}


__all__ = ["Ceilings", "METRICS_SCHEMA", "RunBudget", "SCHEMA", "decoded_tokens"]
