"""Run-level ceilings on what an AutoKernel run may consume (AK-CONT-BUDGET).

Per-call budgets (OAB-23) and `--max-iterations` bound one call and one count. Nothing
bounded a whole run: this adds wall-clock, CPU/GPU arm-second and decoded-token ceilings,
enforced by the run process from its own ledger, never by an actor. `run.py` folds
`tripped` into `should_stop`, so a breach parks the run at the next boundary through the
existing drain and ends with the resumable terminal `budget_parked`.

Debits come from two places the actor cannot edit: the `actor_call_metrics.v1` rows the
loop itself writes after each call, and claimed arm intervals, including idle time. A seat that meters
nothing (codex and claude record no tokens, a local seat can report zero) is an explicit
UNKNOWN: its wall time is debited as always, and because unknown spend cannot be shown
to be under a token cap, any such call fails a configured token cap closed.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from ..loop.arm_accounting import ArmSecondsBudget

SCHEMA = "epyc.autokernel.run_budget.v1"
METRICS_SCHEMA = "epyc.autokernel.actor_call_metrics.v1"
#: Row-scan rate limit: `should_stop` is polled often, the log only grows per actor call.
SCAN_INTERVAL_S = 10.0


@dataclass(frozen=True)
class Ceilings:
    wall_s: float | None = None
    gpu_s: float | None = None
    decoded_tokens: int | None = None
    cpu_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("wall_s", "gpu_s", "decoded_tokens", "cpu_s"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"run budget {name} must be finite and positive")

    @property
    def any(self) -> bool:
        return any(v is not None for v in (self.wall_s, self.gpu_s, self.decoded_tokens, self.cpu_s))


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
                                                            "gpu_s": 0.0, "cpu_s": 0.0, "decoded_tokens": 0,
                                                            "unmetered_calls": 0, "calls": 0})
        self._calls: list[dict] = []
        self.gpu_s, self.tokens, self.unmetered = 0.0, 0, 0
        limits = {name: limit for name, limit in (("cpu", ceilings.cpu_s), ("gpu", ceilings.gpu_s))
                  if limit is not None}
        self._arm_budget = ArmSecondsBudget(limits) if limits else None
        self._arm_spent = {"cpu": 0.0, "gpu": 0.0}
        self._active_arms: dict[int, tuple[str, float | None, str]] = {}
        self._next_arm = 0
        self.breach: str | None = None
        self._next_scan = 0.0

    def debit_stage(self, label: str, wall_s: float, *, gpu: bool, lane: str | None = None) -> None:
        with self._lock:
            row = self._lanes[lane or threading.current_thread().name]
            row["stage_wall_s"] += wall_s
            self._calls.append({"kind": "stage", "label": label, "wall_s": round(wall_s, 3), "gpu": gpu})

    def _settle_arms(self, now: float) -> None:
        """Debit each active claimed interval once; callers hold the ledger lock."""
        for token, (surface, started, lane) in list(self._active_arms.items()):
            if started is None:
                continue
            elapsed = max(0.0, now - started)
            self._arm_spent[surface] += elapsed
            self._lanes[lane][surface + "_s"] += elapsed
            if self._arm_budget is not None and getattr(self.ceilings, surface + "_s") is not None:
                self._arm_budget.charge(surface, {"arm_seconds_claimed": elapsed})
            self._active_arms[token] = (surface, now, lane)
        self.gpu_s = self._arm_spent["gpu"]

    def begin_arm(self, surface: str) -> int:
        """Start accounting only after a real CPU or GPU arm claim succeeds."""
        if surface not in self._arm_spent:
            raise ValueError(f"unknown arm surface {surface!r}")
        with self._lock:
            self._next_arm += 1
            self._active_arms[self._next_arm] = (surface, self._monotonic(), threading.current_thread().name)
            return self._next_arm

    def pause_arm(self, token: int) -> None:
        with self._lock:
            self._settle_arms(self._monotonic())
            surface, _, lane = self._active_arms[token]
            self._active_arms[token] = (surface, None, lane)

    def resume_arm(self, token: int) -> None:
        with self._lock:
            surface, started, lane = self._active_arms[token]
            if started is None:
                self._active_arms[token] = (surface, self._monotonic(), lane)

    def end_arm(self, token: int) -> None:
        with self._lock:
            self._settle_arms(self._monotonic())
            del self._active_arms[token]

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
            self._settle_arms(now)
            if force or now >= self._next_scan:
                self._next_scan = now + SCAN_INTERVAL_S
                self._scan()
            if self.breach is not None:
                return self.breach
            limits = self.ceilings
            elapsed = self._clock() - self.started_at
            if limits.wall_s is not None and elapsed >= limits.wall_s:
                self.breach = f"wall_s {elapsed:.0f} >= {limits.wall_s:g}"
            elif limits.gpu_s is not None and self._arm_budget.remaining("gpu") <= 0:
                self.breach = f"gpu_s {self.gpu_s:.0f} >= {limits.gpu_s:g}"
            elif limits.cpu_s is not None and self._arm_budget.remaining("cpu") <= 0:
                self.breach = f"cpu_s {self._arm_spent['cpu']:.0f} >= {limits.cpu_s:g}"
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
                                                   "decoded_tokens": self.ceilings.decoded_tokens,
                                                   "cpu_s": self.ceilings.cpu_s},
                    "started_at": self.started_at, "elapsed_s": round(self._clock() - self.started_at, 3),
                    "gpu_s": round(self.gpu_s, 3), "arm_seconds_claimed": {k: round(v, 3) for k, v in self._arm_spent.items()},
                    "arm_seconds_budget": self._arm_budget.snapshot() if self._arm_budget else None,
                    "decoded_tokens": self.tokens,
                    "unmetered_calls": self.unmetered, "breach": self.breach,
                    "lanes": {k: dict(v) for k, v in self._lanes.items()}, "calls": list(self._calls)}


@contextmanager
def budgeted_claim(claim, budget, surface: str, token_ref=None):
    """Count a real claim from acquisition through release, including idle time."""
    token = None
    try:
        with claim as receipt:
            token = budget.begin_arm(surface)
            if token_ref is not None:
                token_ref[0] = token
            yield receipt
    finally:
        if token is not None:
            budget.end_arm(token)
            if token_ref is not None:
                token_ref[0] = None


def enter_claim(stack, factory, budget, surface, token_ref=None):
    """Acquire a fresh claim on every retry and register its budgeted lifetime."""
    claim = factory()
    if surface is not None:
        claim = budgeted_claim(claim, budget, surface, token_ref)
    return stack.enter_context(claim)


def lease_change(budget, token_ref, held):
    """Pause an arm while its lease is yielded and resume when reacquired."""
    token = token_ref[0]
    if token is not None:
        (budget.resume_arm if held else budget.pause_arm)(token)


__all__ = ["budgeted_claim", "Ceilings", "METRICS_SCHEMA", "RunBudget", "SCHEMA", "decoded_tokens"]
