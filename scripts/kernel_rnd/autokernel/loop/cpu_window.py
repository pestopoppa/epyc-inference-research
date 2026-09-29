#!/usr/bin/env python3
"""CPU windows: yield the CPU-region claim during actor phases and say so (2026-09-26).

WHY. A batch took its CPU-region claim at start and held it to the end, but most of
an iteration is the planner, the critics and the author talking to a GPU server or a
hosted API -- the claimed CPUs sat idle while every other session's
`region-lock run --cpu-list 0-191 ...` blocked for the whole batch (operator proposal,
relayed by workspace-8d). With `--cpu-window-yield on` (the default) the loop gives
the regions back while every lane is in an actor phase and publishes that as a WINDOW
peers can fit a bounded measurement into.

THE CLAIM (see `claim.CpuClaimLease`). Release is the region-lock provider's own
exit; re-acquire is a fresh acquisition of the SAME regions, verified to land on the
same lock files and to observe as held by this process. The original `HeldCpuClaim`
context -- and therefore the held-resource evidence `serial_run` checks -- is the
same object for the whole batch: its open and close observations are real
observations of this process holding the regions. The gaps are published beside it
(`loop-cpu-window.json` in the batch dir, `segments`), never folded into it.

WHEN. Released on a lane's transition INTO an actor phase (planner propose, critic
pass 1, author, critic pass 2) while no lane holds the serialized tail. Re-acquired
structurally: on entry to every tail session (build, oracle, A/B, commit), on entry
to every CPU measurement window, and at teardown (the claim's close observation
needs it). A re-acquire waits while a peer holds the regions -- past
`wait_bound_s` it logs and KEEPS waiting; nothing is measured without the claim and a
peer's process is never touched. A stop while waiting in a tail refuses the
candidate (`loop.TailRefused`, the q3 quiet-window precedent) with the claim still
released.

THE WINDOW FILE (`WINDOW_SCHEMA`), written atomically to the well-known path and a
per-campaign copy in the store:
  state             open | closing | closed
  loop_holds_claim  the separate fact: whether this loop holds the regions now
  campaign, phase, phase_label, phase_started_at, opened_at
  est_close_at      from the phase budgets (planner budget, author budget, critic
                    timeout) and the recent median walls in actor-calls.jsonl, summed
                    over the actor phases still ahead (`est_close_basis`)
  cpus_reserved_by_loop   the ak-check cores: the author's sandbox compiles and
                    op-tests there during an open window (`contention`)
  closing_reason, wait (a re-acquire in progress), generation
  heartbeat_at / expires_at   refreshed every `heartbeat_s`; a window whose
                    `expires_at` has passed is STALE (the loop died) whatever its
                    state says. `read_window` applies that rule for peers.
A normal exit (including an exception) writes `closed` with `loop_holds_claim` false
and an expired heartbeat; a hard kill leaves a file that goes stale within `ttl_s`,
and the next batch overwrites it at startup.

THE EVENT. Every state change and every phase change inside an open window is also
appended to `<window>.events.jsonl` beside the window file (tail -F it) and, when a
roster id is configured, to the session bus as a `status` broadcast from the loop's
own outbox (`session_bus.py append --agent <id> --target outbox`, `to: "*"`, FYI:
no action_required). The bus append runs on a background thread with a timeout: the
loop never blocks on the bus (BUS_PROTOCOL rule 2), and a refused append is logged
once and disables the bus for this process.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Mapping

from . import status

WINDOW_SCHEMA = "epyc.autokernel.cpu_window.v1"
EVENT_SCHEMA = "epyc.autokernel.cpu_window_event.v1"
LEDGER_SCHEMA = "epyc.autokernel.cpu_window_ledger.v1"
BUS_EVENT = "autokernel-cpu-window"

DEFAULT_PATH = Path("/mnt/raid0/llm/autokernel/cpu-window.json")
CAMPAIGN_NAME = "cpu-window.json"
LEDGER_NAME = "loop-cpu-window.json"
DEFAULT_WAIT_BOUND_S = 1800.0
DEFAULT_BUS_AGENT = "autokernel"
HEARTBEAT_S = 30.0
TTL_S = 120.0
POLL_S = 5.0
MEDIAN_SAMPLES = 9
CALL_LOG_TAIL_BYTES = 1 << 20

STATES = ("open", "closing", "closed")
ACTOR_PHASES = ("planner", "critic1", "author", "critic2")
#: The actor phases still ahead once a phase starts: the window stays open across
#: all of them (a rejection loops back, so this is an estimate, and says so).
_AHEAD = {"planner": ("planner", "critic1", "author", "critic2"),
          "critic1": ("critic1", "author", "critic2"),
          "author": ("author", "critic2"),
          "critic2": ("critic2",)}
_ROLE = {"planner": "planner", "critic1": "critic", "author": "author", "critic2": "critic"}
#: Labels that mean the next thing this lane does is a CPU step.
_CPU_NEXT = ("building and gating", "checking runtime treatment", "measuring A/B",
             "critic pass 2: carried verdict", "rescheduling original invalid CPU")


def classify(label: str) -> str | None:
    """The actor phase a `loop.iterate` step label starts, or None (not an actor call)."""
    if label.startswith("proposing a hypothesis"):
        return "planner"
    if label.startswith("critic pass 1:"):
        return "critic1"
    if label.startswith("authoring the patch"):
        return "author"
    if label.startswith("critic pass 2: reviewing the diff"):
        return "critic2"
    return None


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _epoch(text: Any) -> float | None:
    if not isinstance(text, str) or not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def read_window(path: Path = DEFAULT_PATH, *, now: float | None = None) -> dict | None:
    """A peer's read: the window plus `effective_state` (open|closing|closed|stale).

    Stale means the heartbeat expired: the owning loop is gone, so neither its
    state nor its ETA says anything. The flock (`region-lock status`) stays the
    fact either way; a window is advice about WHEN, never permission.
    """
    try:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("schema") != WINDOW_SCHEMA:
        return None
    expires = _epoch(body.get("expires_at"))
    now = time.time() if now is None else now
    body["effective_state"] = ("stale" if expires is None or expires <= now
                               else body.get("state"))
    return body


def ak_check_cpus(environ: Mapping[str, str] | None = None) -> dict:
    """The cores the author's `ak-check` uses, from this process's view.

    Same rule as `ak_check.default_cpus`: `AK_CHECK_CPUS` if set, else the LAST
    cores of the inherited affinity (actor children inherit the loop's affinity).
    """
    from . import ak_check
    environ = os.environ if environ is None else environ
    text = environ.get(ak_check.ENV_CPUS)
    if text:
        cpus = ak_check.parse_cpus(text)
        compile_cpus, op_cpus, source = (cpus[:ak_check.COMPILE_CPUS],
                                         cpus[:ak_check.OP_TEST_CPUS], ak_check.ENV_CPUS)
    else:
        cpus = sorted(os.sched_getaffinity(0))
        compile_cpus, op_cpus, source = (cpus[-ak_check.COMPILE_CPUS:],
                                         cpus[-ak_check.OP_TEST_CPUS:], "affinity_tail")
    reserved = sorted(set(compile_cpus) | set(op_cpus))
    return {"cpus": _cpu_text(reserved), "compile": _cpu_text(compile_cpus),
            "op_test": _cpu_text(op_cpus), "source": source}


def _cpu_text(cpus) -> str:
    cpus = sorted(set(int(c) for c in cpus))
    parts, start = [], None
    for index, cpu in enumerate(cpus):
        if start is None:
            start = cpu
        if index + 1 == len(cpus) or cpus[index + 1] != cpu + 1:
            parts.append(str(start) if start == cpu else f"{start}-{cpu}")
            start = None
    return ",".join(parts)


#: `actors.SALVAGE_ARM_SUFFIX`, kept literal so this module stays import-free of `actors`
#: (a test pins the two equal).
SALVAGE_ARM_SUFFIX = "+salvage"


def _salvage_row(row: dict) -> bool:
    seat = row.get("seat")
    arm = seat.get("arm") if isinstance(seat, dict) else None
    return isinstance(arm, str) and arm.endswith(SALVAGE_ARM_SUFFIX)


class PhaseEstimator:
    """Seconds an actor phase is expected to take: recent median wall, capped by budget.

    Budgets: the planner's per-call budget (else the actor timeout), the author's
    (else the actor timeout), the critic's timeout. Medians come from the last
    `samples` completed `actor_call.v1` rows of that role in actor-calls.jsonl. A planner
    SALVAGE turn (seat arm ending `SALVAGE_ARM_SUFFIX`) is not a proposal call -- it is a
    short continuation of one that spent its budget -- so its row is skipped rather than
    pulling the planner median down.
    """

    def __init__(self, call_log: Path | None, *, planner_budget_s: float,
                 author_budget_s: float, critic_timeout_s: float,
                 samples: int = MEDIAN_SAMPLES) -> None:
        self.call_log = Path(call_log) if call_log is not None else None
        self.budgets = {"planner": float(planner_budget_s), "author": float(author_budget_s),
                        "critic": float(critic_timeout_s)}
        self.samples = samples

    def _walls(self, models: Mapping[str, str] | None = None) -> dict[str, list[float]]:
        """Recent walls per role. With `models` (role -> the model THIS lane calls; set
        only when lanes run different models, `--lane-actor-models`) a role keeps only
        its model's rows, falling back to every row of the role when that model has
        none yet: one lane's median must not be another model's."""
        walls: dict[str, list[float]] = {"planner": [], "author": [], "critic": []}
        if self.call_log is None:
            return walls
        try:
            with self.call_log.open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - CALL_LOG_TAIL_BYTES))
                data = stream.read()
        except OSError:
            return walls
        matched: dict[str, list[float]] = {"planner": [], "author": [], "critic": []}
        for line in data.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (isinstance(row, dict) and row.get("schema") == "epyc.autokernel.actor_call.v1"
                    and row.get("role") in walls
                    and not _salvage_row(row)
                    and type(row.get("wall_s")) in (int, float) and row["wall_s"] > 0):
                walls[row["role"]].append(float(row["wall_s"]))
                if models is not None:
                    backend = row.get("backend")
                    model = backend.get("model") if isinstance(backend, dict) else None
                    if model is not None and model == models.get(row["role"]):
                        matched[row["role"]].append(float(row["wall_s"]))
        if models is not None:
            walls = {role: (matched[role] or rows) for role, rows in walls.items()}
        return walls

    def estimate(self, phases, models: Mapping[str, str] | None = None) -> tuple[float, dict]:
        walls = self._walls(models)
        total, basis = 0.0, {}
        for phase in phases:
            role = _ROLE[phase]
            recent = walls[role][-self.samples:]
            budget = self.budgets[role]
            median = statistics.median(recent) if recent else None
            seconds = (min(median, budget) if median is not None and budget > 0
                       else median if median is not None else budget)
            basis[phase] = {"seconds": round(seconds, 1), "role": role,
                            "median_wall_s": None if median is None else round(median, 1),
                            "samples": len(recent), "budget_s": budget}
            total += seconds
        return total, basis


class BusPublisher:
    """Append window events to the loop's OWN bus outbox, off the loop's thread."""

    def __init__(self, agent: str | None, *, root: Path | None = None,
                 runner: Callable[..., Any] = subprocess.run, synchronous: bool = False,
                 timeout_s: float = 20.0, log=None) -> None:
        self.agent = agent or None
        self.root = Path(root or os.environ.get("EPYC_ROOT_REPO", "/workspace"))
        self.runner = runner
        self.timeout_s = timeout_s
        self.disabled_reason: str | None = None if self.agent else "no bus agent configured"
        self._log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self._synchronous = synchronous
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None

    def argv(self, message: Mapping[str, Any]) -> list[str]:
        return ["python3", str(self.root / "scripts" / "coordination" / "session_bus.py"),
                "append", "--agent", str(self.agent), "--target", "outbox",
                "--json", json.dumps(message, sort_keys=True)]

    def message(self, payload: Mapping[str, Any]) -> dict:
        return {"to": "*", "kind": "status", "payload": dict(payload)}

    def _send(self, message: Mapping[str, Any]) -> None:
        if self.disabled_reason is not None:
            return
        try:
            done = self.runner(self.argv(message), capture_output=True, text=True,
                               timeout=self.timeout_s)
            code = getattr(done, "returncode", 0)
            if code != 0:
                raise RuntimeError((getattr(done, "stderr", "") or "")[-400:].strip()
                                   or f"exit {code}")
        except Exception as exc:      # noqa: BLE001 -- the bus never takes the loop down
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            self._log(f"cpuwin    bus append by {self.agent!r} refused ({self.disabled_reason}); "
                      "bus events disabled for this run, the window file and "
                      "events.jsonl still carry them")

    def emit(self, payload: Mapping[str, Any]) -> None:
        if self.disabled_reason is not None:
            return
        message = self.message(payload)
        if self._synchronous:
            self._send(message)
            return
        if self._thread is None:
            self._queue = queue.Queue()
            self._thread = threading.Thread(target=self._drain, name="cpu-window-bus",
                                            daemon=True)
            self._thread.start()
        self._queue.put(message)

    def _drain(self) -> None:
        while True:
            message = self._queue.get()
            if message is None:
                return
            self._send(message)

    def close(self, join_s: float = 5.0) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(join_s)


class CpuWindow:
    """One loop process's CPU window: the claim yield policy and its publication."""

    def __init__(self, *, campaign: str, path: Path = DEFAULT_PATH,
                 campaign_path: Path | None = None, ledger_dir: Path | None = None,
                 estimator: PhaseEstimator | None = None,
                 wait_bound_s: float = DEFAULT_WAIT_BOUND_S, poll_s: float = POLL_S,
                 should_stop: Callable[[], bool] = lambda: False,
                 bus: BusPublisher | None = None, reserved: Mapping[str, str] | None = None,
                 heartbeat_s: float | None = HEARTBEAT_S, ttl_s: float = TTL_S,
                 clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] | None = None,
                 on_change: Callable[[dict], None] | None = None,
                 lane_models: Mapping[str, Mapping[str, str]] | None = None) -> None:
        self.campaign = campaign
        self.path = Path(path)
        self.campaign_path = Path(campaign_path) if campaign_path is not None else None
        self.ledger_dir = Path(ledger_dir) if ledger_dir is not None else None
        self.estimator = estimator
        self.wait_bound_s = float(wait_bound_s)
        self.poll_s = float(poll_s)
        self.should_stop = should_stop
        self.bus = bus
        self.reserved = dict(reserved or {})
        self.heartbeat_s = heartbeat_s
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self._log = log or (lambda text: print(text, flush=True))
        self._on_change = on_change
        self.lease = None
        self.disabled: str | None = None
        self._lock = threading.RLock()
        self._tail_active = 0
        self._reacquiring = False
        self._lanes: dict[str, str | None] = {}
        #: lane -> role -> model, only when lanes call different models
        #: (`--lane-actor-models`); None keeps one pooled median per role.
        self.lane_models = ({str(lane): dict(models) for lane, models in lane_models.items()}
                            if lane_models else None)
        #: lane -> (epoch its actor phases are expected to end, the phase it is in).
        self._lane_eta: dict[str, tuple[float | None, str]] = {}
        self._finalized = False
        self._published = False
        self._write_errors = 0
        self._seq = 0
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._events_path = self.path.with_name(self.path.stem + ".events.jsonl")
        now = clock()
        self._window: dict[str, Any] = {
            "schema": WINDOW_SCHEMA, "state": "closed", "loop_holds_claim": False,
            "campaign": campaign, "phase": "startup", "phase_label": None,
            "phase_started_at": _iso(now), "opened_at": None, "est_close_at": None,
            "est_close_basis": None, "closing_reason": "startup",
            "cpu_list": None, "regions": [],
            "cpus_reserved_by_loop": self.reserved.get("cpus"),
            "cpus_reserved_detail": self.reserved or None,
            "contention": self._contention(),
            "owner": self._owner(), "generation": 0, "wait": None,
            "window_path": str(self.path),
            "campaign_path": None if self.campaign_path is None else str(self.campaign_path),
            "ttl_s": self.ttl_s}

    # ---- identity and publication ------------------------------------------------
    def _contention(self) -> str | None:
        if not self.reserved.get("cpus"):
            return None
        return (f"during an open window the author's ak-check may compile (cores "
                f"{self.reserved.get('compile')}) or op-test (cores "
                f"{self.reserved.get('op_test')}) at nice 19, source "
                f"{self.reserved.get('source')}; a peer measuring on those cores shares "
                "them. ak-check refuses while this loop's tail holds its fence.")

    @staticmethod
    def _owner() -> dict:
        owner: dict[str, Any] = {"pid": os.getpid(), "host": socket.gethostname(),
                                 "boot_id": None, "start_ticks": None}
        try:
            owner["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            stat = Path(f"/proc/{os.getpid()}/stat").read_text()
            owner["start_ticks"] = int(stat[stat.rfind(")") + 2:].split()[19])
        except (OSError, ValueError, IndexError):
            pass
        return owner

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._window))

    def _write(self, *, expire_now: bool = False) -> dict:
        now = self.clock()
        self._window["seq"] = self._seq = self._seq + 1
        self._window["heartbeat_at"] = self._window["updated_at"] = _iso(now)
        self._window["expires_at"] = _iso(now if expire_now else now + self.ttl_s)
        body = json.loads(json.dumps(self._window))
        for target in (self.path, self.campaign_path):
            if target is None:
                continue
            try:
                status.write_json(target.parent, target.name, body, prefix=".cpu-window-")
            except OSError as exc:
                self._write_errors += 1
                if self._write_errors <= 3:
                    self._log(f"cpuwin    window write to {target} failed: {exc}")
        return body

    def _event(self, previous: Mapping[str, Any]) -> None:
        body = self._window
        payload = {"event": BUS_EVENT, "schema": EVENT_SCHEMA,
                   "previous_state": previous.get("state"),
                   **{key: body.get(key) for key in (
                       "state", "loop_holds_claim", "campaign", "phase", "phase_label",
                       "opened_at", "est_close_at", "closing_reason", "cpu_list",
                       "cpus_reserved_by_loop", "generation", "seq", "expires_at",
                       "window_path")},
                   "owner_pid": body["owner"]["pid"]}
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError as exc:
            self._write_errors += 1
            if self._write_errors <= 3:
                self._log(f"cpuwin    events append failed: {exc}")
        if self.bus is not None:
            self.bus.emit(payload)

    def _publish(self, **fields) -> None:
        """Caller holds the lock. Writes the file; emits an event on a real change.

        Nothing is written before the first publication, and the first one happens
        only once there is something real to say (a real claim waiting on a busy
        region, or a bound yield lease): a run whose claim provider is a stand-in
        publishes nothing at all.
        """
        if not self._published:
            self._published = True
            if self.heartbeat_s and self._heartbeat_thread is None:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, name="cpu-window-heartbeat", daemon=True)
                self._heartbeat_thread.start()
        previous = dict(self._window)
        self._window.update(fields)
        self._write()
        changed = (previous.get("state") != self._window["state"]
                   or previous.get("loop_holds_claim") != self._window["loop_holds_claim"]
                   or (self._window["state"] != "closed"
                       and previous.get("phase") != self._window["phase"]))
        if changed:
            self._event(previous)
            self._log(f"cpuwin    {self._window['state']:<7} phase={self._window['phase']} "
                      f"claim={'held' if self._window['loop_holds_claim'] else 'released'}"
                      + (f" est_close={self._window['est_close_at']}"
                         if self._window["est_close_at"] else "")
                      + (f" reason={self._window['closing_reason']}"
                         if self._window["closing_reason"] else ""))
            if self._on_change is not None:
                try:
                    self._on_change(dict(self._window))
                except Exception:      # noqa: BLE001 -- reporting only
                    pass

    def heartbeat(self) -> None:
        with self._lock:
            if self._published and not self._finalized:
                self._write()

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self.heartbeat_s):
            self.heartbeat()

    def set_reserved(self, reserved: Mapping[str, str]) -> None:
        """The ak-check cores, read AFTER the loop confined its own affinity."""
        with self._lock:
            self.reserved = dict(reserved)
            self._window.update(cpus_reserved_by_loop=self.reserved.get("cpus"),
                                cpus_reserved_detail=self.reserved or None,
                                contention=self._contention())

    # ---- the claim ---------------------------------------------------------------
    def acquire_initial(self, attempt: Callable[[], Any]):
        """Run the claim acquisition, retrying while a peer holds the regions.

        Between batches the regions are free, and a peer that was waiting in
        `region-lock run` takes them the instant the previous batch exits. A
        one-shot acquisition would then refuse the batch (and a refused batch fails
        the serial roster), so under yield the start waits like any re-acquire.
        Anything but a busy region still raises at once.
        """
        from .claim import ClaimRefused, region_lock_busy
        started, warned, next_log = time.monotonic(), False, 0.0
        while True:
            try:
                return attempt()
            except Exception as exc:
                if not region_lock_busy(exc):
                    raise
                waited = time.monotonic() - started
                if self.should_stop():
                    raise ClaimRefused(
                        f"stop requested while waiting {waited:.0f} s for a peer to release "
                        "the CPU regions at batch start") from None
                with self._lock:
                    if self._window["wait"] is None:
                        self._publish(state="closing", phase="startup",
                                      closing_reason="startup_acquire",
                                      wait={"since": _iso(self.clock() - waited),
                                            "bound_s": self.wait_bound_s, "exceeded": False})
                    if not warned and waited >= self.wait_bound_s:
                        warned = True
                        self._window["wait"]["exceeded"] = True
                        self._write()
                        self._log(f"cpuwin    batch start has waited {waited:.0f} s (bound "
                                  f"{self.wait_bound_s:.0f} s) for a peer holding the CPU "
                                  "regions; still waiting, never pre-empting it")
                if time.monotonic() >= next_log:
                    self._log(f"cpuwin    waiting for a peer to release the CPU regions "
                              f"({waited:.0f} s)")
                    next_log = time.monotonic() + 60
                time.sleep(self.poll_s)

    def bind(self, receipt) -> bool:
        """Attach the acquired CPU context. False (window inert) when it has no lease."""
        from .claim import yield_lease
        lease = yield_lease(receipt)
        with self._lock:
            if lease is None:
                # A stand-in provider (tests) or a claim without a lease: the claim is
                # held for the whole batch exactly as with the knob off.
                self.disabled = "the CPU claim provider exposes no yield lease"
                if self._published:
                    self._publish(state="closed", loop_holds_claim=True, phase="batch",
                                  closing_reason="yield_unavailable", wait=None)
                return False
            self.lease = lease
            self._publish(state="closed", loop_holds_claim=True, phase="startup",
                          closing_reason="claim_acquired", wait=None,
                          cpu_list=receipt.get("cpu_list"),
                          regions=list(receipt.get("regions") or []),
                          generation=lease.generation)
            return True

    def _estimate(self, phase: str, lane: str | None = None) -> tuple[float | None, dict | None]:
        if self.estimator is None:
            return None, None
        try:
            models = (self.lane_models.get(lane) if self.lane_models is not None
                      and lane is not None else None)
            seconds, basis = (self.estimator.estimate(_AHEAD[phase], models=models)
                              if models else self.estimator.estimate(_AHEAD[phase]))
        except Exception:      # noqa: BLE001 -- an ETA is advice
            return None, None
        return seconds, {"phases": list(_AHEAD[phase]), "per_phase": basis,
                         "note": "median recent wall capped by budget, summed over the "
                                 "actor phases still ahead; a rejection loops back"}

    def note_step(self, lane: str, label: str) -> None:
        """A lane's step label (`pool.drive`'s on_step). Reporting must never raise."""
        try:
            self._note_step(lane, label)
        except Exception as exc:      # noqa: BLE001
            self._log(f"cpuwin    step handling failed: {type(exc).__name__}: {exc}")

    def _note_step(self, lane: str, label: str) -> None:
        if self.lease is None or self._finalized:
            return
        phase = classify(label)
        with self._lock:
            self._lanes[lane] = phase
            now = self.clock()
            if phase is None:
                if self._window["state"] == "open" and label.startswith(_CPU_NEXT):
                    self._publish(state="closing", phase="cpu_step", phase_label=label,
                                  phase_started_at=_iso(now), est_close_at=_iso(now),
                                  est_close_basis={"note": "the next step needs the CPU"},
                                  closing_reason="cpu_step_next")
                return
            if self._tail_active or self._reacquiring:
                # A lane owns the tail (the claim stays held), or a re-acquire is in
                # flight: this transition changes nothing about the claim.
                return
            seconds, basis = self._estimate(phase, lane)
            est = None if seconds is None else _iso(now + seconds)
            closing = phase == "critic2"
            self._lane_eta[lane] = (None if seconds is None else now + seconds, phase)
            # Several lanes in actor phases (a pool): the window closes when the FIRST
            # of them needs the CPU, not when the lane that just moved does. One lane:
            # exactly the historical estimate.
            actor_lanes = {name: self._lane_eta[name] for name, current in self._lanes.items()
                           if current is not None and name in self._lane_eta}
            if len(actor_lanes) > 1:
                etas = [eta for eta, _phase in actor_lanes.values()]
                if all(eta is not None for eta in etas):
                    est = _iso(min(etas))
                basis = {**(basis or {}),
                         "lanes": {name: {"phase": lane_phase, "est_close_at": _iso(eta)}
                                   for name, (eta, lane_phase) in sorted(actor_lanes.items())},
                         "lanes_note": "the window closes when the first lane needs the CPU"}
                closing = any(lane_phase == "critic2"
                              for _eta, lane_phase in actor_lanes.values())
            if self.lease.held:
                self.lease.release(reason=f"actor:{phase}")
                opened_at = _iso(now)
            else:
                opened_at = self._window["opened_at"] or _iso(now)
            self._publish(state="closing" if closing else "open", loop_holds_claim=False,
                          phase=phase, phase_label=label, phase_started_at=_iso(now),
                          opened_at=opened_at, est_close_at=est, est_close_basis=basis,
                          closing_reason="critic2_started" if closing else None,
                          wait=None, generation=self.lease.generation)

    def ensure_held(self, *, reason: str, stop_exception=None) -> None:
        """Take the claim back before a CPU step, waiting on a peer (see module doc)."""
        if self.lease is None:
            return
        with self._lock:
            if self.lease.held:
                return
            self._reacquiring = True
            now = self.clock()
            self._publish(state="closing", closing_reason=reason, phase=reason,
                          phase_started_at=_iso(now),
                          est_close_at=self._window["est_close_at"] or _iso(now),
                          wait={"since": _iso(now), "bound_s": self.wait_bound_s,
                                "exceeded": False})
        next_log = [time.monotonic() + 60]

        def on_wait(waited):
            if time.monotonic() >= next_log[0]:
                self._log(f"cpuwin    {reason}: waiting {waited:.0f} s for a peer to release "
                          "the CPU regions")
                next_log[0] = time.monotonic() + 60

        def on_bound(waited):
            with self._lock:
                if self._window["wait"] is not None:
                    self._window["wait"]["exceeded"] = True
                self._write()
            self._log(f"cpuwin    {reason}: re-acquire has waited {waited:.0f} s (bound "
                      f"{self.wait_bound_s:.0f} s) behind a peer; still waiting -- nothing "
                      "is measured without the claim and the peer is never pre-empted")

        from .claim import ClaimRefused
        try:
            self.lease.reacquire(reason=reason, should_stop=(
                self.should_stop if stop_exception is not None else None),
                wait_bound_s=self.wait_bound_s, poll_s=self.poll_s,
                on_wait=on_wait, on_bound=on_bound)
        except ClaimRefused as exc:
            with self._lock:
                self._reacquiring = False
            if stop_exception is not None and self.should_stop():
                raise stop_exception(str(exc)) from None
            raise
        except BaseException:
            with self._lock:
                self._reacquiring = False
            raise
        with self._lock:
            self._reacquiring = False
            now = self.clock()
            self._publish(state="closed", loop_holds_claim=True, closing_reason=reason,
                          phase=reason, phase_started_at=_iso(now), opened_at=None,
                          est_close_at=None, est_close_basis=None, wait=None,
                          generation=self.lease.generation)

    @contextmanager
    def cpu_step(self, reason: str) -> Iterator[None]:
        """Hold the claim for one CPU step (a tail session, a measurement window)."""
        from . import loop as loop_mod
        with self._lock:
            self._tail_active += 1
        try:
            self.ensure_held(reason=reason, stop_exception=loop_mod.TailRefused)
            yield
        finally:
            with self._lock:
                self._tail_active -= 1

    def wrap_fence(self, fence: Callable[[], Any] | None) -> Callable[[], Any]:
        """The tail's fence: re-acquire the claim first, then any inner fence."""
        @contextmanager
        def composed():
            with self.cpu_step("build_and_measure"):
                if fence is None:
                    yield
                else:
                    with fence():
                        yield
        return composed

    # ---- teardown ----------------------------------------------------------------
    def teardown(self) -> None:
        """Before the claim context closes: hold it again for its close observation."""
        if self.lease is None or self._finalized:
            return
        try:
            self.ensure_held(reason="teardown")
        except Exception as exc:      # noqa: BLE001 -- the claim's own close reports it
            self._log(f"cpuwin    teardown re-acquire failed: {type(exc).__name__}: {exc}")

    def finalize(self) -> None:
        """After the claim is released: `closed`, no claim, heartbeat expired. Idempotent."""
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            if not self._published:
                self._stop_heartbeat.set()
                if self.bus is not None:
                    self.bus.close()
                return
            now = self.clock()
            previous = dict(self._window)
            self._window.update(state="closed", loop_holds_claim=False, phase="exited",
                                phase_label=None, phase_started_at=_iso(now),
                                opened_at=None, est_close_at=None, est_close_basis=None,
                                closing_reason="loop_exit", wait=None)
            self._write(expire_now=True)
            self._event(previous)
            if self.ledger_dir is not None:
                ledger = {"schema": LEDGER_SCHEMA, "window": json.loads(json.dumps(self._window)),
                          "disabled": self.disabled,
                          "claim": None if self.lease is None else self.lease.ledger(),
                          "note": ("the held-claim evidence interval spans these segments; "
                                   "between a segment's released_at and the next "
                                   "acquired_at a peer may have held the regions")}
                try:
                    status.write_json(self.ledger_dir, LEDGER_NAME, ledger,
                                      prefix=".cpu-window-ledger-")
                except OSError as exc:
                    self._log(f"cpuwin    ledger write failed: {exc}")
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(5)
        if self.bus is not None:
            self.bus.close()


__all__ = ["ACTOR_PHASES", "BUS_EVENT", "BusPublisher", "CAMPAIGN_NAME", "CpuWindow",
           "DEFAULT_BUS_AGENT", "DEFAULT_PATH", "DEFAULT_WAIT_BOUND_S", "EVENT_SCHEMA",
           "LEDGER_NAME", "LEDGER_SCHEMA", "PhaseEstimator", "STATES", "WINDOW_SCHEMA",
           "ak_check_cpus", "classify", "read_window"]
