#!/usr/bin/env python3
"""No AutoKernel process outlives the actor call, gate or run that spawned it (DS41-C84).

WHY. On 2026-09-29 an author's `ak-check --op-test` was ended by opencode's shell-tool
timeout (120 s default; the tool SIGTERMs its own detached process group). `ak-check`
died of the default SIGTERM action, but its `test-backend-ops` child had been started
with `start_new_session=True` -- a different session, so neither opencode's group kill
nor the loop's actor group kill reached it. It was reparented to the container
subreaper and ran for 25 h at nice 19 on 4-6 cores, through two serving-floor
calibrations and every A/B of the day. Its fence slot died with the Python process, so
the tail took the fence and measured beside it. Nothing noticed.

THE MECHANISM (design: docs/design/ak-no-orphan-processes.md).

* PROVENANCE COOKIE. Every actor/validator process the loop spawns gets
  `AK_PROC_SCOPE=<campaign-key>.<owner pid>.<owner start ticks>.<call uuid>` in its
  environment (`Guard.call_scope`). The environment is inherited through fork/exec,
  setsid and double-forks alike, so every descendant -- opencode's detached shell-tool
  sessions, ak-check, its compiler and test-backend-ops -- carries it in
  `/proc/<pid>/environ`. When the call ends (any path: return, stop, budget, timeout,
  exception) every process still carrying THAT call's cookie is ended. This is the
  Jenkins ProcessTreeKiller technique; it needs no subreaper, no cgroup and no process
  name, and it only ever selects processes this loop provably spawned.
* STALE SWEEP (`Guard.sweep_stale`): at run start and before every CPU measurement,
  processes carrying a cookie of THIS campaign whose owner (pid + start ticks) is dead,
  or of this owner whose call scope is closed, are ended; so is a process with no
  cookie whose cwd or exe lies under this campaign's scratch roots and which is not a
  descendant of this process (the operator's rule: under the campaign's scratch is
  provably ours -- the class of the C84 orphan, which predates the cookie).
* MEASUREMENT WATCH (`MeasurementWatch`): around every CPU serving launch, a low-rate
  census of campaign-related processes that are NOT this loop's own (not self, an
  ancestor, a descendant, or a live call of this owner). One that burns CPU in two
  consecutive intervals becomes an observed `campaign_process_contention` condition
  and the launch is MeasurementInvalid -- silent contamination becomes a visible
  refusal. Never a kill during the timed rounds.
* SCRATCH RELEASE (`Guard.sweep_path`): a marked scratch dir is not deleted while a
  process's cwd or exe is inside it; provably-ours ones are ended first, and a release
  with survivors is refused (journalled) instead of pulling the tree from under them.

SIGNALS. Only processes selected above, only of this uid, never by name, never a
group. Each target is pinned with `pidfd_open` and its identity (start ticks AND the
reason it was selected) is re-read AFTER pinning, so a recycled PID is never signalled.
SIGTERM, a grace period, SIGKILL, then verified dead via the pidfd (a survivor is
reported, never assumed gone).

EXEMPT: shared daemons an actor CLI may start as a side effect and that other sessions
then use (`NEVER_SWEEP_EXE_PREFIXES`: the codex app-server daemon). Exempting is the
safe direction of a path rule; nothing is ever SELECTED for a kill by name or path
prefix of its executable alone.

Standard library only; writes no files.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import select
import signal
import threading
import time
from typing import Callable, Iterable, Mapping
import uuid

SCHEMA = "epyc.autokernel.procguard.v1"
ENV_SCOPE = "AK_PROC_SCOPE"
CONTENTION = "campaign_process_contention"
DEFAULT_GRACE_S = 5.0
KILL_WAIT_S = 5.0
ENVIRON_READ_BYTES = 256 * 1024
DELETED_SUFFIX = " (deleted)"
#: Shared services an actor CLI may start as a side effect and other sessions then
#: use: never swept, never flagged (the managed codex app-server daemon outlives every
#: `codex exec` by design and serves the operator's own codex sessions too).
NEVER_SWEEP_EXE_PREFIXES = (str(Path.home() / ".codex" / "packages" / "app-server-daemon") + "/",)

#: Classes a guard may end: provably ours AND provably orphaned.
KILLABLE = frozenset({"leak", "stale", "orphan_scratch"})
#: Classes a measurement treats as contention when they burn CPU.
FOREIGN = frozenset({"leak", "stale", "orphan_scratch", "campaign_foreign", "peer"})
#: This loop's own live processes: never signalled, never contention.
OWN = frozenset({"self", "ancestor", "descendant", "live"})

try:
    CLK_TCK = os.sysconf("SC_CLK_TCK")
except (ValueError, OSError, AttributeError):   # pragma: no cover - Linux always has it
    CLK_TCK = 100


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    start_ticks: int
    state: str
    cpu_ticks: int
    tty: int
    uid: int | None
    cwd: str | None
    exe: str | None
    scope: str | None

    @property
    def identity(self) -> tuple[int, int]:
        return self.pid, self.start_ticks

    def to_dict(self) -> dict:
        return {"pid": self.pid, "ppid": self.ppid, "start_ticks": self.start_ticks,
                "state": self.state, "cpu_ticks": self.cpu_ticks, "tty": self.tty,
                "cwd": self.cwd,
                "exe": self.exe, "scope": self.scope}


def _strip_deleted(path: str | None) -> str | None:
    if path is not None and path.endswith(DELETED_SUFFIX):
        return path[:-len(DELETED_SUFFIX)]
    return path


def _under(path: str | None, root: str) -> bool:
    path = _strip_deleted(path)
    return bool(path) and (path == root or path.startswith(root.rstrip("/") + "/"))


def read_stat(proc_root: Path, pid: int) -> tuple[int, int, str, int, int] | None:
    """(ppid, start ticks, state, utime+stime, tty_nr) from `/proc/<pid>/stat`, or None."""
    try:
        text = (Path(proc_root) / str(pid) / "stat").read_text(errors="replace")
        head, sep, tail = text.rpartition(") ")
        if not sep or head.split(" (", 1)[0].strip() != str(pid):
            return None
        fields = tail.split()
        return (int(fields[1]), int(fields[19]), fields[0], int(fields[11]) + int(fields[12]),
                int(fields[4]))
    except (OSError, ValueError, IndexError):
        return None


def _readlink(path: Path) -> str | None:
    try:
        return os.readlink(path)
    except OSError:
        return None


def read_scope(proc_root: Path, pid: int) -> str | None:
    """This loop's cookie in the process's (initial) environment, or None."""
    try:
        with open(Path(proc_root) / str(pid) / "environ", "rb") as stream:
            raw = stream.read(ENVIRON_READ_BYTES)
    except OSError:
        return None
    prefix = ENV_SCOPE.encode() + b"="
    for item in raw.split(b"\0"):
        if item.startswith(prefix):
            return item[len(prefix):].decode("ascii", errors="replace") or None
    return None


def read_proc(proc_root: Path, pid: int,
              cache: dict | None = None) -> Proc | None:
    stat = read_stat(proc_root, pid)
    if stat is None:
        return None
    ppid, start, state, cpu, tty = stat
    key = (pid, start)
    static = cache.get(key) if cache is not None else None
    if static is None:
        root = Path(proc_root) / str(pid)
        try:
            uid = os.stat(root).st_uid
        except OSError:
            uid = None
        static = (uid, _readlink(root / "cwd"), _readlink(root / "exe"),
                  read_scope(proc_root, pid))
        if cache is not None:
            cache[key] = static
    return Proc(pid, ppid, start, state, cpu, tty, *static)


def snapshot(proc_root: Path = Path("/proc"), *, cache: dict | None = None) -> dict[int, Proc]:
    table: dict[int, Proc] = {}
    try:
        names = os.listdir(proc_root)
    except OSError:
        return table
    for name in names:
        if not name.isdigit():
            continue
        proc = read_proc(proc_root, int(name), cache)
        if proc is not None:
            table[proc.pid] = proc
    if cache is not None:
        live = {p.identity for p in table.values()}
        for key in [k for k in cache if k not in live]:
            del cache[key]
    return table


def _ancestors(table: Mapping[int, Proc], pid: int) -> set[int]:
    out: set[int] = set()
    cursor = table[pid].ppid if pid in table else None
    while cursor and cursor not in out and cursor in table:
        out.add(cursor)
        cursor = table[cursor].ppid
    return out


def _descendants(table: Mapping[int, Proc], pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for proc in table.values():
        children.setdefault(proc.ppid, []).append(proc.pid)
    out: set[int] = set()
    stack = list(children.get(pid, ()))
    while stack:
        current = stack.pop()
        if current in out:
            continue
        out.add(current)
        stack.extend(children.get(current, ()))
    return out


def campaign_key(store: Path | str | None) -> str:
    if store is None:
        return "none"
    return hashlib.sha256(os.path.realpath(str(store)).encode()).hexdigest()[:12]


def parse_scope(value: str | None) -> tuple[str, int, int, str] | None:
    if not value:
        return None
    parts = value.split(".")
    if len(parts) != 4:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2]), parts[3]
    except ValueError:
        return None


class Guard:
    """One loop process's view: its identity, its campaign roots, its open call scopes."""

    def __init__(self, *, store: Path | str | None = None,
                 scratch_roots: Iterable[Path | str] = (),
                 proc_root: Path | str = "/proc", grace_s: float = DEFAULT_GRACE_S,
                 kill_wait_s: float = KILL_WAIT_S,
                 never_sweep: Iterable[str] = NEVER_SWEEP_EXE_PREFIXES,
                 identity: tuple[int, int] | None = None,
                 uid: int | None = None,
                 terminate: Callable[[list[Proc], Callable[[Proc], bool]], dict] | None = None):
        self.proc_root = Path(proc_root)
        self.store = os.path.realpath(str(store)) if store is not None else None
        roots = [os.path.realpath(str(r)) for r in scratch_roots]
        if self.store is not None:
            roots.append(os.path.join(self.store, "scratch"))
        self.scratch_roots = tuple(sorted(set(roots)))
        self.key = campaign_key(store)
        if identity is None:
            stat = read_stat(self.proc_root, os.getpid())
            identity = (os.getpid(), stat[1] if stat else 0)
        self.pid, self.start_ticks = identity
        self.uid = os.getuid() if uid is None else uid
        self.grace_s, self.kill_wait_s = float(grace_s), float(kill_wait_s)
        self.never_sweep = tuple(never_sweep)
        self._terminate = terminate or self._terminate_verified
        self._open: set[str] = set()
        self._lock = threading.Lock()
        self.records: list[dict] = []

    # -- identity -----------------------------------------------------------------------
    def new_scope(self) -> str:
        return f"{self.key}.{self.pid}.{self.start_ticks}.{uuid.uuid4().hex[:16]}"

    def _alive(self, pid: int, start: int) -> bool:
        stat = read_stat(self.proc_root, pid)
        return stat is not None and stat[1] == start and stat[2] != "Z"

    def _exempt(self, proc: Proc) -> bool:
        exe = _strip_deleted(proc.exe) or ""
        return any(exe.startswith(prefix) for prefix in self.never_sweep)

    def _orphaned(self, proc: Proc, anc: set[int]) -> bool:
        """Reparented to a reaper this loop shares (init, or one of this process's own
        ancestors -- the container subreaper): its spawner is gone. A child of some OTHER
        live process (a peer loop's work) is not orphaned; a process with a controlling
        terminal is somebody's interactive shell. Neither is ever swept without a cookie."""
        return proc.tty == 0 and (proc.ppid == 1 or proc.ppid in anc)

    def location(self, proc: Proc) -> str | None:
        """'scratch' / 'store' when the cwd or exe lies under this campaign's roots."""
        paths = (proc.cwd, proc.exe)
        if any(_under(p, root) for p in paths for root in self.scratch_roots):
            return "scratch"
        if self.store is not None and any(_under(p, self.store) for p in paths):
            return "store"
        return None

    def classify(self, proc: Proc, anc: set[int], desc: set[int]) -> str | None:
        """The ONE ownership decision. None: not this campaign's business."""
        if proc.pid == self.pid:
            return "self"
        if proc.pid in anc:
            return "ancestor"
        if proc.uid is not None and proc.uid != self.uid:
            return None
        if proc.state == "Z":
            return None                     # already dead; its parent reaps it
        if self._exempt(proc):
            return "exempt"
        owner = parse_scope(proc.scope)
        if owner is not None and owner[0] == self.key:
            if (owner[1], owner[2]) == (self.pid, self.start_ticks):
                with self._lock:
                    return "live" if proc.scope in self._open else "leak"
            if not self._alive(owner[1], owner[2]):
                return "stale"
            if owner[1] in anc:
                return "live"               # an ancestor loop's own open call
            return "peer"
        if proc.pid in desc:
            return "descendant"
        where = self.location(proc)
        if where == "scratch" and owner is None and self._orphaned(proc, anc):
            # No cookie (it predates it, or scrubbed its env) but inside this campaign's
            # scratch and orphaned: the operator's rule makes that provably ours.
            return "orphan_scratch"
        if where is not None:
            return "campaign_foreign"
        return None

    def census(self, *, cache: dict | None = None) -> list[tuple[str, Proc]]:
        """Every campaign-related process with its class (own ones included)."""
        table = snapshot(self.proc_root, cache=cache)
        anc = _ancestors(table, self.pid)
        desc = _descendants(table, self.pid)
        out = []
        for proc in table.values():
            cls = self.classify(proc, anc, desc)
            if cls is not None:
                out.append((cls, proc))
        return out

    # -- ending processes ---------------------------------------------------------------
    def _terminate_verified(self, targets: list[Proc],
                            still_selected: Callable[[Proc], bool]) -> dict:
        """TERM, grace, KILL, verify -- each target pinned by pidfd and re-identified."""
        pinned: list[tuple[Proc, int]] = []
        for proc in targets:
            try:
                fd = os.pidfd_open(proc.pid)
            except (ProcessLookupError, OSError):
                continue                                   # already gone
            current = read_proc(self.proc_root, proc.pid)
            if (current is None or current.start_ticks != proc.start_ticks
                    or not still_selected(current)):
                os.close(fd)                               # recycled or no longer ours
                continue
            pinned.append((proc, fd))
        signalled, killed = [], []
        try:
            live = self._signal_and_wait(pinned, signal.SIGTERM, self.grace_s, signalled)
            live = self._signal_and_wait(live, signal.SIGKILL, self.kill_wait_s, killed)
            survivors = [proc.to_dict() for proc, _fd in live]
        finally:
            for _proc, fd in pinned:
                try:
                    os.close(fd)
                except OSError:
                    pass
        return {"terminated": signalled, "killed": killed, "survivors": survivors}

    @staticmethod
    def _signal_and_wait(pinned: list[tuple[Proc, int]], sig: int, wait_s: float,
                         sent: list) -> list[tuple[Proc, int]]:
        if not pinned:
            return []
        for proc, fd in pinned:
            try:
                signal.pidfd_send_signal(fd, sig)
                sent.append(proc.pid)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        remaining = list(pinned)
        deadline = time.monotonic() + max(0.0, wait_s)
        while remaining:
            poller = select.poll()
            for _proc, fd in remaining:
                poller.register(fd, select.POLLIN)
            left = deadline - time.monotonic()
            if left <= 0:
                break
            ready = {fd for fd, _ev in poller.poll(min(left, 0.2) * 1000)}
            remaining = [(p, fd) for p, fd in remaining if fd not in ready]
        return remaining

    def _end(self, reason: str, targets: list[tuple[str, Proc]],
             still_selected: Callable[[Proc], bool]) -> dict:
        record = {"schema": SCHEMA, "reason": reason, "at": time.time(),
                  "owner": {"pid": self.pid, "start_ticks": self.start_ticks},
                  "targets": [{"class": cls, **proc.to_dict()} for cls, proc in targets],
                  "terminated": [], "killed": [], "survivors": []}
        if targets:
            record.update(self._terminate([proc for _cls, proc in targets], still_selected))
        with self._lock:
            self.records.append(record)
            del self.records[:-64]
        return record

    # -- the three sweeps ---------------------------------------------------------------
    def scope_is_open(self, token: str | None) -> bool:
        """An existing cookie is reusable only under THIS guard's actual open scope."""
        with self._lock:
            return token is not None and token in self._open

    @contextmanager
    def call_scope(self, env: Mapping[str, str] | None = None):
        """`with guard.call_scope(env) as child_env: Popen(..., env=child_env)`.

        On exit (every path) every process still carrying this call's cookie is ended;
        `self.records[-1]` says what was found."""
        token = self.new_scope()
        child = dict(os.environ if env is None else env)
        child[ENV_SCOPE] = token
        with self._lock:
            self._open.add(token)
        try:
            yield child
        finally:
            with self._lock:
                self._open.discard(token)
            self.sweep_scope(token)

    def sweep_scope(self, token: str) -> dict:
        table = snapshot(self.proc_root)
        anc = _ancestors(table, self.pid)
        targets = [("leak", proc) for proc in table.values()
                   if proc.scope == token and proc.pid != self.pid and proc.pid not in anc
                   and proc.state != "Z" and proc.uid in (None, self.uid)
                   and not self._exempt(proc)]
        return self._end("call_scope_closed", targets,
                         lambda current: current.scope == token)

    def sweep_stale(self, reason: str = "stale") -> dict:
        table = snapshot(self.proc_root)
        anc = _ancestors(table, self.pid)
        desc = _descendants(table, self.pid)
        targets = [(cls, proc) for proc in table.values()
                   if (cls := self.classify(proc, anc, desc)) in KILLABLE]
        return self._end(reason, targets,
                         lambda current: self.classify(current, anc, desc) in KILLABLE)

    def _anything_inside(self, root: str) -> bool:
        try:
            names = os.listdir(self.proc_root)
        except OSError:
            return True
        for name in names:
            if name.isdigit() and name != str(self.pid):
                base = self.proc_root / name
                if (_under(_readlink(base / "cwd"), root)
                        or _under(_readlink(base / "exe"), root)):
                    return True
        return False

    def sweep_path(self, path: Path | str) -> dict:
        """End provably-ours processes whose cwd or exe is inside `path` (a marked
        scratch dir about to be deleted). Survivors are reported in `survivors`, plus
        any process inside that is NOT ours to end (a live call, a peer's, exempt)."""
        root = os.path.realpath(str(path))
        empty = {"schema": SCHEMA, "reason": f"scratch_release:{root}", "targets": [],
                 "terminated": [], "killed": [], "survivors": []}
        if not self._anything_inside(root):
            return empty                   # the common case: two readlinks per process
        table = snapshot(self.proc_root)
        anc = _ancestors(table, self.pid)
        desc = _descendants(table, self.pid)

        def inside(proc: Proc) -> bool:
            return _under(proc.cwd, root) or _under(proc.exe, root)

        def endable(proc: Proc) -> bool:
            if not inside(proc):
                return False
            cls = self.classify(proc, anc, desc)
            return (cls in KILLABLE or cls == "descendant"
                    or (cls in (None, "campaign_foreign") and proc.scope is None
                        and self._orphaned(proc, anc) and proc.uid in (None, self.uid)
                        and proc.state != "Z"))
        targets = [(self.classify(p, anc, desc) or "inside", p)
                   for p in table.values() if endable(p)]
        record = self._end(f"scratch_release:{root}", targets, endable)
        blockers = [p.to_dict() for p in table.values()
                    if inside(p) and not endable(p) and p.pid != self.pid
                    and p.pid not in anc and p.state != "Z"
                    and p.uid in (None, self.uid)]
        record["survivors"] = list(record["survivors"]) + blockers
        return record


class MeasurementWatch:
    """Before and DURING one CPU measurement: sweep what is provably ours, then watch.

    `finish()` returns observed contradictions in the `cpu_lifecycle_invalidity`
    shape: a FOREIGN-class process that burned >= `min_cpu_fraction` of a core in
    `persistence` consecutive intervals (two samples to act, never one)."""

    def __init__(self, guard: Guard, label: str, *, interval_s: float = 3.0,
                 min_cpu_fraction: float = 0.10, persistence: int = 2,
                 clock: Callable[[], float] = time.monotonic, background: bool = True):
        self.guard, self.label = guard, label
        self.interval_s, self.min_cpu_fraction = float(interval_s), float(min_cpu_fraction)
        self.persistence = max(2, int(persistence))
        self.clock, self.background = clock, background
        self._cache: dict = {}
        self._last: dict[tuple[int, int], tuple[float, int]] = {}
        self._streak: dict[tuple[int, int], int] = {}
        self._flagged: dict[tuple[int, int], dict] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.pre_sweep: dict | None = None
        self.samples = 0
        self.errors: list[str] = []

    def start(self) -> "MeasurementWatch":
        try:
            self.pre_sweep = self.guard.sweep_stale(f"before_measurement:{self.label}")
        except Exception as exc:          # noqa: BLE001 -- a guard never breaks a launch
            self.errors.append(f"pre-sweep: {type(exc).__name__}: {exc}"[:256])
        self.sample()
        if self.background:
            self._thread = threading.Thread(target=self._loop, name="ak-procguard-watch",
                                            daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def sample(self) -> None:
        try:
            rows = self.guard.census(cache=self._cache)
        except Exception as exc:          # noqa: BLE001
            if len(self.errors) < 8:
                self.errors.append(f"census: {type(exc).__name__}: {exc}"[:256])
            return
        now = self.clock()
        self.samples += 1
        seen = set()
        for cls, proc in rows:
            if cls not in FOREIGN:
                continue
            key = proc.identity
            seen.add(key)
            previous = self._last.get(key)
            self._last[key] = (now, proc.cpu_ticks)
            if previous is None or now <= previous[0]:
                continue
            fraction = (proc.cpu_ticks - previous[1]) / (CLK_TCK * (now - previous[0]))
            if fraction >= self.min_cpu_fraction:
                self._streak[key] = self._streak.get(key, 0) + 1
            else:
                self._streak[key] = 0
            if self._streak[key] >= self.persistence:
                row = self._flagged.setdefault(key, {
                    "condition": CONTENTION, "class": cls, "label": self.label,
                    "pid": proc.pid, "start_ticks": proc.start_ticks,
                    "cwd": proc.cwd, "exe": proc.exe, "scope": proc.scope,
                    "observations": 0, "max_cpu_fraction": 0.0})
                row["observations"] = self._streak[key]
                row["max_cpu_fraction"] = round(max(row["max_cpu_fraction"], fraction), 3)
        for key in [k for k in self._last if k not in seen]:
            self._last.pop(key, None)
            self._streak.pop(key, None)

    def finish(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)
        self.sample()
        return [dict(row) for row in self._flagged.values()]


# -- the installed guard (one per loop process, like `scratch.install`) ---------------
_INSTALLED: Guard | None = None
_INSTALL_LOCK = threading.Lock()


def install(guard: Guard) -> Guard:
    global _INSTALLED
    with _INSTALL_LOCK:
        _INSTALLED = guard
    return guard


def uninstall(guard: Guard) -> None:
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED is guard:
            _INSTALLED = None


def installed() -> Guard | None:
    return _INSTALLED


_FALLBACK: Guard | None = None


def current() -> Guard:
    """The installed guard, else a campaign-less one: its call scopes still end every
    process carrying their cookie (no location rules without a campaign root)."""
    global _FALLBACK
    guard = _INSTALLED
    if guard is not None:
        return guard
    with _INSTALL_LOCK:
        if _FALLBACK is None or _FALLBACK.pid != os.getpid():
            _FALLBACK = Guard()
        return _FALLBACK


def measurement_watch(label: str, **kwargs) -> MeasurementWatch | None:
    """A started watch when a campaign guard is installed, else None (unit tests,
    standalone tools: no campaign root, nothing to judge contention against)."""
    guard = _INSTALLED
    if guard is None or guard.store is None:
        return None
    return MeasurementWatch(guard, label, **kwargs).start()


__all__ = ["CONTENTION", "ENV_SCOPE", "FOREIGN", "Guard", "KILLABLE", "MeasurementWatch",
           "NEVER_SWEEP_EXE_PREFIXES", "OWN", "Proc", "SCHEMA", "campaign_key", "current",
           "install", "installed", "measurement_watch", "parse_scope", "read_proc",
           "read_scope", "read_stat", "snapshot", "uninstall"]
