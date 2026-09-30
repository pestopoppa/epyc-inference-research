"""AutoKernel CPU-window gate, VENDORED from the champion sidecar (fail-closed).

Source: ``/mnt/raid0/llm/tmp/champion-sidecar/window_gate.py`` (workspace-8d, 2026-09-29),
sha256 ``2e9460c3f8377ec593df9bc9c3f0b0728418c9d2a68b2aea59cb1a5824b6675b`` (``SOURCE_SHA256``,
also recorded in every run manifest). Vendored rather than imported by path because ``/tmp``
paths rot: an import-by-path would make a future RI-18 resume depend on a scratch file that
nobody owns. The rules below are the source's rules unchanged; only the CLI became a function
(``evaluate`` is pure, ``check`` reads the file), and an unreadable or malformed window file is a
REFUSAL (the source returned exit 2 for it, which its callers also treated as "do not start").

Rules (reference_autokernel_cpu_window; operator ruling 6, 2026-09-29):
  * state must be "open" ("closing" = wrap up, "closed" = no) -- mid-run included;
  * loop_holds_claim must be false;
  * now < expires_at, else the loop's heartbeat is stale (loop dead) -> refuse;
  * est_close_at - now >= need_s (a job starts only if it fits; skipped when mid_run);
  * cpus_reserved_by_loop must not intersect our cpuset.
The window file is timing advice; the flock is the truth. Held region locks are reported for
the record, not gated on (API traffic takes them per call).

ANNOUNCED DS41 PAUSE (opt-in; source updated 2026-09-30 to sha ``SOURCE_SHA256`` below, whose default
rules are byte-for-byte those of the earlier ``2e9460c3...`` = ``DEFAULT_RULES_SOURCE_SHA256``). Only with
``allow_announced_pause=True`` (``run_ri18 --allow-announced-pause`` or env
``WS8D_ALLOW_ANNOUNCED_PAUSE=1``) a window the rules above refuse is still ADMITTED, as mode
``announced-pause``, when ALL of these hold (``evaluate_pause``):
  * ``PAUSE_FILE`` is a JSON object written by the main session with ``by == "workspace-76"``,
    parseable ``announced_at`` / ``expected_end``, ``expected_end`` in the future and at most 24 h after
    ``announced_at``; ``need_s`` fits before ``expected_end`` (skipped when mid_run);
  * the window reads state=closed, loop_holds_claim=false, phase=exited, closing_reason=loop_exit
    (what ``CpuWindow.finalize`` writes when the DS41 loop exits), and that exit is >= ``SETTLE_S`` old;
  * the window's owner PID is not alive (``/proc``; boot_id and start_ticks guard PID reuse);
  * no DS41 serial parent is alive (``<campaign>/state*/launcher.pid``) unless its fresh
    ``loop-status.json`` says the serial control plane is paused (desired = observed = paused);
  * the cpuset intersects neither ``cpus_reserved_by_loop`` nor ``avoid_cpus`` / the file's ``avoid_cpus``.
A DS41 resume (a new batch publishes phase=startup with the claim; a resumed serial parent reports
running) refuses the next check, so a CPU segment stops before its next item (RC_WINDOW). Verdicts
carry ``mode`` (open | announced-pause) only when the opt-in is on; without it everything is unchanged.

RI-18 cpuset choice: the driver process does no CPU compute of its own; its CPU work runs inside
the shared llama-server :8070 (and the :8090-8095 embedders), whose cores are governed by the
per-call region claim (the flock). The driver therefore passes an EMPTY cpuset by default (no
overlap test) and records ``cpus_reserved_by_loop`` in every verdict; ``--cpuset`` restores the
source's overlap test.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
from typing import Any

WINDOW = "/mnt/raid0/llm/autokernel/cpu-window.json"
SOURCE_PATH = "/mnt/raid0/llm/tmp/champion-sidecar/window_gate.py"
SOURCE_SHA256 = "492330057ee29487e7768495ce9430d4ed5b381ae6286e3775c57249ecda88ad"
DEFAULT_RULES_SOURCE_SHA256 = "2e9460c3f8377ec593df9bc9c3f0b0728418c9d2a68b2aea59cb1a5824b6675b"
SCHEMA = "epyc.autokernel.cpu_window.v1"
ALLOWED_STATES = ("open",)
PAUSE_FILE = "/mnt/raid0/llm/tmp/sequencer-8d/DS41_PAUSE_ANNOUNCED.json"
PAUSE_ENV = "WS8D_ALLOW_ANNOUNCED_PAUSE"
PAUSE_BY = "workspace-76"
PAUSE_MAX_SPAN_S = 24 * 3600
SETTLE_S = 600.0
CAMPAIGNS = "/mnt/raid0/llm/autokernel/campaigns"


def parse_cpus(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def held_region_locks() -> list[str]:
    try:
        with open("/proc/locks", encoding="utf-8") as handle:
            locks = handle.read()
    except OSError:
        return []
    held = []
    for path in sorted(glob.glob("/mnt/raid0/llm/tmp/cpu_region.*.lock")):
        try:
            ino = os.stat(path).st_ino
        except OSError:
            continue
        if f":{ino} " in locks:
            held.append(os.path.basename(path))
    return held


def _boot_id() -> str | None:
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _start_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stat = handle.read()
        return int(stat[stat.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _start_epoch(pid: int) -> float | None:
    ticks = _start_ticks(pid)
    if ticks is None:
        return None
    try:
        with open("/proc/stat", encoding="utf-8") as handle:
            btime = next(int(line.split()[1]) for line in handle if line.startswith("btime "))
    except (OSError, ValueError, StopIteration):
        return None
    return btime + ticks / os.sysconf("SC_CLK_TCK")


def owner_alive(win: dict[str, Any]) -> tuple[bool, str]:
    """Is the window's owner (the DS41 loop process that wrote it) still running?"""
    owner = win.get("owner") if isinstance(win.get("owner"), dict) else {}
    pid = owner.get("pid", win.get("owner_pid"))
    if pid is None:
        return False, "no owner pid in the window file"
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return True, f"owner pid {pid!r} unparseable (fail-closed)"
    boot = owner.get("boot_id")
    if boot and _boot_id() and boot != _boot_id():
        return False, f"owner pid {pid} is from another boot"
    if not os.path.isdir(f"/proc/{pid}"):
        return False, f"owner pid {pid} not running"
    want = owner.get("start_ticks")
    if want is not None:
        got = _start_ticks(pid)
        if got is not None and got != int(want):
            return False, f"owner pid {pid} reused (start_ticks {got} != {want})"
    return True, f"owner pid {pid} is RUNNING"


def _campaign_root(win: dict[str, Any]) -> str | None:
    path = win.get("campaign_path")
    if path:
        return os.path.dirname(os.path.dirname(path))       # <campaign>/store/cpu-window.json
    if win.get("campaign"):
        return os.path.join(CAMPAIGNS, str(win["campaign"]))
    return None


def serial_parents(win: dict[str, Any], now: dt.datetime) -> tuple[list[str], list[dict[str, Any]]]:
    """Live DS41 serial parents (launcher.pid) that are NOT control-paused -> refusal reasons."""
    root = _campaign_root(win)
    reasons: list[str] = []
    seen: list[dict[str, Any]] = []
    if not root or not os.path.isdir(root):
        return [f"campaign dir {root!r} not found (cannot prove no DS41 serial parent runs)"], seen
    for pidfile in sorted(glob.glob(os.path.join(root, "state*", "launcher.pid"))):
        try:
            with open(pidfile, encoding="utf-8") as handle:
                pid = int("".join(ch for ch in handle.read() if ch.isdigit()) or "0")
            written = os.stat(pidfile).st_mtime
        except (OSError, ValueError):
            continue
        if pid <= 0 or not os.path.isdir(f"/proc/{pid}"):
            continue
        started = _start_epoch(pid)
        if started is not None and started > written + 5:
            continue                                        # PID reused after the pidfile
        state_dir = os.path.dirname(pidfile)
        entry: dict[str, Any] = {"state_dir": state_dir, "pid": pid, "control": None}
        seen.append(entry)
        control: dict[str, Any] = {}
        try:
            with open(os.path.join(state_dir, "loop-status.json"), encoding="utf-8") as handle:
                status = json.load(handle)
            control = status.get("serial_control") or {}
            entry["control"] = {k: control.get(k) for k in ("desired_state", "observed_state")}
            generated = status.get("generated_at")
            gen = (dt.datetime.fromtimestamp(float(generated), dt.timezone.utc)
                   if isinstance(generated, (int, float)) else ts(generated))
            fresh = gen is not None and (now - gen).total_seconds() <= float(
                status.get("stale_after_s") or 180)
        except (OSError, ValueError, TypeError, AttributeError):
            fresh = False
        if not (fresh and control.get("desired_state") == "paused"
                and control.get("observed_state") == "paused"):
            reasons.append(f"DS41 serial parent pid {pid} ({state_dir}) is alive and not "
                           f"control-paused (control={entry['control']}, status fresh={fresh})")
    return reasons, seen


def read_pause(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
    except FileNotFoundError:
        return None, f"no pause file {path}"
    except (OSError, ValueError) as exc:
        return None, f"pause file {path} unreadable: {exc}"
    if not isinstance(doc, dict):
        return None, f"pause file {path} is not a JSON object"
    return doc, None


def evaluate_pause(win: dict[str, Any], *, now: dt.datetime, need_s: float, mid_run: bool,
                   cpuset: str, avoid_cpus: str = "", pause_path: str = PAUSE_FILE,
                   settle_s: float = SETTLE_S) -> tuple[list[str], dict[str, Any]]:
    """Refusal reasons for the announced-pause mode (empty = admitted) and the evidence."""
    reasons: list[str] = []
    info: dict[str, Any] = {"pause_file": pause_path, "settle_s": settle_s}
    pause, err = read_pause(pause_path)
    remaining = None
    if err:
        reasons.append(err)
    else:
        assert pause is not None
        info["pause"] = {k: pause.get(k) for k in ("announced_at", "by", "expected_end", "note",
                                                   "avoid_cpus")}
        if pause.get("by") != PAUSE_BY:
            reasons.append(f"pause file by={pause.get('by')!r} (need {PAUSE_BY!r})")
        try:
            announced, end = ts(pause.get("announced_at")), ts(pause.get("expected_end"))
        except (TypeError, ValueError) as exc:
            announced = end = None
            reasons.append(f"pause file timestamps malformed: {exc}")
        if announced is None or end is None:
            reasons.append("pause file needs announced_at and expected_end")
        else:
            remaining = (end - now).total_seconds()
            if remaining <= 0:
                reasons.append(f"pause expected_end {pause.get('expected_end')} has passed")
            if (end - announced).total_seconds() > PAUSE_MAX_SPAN_S:
                reasons.append("pause expected_end is more than 24 h after announced_at")
            if not mid_run and remaining < need_s:
                reasons.append(f"only {round(remaining)} s to pause expected_end, need {round(need_s)} s")
    info["remaining_s"] = remaining
    if win.get("state") != "closed":
        reasons.append(f"state={win.get('state')!r} (announced pause needs 'closed')")
    if win.get("loop_holds_claim") is not False:
        reasons.append(f"loop_holds_claim={win.get('loop_holds_claim')!r}")
    if win.get("phase") != "exited":
        reasons.append(f"phase={win.get('phase')!r} (announced pause needs 'exited': the loop is not running)")
    if win.get("closing_reason") != "loop_exit":
        reasons.append(f"closing_reason={win.get('closing_reason')!r} (need 'loop_exit')")
    try:
        exited_at = ts(win.get("phase_started_at")) or ts(win.get("expires_at"))
    except (TypeError, ValueError):
        exited_at = None
    age = (now - exited_at).total_seconds() if exited_at else None
    info["exited_age_s"] = age
    if age is None or age < settle_s:
        reasons.append(f"loop exit is {None if age is None else round(age)} s old, need {round(settle_s)} s")
    alive, detail = owner_alive(win)
    info["owner"] = detail
    if alive:
        reasons.append(detail)
    parent_reasons, parents = serial_parents(win, now)
    info["serial_parents_alive"] = parents
    reasons.extend(parent_reasons)
    try:
        avoid = parse_cpus(avoid_cpus) | parse_cpus((pause or {}).get("avoid_cpus", ""))
        overlap = sorted((parse_cpus(win.get("cpus_reserved_by_loop", "")) | avoid)
                         & parse_cpus(cpuset))
    except ValueError as exc:
        overlap = []
        reasons.append(f"malformed cpu list: {exc}")
    if overlap:
        reasons.append(f"cpuset {cpuset} overlaps loop-reserved/avoided cpus {overlap}")
    return reasons, info


def evaluate(win: dict[str, Any], *, now: dt.datetime, need_s: float, cpuset: str = "",
             mid_run: bool = False, allow_announced_pause: bool = False,
             pause_path: str = PAUSE_FILE, avoid_cpus: str = "",
             settle_s: float = SETTLE_S) -> dict[str, Any]:
    """The source's verdict logic, as a pure function of the window document.

    With ``allow_announced_pause`` a refusal is re-evaluated by ``evaluate_pause`` (which also reads
    the pause file and /proc); the verdict then carries ``mode`` (open | announced-pause | None).
    """
    reasons: list[str] = []
    try:
        expires = ts(win.get("expires_at"))
        close = ts(win.get("est_close_at"))
    except (TypeError, ValueError) as exc:
        expires = close = None
        reasons.append(f"malformed timestamps: {exc}")
    remaining = (close - now).total_seconds() if close else None
    try:
        overlap = sorted(parse_cpus(win.get("cpus_reserved_by_loop", "")) & parse_cpus(cpuset))
    except ValueError as exc:
        overlap = []
        reasons.append(f"malformed cpu list: {exc}")
    if win.get("schema") != SCHEMA:
        reasons.append(f"unexpected schema {win.get('schema')!r}")
    if win.get("state") not in ALLOWED_STATES:
        reasons.append(f"state={win.get('state')!r} (need one of {ALLOWED_STATES})")
    if win.get("loop_holds_claim") is not False:
        reasons.append(f"loop_holds_claim={win.get('loop_holds_claim')!r}")
    if expires is None or now >= expires:
        reasons.append(f"stale window: now >= expires_at {win.get('expires_at')}")
    if not mid_run and (remaining is None or remaining < need_s):
        reasons.append(
            f"only {remaining if remaining is None else round(remaining)} s to est_close_at "
            f"{win.get('est_close_at')}, need {round(need_s)} s"
        )
    if overlap:
        reasons.append(f"cpuset {cpuset} overlaps loop-reserved cpus {overlap}")
    verdict = {
        "ok": not reasons,
        "checked_at": now.isoformat(),
        "state": win.get("state"),
        "phase": win.get("phase"),
        "loop_holds_claim": win.get("loop_holds_claim"),
        "est_close_at": win.get("est_close_at"),
        "remaining_s": remaining,
        "need_s": need_s,
        "cpus_reserved_by_loop": win.get("cpus_reserved_by_loop"),
        "reasons": reasons,
    }
    if not allow_announced_pause:
        return verdict
    if win.get("schema") != SCHEMA:
        pause_reasons, info = [f"unexpected schema {win.get('schema')!r}"], {}
    elif not reasons:
        pause_reasons, info = ["not evaluated (open admitted)"], {}
    else:
        pause_reasons, info = evaluate_pause(win, now=now, need_s=need_s, mid_run=mid_run,
                                             cpuset=cpuset, avoid_cpus=avoid_cpus,
                                             pause_path=pause_path, settle_s=settle_s)
    mode = "open" if not reasons else ("announced-pause" if not pause_reasons else None)
    verdict.update(ok=mode is not None, mode=mode, allow_announced_pause=True,
                   open_reasons=reasons, announced_pause={"reasons": pause_reasons, **info},
                   reasons=[] if mode else [f"open: {r}" for r in reasons]
                   + [f"announced-pause: {r}" for r in pause_reasons])
    if mode == "announced-pause":
        verdict["pause_remaining_s"] = info.get("remaining_s")
    return verdict


def check(need_s: float, *, window_path: str = WINDOW, cpuset: str = "",
          mid_run: bool = False, now: dt.datetime | None = None,
          allow_announced_pause: bool = False, pause_path: str = PAUSE_FILE,
          avoid_cpus: str = "", settle_s: float | None = None) -> dict[str, Any]:
    """Read the window file and evaluate it. Unreadable or malformed = refused (fail-closed)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        with open(window_path, encoding="utf-8") as handle:
            win = json.load(handle)
        if not isinstance(win, dict):
            raise ValueError("window document is not an object")
    except (OSError, ValueError) as exc:
        return {"ok": False, "checked_at": now.isoformat(), "need_s": need_s,
                "reasons": [f"GATE ERROR: cannot read {window_path}: {exc}"]}
    if settle_s is None:
        settle_s = float(os.environ.get("WS8D_PAUSE_SETTLE_S", SETTLE_S))
    verdict = evaluate(win, now=now, need_s=need_s, cpuset=cpuset, mid_run=mid_run,
                       allow_announced_pause=allow_announced_pause, pause_path=pause_path,
                       avoid_cpus=avoid_cpus, settle_s=settle_s)
    verdict["held_region_locks"] = held_region_locks()
    return verdict
