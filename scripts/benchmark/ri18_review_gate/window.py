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
SOURCE_SHA256 = "2e9460c3f8377ec593df9bc9c3f0b0728418c9d2a68b2aea59cb1a5824b6675b"
SCHEMA = "epyc.autokernel.cpu_window.v1"
ALLOWED_STATES = ("open",)


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


def evaluate(win: dict[str, Any], *, now: dt.datetime, need_s: float, cpuset: str = "",
             mid_run: bool = False) -> dict[str, Any]:
    """The source's verdict logic, as a pure function of the window document."""
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
    return {
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


def check(need_s: float, *, window_path: str = WINDOW, cpuset: str = "",
          mid_run: bool = False, now: dt.datetime | None = None) -> dict[str, Any]:
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
    verdict = evaluate(win, now=now, need_s=need_s, cpuset=cpuset, mid_run=mid_run)
    verdict["held_region_locks"] = held_region_locks()
    return verdict
