#!/usr/bin/env python3
"""Gate CPU-leg work on the AutoKernel CPU window (/mnt/raid0/llm/autokernel/cpu-window.json).

AutoKernel's campaign holds q0-q3 (cpus 0-95) and is CPU-only. Its SMT siblings (96-191) are the same
physical cores, so there is NO non-overlapping NUMA budget while the campaign runs. CPU-leg work
therefore starts only when:
  * state == "open", and the heartbeat is fresh (expires_at in the future);
  * est_close_at is known and leaves need + margin minutes (est_close_at = null means NO: an open
    window with an unknown close time is not ours to fill with a multi-minute prefill);
  * and then region-lock still has to ACQUIRE the regions (the flock is the truth; this gate is advisory).
A NUMA budget (--budget-cpus) is honoured only if it does not overlap the window's cpu_list or
cpus_reserved_by_loop while the window is closed.

    cpu_window_gate.py --need-min 30 [--wait] [--margin-min 5] [--budget-cpus 72-95]
exit 0 = go, 1 = no (reason printed), with --wait it polls every 60 s until go.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

WINDOW = Path("/mnt/raid0/llm/autokernel/cpu-window.json")


def _ts(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _cpus(spec: str | None) -> set[int]:
    out: set[int] = set()
    for part in (spec or "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return out


def check(need_min: float, margin_min: float = 5.0, budget_cpus: str | None = None) -> tuple[bool, str, dict]:
    try:
        w = json.loads(WINDOW.read_text())
    except (OSError, ValueError) as e:
        return False, f"window file unreadable: {e}", {}
    now = dt.datetime.now(dt.timezone.utc)
    exp = _ts(w.get("expires_at"))
    fresh = exp is not None and exp > now
    info = {"state": w.get("state"), "est_close_at": w.get("est_close_at"), "expires_at": w.get("expires_at"),
            "phase": w.get("phase_label"), "cpu_list": w.get("cpu_list"), "fresh": fresh}
    if budget_cpus:
        held = _cpus(w.get("cpu_list")) | _cpus(w.get("cpus_reserved_by_loop"))
        mine = _cpus(budget_cpus)
        sib = {c + 96 if c < 96 else c - 96 for c in mine}
        if not (mine | sib) & held:
            return True, f"budget {budget_cpus} (and SMT siblings) does not overlap AutoKernel {w.get('cpu_list')}", info
    if w.get("state") != "open":
        return False, f"window {w.get('state')} ({w.get('phase_label')})", info
    if not fresh:
        return False, "window heartbeat stale (expires_at passed): treat as closed", info
    close = _ts(w.get("est_close_at"))
    if close is None:
        return False, "window open but est_close_at is null: not starting a multi-minute step", info
    left = (close - now).total_seconds() / 60
    info["minutes_left"] = round(left, 1)
    if left < need_min + margin_min:
        return False, f"window closes in {left:.0f} min < need {need_min:.0f} + margin {margin_min:.0f}", info
    return True, f"window open, {left:.0f} min left >= {need_min:.0f} + {margin_min:.0f}", info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-min", type=float, required=True)
    ap.add_argument("--margin-min", type=float, default=5.0)
    ap.add_argument("--budget-cpus")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    while True:
        ok, why, info = check(a.need_min, a.margin_min, a.budget_cpus)
        print(json.dumps({"go": ok, "why": why, **info}), flush=True)
        if ok or not a.wait:
            return 0 if ok else 1
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
