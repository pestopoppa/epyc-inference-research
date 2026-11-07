#!/usr/bin/env python3
"""Per-stage wall timing of ONE AutoKernel ITERATION, tail included (observability only).

WHY. AK-TAIL-SPLIT asks whether splitting the serialized tail (pipeline.SerializedTail:
build, op oracle, A/B, commit) pays off: builds and correctness oracles concurrent, only
the quiet timing slot and the commit serialized. Answering that needs what the store
does not record today: how long each lane WAITS for the tail, how much tail time goes
to candidates that then fail the build or correctness, and how the device-held time
splits between correctness and timing. Sibling of `keep_timing` (keep path only).

WHAT. `IterationTimer` collects, per pool draw:
  * actor stages from the `on_step` labels `loop.iterate` already emits
    (propose / critic1 / author / critic2), each closed by the next label;
  * per tail SESSION (one per patch round that reaches the tail): `wait_s`
    ("ready for tail" -> lock acquired), `held_s`, and inside it `gate` (with the
    verdicts' own `elapsed_s` split into build vs correctness), `measure` (A/B), `commit`;
  * the terminal outcome and `died_at` (the tail stage that ended the candidate).
`pipeline.run_pool` sets the dict on the outcome as attribute `ATTR`; `run.py` copies it
into the experiment row payload as the additive field `FIELD`. Every method swallows its
own faults; a wrapped call that raises is recorded with `error` and re-raised unchanged.

`summarize` / `main` is the read-only report over a store's experiment rows.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Iterable, Iterator, Mapping

SCHEMA = "epyc.autokernel.tail_path_timing.v1"
#: Payload key on every experiment row a pool draw produces.
FIELD = "tail_path_timing"
#: Attribute `pipeline.run_pool` sets on the Outcome; `run.py` copies it to FIELD.
ATTR = "tail_path_timing"
#: Same set as `history_fields.BUILD_VERDICT_GATES`: verdicts that are the build.
BUILD_VERDICT_GATES = frozenset({"configure", "compile"})
#: Verdict gates that are pre-build deterministic screens, neither build nor oracle.
PREBUILD_VERDICT_GATES = frozenset({"op_scope", "cpu_screen", "runtime_treatment",
                                    "runtime_preparation"})

#: on_step label prefix -> actor stage. A label opening "ready" closes the actor phase.
_LABELS = (("proposing", "propose"), ("degraded mode", "propose"),
           ("prevalidated runtime option", "critic1"), ("critic pass 1", "critic1"),
           ("restoring the retained patch", "author"), ("authoring", "author"),
           ("critic pass 2", "critic2"),
           ("building and gating", "ready_for_tail"),
           ("checking runtime treatment", "ready_for_tail"))


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def classify_label(label: str) -> str | None:
    text = str(label or "")
    for prefix, stage in _LABELS:
        if text.startswith(prefix):
            return stage
    return None


def verdict_split(verdicts: Iterable[Any]) -> dict[str, float]:
    """Seconds by class from the verdicts' own `elapsed_s` (gates.run_all)."""
    out = {"build_s": 0.0, "correctness_s": 0.0, "prebuild_s": 0.0}
    for v in verdicts or ():
        gate = getattr(v, "gate", None)
        sec = getattr(v, "elapsed_s", None)
        if not isinstance(sec, (int, float)):
            continue
        key = ("build_s" if gate in BUILD_VERDICT_GATES else
               "prebuild_s" if gate in PREBUILD_VERDICT_GATES else "correctness_s")
        out[key] += float(sec)
    return {k: round(v, 3) for k, v in out.items()}


def died_class(gate: str | None) -> str:
    if gate in BUILD_VERDICT_GATES:
        return "build_fail"
    if gate in PREBUILD_VERDICT_GATES:
        return "prebuild_fail"
    return "correctness_fail"


class IterationTimer:
    """Stage records for ONE pool draw on ONE lane thread. Never raises from bookkeeping."""

    def __init__(self, *, lane: str | None = None, clock=time.monotonic, wall=_iso) -> None:
        self._clock, self._wall = clock, wall
        self.lane = lane
        self.t0 = self._now()
        self.actor: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        self._open: dict[str, Any] | None = None
        self._ready_at: float | None = None
        self._session: dict[str, Any] | None = None
        self.died_at: str | None = None

    def _now(self) -> float:
        try:
            return float(self._clock())
        except Exception:      # noqa: BLE001
            return 0.0

    def _close_actor(self, now: float) -> None:
        if self._open is not None:
            self._open["end"] = round(now - self.t0, 3)
            self._open["duration_s"] = round(now - self.t0 - self._open["start"], 3)
            self.actor.append(self._open)
            self._open = None

    def step(self, label: str) -> None:
        try:
            stage = classify_label(label)
            if stage is None:
                return
            now = self._now()
            self._close_actor(now)
            if stage == "ready_for_tail":
                self._ready_at = now
            else:
                self._open = {"stage": stage, "start": round(now - self.t0, 3),
                              "started_at": self._wall()}
        except Exception:      # noqa: BLE001
            pass

    @contextmanager
    def tail(self) -> Iterator[None]:
        """Wrap the tail session's acquisition AND body: wait = acquire - ready."""
        requested = self._now()
        try:
            self._close_actor(requested)
        except Exception:      # noqa: BLE001
            pass
        ready = self._ready_at if self._ready_at is not None else requested
        rec: dict[str, Any] = {"ready": round(ready - self.t0, 3), "stages": []}
        self._ready_at = None
        try:
            self.sessions.append(rec)
        except Exception:      # noqa: BLE001
            pass
        try:
            yield
        except BaseException as exc:
            if "acquired" not in rec:
                rec["refused"] = type(exc).__name__
            raise
        finally:
            try:
                end = self._now()
                rec["end"] = round(end - self.t0, 3)
                if "acquired" in rec:
                    rec["held_s"] = round(end - self.t0 - rec["acquired"], 3)
                    rec["wait_s"] = round(rec["acquired"] - rec["ready"], 3)
                else:
                    rec["wait_s"] = round(end - self.t0 - rec["ready"], 3)
            except Exception:      # noqa: BLE001
                pass

    def acquired(self) -> None:
        """Called once the tail lock is held (inside the session)."""
        try:
            if self.sessions:
                rec = self.sessions[-1]
                rec["acquired"] = round(self._now() - self.t0, 3)
        except Exception:      # noqa: BLE001
            pass

    def _stage_rec(self, name: str) -> dict[str, Any]:
        rec = {"stage": name, "start": round(self._now() - self.t0, 3)}
        try:
            if not self.sessions:
                self.sessions.append({"ready": rec["start"], "stages": [],
                                      "outside_session": True})
            self.sessions[-1]["stages"].append(rec)
        except Exception:      # noqa: BLE001
            pass
        return rec

    def _finish(self, rec: dict[str, Any]) -> None:
        try:
            rec["end"] = round(self._now() - self.t0, 3)
            rec["duration_s"] = round(rec["end"] - rec["start"], 3)
        except Exception:      # noqa: BLE001
            pass

    def timed(self, name: str, fn, *args):
        """Run `fn(*args)` as tail stage `name`; for `gate`, split by verdicts."""
        rec = self._stage_rec(name)
        try:
            result = fn(*args)
        except BaseException as exc:
            rec["error"] = type(exc).__name__
            self._finish(rec)
            if self.died_at is None:
                self.died_at = f"{name}_error"
            raise
        self._finish(rec)
        if name == "gate":
            try:
                passed, verdicts = result
                rec.update(verdict_split(verdicts))
                rec["verdicts"] = [[getattr(v, "gate", None), bool(getattr(v, "passed", False)),
                                    getattr(v, "elapsed_s", None)] for v in verdicts]
                rec["passed"] = bool(passed)
                if not passed:
                    failing = getattr(verdicts[-1], "gate", None) if verdicts else None
                    rec["died_at"] = died_class(failing)
                    rec["failing_gate"] = failing
                    self.died_at = rec["died_at"]
                else:
                    self.died_at = None
            except Exception:      # noqa: BLE001
                pass
        return result

    def to_dict(self, outcome_status: str | None = None) -> dict[str, Any]:
        try:
            self._close_actor(self._now())
            total = round(self._now() - self.t0, 3)
            sessions = [s for s in self.sessions if isinstance(s, dict)]
            died = self.died_at
            if outcome_status == "superseded" and sessions and "acquired" not in sessions[-1]:
                died = "superseded_at_tail_entry"
            return {"schema": SCHEMA, "lane": self.lane, "total_s": total,
                    "outcome": outcome_status, "died_at": died,
                    "actor": list(self.actor), "tail_sessions": sessions}
        except Exception as exc:      # noqa: BLE001
            return {"schema": SCHEMA, "error": f"{type(exc).__name__}: {exc}"}


@contextmanager
def timed_session(timer: "IterationTimer | None", session_cm) -> Iterator[Any]:
    """Enter the real tail session inside `timer.tail()`; mark acquisition on entry.
    With no timer, exactly `with session_cm:`."""
    if timer is None:
        with session_cm as value:
            yield value
        return
    with timer.tail():
        with session_cm as value:
            timer.acquired()
            yield value


# ---- report ------------------------------------------------------------------------

def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "total_s": 0.0, "median_s": None, "p90_s": None}
    v = sorted(values)
    p90 = v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))]
    return {"n": len(v), "total_s": round(sum(v), 3),
            "median_s": round(statistics.median(v), 3), "p90_s": round(p90, 3)}


def summarize(payloads: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    per: dict[str, list[float]] = {}
    died: dict[str, int] = {}
    timed = untimed = 0
    freed = {"correctness_fail_held_s": 0.0, "build_fail_held_s": 0.0,
             "prebuild_fail_held_s": 0.0}
    device = {"correctness_s": 0.0, "timing_s": 0.0}

    def add(name, val):
        if isinstance(val, (int, float)):
            per.setdefault(name, []).append(float(val))

    for payload in payloads:
        t = payload.get(FIELD) if isinstance(payload, Mapping) else None
        if not isinstance(t, Mapping) or not isinstance(t.get("tail_sessions"), list):
            untimed += 1
            continue
        timed += 1
        key = str(t.get("died_at") or t.get("outcome") or "unknown")
        died[key] = died.get(key, 0) + 1
        for a in t.get("actor") or ():
            if isinstance(a, Mapping):
                add(f"actor:{a.get('stage')}", a.get("duration_s"))
        for s in t["tail_sessions"]:
            if not isinstance(s, Mapping):
                continue
            add("tail_wait", s.get("wait_s"))
            add("tail_held", s.get("held_s"))
            for st in s.get("stages") or ():
                if not isinstance(st, Mapping):
                    continue
                add(f"tail:{st.get('stage')}", st.get("duration_s"))
                if st.get("stage") == "gate":
                    add("tail:gate.build", st.get("build_s"))
                    add("tail:gate.correctness", st.get("correctness_s"))
                    if isinstance(st.get("correctness_s"), (int, float)):
                        device["correctness_s"] += float(st["correctness_s"])
                    cls = st.get("died_at")
                    if cls and isinstance(s.get("held_s"), (int, float)):
                        freed[f"{cls}_held_s"] = freed.get(f"{cls}_held_s", 0.0) + float(s["held_s"])
                elif st.get("stage") == "measure" and isinstance(st.get("duration_s"), (int, float)):
                    device["timing_s"] += float(st["duration_s"])
    stages = {name: _dist(v) for name, v in per.items()}
    return {"schema": SCHEMA + ".report", "iterations_timed": timed,
            "iterations_untimed": untimed,
            "tail_wait_total_s": stages.get("tail_wait", {}).get("total_s", 0.0),
            "tail_held_on_failed_candidates": {k: round(v, 3) for k, v in freed.items()},
            "device_held_split": {k: round(v, 3) for k, v in device.items()},
            "died_at": died, "stages": stages}


def read_store(store_root: Path) -> list[Mapping[str, Any] | None]:
    """Read-only: every experiment row's decoded payload from `experiments.db`."""
    db = Path(store_root) / "experiments.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT payload FROM experiments ORDER BY recorded_at, rowid"
                            ).fetchall()
    finally:
        conn.close()
    out: list[Mapping[str, Any] | None] = []
    for (raw,) in rows:
        try:
            out.append(json.loads(raw))
        except (TypeError, ValueError):
            out.append(None)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize iteration/tail stage timing (read-only).")
    parser.add_argument("store", type=Path, help="AutoKernel store root (holds experiments.db)")
    parser.add_argument("--json", action="store_true", help="print the JSON report")
    args = parser.parse_args(argv)
    r = summarize(read_store(args.store))
    if args.json:
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0
    print(f"iterations timed {r['iterations_timed']}, untimed {r['iterations_untimed']}, "
          f"tail wait {r['tail_wait_total_s'] / 60:.1f} min")
    f = r["tail_held_on_failed_candidates"]
    print(f"  tail held on candidates that failed: correctness "
          f"{f['correctness_fail_held_s'] / 60:.1f} min, build {f['build_fail_held_s'] / 60:.1f} min")
    d = r["device_held_split"]
    print(f"  device-held: correctness {d['correctness_s'] / 60:.1f} min, "
          f"timing {d['timing_s'] / 60:.1f} min")
    print(f"  died_at: {r['died_at']}")
    for name, s in sorted(r["stages"].items(), key=lambda kv: -kv[1]["total_s"]):
        print(f"  {name:24s} n={s['n']:<4d} total={s['total_s'] / 60:8.1f} min "
              f"median={s['median_s']} s p90={s['p90_s']} s")
    return 0


__all__ = ["ATTR", "FIELD", "SCHEMA", "IterationTimer", "classify_label", "main",
           "read_store", "summarize", "timed_session", "verdict_split"]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
