"""RI-18 per-item stages, decoupled into resumable segments by DEVICE.

| segment        | design stage                         | device                   | window gate        |
|----------------|--------------------------------------|--------------------------|--------------------|
| answer         | 1 (A1 via the API) + 2 (quality mark) | CPU :8070 through the API | AutoKernel CPU window |
| verdict        | 3 (production cap) + 5 (V-full, S1)  | GPU :8083, one slot      | none; :8083 preflight |
| revise         | 4 (WRONG only)                       | CPU :8070 in-process     | AutoKernel CPU window |
| noise-verdict  | control: verdict re-run, seeded 50   | GPU :8083                | none; :8083 preflight |
| noise-revise   | control: revision re-run, first 30 WRONG | CPU :8070 in-process | AutoKernel CPU window |
| gate           | 6 (offline gate vs the snapshot)     | CPU embedders :8090-8095 | AutoKernel CPU window |

Every segment: resumable per item (a record is written only once the item finished, fsynced),
``--limit N``, a wall budget (stop scheduling after ``budget_s``), and ``--dry-run`` fakes.
CPU segments re-check the window before EVERY item (``need_s`` = the item's planning or
measured p95 time + margin) and stop on anything but a fitting ``open`` window. GPU segments
run the RI-23b canary pair at their start and end; a failure in either direction voids the
segment (its records are re-run on resume).
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..thesis_ufh13.arms import ARMS, GENERATION
from ..thesis_ufh13.records import summarize_receipts
from ..thesis_ufh13.transports import Transport
from . import window as window_gate
from .bridge import VFULL_QUESTION_CAP, Bridge, InstrumentFault, WindowLost, git_state
from .scoring import is_correct
from .store import RunDir

ARM = "A1"
USER_ID = "ri18"
UNAVAILABLE_HALT = 0.05
UNAVAILABLE_MIN_N = 20
NOISE_REVISE_N = 30
GRID = [round(0.30 + 0.05 * i, 2) for i in range(15)] + [1.01]
CONTROL_T = (0.0, 1.01)
PROD_T = 0.6

# Exit codes (distinct, so a wrapper can tell "resume next window" from "fix the instrument").
RC_OK = 0
RC_USAGE = 2
RC_WINDOW = 3        # CPU window closed/closing/claimed/too short, or region claim denied
RC_BUDGET = 4        # wall budget or --limit reached; resume
RC_VOID = 5          # canary failed, gate control failed, served commit / code drift
RC_INSTRUMENT = 6    # unavailable > 5%, dead embedder, flag not ON, no-op region lock
RC_BUSY = 7          # :8083 preflight: every slot processing
RC_PREREQ = 8        # an earlier stage has not produced what this one needs
RC_DRIFT = 9         # run manifest pins differ (refused resume)

# Design §3.1 planning figures, per item, seconds (lo, hi).
PLANNING_S = {
    ("answer", "S1"): (35 * 60 / 453, 50 * 60 / 453),
    ("answer", "S2"): (30 * 60 / 155, 50 * 60 / 155),
    ("verdict", "S1"): (10 * 60 / 608 + 8 * 60 / 453, 15 * 60 / 608 + 12 * 60 / 453),
    ("verdict", "S2"): (10 * 60 / 608, 15 * 60 / 608),
    ("revise", None): (25 * 60 / 300, 45 * 60 / 200),
    ("gate", None): (0.05, 2 * 60 / 608),
    ("noise-verdict", None): (10 * 60 / 608, 15 * 60 / 608),
    ("noise-revise", None): (25 * 60 / 300, 45 * 60 / 200),
}


def planning(kind: str, stratum: str | None) -> tuple[float, float]:
    return PLANNING_S.get((kind, stratum)) or PLANNING_S[(kind, None)]


@dataclass
class Ctx:
    run: RunDir
    items: list[dict[str, Any]]
    workload: dict[str, Any]
    mode: str                           # real | dry | stub
    limit: int | None = None
    budget_s: float = 3600.0
    window_path: str = window_gate.WINDOW
    margin_s: float = 120.0
    cpuset: str = ""
    skip_window: bool = False           # dry-run / tests only
    allow_announced_pause: bool = False  # --allow-announced-pause / WS8D_ALLOW_ANNOUNCED_PAUSE=1
    pause_file: str = window_gate.PAUSE_FILE
    last_window_mode: str | None = None  # the admitting mode last recorded (opt-in only)
    pause_only: bool = False             # WS8D_PAUSE_ONLY=1: only mode=announced-pause admits
    now: Callable[[], Any] | None = None
    log: Callable[[str], None] = print
    t0: float = field(default_factory=time.monotonic)

    @property
    def by_id(self) -> dict[str, dict[str, Any]]:
        return {row["id"]: row for row in self.items}

    def split_of(self, item_id: str) -> str:
        return "A" if item_id in set(self.workload["split"]["A"]) else "B"

    def over_budget(self) -> bool:
        return time.monotonic() - self.t0 > self.budget_s


class Stop(Exception):
    def __init__(self, rc: int, reason: str) -> None:
        super().__init__(reason)
        self.rc = rc
        self.reason = reason


# ── shared helpers ───────────────────────────────────────────────────────────


def measured_p95(records: dict[str, dict[str, Any]], key: str = "wall_s") -> float | None:
    walls = sorted(float(r[key]) for r in records.values() if isinstance(r.get(key), (int, float)))
    if len(walls) < 20:
        return None
    return walls[min(len(walls) - 1, int(0.95 * len(walls)))]


def window_ok(ctx: Ctx, need_s: float) -> dict[str, Any]:
    if ctx.skip_window:
        return {"ok": True, "skipped": True}
    now = ctx.now() if ctx.now else None
    if not ctx.allow_announced_pause:
        return window_gate.check(need_s, window_path=ctx.window_path, cpuset=ctx.cpuset, now=now)
    return window_gate.check(need_s, window_path=ctx.window_path, cpuset=ctx.cpuset, now=now,
                             allow_announced_pause=True, pause_path=ctx.pause_file)


def pause_only_refusal(verdict: dict[str, Any]) -> dict[str, Any]:
    """The pause lane's rule: an `open` window means DS41 is running again, so it does not admit."""
    return {**verdict, "ok": False, "reasons": [
        f"pause-only: admitted as mode={verdict.get('mode')!r}, not 'announced-pause' "
        "(the announced pause is over or DS41 resumed)"]}


def require_window(ctx: Ctx, seg: str, need_s: float) -> None:
    verdict = window_ok(ctx, need_s)
    if verdict["ok"] and ctx.pause_only and verdict.get("mode") != "announced-pause":
        verdict = pause_only_refusal(verdict)
    if not verdict["ok"]:
        ctx.run.segment_event({"event": "window_refused", "segment_id": seg, "verdict": verdict})
        raise Stop(RC_WINDOW, "CPU window refused: " + "; ".join(verdict.get("reasons") or []))
    mode = verdict.get("mode")
    if mode and mode != ctx.last_window_mode:
        # opt-in only: record which mode admitted the segment's items (open | announced-pause)
        ctx.run.segment_event({"event": "window_admitted", "segment_id": seg, "mode": mode,
                               "verdict": verdict})
        ctx.log(f"[{seg}] CPU window admitted (mode={mode})")
        ctx.last_window_mode = mode


def probe_served(base_url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Read-only: the API's launch commit and start time (``/dashboard/api/version``)."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/dashboard/api/version",
                                timeout=timeout) as resp:  # noqa: S310 - localhost
        doc = json.loads(resp.read().decode())
    return {"server_launch_git_sha": doc.get("server_launch_git_sha"),
            "server_started_at": doc.get("server_started_at"), "git_sha": doc.get("git_sha")}


def served_matches(pinned: dict[str, Any], now: dict[str, Any]) -> bool:
    return (now.get("server_launch_git_sha") == pinned.get("server_launch_git_sha")
            and now.get("server_started_at") == pinned.get("server_started_at"))


def _answer_ok(rec: dict[str, Any] | None) -> bool:
    return bool(rec) and rec.get("status") == "ok" and bool(rec.get("answer_text"))


def _finish(ctx: Ctx, seg: str, *, exit_rc: int, reason: str, canaries_ok: bool | None = None,
            code_root: str | None = None, pinned_commit: str | None = None,
            extra: dict[str, Any] | None = None) -> None:
    if code_root and pinned_commit and git_state(code_root)["commit"] != pinned_commit:
        ctx.run.segment_event({"event": "void", "segment_id": seg,
                               "reason": "code_root commit drifted during the segment"})
        exit_rc = RC_VOID
    if canaries_ok is False:
        ctx.run.segment_event({"event": "void", "segment_id": seg,
                               "reason": "canary pair failed (BOUNDED-NULL-1)"})
    ctx.run.segment_event({"event": "end", "segment_id": seg, "exit": exit_rc, "reason": reason,
                           "canaries_ok": canaries_ok, **(extra or {})})


def _todo(ctx: Ctx, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """(items to run now, whether ``--limit`` truncated the list)."""
    if ctx.limit is None or len(candidates) <= ctx.limit:
        return candidates, False
    return candidates[: ctx.limit], True


# ── stage 1 + 2: answer ──────────────────────────────────────────────────────


def session_id(run_id: str, item_id: str) -> str:
    import hashlib
    import re

    digest = hashlib.sha256(item_id.encode()).hexdigest()[:12]
    return f"ri18-{re.sub(r'[^A-Za-z0-9._-]', '-', run_id)}-{ARM}-{digest}"


def v1_body_keys() -> dict[str, Any]:
    return {"x_tool_mode": "client", "x_user_id": USER_ID, "x_show_routing": True,
            **GENERATION, **ARMS[ARM].body_keys}


def seg_answer(ctx: Ctx, bridge: Bridge, transport: Transport, *, suite: str,
               served_probe: Callable[[], dict[str, Any]]) -> int:
    stratum = {"s1": "S1", "s2": "S2"}[suite]
    manifest = ctx.run.manifest()
    seg = ctx.run.new_segment("answer", {"stratum": stratum, "header": bridge.header,
                                         "transport": transport.describe()})
    done = ctx.run.records("answer", current=seg)
    todo, truncated = _todo(ctx, [row for row in ctx.items
                                  if row["stratum"] == stratum and row["id"] not in done])
    ctx.log(f"[answer {stratum}] {len(done)} on disk, {len(todo)} to run (segment {seg})")
    rc, reason, n = RC_OK, "complete", 0
    try:
        # The served COMMIT is pinned run-wide; the start time is this segment's baseline, so a
        # reload (same commit or not) in the middle of the segment voids the segment.
        baseline = served_probe()
        pinned_sha = manifest["served"]["server_launch_git_sha"]
        if baseline.get("server_launch_git_sha") != pinned_sha:
            raise Stop(RC_DRIFT, f"API serves {baseline} but the run pins {pinned_sha}")
        ctx.run.segment_event({"event": "served", "segment_id": seg, "baseline": baseline})
        for row in todo:
            if ctx.over_budget():
                raise Stop(RC_BUDGET, "wall budget spent")
            lo, hi = planning("answer", stratum)
            p95 = measured_p95({k: v for k, v in done.items() if v.get("stratum") == stratum})
            require_window(ctx, seg, max(hi, p95 or 0.0) + ctx.margin_s)
            served = served_probe()
            if not served_matches(baseline, served):
                ctx.run.segment_event({"event": "void", "segment_id": seg,
                                       "reason": f"served API identity changed: {served}"})
                raise Stop(RC_VOID, f"API reloaded mid-segment: {served} != {baseline}")
            started = time.time()
            t0 = time.perf_counter()
            result = transport.ask(ARM, row["id"], row["prompt"],
                                   session_id(manifest["run_id"], row["id"]))
            wall = round(time.perf_counter() - t0, 3)
            if result.status == "http_error" and result.http_status == 503:
                ctx.run.segment_event({"event": "infra", "segment_id": seg, "item_id": row["id"],
                                       "detail": (result.error or "")[:300]})
                raise Stop(RC_WINDOW, f"503 from the API (region contention): {result.error}")
            request_ds = [r.get("request_device_seconds") for r in result.receipts]
            cost_ok = bool(request_ds) and all(isinstance(v, (int, float)) for v in request_ds)
            text = result.text or ""
            record = {
                "segment_id": seg, "item_id": row["id"], "suite": row["suite"],
                "stratum": stratum, "split": ctx.split_of(row["id"]),
                "status": result.status, "answer_text": text, "answer_chars": len(text),
                "finish_reason": result.finish_reason, "http_status": result.http_status,
                "error": result.error, "session_id": result.session_id,
                "served_role": result.served_role, "usage": result.usage,
                "wall_s": wall, "started_at": started,
                "request_device_seconds": round(sum(request_ds), 6) if cost_ok else None,
                "receipts": result.receipts,
                "receipt_summary": summarize_receipts(ARM, result.receipts),
                "quality_issue": bridge.quality(text) if result.status == "ok" else None,
                "correct": result.status == "ok" and is_correct(row, text),
                "served": served,
            }
            ctx.run.append("answer", record)
            done[row["id"]] = record
            n += 1
            ctx.log(f"  [{n}/{len(todo)}] {row['id']} status={result.status} "
                    f"chars={len(text)} correct={record['correct']} wall={wall}s "
                    f"qd={record['quality_issue']}")
        if truncated:
            rc, reason = RC_BUDGET, "--limit reached"
    except Stop as stop:
        rc, reason = stop.rc, stop.reason
    _finish(ctx, seg, exit_rc=rc, reason=reason, extra={"items_run": n})
    return rc


# ── GPU segments: verdict (+V-full), noise-verdict ───────────────────────────


def preflight_8083(gpu_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Refuse while EVERY :8083 slot is processing (RI-23b preflight, all-busy form)."""
    with urllib.request.urlopen(f"{gpu_url}/slots", timeout=timeout) as resp:  # noqa: S310
        slots = json.loads(resp.read().decode())
    busy = [s.get("id") for s in slots if isinstance(s, dict) and s.get("is_processing")]
    return {"n_slots": len(slots), "busy_slots": busy,
            "all_busy": bool(slots) and len(busy) == len(slots)}


def _unavailable_halt(records: list[dict[str, Any]]) -> tuple[bool, float]:
    n = len(records)
    bad = sum(1 for r in records if r.get("status") == "unavailable")
    return bad > UNAVAILABLE_HALT * max(n, UNAVAILABLE_MIN_N), (bad / n if n else 0.0)


def _gpu_segment(ctx: Ctx, bridge: Bridge, kind: str, preflight: Callable[[], dict[str, Any]],
                 body: Callable[[str], tuple[int, str, int]]) -> int:
    pre = preflight()
    if pre.get("all_busy"):
        ctx.log(f"[{kind}] :8083 preflight refused: {pre}")
        return RC_BUSY
    pinned = ctx.run.manifest()["code_root_commit"]
    seg = ctx.run.new_segment(kind, {"header": bridge.header, "preflight": pre})
    start = bridge.canaries()
    ctx.run.segment_event({"event": "canary", "segment_id": seg, "when": "start", **start})
    ctx.log(f"[{kind}] start canaries ok={start['ok']} "
            f"({start['right']['status']}/{start['wrong']['status']})")
    if not start["ok"]:
        _finish(ctx, seg, exit_rc=RC_VOID, reason="start canaries failed", canaries_ok=False,
                code_root=bridge.code_root, pinned_commit=pinned)
        return RC_VOID
    rc, reason, n = RC_OK, "complete", 0
    try:
        rc, reason, n = body(seg)
    except Stop as stop:
        rc, reason = stop.rc, stop.reason
    finally:
        end = bridge.canaries()
        ctx.run.segment_event({"event": "canary", "segment_id": seg, "when": "end", **end})
        ctx.log(f"[{kind}] end canaries ok={end['ok']} "
                f"({end['right']['status']}/{end['wrong']['status']})")
        ok = bool(start["ok"] and end["ok"])
        if not ok:
            rc, reason = RC_VOID, "end canaries failed"
        _finish(ctx, seg, exit_rc=rc, reason=reason, canaries_ok=ok, code_root=bridge.code_root,
                pinned_commit=pinned, extra={"items_run": n})
    return rc


def _verdict_record(seg: str, row: dict[str, Any], split: str, stage_info: dict[str, Any]
                    ) -> dict[str, Any]:
    return {"segment_id": seg, "item_id": row["id"], "suite": row["suite"],
            "stratum": row["stratum"], "split": split, **stage_info}


def seg_verdict(ctx: Ctx, bridge: Bridge, *, preflight: Callable[[], dict[str, Any]]) -> int:
    answers = ctx.run.records("answer")

    def body(seg: str) -> tuple[int, str, int]:
        verdicts = ctx.run.records("verdict", current=seg)
        vfull = ctx.run.records("vfull", current=seg)
        todo, truncated = _todo(ctx, [
            row for row in ctx.items if _answer_ok(answers.get(row["id"]))
            and (row["id"] not in verdicts or (row["stratum"] == "S1" and row["id"] not in vfull))])
        ctx.log(f"[verdict] {len(verdicts)} verdicts / {len(vfull)} V-full on disk, "
                f"{len(todo)} items to run (answered: {len(answers)})")
        n = 0
        for row in todo:
            if ctx.over_budget():
                return RC_BUDGET, "wall budget spent", n
            answer = answers[row["id"]]["answer_text"]
            if row["id"] not in verdicts:
                v = bridge.verdict(row["prompt"], answer)
                rec = _verdict_record(seg, row, ctx.split_of(row["id"]), v)
                ctx.run.append("verdict", rec)
                verdicts[row["id"]] = rec
                halt, rate = _unavailable_halt(list(verdicts.values()))
                if halt:
                    raise Stop(RC_INSTRUMENT, f"verdict unavailable rate {rate:.3f} > 5% "
                                              "(instrument fault, RI-22 regression)")
            if row["stratum"] == "S1" and row["id"] not in vfull:
                v = bridge.verdict(row["prompt"], answer, question_cap=VFULL_QUESTION_CAP)
                rec = _verdict_record(seg, row, ctx.split_of(row["id"]), v)
                ctx.run.append("vfull", rec)
                vfull[row["id"]] = rec
            n += 1
            ctx.log(f"  [{n}/{len(todo)}] {row['id']} verdict={verdicts[row['id']]['status']}"
                    + (f" vfull={vfull[row['id']]['status']}" if row["id"] in vfull else ""))
        if truncated:
            return RC_BUDGET, "--limit reached", n
        return RC_OK, "complete", n

    return _gpu_segment(ctx, bridge, "verdict", preflight, body)


def seg_noise_verdict(ctx: Ctx, bridge: Bridge, *, preflight: Callable[[], dict[str, Any]]) -> int:
    answers = ctx.run.records("answer")
    subset = ctx.workload["noise_subset"]

    def body(seg: str) -> tuple[int, str, int]:
        done = ctx.run.records("noise_verdict", current=seg)
        todo, truncated = _todo(ctx, [ctx.by_id[i] for i in subset
                                      if _answer_ok(answers.get(i)) and i not in done])
        missing = [i for i in subset if not _answer_ok(answers.get(i))]
        ctx.log(f"[noise-verdict] {len(done)} on disk, {len(todo)} to run; "
                f"{len(missing)} subset items have no usable answer yet")
        n = 0
        for row in todo:
            if ctx.over_budget():
                return RC_BUDGET, "wall budget spent", n
            v = bridge.verdict(row["prompt"], answers[row["id"]]["answer_text"])
            ctx.run.append("noise_verdict", _verdict_record(seg, row, ctx.split_of(row["id"]), v))
            n += 1
            ctx.log(f"  [{n}/{len(todo)}] {row['id']} noise verdict={v['status']}")
        if truncated:
            return RC_BUDGET, "--limit reached", n
        return RC_OK, "complete", n

    return _gpu_segment(ctx, bridge, "noise-verdict", preflight, body)


# ── CPU in-process: revise, noise-revise ─────────────────────────────────────


def _cpu_loop(ctx: Ctx, bridge: Bridge, kind: str, stage: str, todo: list[dict[str, Any]],
              make: Callable[[str, dict[str, Any]], dict[str, Any]]) -> int:
    pinned = ctx.run.manifest()["code_root_commit"]
    seg = ctx.run.new_segment(kind, {"header": bridge.header})
    done = ctx.run.records(stage, current=seg)
    todo, truncated = _todo(ctx, [row for row in todo if row["id"] not in done])
    ctx.log(f"[{kind}] {len(done)} on disk, {len(todo)} to run (segment {seg})")
    rc, reason, n = RC_OK, "complete", 0
    try:
        for row in todo:
            if ctx.over_budget():
                raise Stop(RC_BUDGET, "wall budget spent")
            lo, hi = planning(kind, None)
            require_window(ctx, seg, max(hi, measured_p95(done) or 0.0) + ctx.margin_s)
            try:
                rec = make(seg, row)
            except WindowLost as exc:
                ctx.run.segment_event({"event": "infra", "segment_id": seg, "item_id": row["id"],
                                       "detail": f"region claim denied: {exc}"[:300]})
                raise Stop(RC_WINDOW, f"region claim denied: {exc}") from exc
            ctx.run.append(stage, rec)
            done[row["id"]] = rec
            n += 1
            ctx.log(f"  [{n}/{len(todo)}] {row['id']} changed={rec['changed']} "
                    f"failed={rec['revise_failed']} wall={rec['wall_s']}s")
        if truncated:
            rc, reason = RC_BUDGET, "--limit reached"
    except Stop as stop:
        rc, reason = stop.rc, stop.reason
    _finish(ctx, seg, exit_rc=rc, reason=reason, code_root=bridge.code_root,
            pinned_commit=pinned, extra={"items_run": n})
    return rc


def _revision_record(seg: str, ctx: Ctx, row: dict[str, Any], answer: str, verdict: dict[str, Any],
                     rev: dict[str, Any]) -> dict[str, Any]:
    return {"segment_id": seg, "item_id": row["id"], "suite": row["suite"],
            "stratum": row["stratum"], "split": ctx.split_of(row["id"]),
            "verdict_segment_id": verdict["segment_id"], "corrections": verdict["verdict"],
            # Production control flow: _fast_revise returns the original on an empty or failed
            # revision, so its return value IS the final answer.
            "final_text": rev["revised"],
            "correct_final": is_correct(row, rev["revised"]),
            "correct_original": is_correct(row, answer),
            **{k: v for k, v in rev.items() if k != "revised"}}


def wrong_items(ctx: Ctx) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    verdicts = ctx.run.records("verdict")
    return [row for row in ctx.items
            if verdicts.get(row["id"], {}).get("status") == "wrong"], verdicts


def seg_revise(ctx: Ctx, bridge: Bridge) -> int:
    answers = ctx.run.records("answer")
    wrong, verdicts = wrong_items(ctx)

    def make(seg: str, row: dict[str, Any]) -> dict[str, Any]:
        answer = answers[row["id"]]["answer_text"]
        rev = bridge.revise(row["prompt"], answer, verdicts[row["id"]]["verdict"])
        return _revision_record(seg, ctx, row, answer, verdicts[row["id"]], rev)

    return _cpu_loop(ctx, bridge, "revise", "revise", wrong, make)


def seg_noise_revise(ctx: Ctx, bridge: Bridge) -> int:
    answers = ctx.run.records("answer")
    answered = [row for row in ctx.items if _answer_ok(answers.get(row["id"]))]
    verdicts = ctx.run.records("verdict")
    missing = [row["id"] for row in answered if row["id"] not in verdicts]
    if missing:
        ctx.log(f"[noise-revise] refused: {len(missing)} answered items lack a verdict, so "
                "'the first 30 WRONG' is not fixed yet (run the verdict segment first)")
        return RC_PREREQ
    first = [row for row in answered if verdicts[row["id"]]["status"] == "wrong"][:NOISE_REVISE_N]

    def make(seg: str, row: dict[str, Any]) -> dict[str, Any]:
        answer = answers[row["id"]]["answer_text"]
        rev = bridge.revise(row["prompt"], answer, verdicts[row["id"]]["verdict"])
        return _revision_record(seg, ctx, row, answer, verdicts[row["id"]], rev)

    return _cpu_loop(ctx, bridge, "noise-revise", "noise_revise", first, make)


# ── stage 6: gate (offline, against the snapshot) ────────────────────────────


def prepare_gate_store(snapshot_dir: Path, work_dir: Path, snap_manifest: dict[str, Any]) -> None:
    """Copy the pinned snapshot into a per-run working copy and verify every sha256.

    Opening an ``EpisodicStore`` runs schema DDL and takes flocks, so the pristine snapshot is
    never opened: the gate opens this copy, re-made at every gate segment start.
    """
    from .snapshot import sha256_file

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    for rel, meta in snap_manifest["files"].items():
        src = snapshot_dir / rel
        dst = work_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) != meta["sha256"]:
            raise InstrumentFault(f"gate working copy {dst} does not match the snapshot sha")


def seg_gate(ctx: Ctx, bridge: Bridge, *, snapshot_dir: Path | None, retriever_kind: str) -> int:
    answers = ctx.run.records("answer")
    manifest = ctx.run.manifest()
    seg = ctx.run.new_segment("gate", {"header": bridge.header, "retriever": retriever_kind})
    rc, reason, n = RC_OK, "complete", 0
    try:
        done = ctx.run.records("gate", current=seg)
        todo, truncated = _todo(ctx, [row for row in ctx.items
                                      if _answer_ok(answers.get(row["id"])) and row["id"] not in done])
        ctx.log(f"[gate] {len(done)} on disk, {len(todo)} to run (segment {seg})")
        if not todo:
            raise Stop(RC_OK, "nothing to do")
        snap = manifest.get("snapshot") or {}
        if bridge.mode == "real":
            from .snapshot import verify_snapshot

            verify_snapshot(Path(snapshot_dir), snap)
            prepare_gate_store(Path(snapshot_dir), ctx.run.out / "gate-store", snap)
            state, info = bridge.snapshot_state(ctx.run.out / "gate-store", retriever_kind)
            pinned_cfg = snap.get("retrieval_config")
            if pinned_cfg and info["retrieval_config"] != pinned_cfg:
                raise Stop(RC_DRIFT, f"retrieval config {info['retrieval_config']} != pinned "
                                     f"{pinned_cfg}")
        else:
            state, info = bridge.snapshot_state(Path("."), retriever_kind)
        ctx.run.segment_event({"event": "gate_store", "segment_id": seg, **info})
        errors = 0
        for row in todo:
            if ctx.over_budget():
                raise Stop(RC_BUDGET, "wall budget spent")
            require_window(ctx, seg, planning("gate", None)[1] + ctx.margin_s)
            ans = answers[row["id"]]
            role = ans.get("served_role") or "frontdoor"
            g = bridge.gate(state, role, ans["answer_text"], row["prompt"], f"ri18-gate-{row['id']}")
            gs, gq = g.pop("_gs"), g.pop("_gq")
            if gs.skip_reason == "error" or gq.skip_reason == "error":
                errors += 1
                ctx.run.segment_event({"event": "infra", "segment_id": seg, "item_id": row["id"],
                                       "detail": "gate skip_reason=error (retriever/embedder)"})
                if errors >= 3:
                    raise Stop(RC_INSTRUMENT, "3 gate evaluations raised: embedder/retriever down")
                continue
            scored = gs.skip_reason == "scored"
            controls = {
                "gate_1_01_iff_scored": gs.fires_at(1.01) == scored,
                "gate_0_0_false": gs.fires_at(0.0) is False,
                "should_review_eq_gate_0_6": bool(g["should_review"]) == gs.fires_at(PROD_T),
                "should_review_eq_triggered": bool(g["should_review"]) == bool(gs.triggered),
            }
            record = {
                "segment_id": seg, "item_id": row["id"], "suite": row["suite"],
                "stratum": row["stratum"], "split": ctx.split_of(row["id"]), "role": role,
                **g,
                "fires": {f"{t:.2f}": gs.fires_at(t) for t in (*CONTROL_T, *GRID)},
                "fires_question": {f"{t:.2f}": gq.fires_at(t) for t in (*CONTROL_T, *GRID)},
                "controls": controls, "controls_ok": all(controls.values()),
            }
            ctx.run.append("gate", record)
            n += 1
            ctx.log(f"  [{n}/{len(todo)}] {row['id']} skip={gs.skip_reason} avg_q={gs.avg_q} "
                    f"avg_q_question={gq.avg_q} prod={g['should_review']}")
            if not record["controls_ok"]:
                ctx.run.segment_event({"event": "void", "segment_id": seg,
                                       "reason": f"gate control failed on {row['id']}: {controls}"})
                raise Stop(RC_VOID, f"gate control failed on {row['id']}: {controls}")
        if truncated:
            rc, reason = RC_BUDGET, "--limit reached"
    except Stop as stop:
        rc, reason = stop.rc, stop.reason
    if bridge.mode == "real" and snapshot_dir is not None:
        from .snapshot import verify_snapshot

        try:
            verify_snapshot(Path(snapshot_dir), manifest.get("snapshot") or {})
        except InstrumentFault as exc:
            ctx.run.segment_event({"event": "void", "segment_id": seg, "reason": str(exc)})
            rc, reason = RC_VOID, str(exc)
    _finish(ctx, seg, exit_rc=rc, reason=reason, code_root=bridge.code_root,
            pinned_commit=manifest["code_root_commit"], extra={"items_run": n})
    return rc


# ── plan ─────────────────────────────────────────────────────────────────────


def plan(ctx: Ctx) -> dict[str, Any]:
    """Remaining work per segment and window estimates (design §3.1, re-measured when >=20)."""
    run = ctx.run
    have = run.manifest_path.exists()
    answers = run.records("answer") if have else {}
    verdicts = run.records("verdict") if have else {}
    vfull = run.records("vfull") if have else {}
    revs = run.records("revise") if have else {}
    gate = run.records("gate") if have else {}
    nv = run.records("noise_verdict") if have else {}
    nr = run.records("noise_revise") if have else {}
    out: dict[str, Any] = {}

    def est(kind: str, stratum: str | None, n: int, recs: dict[str, Any]) -> dict[str, Any]:
        lo, hi = planning(kind, stratum)
        walls = [float(r["wall_s"]) for r in recs.values()
                 if isinstance(r.get("wall_s"), (int, float))
                 and (stratum is None or r.get("stratum") == stratum)]
        med = statistics.median(walls) if len(walls) >= 20 else None
        return {"remaining": n, "planning_min": [round(n * lo / 60, 1), round(n * hi / 60, 1)],
                "measured_median_s": med,
                "measured_min": round(n * med / 60, 1) if med is not None else None}

    for stratum, key in (("S1", "answer-s1"), ("S2", "answer-s2")):
        rem = [r for r in ctx.items if r["stratum"] == stratum and r["id"] not in answers]
        out[key] = {"device": "CPU :8070 via the API (AutoKernel window)",
                    **est("answer", stratum, len(rem), answers)}
    answered = [r for r in ctx.items if _answer_ok(answers.get(r["id"]))]
    rem_v = [r for r in answered if r["id"] not in verdicts]
    rem_vf = [r for r in answered if r["stratum"] == "S1" and r["id"] not in vfull]
    lo_v, hi_v = planning("verdict", "S2")
    out["verdict"] = {"device": "GPU :8083 one slot (no CPU window)",
                      "remaining_verdicts": len(rem_v), "remaining_vfull": len(rem_vf),
                      "unanswered_items_pending": len(ctx.items) - len(answered),
                      "planning_min": [round((len(rem_v) + len(rem_vf)) * lo_v / 60, 1),
                                       round((len(rem_v) + len(rem_vf)) * hi_v / 60, 1)]}
    wrong = [r for r in answered if verdicts.get(r["id"], {}).get("status") == "wrong"]
    pending_v = len(rem_v)
    rem_r = [r for r in wrong if r["id"] not in revs]
    guess = (round(pending_v * 0.33), round(pending_v * 0.5))
    lo_r, hi_r = planning("revise", None)
    out["revise"] = {"device": "CPU :8070 in-process (AutoKernel window)",
                     "remaining_known_wrong": len(rem_r),
                     "plus_expected_from_pending_verdicts": list(guess),
                     "planning_min": [round((len(rem_r) + guess[0]) * lo_r / 60, 1),
                                      round((len(rem_r) + guess[1]) * hi_r / 60, 1)],
                     "measured_median_s": est("revise", None, 0, revs)["measured_median_s"]}
    rem_g = [r for r in answered if r["id"] not in gate]
    out["gate"] = {"device": "CPU embedders :8090-8095 (AutoKernel window)",
                   **est("gate", None, len(rem_g), gate)}
    subset = ctx.workload["noise_subset"]
    out["noise-verdict"] = {"device": "GPU :8083", **est("noise-verdict", None,
                                                          len([i for i in subset if i not in nv]), nv)}
    out["noise-revise"] = {"device": "CPU :8070 in-process (AutoKernel window)",
                           **est("noise-revise", None, max(0, NOISE_REVISE_N - len(nr)), nr)}
    return out
