"""VB-AKX-LB32-WRITE, write side: VERDICT.json -> akx_lb32.beliefs.jsonl (self-hashed rows) beside it.

The belief-kernel reader is root `scripts/vidya/adapters/akx_lb32.py` (`cli.py ingest akx-lb32`). The row vocabulary is
root `adapters/_native_sidecar.py`'s (measurement_id / metric / value / unit / metric_direction / category / claim /
protocol_id / reps / reps_basis / extra + row_sha256 over the canonical JSON of the rest); this module re-implements
only the hashing so the research repo needs no root import. The reader re-derives every row and voids the whole file on
any difference, so the two cannot drift silently.

Rows:
  kind=cell     one per (threads, N) whose cell carries an F (t48 gating cells; t96 cells only if the t96 report is
                itself not INVALID): value = F(N) = FA node wall share of the 64k step (M arm).
  kind=verdict  one per run: value = max F(N), N=2..5; extra carries verdict, seed class, max S.
Attestations: results_path/results_sha256 = VERDICT.json bytes; extra.predeclaration_path/sha256 = the frozen rule.
An INVALID gating verdict writes NOTHING (and removes a stale sidecar): absence stays absent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CAPTURE_SCHEMA = "epyc.vidya.akx_lb32_capture.v1"
SIDECAR_NAME = "akx_lb32.beliefs.jsonl"
PRODUCER = "epyc-inference-research/scripts/benchmark/akx_lb3_attribution/verdict.py"
METRIC_CELL = "akx_lb32_fa_wall_share_d65536"
METRIC_VERDICT = "akx_lb32_max_fa_wall_share_n2to5_d65536"
UNIT = "fraction of the timed llama-bench pp-N step at depth 65536 (FLASH_ATTN_EXT node wall through the graph barrier)"
VALID_VERDICTS = ("LANE_LARGE", "LANE_SMALL", "SPLIT_UNRESOLVED")
GATE_NS = (2, 3, 4, 5)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(v: Any) -> Any:
    """Canonical-JSON-safe: non-finite floats become text (never a silent number)."""
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def measurement_id(run_id: str, kind: str, threads: int, n: Optional[int], results_sha256: str) -> str:
    return "akxlb32_" + content_hash({"run_id": run_id, "kind": kind, "threads": threads, "n_prompt": n,
                                      "results": results_sha256})[:24]


def seal(row: dict) -> dict:
    row["row_sha256"] = content_hash({k: v for k, v in row.items() if k != "row_sha256"})
    return row


def build_rows(verdict: dict, *, verdict_path: str, verdict_sha256: str, predeclaration_path: str,
               predeclaration_sha256: str, run_id: str, emitted_at: str) -> list[dict]:
    g = verdict.get("gating") or {}
    if g.get("verdict") not in VALID_VERDICTS:
        return []
    base = dict(schema=CAPTURE_SCHEMA, run_id=run_id, producer=PRODUCER, emitted_at=emitted_at, date=emitted_at,
                unit=UNIT, metric_direction="lower_better", category="BASELINE", protocol_id="",
                results_path=verdict_path, results_sha256=verdict_sha256)
    common_extra = dict(predeclaration_path=predeclaration_path, predeclaration_sha256=predeclaration_sha256,
                        n_depth=65536, n_depth_lo=512)
    rows: list[dict] = []
    for threads, part in ((48, g), (96, verdict.get("reported_t96"))):
        if not part or part.get("verdict") not in VALID_VERDICTS:
            continue
        for n_s, cell in sorted((part.get("cells") or {}).items(), key=lambda kv: int(kv[0])):
            F = cell.get("F_fa_wall_share")
            reps = cell.get("reps_used_d64k")
            if not isinstance(F, (int, float)) or isinstance(F, bool) or not math.isfinite(F) or not isinstance(reps, int):
                continue
            n = int(n_s)
            extra = dict(common_extra, kind="cell", threads=threads, gating=threads == 48, n_prompt=n,
                         F_fa_wall_share=F, S_depth_share=cell.get("S_depth_share"),
                         S_M_depth_share=cell.get("S_M_depth_share"), overhead_d64k=cell.get("overhead_d64k"),
                         overhead_d512=cell.get("overhead_d512"), coverage_defect_d64k=cell.get("coverage_defect_d64k"),
                         reps_used_d64k=reps, t_ms=cell.get("t_ms"),
                         F_cpu_sample_share_G=cell.get("F_cpu_sample_share_G"),
                         class_share_d64k=cell.get("class_share_d64k"),
                         class_depth_delta_share=cell.get("class_depth_delta_share"),
                         run_verdict=part["verdict"])
            row = dict(base, metric=METRIC_CELL, value=float(F), reps=reps,
                       reps_basis="usable timed llama-bench repetitions at depth 65536 in the M arm",
                       claim=(f"AKX-ALL-LB3.2 Q38FN t{threads} pp{n} @ d65536: FA node wall = {F:.4f} of the step "
                              f"(critical path through the graph barrier, measurement build; S={cell.get('S_depth_share')}"
                              f" on the champion store image); one run, predeclared rule, no codified protocol"),
                       extra=_clean(extra))
            row["measurement_id"] = measurement_id(run_id, "cell", threads, n, verdict_sha256)
            rows.append(seal(row))
    gcells = {int(k): v for k, v in (g.get("cells") or {}).items()}   # int keys in memory, str after a JSON round trip
    reps_gate = [gcells[n].get("reps_used_d64k", 0) for n in GATE_NS if n in gcells]
    extra = dict(common_extra, kind="verdict", threads=48, gating=True, verdict=g["verdict"], seed_class=g.get("seed_class"),
                 max_F_N2to5=g.get("max_F_N2to5"), max_S_N2to5=g.get("max_S_N2to5"),
                 mean_depth_delta_share_N2to5=g.get("mean_depth_delta_share_N2to5"), thresholds=g.get("thresholds"))
    row = dict(base, metric=METRIC_VERDICT, value=float(g["max_F_N2to5"]), reps=min(reps_gate) if reps_gate else 0,
               reps_basis="fewest usable timed repetitions over the N=2..5 gating cells (M arm, depth 65536)",
               claim=(f"AKX-ALL-LB3.2 verdict {g['verdict']}: max F(2..5) = {g['max_F_N2to5']:.4f}, max S(2..5) = "
                      f"{g.get('max_S_N2to5')}, seed class {g.get('seed_class')}; one run, predeclared rule"),
               extra=_clean(extra))
    row["measurement_id"] = measurement_id(run_id, "verdict", 48, None, verdict_sha256)
    rows.append(seal(row))
    return rows


def write_sidecar(verdict_path: str | Path, predeclaration_path: str | Path, *, run_id: str,
                  emitted_at: Optional[str] = None) -> Optional[Path]:
    vp, pp = Path(verdict_path), Path(predeclaration_path)
    side = vp.parent / SIDECAR_NAME
    verdict = json.loads(vp.read_text())
    when = emitted_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = build_rows(verdict, verdict_path=str(vp.resolve()), verdict_sha256=file_sha256(vp),
                      predeclaration_path=str(pp.resolve()), predeclaration_sha256=file_sha256(pp),
                      run_id=run_id, emitted_at=when)
    if not rows:
        if side.exists():
            side.unlink()
        return None
    tmp = side.with_suffix(side.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    os.replace(tmp, side)
    return side
