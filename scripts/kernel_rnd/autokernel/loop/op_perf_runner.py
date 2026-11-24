"""RSL-R3 live op-perf runner: paired `test-backend-ops perf` over the op_perf_screen grid, plus the A/A floor writer.

Per round the whole grid is run once per binary and per op (MUL_MAT, MUL_MAT_ID); the arm order alternates every round
(AB, BA, AB, ...), the same ABBA idiom `cpu_fa_reference.perf_screen` uses. Each (shape, n) cell is the SUM of us/run over
the types its lane serves. Floors come ONLY from a per-cell paired A/A file (never borrowed): a cell without one is
UNMEASURED and the stage reports FUTILITY_HARNESS_INSTRUMENT. `calibrate` writes that file; a measurement row runs it.

Floor file: <store>/calibration/op-perf-aa.<path-key>.json
  {schema, path, gguf_stem, backend, unit: bench.FLOOR_UNIT, n_pairs (>= 24), binary_sha256, source_row,
   cells: {cell_key: {shape, n, anchor_us[], candidate_us[], floor_pct: {"5": p95 |median effect| %, "9": .., "20": ..}}}}

Prerequisite stated, not assumed: stock `perf` mode only builds `make_test_cases_perf()`, which has none of the served
shapes. A binary must carry PERF_CASE_SET_ID (a perf-mode registration of the served-shape cases) or the run refuses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import bench, census_coverage as cc, op_perf_screen as ops, served_shape_cases as ssc
from .tool_output import strip_ansi

FLOOR_SCHEMA = "epyc.autokernel.op_perf_floor.v1"
CALIBRATION_MIN_PAIRS = 24          # FLOOR-UNIT-1 (serving.MATCHED_CALIBRATION_PAIRS)
FLOOR_KS = (5, 9, 20)
PERF_CASE_SET_ID = "served_shape_perf_v1"
STORE = Path("/mnt/raid0/llm/autokernel/loop-memory")
CENSUS = Path(__file__).resolve().parent / "census" / "mi210_bandwidth_census.json"
PERF_TIMEOUT_S = 1800
_LINE = re.compile(r"^\s*([A-Z0-9_]+)\((.*)\):\s+\d+ runs -\s+([0-9.]+) us/run")


class RunnerRefused(RuntimeError):
    """A runner refused before/while measuring; `token` is the FUTILITY_HARNESS_* verdict the stage writes."""

    def __init__(self, token: str, why: str):
        super().__init__(why)
        self.token = token


def floor_path(store: Path, key: str) -> Path:
    return Path(store) / "calibration" / f"op-perf-aa.{re.sub(r'[^A-Za-z0-9._-]', '_', key)}.json"


def vars_prefix(cell: Mapping, type_a: str) -> str:
    """The `vars()` text of the case, up to (not including) any `,max_nmse=` the patched subclasses append."""
    d, t = cell["dims"], ssc.GGML_TYPE_NAMES[type_a]
    if cell["op"] == "MUL_MAT":
        return (f"type_a={t},type_b=f32,m={d['m']},n={cell['n']},k={d['k']},bs=[1,1],nr=[1,1],"
                "per=[0,1,2,3],k_v=0,o=1")
    return (f"type_a={t},type_b=f32,n_mats={d['n_mats']},n_used={d['n_used']},b=0,m={d['m']},n={cell['n']},k={d['k']}")


def plan_cells(spec: Mapping, cells: Sequence[Mapping]) -> list[tuple[Mapping, list[tuple[str, re.Pattern]]]]:
    lane, dense = ssc.lane_for_model(spec["gguf"]), ssc.dense_lane_for_model(spec["gguf"])
    if not (lane or dense):
        raise RunnerRefused("FUTILITY_HARNESS_MISSING_INPUT", f"no served-shape lane for {spec['key']}")
    shapes = {s.name: s for s in ssc.lane_served_shapes(lane or dense)}
    return [(c, [(t, re.compile("^" + re.escape(vars_prefix(c, t))))
                 for t in (("Q8_0",) if dense else ssc.routed_types(lane, shapes[c["shape"]]))]) for c in cells]


def parse_perf(text: str) -> list[tuple[str, str, float]]:
    return [(m.group(1), m.group(2), float(m.group(3))) for ln in strip_ansi(text).splitlines()
            if (m := _LINE.match(ln))]


def cell_us(plan, lines) -> dict[str, float]:
    """{cell key: sum of us/run over the cell's types}; a cell missing any type's line is left OUT (unmeasured)."""
    out: dict[str, float] = {}
    for cell, rxs in plan:
        hits = [[us for op, v, us in lines if op == cell["op"] and rx.match(v)] for _, rx in rxs]
        if any(len(h) > 1 for h in hits):
            raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"{cell['key']}: two lines match one case")
        if all(hits):
            out[cell["key"]] = sum(h[0] for h in hits)
    return out


def full_grid(spec: Mapping) -> list[dict]:
    """Every served cell of the path's lane (no shape-class narrowing): the floor file must cover any class."""
    lane = ssc.lane_for_model(spec["gguf"]) or ssc.dense_lane_for_model(spec["gguf"])
    shapes = ssc.lane_served_shapes(lane)
    return [c for fam in sorted({s.op for s in shapes}) for c in ops.grid({"op_family": fam}, shapes, spec["n"])]


def load_floors(store: Path, spec: Mapping, backend: str, pairs: int) -> dict[str, float]:
    try:
        rec = json.loads(floor_path(store, spec["key"]).read_text())
    except (OSError, ValueError):
        return {}
    if (rec.get("schema") != FLOOR_SCHEMA or rec.get("path") != spec["key"] or rec.get("backend") != backend
            or rec.get("gguf_stem") != Path(spec["gguf"]).stem or rec.get("unit") != bench.FLOOR_UNIT
            or int(rec.get("n_pairs") or 0) < CALIBRATION_MIN_PAIRS):
        return {}
    return {k: float(v["floor_pct"][str(pairs)]) for k, v in (rec.get("cells") or {}).items()
            if str(pairs) in (v.get("floor_pct") or {})}


def weights(cells: Sequence[Mapping], spec: Mapping, census: Mapping | None = None) -> tuple[dict, dict]:
    """weight = census calls_per_round x share_of_round for the cell's (m x k, n) on THIS path's census model; an n with
    no row falls back to the shape's median over n (basis `shape-median`); no row at all -> no weight (unmeasured)."""
    ck = cc.CENSUS_KEY.get((spec["model_role"], spec["device"]))
    recs = (census or json.loads(CENSUS.read_text()))["records"].values()
    by: dict[tuple, list[float]] = {}
    for r in recs:
        if ck and r.get("model") == ck[0] and r.get("calls_per_round") and r.get("share_of_round"):
            by.setdefault((r["shape"], int(r["n"])), []).append(r["calls_per_round"] * r["share_of_round"])
    out, basis = {}, {}
    for c in cells:
        shp = f"{c['dims']['m']}x{c['dims']['k']}"
        if (v := by.get((shp, c["n"]))):
            out[c["key"]], basis[c["key"]] = max(v), "exact"
        elif (v := [x for (s, _), xs in by.items() if s == shp for x in xs]):
            out[c["key"]], basis[c["key"]] = statistics.median(v), "shape-median"
    return out, basis


def _placement(spec: Mapping) -> tuple[list[str], int | None, dict]:
    if spec["lane"] == "gpu":
        return ["taskset", "-c", bench.CPU_LIST], None, {}
    from . import cpu_timing_runner as ctr
    inst = ctr.topology(spec)[0]
    return inst["prefix"], inst["threads"], ctr.recipe_env(spec)


def _invoke(binary: Path, op: str, rxs: Sequence[re.Pattern], spec: Mapping, runner: Callable, live: bool) -> list:
    prefix, threads, extra = _placement(spec) if live else ([], None, {})
    backend = "ROCm0" if spec["lane"] == "gpu" else "CPU"
    env = {**os.environ, "LD_LIBRARY_PATH": f"{binary.parent}:/opt/rocm/lib", **extra, ssc.CASE_SET_ENV: PERF_CASE_SET_ID,
           **({ssc.BACKEND_THREADS_ENV: str(threads)} if threads else {})}
    done = runner([*prefix, str(binary), "perf", "-o", op, "-b", backend, "-p", "|".join(r.pattern for r in rxs)],
                  capture_output=True, text=True, timeout=PERF_TIMEOUT_S, env=env)
    if done.returncode:
        raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"{binary.name} perf rc={done.returncode}: {done.stderr[-200:]}")
    if not re.search(rf"Backend \d+/\d+: {backend}\b", strip_ansi(done.stdout)):
        raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"no `Backend n/m: {backend}` section: not a {backend} run")
    return parse_perf(done.stdout)


def prove_binaries(binaries: Sequence[Path], literal: str | None = PERF_CASE_SET_ID, ld_tail: str = ":/opt/rocm/lib") -> None:
    """Content proof per binary: carries `literal` (the perf case set) when asked, and loads its OWN ggml (verify_ggml_linkage.sh)."""
    from . import hip_launch_proof as hp
    for b in dict.fromkeys(map(Path, binaries)):
        try:
            carries = literal is None or literal.encode() in b.read_bytes()
        except OSError as e:
            raise RunnerRefused("FUTILITY_HARNESS_MISSING_INPUT", f"{b}: {e}")
        if not carries:
            raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"{b} lacks the {literal} perf cases (stock perf has no served shapes)")
        link = hp.linkage(b, f"{b.parent}{ld_tail}")
        if link["status"] != hp.PROVEN:
            raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"ggml linkage {link['status']} for {b}: {link.get('reason') or link.get('tail', '')[-120:]}")


def measure(anchor: Path, cand: Path, cells, spec, *, pairs: int, runner=subprocess.run, live=True) -> dict[str, dict]:
    """{cell key: {anchor: [us..], candidate: [us..]}}; arm order alternates per round."""
    bins = {"anchor": Path(anchor) / "bin" / "test-backend-ops", "candidate": Path(cand) / "bin" / "test-backend-ops"}
    if live:
        prove_binaries(list(bins.values()))
    plan = plan_cells(spec, cells)
    per_op = {op: [p for p in plan if p[0]["op"] == op] for op in sorted({c["op"] for c in cells})}
    out: dict[str, dict] = {c["key"]: {"anchor": [], "candidate": []} for c in cells}
    for i in range(pairs):
        for role in (("anchor", "candidate") if i % 2 == 0 else ("candidate", "anchor")):
            for op, sub in per_op.items():
                got = cell_us(sub, _invoke(bins[role], op, [rx for _, rxs in sub for _, rx in rxs], spec, runner, live))
                for k, us in got.items():
                    out[k][role].append(us)
    return out


def run(anchor: Path, cand: Path, cells, spec, *, pairs: int | None = None, store: Path = STORE,
        runner=subprocess.run, live=True, floors: Mapping | None = None, census: Mapping | None = None) -> dict:
    """The stage primitive's result: {"cells": [{key, anchor, candidate, floor_pct}], "weights": {...}}."""
    pairs = max(int(pairs or ops.MIN_PAIRS), ops.MIN_PAIRS)
    backend = "ROCm0" if spec["lane"] == "gpu" else "CPU"
    fl = load_floors(store, spec, backend, pairs) if floors is None else dict(floors)
    m = measure(anchor, cand, cells, spec, pairs=pairs, runner=runner, live=live)
    w, basis = weights(cells, spec, census)
    return {"cells": [{"key": k, **v, "floor_pct": fl.get(k)} for k, v in m.items()], "weights": w,
            "weight_basis": basis, "pairs": pairs, "floors_found": len(fl)}


def calibrate(build: Path, spec: Mapping, *, source_row: str, pairs: int = CALIBRATION_MIN_PAIRS, store: Path = STORE,
              runner=subprocess.run, live=True) -> Path:
    """A/A: BOTH arms are the same build. Writes the per-cell floor file (atomically). Used by a measurement row only."""
    if pairs < CALIBRATION_MIN_PAIRS or not source_row:
        raise RunnerRefused("FUTILITY_HARNESS_INPUT", f"calibration needs >= {CALIBRATION_MIN_PAIRS} pairs and a source row")
    cells = full_grid(spec)
    m = measure(build, build, cells, spec, pairs=pairs, runner=runner, live=live)
    if short := [k for k, v in m.items() if len(v["anchor"]) != pairs or len(v["candidate"]) != pairs]:
        raise RunnerRefused("FUTILITY_HARNESS_INSTRUMENT", f"{len(short)} cells lack {pairs} paired samples: {short[0]}")
    binary = Path(build) / "bin" / "test-backend-ops"
    rec = {"schema": FLOOR_SCHEMA, "path": spec["key"], "gguf_stem": Path(spec["gguf"]).stem,
           "backend": "ROCm0" if spec["lane"] == "gpu" else "CPU", "unit": bench.FLOOR_UNIT, "n_pairs": pairs,
           "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest() if live else "test-mode",
           "source_row": source_row,
           "cells": {c["key"]: {"shape": c["shape"], "n": c["n"], "anchor_us": m[c["key"]]["anchor"],
                                "candidate_us": m[c["key"]]["candidate"],
                                "floor_pct": {str(k): v for k, v in bench.bootstrap_floor(
                                    m[c["key"]]["anchor"], m[c["key"]]["candidate"], ks=FLOOR_KS).items()}}
                     for c in cells}}
    p = floor_path(store, spec["key"])
    p.parent.mkdir(parents=True, exist_ok=True)
    (tmp := p.with_suffix(".tmp")).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)
    return p


def synth_runner(cells, spec, anchor_bin: Path, outcome: str = "pass") -> Callable:
    """--test-mode: real-format `perf` output built from the cells; NO binary runs. outcome: pass | regress | bad-rc |
    no-backend | missing-line (the last drops the first cell so it is unmeasured)."""
    plan, backend, calls = plan_cells(spec, cells), "ROCm0" if spec["lane"] == "gpu" else "CPU", [0]

    def runner(argv, **kw):
        calls[0] += 1
        jitter = 1.0 + 0.004 * ((calls[0] * 7) % 5 - 2)   # deterministic scatter, so an A/A floor is positive
        op = argv[argv.index("-o") + 1]
        cand = Path(next(a for a in argv if a.endswith("test-backend-ops"))) != Path(anchor_bin)
        body = [f"Testing 1 devices\n\nBackend 1/1: {backend if outcome != 'no-backend' else 'Vulkan0'}\n"]
        for i, (c, rxs) in enumerate(plan):
            if c["op"] != op or (outcome == "missing-line" and i == 0):
                continue
            for t, _ in rxs:
                us = 100.0 * jitter * (1.0 if not cand else 1.5 if outcome == "regress" and i == 0 else 0.9)
                body.append(f"  {c['op']}({vars_prefix(c, t)},max_nmse=0.0005):  {1000:8d} runs - {us:8.2f} us/run - "
                            "12.00 MFLOP/run - \033[1;34m 1.0 GFLOPS\033[0m\n")
        return subprocess.CompletedProcess(argv, 3 if outcome == "bad-rc" else 0, "".join(body), "boom")
    return runner


def base_parser(doc: str, key_help: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help=key_help), ap.add_argument("--anchor-build", type=Path, required=True)
    ap.add_argument("--candidate-build", type=Path), ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--calibrate", action="store_true", help="write the A/A floor file (measurement row only)")
    ap.add_argument("--row"), ap.add_argument("--pairs", type=int), ap.add_argument("--out", type=Path)
    ap.add_argument("--test-mode", action="store_true", help="fixture outputs, NO binary runs")
    ap.add_argument("--outcome", default="pass", help="test-mode only")
    return ap


def emit(a: argparse.Namespace, res: Mapping) -> int:
    (a.out.write_text if a.out else print)(json.dumps(res, indent=1, sort_keys=True, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    a = base_parser(__doc__, 'served path key, e.g. "27b-q8_0|mi210" (registry model|device)').parse_args(argv)
    from . import stage_bodies as sb
    spec = sb.path_specs(cc.load_registry() or {})[a.path]
    cells = full_grid(spec)
    runner = synth_runner(cells, spec, a.anchor_build / "bin" / "test-backend-ops", a.outcome) if a.test_mode else subprocess.run
    try:
        if a.calibrate:
            return emit(a, {"floor_file": str(calibrate(a.anchor_build, spec, source_row=a.row or "", pairs=a.pairs or CALIBRATION_MIN_PAIRS,
                                                        store=a.store, runner=runner, live=not a.test_mode))})
        res = run(a.anchor_build, a.candidate_build or a.anchor_build, cells, spec, pairs=a.pairs, store=a.store, runner=runner,
                  live=not a.test_mode, floors={c["key"]: 1.0 for c in cells} if a.test_mode else None)
        return emit(a, {**res, "decision": ops.decide(res["cells"], res["weights"])})
    except RunnerRefused as e:
        return emit(a, {"verdict": f"{e.token} ({e})"})


if __name__ == "__main__":
    sys.exit(main())
