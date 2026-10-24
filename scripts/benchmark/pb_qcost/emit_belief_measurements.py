#!/usr/bin/env python3
"""PB-QCOST-1 post-hoc belief emitter: write a self-hashed ``belief_measurements.jsonl`` for one run unit (VB-PBQ-1).

    python3 -B emit_belief_measurements.py --run-dir <unit> --out <file> [--emission readout|backfill] [--overwrite]

Neither PB-QCOST-1 producer writes belief rows, and the drivers that ran are sha-pinned by their predeclarations, so
they are NOT edited. This emitter runs after the run (the confirm-R read-out calls it after ``verdict_q.py``) and reads
only what the run wrote. Two layouts, the same ones the root adapter ``scripts/vidya/adapters/pb_qcost.py`` reads:

* ORIGINAL (``pb_qcost_1.py``): ``<unit>/pb_qcost_1.jsonl`` + ``<unit>/summary.json``.
* CONFIRM (``pb_qcost_1_q.py`` + ``verdict_q.py``): ``<unit>/out/run/pb_qcost_1.jsonl``, ``out/run/summary.json``,
  ``out/run/graphs_proof_ub*.json``, ``<unit>/VERDICT.json``, ``<unit>/RUN_RECORD.json``, ``<unit>/PREDECLARATION.md``
  (and, for confirm-R, ``AMENDMENT-confirm-R.md``).

Every number is RE-DERIVED from the jsonl rows with the PRODUCER'S OWN pure statistics. They are not copied: the
functions (``pct``, ``ols``, ``resid_stats``, ``usable``, ``analyse_q`` and its solver) are extracted by name from the
sha-pinned byte-identical producer copies beside this file (``pb_qcost_1.py``, ``pb_qcost_1_q.py``) and executed
without importing the driver, whose module import pulls in the live harness. A re-derived number that differs from
the value the run itself stated (``summary.json`` / ``VERDICT.json``) refuses the emission (exit 3); nothing is
repaired. One row per (phase, n generating slots, metric), the same keys the root adapter projects, so the reader can
compare the two sets one for one.

Each row carries: the schema id, ``content_sha256`` (sha256 over the canonical JSON of the row without that key), the
sha256 of every input file read, the PREDECLARATION / AMENDMENT digests, the grade with its cause IN WORDS, the unit
(ms per server iteration, or ms per token for b), n, whether q was held fixed (and at what value), ``-np``, the binary
sha256 and the argv proved by the run's ``port_ownership`` row(s). This tool decides nothing: ``claim_tuple.grade()`` in
epyc-root grades the projected tuples; there is no ladder here.

``--emission backfill`` marks a sidecar emitted for a run that finished before this tool existed (the two 2026-10-21 /
10-23 runs); the reader then still records the projection as a retrofit. ``readout`` (default) is the confirm-R path.
Exit codes: 0 written, 2 usage / not a PB-QCOST-1 unit, 3 refused (a stated value does not re-derive, or a proof is
missing).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEMA = "epyc.pbqcost1.belief_measurement.v1"
RUN_SCHEMA = "epyc.pbqcost1.run.v1"
EMITTER = "scripts/benchmark/pb_qcost/emit_belief_measurements.py"

#: layout -> (producer file beside this one, its pinned sha256). The copies are byte-identical to the drivers that ran:
#: pb_qcost_1.py == /mnt/raid0/llm/tmp/pb-qcost-1/pb_qcost_1.py (ORIG_DRIVER.sha256 of both confirm launchers);
#: pb_qcost_1_q.py == RUN_RECORD.driver_sha256 of pb-qcost-1-confirm and the AMENDMENT-confirm-R "Unchanged" list.
PRODUCERS = {
    "original": ("pb_qcost_1.py", "9899ffba3aef3127d97272432cf743b58e0f56fc278bbfc279ce3dba1ed56630"),
    "confirm": ("pb_qcost_1_q.py", "89a35f2de62c43e86eb4b58bcce4ad8470219fa99461d99596802a2d89f180cb"),
}
PURE_FUNCS = ("pct", "ols", "resid_stats", "_solve3", "_feat", "_suff", "_coef", "_pred", "usable", "analyse_q")
PURE_CONSTS = ("NS", "TRAIN", "HELD", "BASE_UB", "IT_Q", "C_REF", "BOOT_SEED", "BOOT_N", "OLD21", "B_OFF", "PASS_MAPE",
               "NP")

#: The 2026-10-21 original run states no grade of its own. Its grade is the one the confirm's predeclaration (section 0)
#: states for that exact jsonl, pinned here by the jsonl's digest (the same pin the root adapter holds).
ORIGINAL_GRADES = {
    "a82cdf93dc8ca304703f057822586d61f746600bc8791036a3d7023e41c2699e": {
        "grade": "INDICATIVE",
        "cause": "The 21:29Z run is graded INDICATIVE because it ran beside the KVU-12 np8 CPU timing.",
        "source": "pb-qcost-1-confirm PREDECLARATION.md section 0 (sha256 "
                  "f45aef1a43976af3d707aea4e7d61875a83e343ec726a61e1725316b385d3e06); not in the run record",
        "source_sha256": "f45aef1a43976af3d707aea4e7d61875a83e343ec726a61e1725316b385d3e06",
    },
}

U_NORM = "ms per server iteration (q normalised to 512 tokens)"
U_RAW = "ms per server iteration (raw)"
#: metric key -> (metric name, unit): the root adapter's vocabulary, one for one (it refuses a row whose unit differs).
METRICS = {
    "intercept": ("pbqcost_intercept_ms_per_iter_c0", U_NORM),
    "c_slope": ("pbqcost_c_slope_ms_per_iter_per_cached_token", "ms per server iteration per cached token "
                                                                 "(q normalised to 512 tokens)"),
    "noise_floor": ("pbqcost_noise_floor_train_rmse_ms_per_iter", U_NORM),
    "heldout_p99_abs": ("pbqcost_heldout_p99_abs_residual_ms_per_iter", U_NORM),
    "q_a": ("pbqcost_q_a_ms_per_iter", U_RAW),
    "q_b": ("pbqcost_q_b_ms_per_token", "ms per token"),
    "q_d": ("pbqcost_q_d_ms_per_token_cached_token", "ms per token x cached token"),
    "q_noise_floor": ("pbqcost_q_noise_floor_train_rmse_ms_per_iter", U_RAW),
    "q_heldout_p99_abs": ("pbqcost_q_heldout_p99_abs_residual_ms_per_iter", U_RAW),
}
TRAIN_BASIS = "train-length (4k, 16k, 64k) server iterations, pooled over reps"
HELD_BASIS = "held-out-length (8k, 32k) server iterations, pooled over reps"


class Refused(ValueError):
    """A stated value does not re-derive, or the run does not prove what a row would claim."""


# --- canonical hashing ------------------------------------------------------------------------------------------------
def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_sha256(row: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in row.items() if k != "content_sha256"}).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rt(value: Any) -> Any:
    """JSON round trip, as the producer wrote its records (int keys become strings)."""
    return json.loads(json.dumps(value, default=str))


# --- the producer's own pure statistics -------------------------------------------------------------------------------
def load_pure(layout: str) -> tuple[Any, str, str]:
    """Exec the producer's pure functions and constants (by name, from its pinned copy) in a fresh namespace."""
    name, pin = PRODUCERS[layout]
    path = HERE / name
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != pin:
        raise Refused(f"{name} hashes {digest}, not the pinned producer {pin}")
    tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PURE_FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in PURE_CONSTS for t in node.targets):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) and
                                                  any(isinstance(e, ast.Name) and e.id in PURE_CONSTS for e in t.elts)
                                                  for t in node.targets):
            keep.append(node)
    ns: dict = {"math": math, "statistics": statistics, "__name__": f"pbq_pure_{layout}"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)  # noqa: S102 - pinned producer code
    ns.setdefault("BASE_UB", 512)
    return argparse.Namespace(**{k: v for k, v in ns.items() if not k.startswith("__")}), str(path), digest


def interval_fit(P: Any, rows: list[dict], n: int) -> dict | None:
    """The interval-level block of the producer's ``analyse()`` at n (ub 512): ``iter_ms = A + B*c`` on TRAIN lengths,
    residuals on HELD-OUT lengths, rounded exactly as ``analyse()`` rounds them."""
    runs = [r for r in rows if r.get("kind") == "run" and r.get("ttft_ms") and not r.get("skipped") and r.get("http") == 200
            and r.get("ub", P.BASE_UB) == P.BASE_UB and r.get("n_gen") == n]
    iv = [{**i, "length": r["length"]} for r in runs for i in (r.get("intervals") or [])]
    ivt = [i for i in iv if i["length"] in P.TRAIN]
    ivh = [i for i in iv if i["length"] in P.HELD]
    fit = P.ols([i["c"] for i in ivt], [i["iter_ms"] for i in ivt])
    if not fit:
        return None
    noise = P.resid_stats([i["iter_ms"] for i in ivt], [fit[0] + fit[1] * i["c"] for i in ivt])
    held = P.resid_stats([i["iter_ms"] for i in ivh], [fit[0] + fit[1] * i["c"] for i in ivh])
    dns = [i["dn"] for i in iv]
    return {"A": round(fit[0], 1), "B": round(fit[1], 6), "noise_rmse": noise.get("rmse"), "noise_n": noise.get("n"),
            "p99_abs": held.get("p99_abs"), "p99_under": held.get("p99_under_prediction"), "held_n": held.get("n"),
            "q_median": statistics.median(dns) if dns else None, "requests": len(runs)}


# --- reading ---------------------------------------------------------------------------------------------------------
def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"{path.name} unreadable: {exc}") from exc


def _rows(path: Path) -> list[dict]:
    rows = []
    for k, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise Refused(f"{path.name} line {k} is torn: {exc}") from exc
        if not isinstance(row, dict):
            raise Refused(f"{path.name} line {k} is not an object")
        rows.append(row)
    return rows


def _same(stated: Any, derived: Any, what: str) -> None:
    if _rt(stated) != _rt(derived):
        raise Refused(f"{what} does not re-derive from pb_qcost_1.jsonl: stated {stated!r}, re-derived {derived!r}")


def _argv(owner: dict, what: str) -> list[str]:
    argv = owner.get("argv")
    if not (owner.get("ok") is True and isinstance(argv, list) and argv and all(isinstance(t, str) for t in argv)):
        raise Refused(f"{what}: the port_ownership row does not prove the launched argv")
    return argv


def _flag(argv: list[str], *names: str) -> str | None:
    for i, tok in enumerate(argv[:-1]):
        if tok in names:
            return argv[i + 1]
    return None


def _q_int(value: Any) -> int | None:
    return int(value) if type(value) in (int, float) and float(value).is_integer() and value > 0 else None


def _owner_brief(owner: dict) -> dict:
    return {k: owner.get(k) for k in ("ok", "pid", "detail", "ub")}


def confirm_grade(unit: Path, verdict: dict) -> tuple[str, str]:
    """The grade VERDICT.json states, and its cause in words, read from the run's own files."""
    grade = verdict.get("grade")
    if not isinstance(grade, str) or not grade:
        raise Refused("VERDICT.json states no grade")
    proof_ok = (verdict.get("proof") or {}).get("all_ok") is True
    per_phase = (verdict.get("sibling_per_phase") or {}).get("per_phase")
    if grade == "UNCONTENDED":
        if isinstance(per_phase, dict):
            return grade, ("the run-wide sibling-load bars (PREDECLARATION section 2: median 30-s busy fraction "
                           "of CPUs 88-95 <= 5 % and <= 5 % of windows > 25 %) passed, every completed ub phase passed them "
                           "in its own window (AMENDMENT A2), and every graphs/HSA/VRAM proof is ok")
        return grade, "the section-2 sibling-load bars passed and every graphs/HSA/VRAM proof is ok"
    if verdict.get("sibling_summary") is None:
        read_at, written_at = unit / "out" / "run" / "sibling_summary.json", unit / "out" / "sibling_summary.json"
        if not read_at.exists() and written_at.exists() and not isinstance(per_phase, dict):
            return grade, ("sibling-summary path defect: verdict_q reads out/run/sibling_summary.json but the sampler wrote "
                           "out/sibling_summary.json, so VERDICT.sibling_summary is null and the predeclared section-2 rule "
                           "could only grade INDICATIVE (fail-closed); the measured sibling load never entered the grade")
        return grade, "VERDICT.sibling_summary is null: no sibling-load summary entered the grade (fail-closed INDICATIVE)"
    why = []
    if not proof_ok:
        bad = [u for u, p in ((verdict.get("proof") or {}).get("per_ub") or {}).items() if not (p or {}).get("ok")]
        why.append("a graphs/HSA/VRAM proof failed (ub " + ", ".join(bad or ["?"]) + ")")
    sib = verdict["sibling_summary"]
    if verdict.get("grade_run_level", "INDICATIVE") != "UNCONTENDED":
        why.append(f"the run-wide sibling load failed the section-2 bars (median busy {sib.get('median_busy_frac')}, "
                   f"{sib.get('frac_windows_gt_25pct')} of 30-s windows above 25 %)")
    if isinstance(per_phase, dict):
        for g in sorted(per_phase.values(), key=lambda g: str(g.get("span_utc"))):
            if g.get("grade") != "UNCONTENDED":
                why.append(f"ub {g.get('ub')} phase {g.get('why') or 'failed the A2 per-phase bars'} (median busy "
                           f"{g.get('median_busy_frac')}, {g.get('frac_windows_gt_25pct')} of {g.get('windows')} windows "
                           f"above 25 %, max {g.get('max_busy_frac')})")
    return grade, "; ".join(why) if why else "the predeclared sibling-load / proof condition was not met"


def _proofs(run_dir: Path, rows: list[dict]) -> dict:
    """``verdict_q.proofs`` (unchanged between the two verdict_q versions)."""
    done = sorted({r["ub"] for r in rows if r.get("kind") == "phase_done"})
    res, ok = {}, True
    for ub in done:
        f = run_dir / f"graphs_proof_ub{ub}.json"
        d = _json(f) if f.exists() else {}
        good = (d.get("graphs") == "ON" and (d.get("own_pid_kfd_vram_mib_at_ready") or 0) > 0
                and d.get("HSA_OVERRIDE_CPU_AFFINITY_DEBUG_in_environ") == ["HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0"])
        res[str(ub)] = {"graphs": d.get("graphs"), "own_pid_vram_mib_ready": d.get("own_pid_kfd_vram_mib_at_ready"),
                        "hsa": d.get("HSA_OVERRIDE_CPU_AFFINITY_DEBUG_in_environ"), "ok": good}
        ok &= good
    return {"per_ub": res, "all_ok": bool(done) and ok}


# --- per-layout derivation -------------------------------------------------------------------------------------------
def _metric_rows(common: dict, fit: dict, *, anchors: dict | None = None) -> list[dict]:
    anchors = anchors or {}
    return [{**common, "metric_key": "intercept", "value": fit["A"], "reps": fit["noise_n"], "anchor": anchors.get("A")},
            {**common, "metric_key": "c_slope", "value": fit["B"], "reps": fit["noise_n"], "anchor": anchors.get("B")},
            {**common, "metric_key": "noise_floor", "value": fit["noise_rmse"], "reps": fit["noise_n"]},
            {**common, "metric_key": "heldout_p99_abs", "value": fit["p99_abs"], "reps": fit["held_n"]}]


def derive_original(unit: Path) -> tuple[list[dict], dict]:
    P, prod_path, prod_sha = load_pure("original")
    summary = _json(unit / "summary.json")
    if not isinstance(summary, dict) or summary.get("schema") != RUN_SCHEMA or "analysis_q" in summary:
        raise Refused("summary.json is not an original-layout PB-QCOST-1 run record")
    rows = _rows(unit / "pb_qcost_1.jsonl")
    owners = [r for r in rows if r.get("kind") == "port_ownership"]
    if len(owners) != 1:
        raise Refused(f"pb_qcost_1.jsonl must hold exactly one port_ownership row (has {len(owners)})")
    argv = _argv(owners[0], "original")
    rb = (summary.get("server") or {}).get("readback") or {}
    if not (rb.get("proc_cmdline") == argv and rb.get("proc_cmdline_matches_argv") is True
            and rb.get("exe_matches_binary") is True and argv[0] == summary.get("binary")):
        raise Refused("summary.json server readback does not match the port_ownership argv / binary")
    digest = file_sha256(unit / "pb_qcost_1.jsonl")
    known = ORIGINAL_GRADES.get(digest)
    grade = known or {"grade": "UNGRADED", "cause": "the run record states no grade", "source": "", "source_sha256": None}
    inputs = {"pb_qcost_1.jsonl": digest, "summary.json": file_sha256(unit / "summary.json")}
    if (unit / "run_meta.json").is_file():
        inputs["run_meta.json"] = file_sha256(unit / "run_meta.json")
    base = {"layout": "original", "run_name": unit.name, "date": str(summary.get("finished") or ""),
            "np": summary.get("np"), "binary": summary.get("binary"), "binary_sha256": summary.get("binary_sha256"),
            "argv": argv, "argv_by_ub": {"512": argv}, "port_ownership": [_owner_brief(owners[0])],
            "predeclaration_sha256": None, "amendment_sha256": None,
            "rule": "the driver's docstring PRE-DECLARED PASS RULE (summary.json pass_rule); no PREDECLARATION.md",
            "grade": grade["grade"], "grade_cause": grade["cause"], "grade_source": grade["source"],
            "grade_source_sha256": grade["source_sha256"], "status": summary.get("status"),
            "overall_native": summary.get("overall"), "label": None}
    out = []
    for n in P.NS:
        il = (((summary.get("analysis") or {}).get("per_n") or {}).get(str(n)) or {}).get("interval_level") or {}
        fit = interval_fit(P, rows, n)
        if fit is None:
            continue
        held = il.get("refit_model_heldout") or {}
        _same(il.get("refit_intercept_ms_at_c0"), fit["A"], f"n={n} refit_intercept_ms_at_c0")
        _same(il.get("refit_slope_ms_per_cached_token"), fit["B"], f"n={n} refit_slope_ms_per_cached_token")
        _same(il.get("noise_floor_train_rmse_ms_per_iter"), fit["noise_rmse"], f"n={n} noise floor")
        _same(il.get("noise_floor_n_intervals"), fit["noise_n"], f"n={n} noise-floor interval count")
        _same(held.get("p99_abs"), fit["p99_abs"], f"n={n} refit held-out p99_abs")
        _same(held.get("n"), fit["held_n"], f"n={n} refit held-out count")
        common = {**base, "phase": "ub512", "n": n, "decode_rows": 8 * n, "q_tokens_per_iteration": _q_int(fit["q_median"]),
                  "q_held_fixed": True, "q_levels_by_ub": None, "verdict_native": (summary["analysis"]["per_n"][str(n)]
                                                                                   .get("verdict")), "ci95": None}
        out += _metric_rows(common, fit)
    return out, {"inputs": inputs, "producer": {"file": prod_path, "sha256": prod_sha,
                                                "identity_basis": "byte-identical copy of /mnt/raid0/llm/tmp/pb-qcost-1/"
                                                                  "pb_qcost_1.py, whose sha256 both confirm launchers "
                                                                  "pinned in ORIG_DRIVER.sha256; the run dir holds no "
                                                                  "driver copy of its own"}}


def derive_confirm(unit: Path) -> tuple[list[dict], dict]:
    P, prod_path, prod_sha = load_pure("confirm")
    run_dir = unit / "out" / "run"
    verdict = _json(unit / "VERDICT.json")
    record = _json(unit / "RUN_RECORD.json")
    summary = _json(run_dir / "summary.json")
    if not (isinstance(verdict, dict) and isinstance(verdict.get("confirm"), dict) and isinstance(verdict.get("q_axis"), dict)):
        raise Refused("VERDICT.json is not a PB-QCOST-1 confirm verdict")
    if not isinstance(summary, dict) or summary.get("schema") != RUN_SCHEMA:
        raise Refused("out/run/summary.json is not a PB-QCOST-1 run record")
    if record.get("driver_sha256") != prod_sha:
        raise Refused(f"RUN_RECORD.driver_sha256 {record.get('driver_sha256')} is not the pinned producer {prod_sha}")
    if (unit / "pb_qcost_1_q.py").is_file() and file_sha256(unit / "pb_qcost_1_q.py") != prod_sha:
        raise Refused("the unit's pb_qcost_1_q.py is not the pinned producer")
    pre_sha = verdict.get("predeclaration_sha256")
    if not (pre_sha and record.get("predeclaration_sha256") == pre_sha and (unit / "PREDECLARATION.md").is_file()
            and file_sha256(unit / "PREDECLARATION.md") == pre_sha):
        raise Refused("PREDECLARATION.md, RUN_RECORD and VERDICT do not agree on the predeclaration digest")
    am_sha = verdict.get("amendment_sha256")
    amendments = sorted(unit.glob("AMENDMENT*.md"))
    if am_sha or record.get("amendment_sha256") or amendments:
        if not (len(amendments) == 1 and record.get("amendment_sha256") == am_sha
                and file_sha256(amendments[0]) == am_sha):
            raise Refused("the AMENDMENT file, RUN_RECORD and VERDICT do not agree on the amendment digest")
    if (unit / "verdict_q.py").is_file() and file_sha256(unit / "verdict_q.py") != record.get("verdict_q_sha256"):
        raise Refused("verdict_q.py does not hash to RUN_RECORD.verdict_q_sha256")
    if record.get("label") != verdict.get("label"):
        raise Refused("RUN_RECORD and VERDICT disagree on the label")
    binary, binary_sha = record.get("binary"), record.get("binary_sha256")
    if not (summary.get("binary") == binary and summary.get("binary_sha256") == binary_sha
            and isinstance(binary_sha, str) and len(binary_sha) == 64):
        raise Refused("RUN_RECORD and summary disagree on the binary or its sha256")
    rows = _rows(run_dir / "pb_qcost_1.jsonl")
    owners = {r.get("ub"): r for r in rows if r.get("kind") == "port_ownership"}
    done = sorted({r["ub"] for r in rows if r.get("kind") == "phase_done"})
    if not done or set(done) - set(owners):
        raise Refused("a completed phase has no port_ownership row")
    argv_by_ub = {}
    for ub in done:
        argv = _argv(owners[ub], f"ub {ub}")
        gpf = run_dir / f"graphs_proof_ub{ub}.json"
        gp = _json(gpf) if gpf.exists() else {}
        if not (gp.get("proc_cmdline") == argv and gp.get("exe_matches_binary") is True and argv[0] == binary
                and _flag(argv, "-ub", "--ubatch-size") == str(ub) and _flag(argv, "-b", "--batch-size") == str(ub)):
            raise Refused(f"ub {ub} phase readback does not prove its argv / binary")
        argv_by_ub[str(ub)] = argv
    proofs = _proofs(run_dir, rows)
    _same(verdict.get("proof"), proofs, "VERDICT.proof")
    _same(verdict.get("phases_done"), done, "VERDICT.phases_done")
    usable = P.usable(rows)
    q_all = P.analyse_q(usable)
    _same(verdict["q_axis"], q_all, "VERDICT.q_axis")
    _same((summary.get("analysis_q") or {}).get("per_n"), q_all["per_n"], "summary.analysis_q.per_n")
    grade, cause = confirm_grade(unit, verdict)
    inputs = {"VERDICT.json": file_sha256(unit / "VERDICT.json"), "RUN_RECORD.json": file_sha256(unit / "RUN_RECORD.json"),
              "PREDECLARATION.md": pre_sha, "out/run/pb_qcost_1.jsonl": file_sha256(run_dir / "pb_qcost_1.jsonl"),
              "out/run/summary.json": file_sha256(run_dir / "summary.json")}
    for ub in done:
        if (run_dir / f"graphs_proof_ub{ub}.json").is_file():
            inputs[f"out/run/graphs_proof_ub{ub}.json"] = file_sha256(run_dir / f"graphs_proof_ub{ub}.json")
    for extra in ("out/sibling_summary.json", "out/sibling_load.csv"):
        if (unit / extra).is_file():
            inputs[extra] = file_sha256(unit / extra)
    if amendments:
        inputs[amendments[0].name] = am_sha
    base = {"layout": "confirm", "run_name": str(record.get("window_id") or unit.name),
            "date": str(record.get("time") or summary.get("finished") or ""), "np": summary.get("np"), "binary": binary,
            "binary_sha256": binary_sha, "argv": argv_by_ub.get("512") or argv_by_ub[str(done[0])],
            "argv_by_ub": argv_by_ub, "port_ownership": [_owner_brief(owners[u]) for u in done],
            "predeclaration_sha256": pre_sha, "amendment_sha256": am_sha,
            "rule": "PREDECLARATION.md" + (f" + {amendments[0].name}" if amendments else ""),
            "grade": grade, "grade_cause": cause, "grade_source": "VERDICT.json grade; cause read from the run's files",
            "grade_source_sha256": inputs["VERDICT.json"], "status": summary.get("status"),
            "overall_native": verdict.get("confirm_overall"), "label": verdict.get("label"), "phases_done": done,
            "proofs": proofs}
    out = []
    for n in P.NS:
        c = verdict["confirm"].get(str(n)) or {}
        fit = interval_fit(P, usable, n)
        if fit is not None and 512 in done:
            il = (((summary.get("analysis") or {}).get("per_n") or {}).get(str(n)) or {}).get("interval_level") or {}
            _same(c.get("A_new"), fit["A"], f"VERDICT confirm n={n} A_new")
            _same(c.get("B_new"), fit["B"], f"VERDICT confirm n={n} B_new")
            _same(c.get("noise_floor_ms_per_iter"), fit["noise_rmse"], f"VERDICT confirm n={n} noise floor")
            _same(c.get("refit_heldout_p99_abs_ms_per_iter"), fit["p99_abs"], f"VERDICT confirm n={n} p99_abs")
            _same(c.get("refit_heldout_p99_under_ms_per_iter"), fit["p99_under"], f"VERDICT confirm n={n} p99_under")
            _same(il.get("refit_intercept_ms_at_c0"), fit["A"], f"summary n={n} refit_intercept_ms_at_c0")
            common = {**base, "phase": "ub512", "n": n, "decode_rows": 8 * n,
                      "q_tokens_per_iteration": _q_int(fit["q_median"]), "q_held_fixed": True, "q_levels_by_ub": None,
                      "verdict_native": c.get("verdict"), "ci95": None}
            out += _metric_rows(common, fit, anchors={"A": c.get("A_old"), "B": c.get("B_old")})
        q = q_all["per_n"].get(n) or {}
        if not isinstance(q.get("coef"), dict):
            continue
        co = q["coef"]
        held = q.get("heldout_interval") or {}
        common = {**base, "phase": "q_axis", "n": n, "decode_rows": 8 * n, "q_tokens_per_iteration": None,
                  "q_held_fixed": False, "q_levels_by_ub": _rt(q.get("q_levels_by_ub")), "verdict_native": q.get("verdict")}
        nt = q.get("n_train_intervals")
        d_old = (q.get("d_vs_2026_10_21_refit") or {}).get("d_old")
        out += [{**common, "metric_key": "q_a", "value": co["a_ms"], "reps": nt, "ci95": q.get("ci95_a"), "anchor": None},
                {**common, "metric_key": "q_b", "value": co["b_ms_per_token"], "reps": nt, "ci95": q.get("ci95_b"),
                 "anchor": P.B_OFF},
                {**common, "metric_key": "q_d", "value": co["d_ms_per_token_cached"], "reps": nt, "ci95": q.get("ci95_d"),
                 "anchor": d_old},
                {**common, "metric_key": "q_noise_floor", "value": q.get("noise_floor_train_rmse_ms_per_iter"), "reps": nt,
                 "ci95": None},
                {**common, "metric_key": "q_heldout_p99_abs", "value": held.get("p99_abs"), "reps": held.get("n"),
                 "ci95": None}]
    return out, {"inputs": inputs, "producer": {"file": prod_path, "sha256": prod_sha,
                                                "identity_basis": "RUN_RECORD.driver_sha256 equals the pinned copy"}}


def layout_of(unit: Path) -> str | None:
    if (unit / "out" / "run" / "pb_qcost_1.jsonl").is_file() and (unit / "VERDICT.json").is_file():
        return "confirm"
    if (unit / "pb_qcost_1.jsonl").is_file() and (unit / "summary.json").is_file():
        return "original"
    return None


def build_rows(unit: Path, *, emission: str = "readout", emitted_at: str | None = None) -> list[dict]:
    unit = unit.resolve()
    layout = layout_of(unit)
    if layout is None:
        raise FileNotFoundError(f"{unit} is not a PB-QCOST-1 run unit (no original or confirm layout)")
    natives, ident = derive_original(unit) if layout == "original" else derive_confirm(unit)
    if not natives:
        raise Refused("the run has no fit to emit at any n")
    when = emitted_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    emitter_sha = file_sha256(Path(__file__).resolve())
    out = []
    for nat in natives:
        key = nat["metric_key"]
        metric, unit_s = METRICS[key]
        value = nat["value"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise Refused(f"{nat['phase']} n={nat['n']} {key} has no finite value")
        row = {"schema": SCHEMA, "run_dir": str(unit), **nat, "metric": metric, "unit": unit_s,
               "metric_direction": "lower_better",
               "reps_basis": HELD_BASIS if key.endswith("p99_abs") else TRAIN_BASIS,
               "measurement_key": f"{nat['run_name']}:{nat['phase']}:n{nat['n']}:{key}",
               "single_q_caveat": ("q varied by -b = -ub (q-axis tuple)" if nat["phase"] == "q_axis" else
                                   f"q held at {nat.get('q_tokens_per_iteration')} tokens/iteration at n={nat['n']}; the q "
                                   "coefficient is NOT measured by this row"),
               "inputs": ident["inputs"], "producer": ident["producer"],
               "emitter": {"file": EMITTER, "sha256": emitter_sha}, "emission": emission, "emitted_at": when}
        row.setdefault("anchor", None)
        row["content_sha256"] = content_sha256(row)
        out.append(row)
    return out


def write_rows(rows: list[dict], out: Path, *, overwrite: bool = False) -> Path:
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists (pass --overwrite to replace it)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=str(out.parent))
    with os.fdopen(fd, "w") as fh:
        for row in rows:
            fh.write(canonical(row) + "\n")
    os.replace(tmp, out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, help="the run unit (original run dir, or the confirm launcher dir)")
    ap.add_argument("--out", required=True, help="the belief_measurements.jsonl to write")
    ap.add_argument("--emission", choices=("readout", "backfill"), default="readout")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        rows = build_rows(Path(a.run_dir), emission=a.emission)
        path = write_rows(rows, Path(a.out), overwrite=a.overwrite)
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"belief_measurements NOT written: {exc}", file=sys.stderr)
        return 2
    except Refused as exc:
        print(f"belief_measurements REFUSED: {exc}", file=sys.stderr)
        return 3
    keys = sorted({(r["phase"], r["n"]) for r in rows})
    print(json.dumps({"written": str(path), "rows": len(rows), "layout": rows[0]["layout"], "grade": rows[0]["grade"],
                      "phases_n": [f"{p}:n{n}" for p, n in keys], "sha256": file_sha256(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
