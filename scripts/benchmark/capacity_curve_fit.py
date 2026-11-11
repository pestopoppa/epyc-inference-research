#!/usr/bin/env python3
"""INF-45 capacity curves: aggregate decode tok/s = f(np, context_depth) per role x shape.

Box: epyc-root numa-placement-and-contention.md -> artifact (2) "Per-role capacity curve" /
"Fit per role x shape". Input is the E5 v2 cell results (one hot server per model; concurrency
varied by requests fired, depth by prompt). Banked W0 e1parity twins are admitted only when
banked_qualification.jsonl tags them QUALIFIED.

FORM (4 parameters, linear in log space, fitted by ordinary least squares):

    log y = a + b*log(np) - c*d - e*(np - 1)*d          d = depth / 1000 tokens

``e`` is the np x depth INTERACTION. A separable A*f(depth)*g(np) (e = 0) cannot express the
measured np=2 turnover at ~15k: d/d(np) of log y = b/np - e*d, so the aggregate peaks at
np* = b / (e*d), which moves with depth only when e != 0. ``--form separable`` fits e = 0 for
comparison. A fit is ACCEPTED only when every point's relative residual is < 3 %.

Output (``--output`` or, with ``--write``, the orchestrator's
``orchestration/capacity_curves.v1.json``, the router's cold-start prior): per (role/model, shape)
the coefficients, the residual per point, and the raw points with their source rows as
attestation. Claims cite the measurement rows (``points[].source``), never the curve.
Zero inference: this reads results files only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "capacity-curves/1"
RESIDUAL_MAX = 0.03
ORCH_TARGET = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/capacity_curves.v1.json")
PARAMS = ("a", "b", "c", "e")


# ---------------------------------------------------------------------------
# least squares (pure python; 3-4 columns)
# ---------------------------------------------------------------------------

def _solve(m: list[list[float]], v: list[float]) -> list[float]:
    n = len(v)
    aug = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular design: the points do not identify every parameter")
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col:
                f = aug[r][col] / aug[col][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _design_row(np_: float, depth: float, form: str) -> list[float]:
    d = depth / 1000.0
    row = [1.0, math.log(np_), -d]
    if form == "interaction":
        row.append(-(np_ - 1.0) * d)
    return row


def predict(coef: dict, np_: float, depth: float) -> float:
    d = depth / 1000.0
    return math.exp(coef["a"] + coef["b"] * math.log(np_) - coef["c"] * d - coef.get("e", 0.0) * (np_ - 1.0) * d)


def fit(points: list[dict], form: str = "interaction") -> dict:
    """points: [{np, depth, tps, ...}]. Returns coefficients, residuals, acceptance."""
    if form not in ("interaction", "separable"):
        raise ValueError(form)
    k = 4 if form == "interaction" else 3
    if len(points) < k + 1:
        raise ValueError(f"{len(points)} points cannot fit {k} parameters with a residual check")
    X = [_design_row(float(p["np"]), float(p["depth"]), form) for p in points]
    y = [math.log(float(p["tps"])) for p in points]
    xtx = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
    beta = _solve(xtx, xty)
    coef = dict(zip(PARAMS, beta))
    if form == "separable":
        coef["e"] = 0.0
    resid = []
    for p in points:
        pred = predict(coef, float(p["np"]), float(p["depth"]))
        resid.append({"np": p["np"], "depth": p["depth"], "observed_tps": p["tps"], "predicted_tps": pred,
                      "rel_residual": (p["tps"] - pred) / p["tps"]})
    worst = max(abs(r["rel_residual"]) for r in resid)
    out = {"form": form, "coefficients": coef, "residuals": resid, "max_abs_rel_residual": worst,
           "accepted": worst < RESIDUAL_MAX, "residual_max": RESIDUAL_MAX}
    if form == "interaction" and coef["e"] > 0 and coef["b"] > 0:
        out["np_peak_at_depth"] = {str(dd): coef["b"] / (coef["e"] * dd / 1000.0) for dd in (4096, 8192, 16384, 32768)}
    return out


# ---------------------------------------------------------------------------
# E5 v2 run ingestion
# ---------------------------------------------------------------------------

def shape_of(row: dict) -> str:
    inst = row.get("instances") or []
    return "|".join(f"{i.get('cpu_list')}@{i.get('numactl_policy', 'none')}" for i in inst)


def row_tps(row: dict) -> float | None:
    v = row.get("aggregate_decode_tps")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def row_depth(row: dict) -> float | None:
    pt, n = row.get("prompt_tokens_total"), row.get("success_count")
    if isinstance(pt, (int, float)) and isinstance(n, int) and n > 0 and pt > 0:
        return pt / n
    return None


def load_points(run_dirs: list[Path], qualified_banked: set[str] | None = None,
                banked_runs: list[Path] | None = None) -> tuple[dict, list[dict]]:
    """Group usable cells by (model_key, shape). A v2 cell needs decision_grade true and a PASS
    policy attestation; a banked cell needs a QUALIFIED tag. Every exclusion is reported."""
    groups: dict[tuple[str, str], list[dict]] = {}
    excluded = []

    def ingest(run: Path, banked: bool):
        att = {}
        ap = run / "policy_attestation.jsonl"
        if ap.exists():
            for line in ap.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    att[r["cell_id"]] = r.get("policy_attestation")
        for line in (run / "cells.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("cell_id")
            why = None
            if banked:
                if cid not in (qualified_banked or set()):
                    why = "banked cell not tagged QUALIFIED"
            else:
                if row.get("decision_grade") is not True:
                    why = f"decision_grade false: {row.get('decision_grade_blockers')}"
                elif att.get(cid) != "PASS":
                    why = f"policy attestation {att.get(cid)!r}"
            tps, depth = row_tps(row), row_depth(row)
            if why is None and (tps is None or depth is None):
                why = "no aggregate_decode_tps or prompt depth"
            src = {"run_dir": str(run), "cell_id": cid,
                   "row_sha256": hashlib.sha256(line.encode()).hexdigest()}
            if why:
                excluded.append({**src, "reason": why})
                continue
            key = (row.get("model_key"), shape_of(row))
            groups.setdefault(key, []).append({"np": row["np"], "depth": depth, "tps": tps, "source": src})

    for r in run_dirs:
        ingest(r, banked=False)
    for r in banked_runs or []:
        ingest(r, banked=True)
    return groups, excluded


def build_document(groups: dict, excluded: list[dict]) -> dict:
    curves = []
    for (model, shape), pts in sorted(groups.items()):
        entry = {"model_key": model, "shape": shape, "n_points": len(pts), "points": pts}
        try:
            entry["fit"] = fit(pts, "interaction")
            entry["separable_reference"] = fit(pts, "separable")
        except ValueError as exc:
            entry["fit"] = {"accepted": False, "error": str(exc)}
        curves.append(entry)
    return {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
            "form": "log y = a + b*log(np) - c*d - e*(np-1)*d, d = depth/1000",
            "consumer": "router cold-start prior; claims cite points[].source, never the curve",
            "curves": curves, "excluded": excluded}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", type=Path, default=[], help="E5 v2 run dir (repeatable)")
    ap.add_argument("--banked-run-dir", action="append", type=Path, default=[],
                    help="banked run dir (W0 e1parity twins); cells admitted only if QUALIFIED")
    ap.add_argument("--qualification", type=Path, default=None,
                    help="banked_qualification.jsonl from e5_v2_qualify_banked.py")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--write", action="store_true", help=f"write {ORCH_TARGET}")
    a = ap.parse_args(argv)
    if not a.run_dir:
        ap.error("at least one --run-dir")
    qualified = set()
    if a.qualification:
        for line in a.qualification.read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                if t.get("verdict") == "QUALIFIED":
                    qualified.add(t["cell_id"])
    elif a.banked_run_dir:
        ap.error("--banked-run-dir needs --qualification (no re-use without a tag)")
    groups, excluded = load_points(a.run_dir, qualified, a.banked_run_dir)
    doc = build_document(groups, excluded)
    text = json.dumps(doc, indent=2) + "\n"
    dest = ORCH_TARGET if a.write else a.output
    if dest:
        dest.write_text(text)
        print(f"wrote {dest}: {len(doc['curves'])} curves, {len(excluded)} excluded cells")
    else:
        sys.stdout.write(text)
    return 0 if all(c["fit"].get("accepted") for c in doc["curves"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
