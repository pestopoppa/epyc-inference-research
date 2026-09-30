"""RI-18 scorer: exact policy evaluation, paired bootstrap CIs, and the pre-registered rule.

Offline and re-runnable over banked records forever (deterministic replay). No inference.

POPULATION. Primary = items whose original answer was SCORED (stage 1 ``status == ok`` with a
non-empty text) AND whose quality detector did NOT fire (production would re-answer those with
``coder_escalation`` before the gate). Detector-fired items are their own reported stratum;
answer infra failures are reported by reason. Every metric below is on the primary population.

POLICIES (design §2.3), each acting on the same banked outcomes. For an item i: ``Y0`` =
correct(original), ``Y1`` = correct(final) where final = the revision when the production
verdict is WRONG (``_fast_revise`` already returns the original on an empty/failed revision) and
the original otherwise. A policy reviews a set R; ``d_i = Y1_i - Y0_i`` on R, 0 elsewhere.

* ``pi0`` never reviews; ``pi1`` reviews every item; ``piQ(t)`` reviews iff the production gate
  (answer-keyed KNN) fires at t (``avg_q < t``; never when ``skip_reason != scored``);
  ``piQq(t)`` is the exploratory question-keyed gate. Grid t = 0.30..1.00 step 0.05, plus 1.01.
* ``fixed``/``broken``/``net`` over R; ``net_per100 = 100 * net / |S|`` (S = the subset scored,
  not R); ``acc`` = mean of Y1 on R and Y0 elsewhere.

COST (GPU and CPU never summed). ``C_gpu(pi) = sum_R DS_verdict``, ``C_cpu(pi) = sum_{R, WRONG}
DS_rev`` (llama-server prompt+decode seconds, the ``_counters`` delta). Device-seconds per net
fix = C / net when net > 0, else undefined (cap not met).

LATENCY CAP (operator Q1, 2026-09-30), defined here explicitly: for the reviewed requests R of a
policy, ``added_i = wall(verdict_i) + wall(revision_i) * [WRONG_i]`` and ``base_i = wall(stage-1
answer_i)``. The cap holds iff ``p50(added over R) <= 0.20 * p50(base over R)`` -- the added p50
latency is at most +20% of the frontdoor answer's own p50 wall latency on those same requests.
Cost cap X = GPU <= 60 device-s per net fix AND CPU <= 120 device-s per net fix AND the latency
cap. ``cost_ok`` is False whenever a term is undefined.

CIs: 95% percentile intervals of a PAIRED bootstrap (items resampled with replacement, every
policy evaluated on the same resample), 10,000 resamples, stratified by suite
(gpqa / mmlu_pro / olympiadbench_hard), numpy ``default_rng(seed)``; the seed is recorded.
AUROC of ``-avg_q`` for ``Y0 = 0`` (Mann-Whitney, ties 1/2) on gate-eligible items
(``skip_reason == scored``), bootstrapped on the same resamples.

DECISION RULE: design §5, applied mechanically in ``apply_rule`` (see its docstring for each
clause's exact test, including the stated reading of clause 5).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import render
from .scoring import SCORER_PATH, is_correct, scorer_sha256
from .store import RunDir

SCORE_SCHEMA = "ri18-review-gate-score/v1"
BELIEF_SCHEMA = "ri18-review-gate-belief/v1"
BELIEF_SIDECAR = "belief_measurements.jsonl"
GRID = [round(0.30 + 0.05 * i, 2) for i in range(15)]           # t* search space
REPORT_GRID = GRID + [1.01]
PROD_T = 0.60
DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOT_SEED = 20260930
X_GPU_S_PER_FIX = 60.0
X_CPU_S_PER_FIX = 120.0
X_LATENCY_P50_RATIO = 0.20
MARGIN_PER100 = 2.0
REVIEW_RATE_MAX = 0.50
MIN_TRIGGERS = 30
UNAVAILABLE_MAX = 0.05
SUITES = ("gpqa", "mmlu_pro", "olympiadbench_hard")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sympy() -> None:
    try:
        import sympy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment
        raise SystemExit("S2 scoring needs sympy: run score under the orchestrator venv") from exc


# ── the per-item table ───────────────────────────────────────────────────────


def fires(avg_q: float | None, t: float) -> bool:
    return avg_q is not None and avg_q < t


def table_from_records(run: RunDir, items: list[dict[str, Any]], workload: dict[str, Any],
                       *, allow_incomplete: bool = False) -> dict[str, Any]:
    """One row per item from the VALID records of each stage, plus the problems found."""
    split = {i: "A" for i in workload["split"]["A"]} | {i: "B" for i in workload["split"]["B"]}
    ans, ver, vf = run.records("answer"), run.records("verdict"), run.records("vfull")
    rev, gate = run.records("revise"), run.records("gate")
    manifest = run.manifest()
    rows: list[dict[str, Any]] = []
    missing: Counter[str] = Counter()
    problems: list[str] = []
    for it in items:
        i = it["id"]
        a = ans.get(i)
        row: dict[str, Any] = {"id": i, "suite": it["suite"], "stratum": it["stratum"],
                               "split": split[i], "answer_status": None, "in_pop": False}
        rows.append(row)
        if a is None:
            missing["answer"] += 1
            row["answer_status"] = "missing"
            continue
        row["answer_status"] = a["status"] if a.get("answer_text") or a["status"] != "ok" else "empty"
        pinned_sha = (manifest.get("served") or {}).get("server_launch_git_sha")
        if a.get("served") and a["served"].get("server_launch_git_sha") != pinned_sha:
            problems.append(f"{i}: answer served commit {a['served']} != pinned {pinned_sha}")
        if row["answer_status"] != "ok":
            continue
        text = a["answer_text"]
        if it["scoring_method"] == "multiple_choice":
            from answer_scoring import extract_letter_answer  # noqa: PLC0415

            row["unparsed"] = extract_letter_answer(text) == ""
        row.update(answer_chars=len(text), qd=a.get("quality_issue"),
                   y0=int(is_correct(it, text)), ds_fd=a.get("request_device_seconds"),
                   wall_fd=a.get("wall_s"), served_role=a.get("served_role"))
        if row["qd"]:
            continue
        v = ver.get(i)
        g = gate.get(i)
        if v is None:
            missing["verdict"] += 1
        if g is None:
            missing["gate"] += 1
        if it["stratum"] == "S1" and i not in vf:
            missing["vfull"] += 1
        if v is not None and v["status"] == "wrong" and i not in rev:
            missing["revise"] += 1
        complete = v is not None and g is not None and not (
            v["status"] == "wrong" and i not in rev)
        if not complete:
            continue
        r = rev.get(i) if v["status"] == "wrong" else None
        final = r["final_text"] if r else text
        gs, gq = g["gate"], g["gate_question"]
        if v["status"] == "unavailable":
            row["unavailable_reason"] = ("call_error" if v.get("call_error") else
                                         "empty" if not (v.get("raw") or "").strip() else
                                         "unparseable")
        row.update(
            in_pop=True, vstatus=v["status"], wrong=v["status"] == "wrong",
            y1=int(is_correct(it, final)) if r else row["y0"],
            ds_verdict=v.get("device_seconds"), wall_verdict=v.get("wall_s"),
            ds_rev=(r or {}).get("device_seconds"), wall_rev=(r or {}).get("wall_s"),
            rev_changed=(r or {}).get("changed"), rev_failed=(r or {}).get("revise_failed"),
            vfull_status=(vf.get(i) or {}).get("status"),
            skip=gs["skip_reason"], avg_q=gs["avg_q"], skip_q=gq["skip_reason"],
            avg_q_q=gq["avg_q"], gate_controls_ok=g.get("controls_ok"),
            should_review=g.get("should_review"),
        )
        stored = g.get("fires") or {}
        for t in (0.0, *REPORT_GRID):
            key = f"{t:.2f}"
            if key in stored and bool(stored[key]) != fires(gs["avg_q"], t):
                problems.append(f"{i}: stored gate@{key} disagrees with avg_q {gs['avg_q']}")
        if not g.get("controls_ok"):
            problems.append(f"{i}: gate controls failed {g.get('controls')}")
    incomplete = sum(missing.values())
    if incomplete and not allow_incomplete:
        problems.append(f"incomplete run: missing records {dict(missing)}")
    from .store import SEGMENT_SCHEMA, SEGMENTS_NAME, read_jsonl

    infra_events = Counter(
        (ev.get("detail") or "")[:60] for ev in read_jsonl(run.out / SEGMENTS_NAME, SEGMENT_SCHEMA)
        if ev.get("event") == "infra")
    return {"rows": rows, "missing": dict(missing), "problems": problems,
            "infra_events": dict(infra_events)}


# ── exact policy arithmetic ──────────────────────────────────────────────────


def policy_sets(rows: list[dict[str, Any]]) -> dict[str, list[bool]]:
    """Policy name -> review indicator per row (rows are primary-population rows)."""
    out = {"pi0": [False] * len(rows), "pi1": [True] * len(rows)}
    for t in REPORT_GRID:
        out[f"piQ@{t:.2f}"] = [fires(r.get("avg_q"), t) for r in rows]
        out[f"piQq@{t:.2f}"] = [fires(r.get("avg_q_q"), t) for r in rows]
    out[f"piQ@{PROD_T:.2f}"] = [fires(r.get("avg_q"), PROD_T) for r in rows]
    return out


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q * 100))


def evaluate_policy(rows: list[dict[str, Any]], reviewed: list[bool]) -> dict[str, Any]:
    n = len(rows)
    rset = [r for r, on in zip(rows, reviewed) if on]
    fixed = sum(1 for r in rset if r["y0"] == 0 and r["y1"] == 1)
    broken = sum(1 for r in rset if r["y0"] == 1 and r["y1"] == 0)
    net = fixed - broken
    acc = (sum(r["y1"] if on else r["y0"] for r, on in zip(rows, reviewed)) / n) if n else None
    gpu = [r.get("ds_verdict") for r in rset]
    cpu = [r.get("ds_rev") for r in rset if r.get("wrong")]
    c_gpu = sum(v for v in gpu if isinstance(v, (int, float)))
    c_cpu = sum(v for v in cpu if isinstance(v, (int, float)))
    added = [(r.get("wall_verdict") or 0.0) + ((r.get("wall_rev") or 0.0) if r.get("wrong") else 0.0)
             for r in rset]
    base = [r["wall_fd"] for r in rset if isinstance(r.get("wall_fd"), (int, float))]
    p50_added, p50_base = _pct(added, 0.5), _pct(base, 0.5)
    ratio = (p50_added / p50_base) if (p50_added is not None and p50_base) else None
    gpu_fix = c_gpu / net if net > 0 else None
    cpu_fix = c_cpu / net if net > 0 else None
    cost_ok = (gpu_fix is not None and gpu_fix <= X_GPU_S_PER_FIX
               and cpu_fix is not None and cpu_fix <= X_CPU_S_PER_FIX
               and ratio is not None and ratio <= X_LATENCY_P50_RATIO)
    return {
        "n": n, "n_reviewed": len(rset), "review_rate": len(rset) / n if n else None,
        "fixed": fixed, "broken": broken, "net": net,
        "net_per100": 100.0 * net / n if n else None, "acc": acc,
        "cost_gpu_device_s": round(c_gpu, 6), "cost_cpu_device_s": round(c_cpu, 6),
        "cost_unmeasured_calls": sum(1 for v in gpu if not isinstance(v, (int, float)))
        + sum(1 for v in cpu if not isinstance(v, (int, float))),
        "gpu_s_per_net_fix": gpu_fix, "cpu_s_per_net_fix": cpu_fix,
        "added_latency_s": {"p50": p50_added, "p95": _pct(added, 0.95), "n": len(added)},
        "answer_latency_p50_s": p50_base, "latency_p50_ratio": ratio,
        "cost_ok": bool(cost_ok),
    }


def gate_pr(rows: list[dict[str, Any]], key: str = "avg_q") -> dict[str, Any]:
    """Precision/recall at every t; recall among eligible items and over the population."""
    wrong = [r for r in rows if r["y0"] == 0]
    elig_wrong = [r for r in wrong if r.get(key) is not None]
    out = {}
    for t in REPORT_GRID:
        fired = [r for r in rows if fires(r.get(key), t)]
        tp = sum(1 for r in fired if r["y0"] == 0)
        out[f"{t:.2f}"] = {
            "triggers": len(fired), "precision": tp / len(fired) if fired else None,
            "recall_eligible": tp / len(elig_wrong) if elig_wrong else None,
            "recall_population": tp / len(wrong) if wrong else None,
        }
    return out


def auroc(rows: list[dict[str, Any]], key: str = "avg_q") -> float | None:
    pos = [r[key] for r in rows if r.get(key) is not None and r["y0"] == 0]
    neg = [r[key] for r in rows if r.get(key) is not None and r["y0"] == 1]
    if not pos or not neg:
        return None
    # score = -avg_q; a wrong answer should have LOWER avg_q.
    wins = sum((1.0 if p < q else 0.5 if p == q else 0.0) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def reviewer_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    have = [r for r in rows if r.get(key) is not None]
    wrong_ans = [r for r in have if r["y0"] == 0]
    right_ans = [r for r in have if r["y0"] == 1]
    return {
        "n": len(have),
        "sensitivity": (sum(1 for r in wrong_ans if r[key] == "wrong") / len(wrong_ans)
                        if wrong_ans else None),
        "specificity": (sum(1 for r in right_ans if r[key] == "ok") / len(right_ans)
                        if right_ans else None),
        "unavailable_rate": (sum(1 for r in have if r[key] == "unavailable") / len(have)
                             if have else None),
        "status_counts": dict(Counter(r[key] for r in have)),
    }


def revision_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    w = [r for r in rows if r.get("wrong")]
    w0 = [r for r in w if r["y0"] == 0]
    w1 = [r for r in w if r["y0"] == 1]
    return {
        "n_wrong_verdicts": len(w),
        "fix_rate": sum(1 for r in w0 if r["y1"] == 1) / len(w0) if w0 else None,
        "break_rate": sum(1 for r in w1 if r["y1"] == 0) / len(w1) if w1 else None,
        "noop_rate": sum(1 for r in w if r.get("rev_changed") is False) / len(w) if w else None,
        "failed": sum(1 for r in w if r.get("rev_failed")),
    }


# ── bootstrap ────────────────────────────────────────────────────────────────


class Bootstrap:
    """Stratified (by suite) paired resampling weights over one subset of rows."""

    def __init__(self, rows: list[dict[str, Any]], resamples: int, seed: int) -> None:
        self.rows = rows
        self.n = len(rows)
        rng = np.random.default_rng(seed)
        w = np.zeros((resamples, self.n), dtype=np.float64)
        for suite in sorted({r["suite"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["suite"] == suite]
            counts = rng.multinomial(len(idx), [1.0 / len(idx)] * len(idx), size=resamples)
            w[:, idx] = counts
        self.w = w

    def ci(self, samples: np.ndarray) -> list[float | None]:
        s = samples[np.isfinite(samples)]
        if s.size == 0:
            return [None, None]
        lo, hi = np.percentile(s, [2.5, 97.5])
        return [float(lo), float(hi)]

    def net_per100(self, reviewed: list[bool]) -> np.ndarray:
        d = np.array([(r["y1"] - r["y0"]) if on else 0 for r, on in zip(self.rows, reviewed)],
                     dtype=np.float64)
        return 100.0 * (self.w @ d) / self.n

    def auroc(self, key: str = "avg_q") -> np.ndarray:
        pos = [i for i, r in enumerate(self.rows) if r.get(key) is not None and r["y0"] == 0]
        neg = [i for i, r in enumerate(self.rows) if r.get(key) is not None and r["y0"] == 1]
        if not pos or not neg:
            return np.full(self.w.shape[0], np.nan)
        qp = np.array([self.rows[i][key] for i in pos])[:, None]
        qn = np.array([self.rows[i][key] for i in neg])[None, :]
        k = (qp < qn).astype(np.float64) + 0.5 * (qp == qn)
        wp, wn = self.w[:, pos], self.w[:, neg]
        num = ((wp @ k) * wn).sum(axis=1)
        den = wp.sum(axis=1) * wn.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def rate(self, num_mask: list[bool], den_mask: list[bool]) -> np.ndarray:
        num = self.w @ np.array(num_mask, dtype=np.float64)
        den = self.w @ np.array(den_mask, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def tune_t_star(rows_a: list[dict[str, Any]]) -> dict[str, Any]:
    """argmax net on split A over t in 0.30..1.00 step 0.05 with review rate <= 50%.

    Ties: fewer reviews, then the lower t. ``None`` if no t is feasible.
    """
    cand = []
    for t in GRID:
        ev = evaluate_policy(rows_a, [fires(r.get("avg_q"), t) for r in rows_a])
        feasible = ev["review_rate"] is not None and ev["review_rate"] <= REVIEW_RATE_MAX
        cand.append({"t": t, "net": ev["net"], "n_reviewed": ev["n_reviewed"],
                     "review_rate": ev["review_rate"], "feasible": feasible})
    feasible = [c for c in cand if c["feasible"]]
    best = (max(feasible, key=lambda c: (c["net"], -c["n_reviewed"], -c["t"]))
            if feasible else None)
    return {"t_star": best["t"] if best else None, "candidates": cand}


# ── the rule ─────────────────────────────────────────────────────────────────


def _lo(ci: list[Any]) -> float | None:
    return ci[0] if ci else None


def apply_rule(inp: dict[str, Any]) -> dict[str, Any]:
    """Design §5, clauses 0-5 in order. ``inp`` carries every number the rule reads.

    Keys: ``void_reasons``; ``pi1_full`` / ``piQ06_full`` / ``pi1_B`` / ``piQt_B`` (each
    ``{"net_per100", "ci"}``); ``diff_B`` = {"piQt_minus_piQ06", "pi1_minus_piQ06"} each
    ``{"point", "ci"}``; ``auroc`` = {"point", "ci"}; ``cost_ok_B`` = {"pi1", "piQt"};
    ``cost_ok_full_pi1``; ``triggers_06``; ``t_star``.

    0 VOID   void_reasons non-empty (canary, unavailable > 5%, gate controls, drift, incomplete).
    1 DROP   pi1 full-set net CI upper < 0, OR neither pi1 nor piQ(t*) has a split-B net CI
             lower > 0.
    5        (modifier, evaluated here) eligible gate@0.6 triggers < 30 => piQ(0.6)'s own net is
             "untested"; clauses 2 and 3 then fall back to AUROC + pi1: every test of piQ(0.6)'s
             own net is replaced by "pi1 full-set net CI lower > 0", and clause 3 additionally
             requires it. (Stated reading: the comparative margins against piQ(0.6) stay the
             computed point differences.)
    2 KEEP@0.6  piQ(0.6) full net CI lower > 0 [clause 5: pi1 instead] AND AUROC CI lower > 0.5
             AND no challenger c in {piQ(t*), pi1} with split-B diff(c - piQ(0.6)) >= 2 per 100
             AND cost_ok_B[c].
    3 RETUNE t*  piQ(t*) split-B net CI lower > 0 AND AUROC CI lower > 0.5 AND split-B
             diff(piQ(t*) - piQ(0.6)) >= 2 per 100 AND cost_ok_B[piQt] [clause 5: AND pi1 full
             net CI lower > 0].
    4 REPLACE TRIGGER  AUROC CI contains 0.5 (an undefined AUROC counts as uninformative) AND
             pi1 full net CI lower > 0 AND pi1 full-set cost within X.
    else INCONCLUSIVE (write a BOUNDED-NULL-1 statement; the power note applies).
    """
    trail: list[dict[str, Any]] = []

    def note(clause: str, fired: bool, why: str) -> None:
        trail.append({"clause": clause, "fired": fired, "why": why})

    if inp["void_reasons"]:
        note("0", True, "; ".join(inp["void_reasons"]))
        return {"decision": "VOID", "clause": "0", "trail": trail}
    note("0", False, "no void condition")

    auc_ci = inp["auroc"]["ci"] or [None, None]
    auc_lo_ok = auc_ci[0] is not None and auc_ci[0] > 0.5
    auc_contains = auc_ci[0] is None or (auc_ci[0] <= 0.5 <= auc_ci[1])
    pi1_full_lo = _lo(inp["pi1_full"]["ci"])
    pi1_full_hi = inp["pi1_full"]["ci"][1] if inp["pi1_full"]["ci"] else None
    pi1_full_pos = pi1_full_lo is not None and pi1_full_lo > 0
    b_pos = {k: (_lo(inp[k]["ci"]) is not None and _lo(inp[k]["ci"]) > 0)
             for k in ("pi1_B", "piQt_B")}
    if inp["t_star"] is None:
        b_pos["piQt_B"] = False

    drop_a = pi1_full_hi is not None and pi1_full_hi < 0
    drop_b = not (b_pos["pi1_B"] or b_pos["piQt_B"])
    note("1", drop_a or drop_b,
         f"pi1 full net/100 CI {inp['pi1_full']['ci']} (upper<0: {drop_a}); split-B lower>0: "
         f"pi1 {inp['pi1_B']['ci']} -> {b_pos['pi1_B']}, piQ(t*={inp['t_star']}) "
         f"{inp['piQt_B']['ci']} -> {b_pos['piQt_B']}")
    if drop_a or drop_b:
        return {"decision": "DROP", "clause": "1", "trail": trail}

    untested = inp["triggers_06"] < MIN_TRIGGERS
    note("5", untested, f"eligible gate@0.6 triggers = {inp['triggers_06']} "
         f"({'<' if untested else '>='} {MIN_TRIGGERS}); piQ(0.6) own net "
         f"{'UNTESTED -> clauses 2/3 fall back to AUROC + pi1' if untested else 'tested'}")

    beats = {}
    for c, dkey in (("piQt", "piQt_minus_piQ06"), ("pi1", "pi1_minus_piQ06")):
        d = inp["diff_B"][dkey]
        ok = (d["point"] is not None and d["point"] >= MARGIN_PER100 and inp["cost_ok_B"][c]
              and (c != "piQt" or inp["t_star"] is not None))
        beats[c] = ok
    q06_lo = _lo(inp["piQ06_full"]["ci"])
    own = pi1_full_pos if untested else (q06_lo is not None and q06_lo > 0)
    keep = own and auc_lo_ok and not any(beats.values())
    note("2", keep, f"{'pi1 (clause-5 fallback)' if untested else 'piQ(0.6)'} full net CI lower "
         f"> 0: {own} (piQ(0.6) CI {inp['piQ06_full']['ci']}, pi1 CI {inp['pi1_full']['ci']}); "
         f"AUROC CI {auc_ci} lower > 0.5: {auc_lo_ok}; challengers beating by >= "
         f"{MARGIN_PER100}/100 within X on B: {beats} (diffs {inp['diff_B']})")
    if keep:
        return {"decision": "KEEP", "clause": "2", "threshold": PROD_T, "trail": trail}

    retune = (b_pos["piQt_B"] and auc_lo_ok and beats["piQt"]
              and (pi1_full_pos if untested else True))
    note("3", retune, f"piQ(t*={inp['t_star']}) split-B CI {inp['piQt_B']['ci']} lower > 0: "
         f"{b_pos['piQt_B']}; AUROC lower > 0.5: {auc_lo_ok}; beats piQ(0.6) on B by >= "
         f"{MARGIN_PER100} within X: {beats['piQt']}"
         + (f"; clause-5 pi1 full lower > 0: {pi1_full_pos}" if untested else ""))
    if retune:
        return {"decision": "RETUNE", "clause": "3", "threshold": inp["t_star"], "trail": trail}

    replace = auc_contains and pi1_full_pos and inp["cost_ok_full_pi1"]
    note("4", replace, f"AUROC CI {auc_ci} contains 0.5: {auc_contains}; pi1 full CI "
         f"{inp['pi1_full']['ci']} lower > 0: {pi1_full_pos}; pi1 cost within X: "
         f"{inp['cost_ok_full_pi1']}")
    if replace:
        return {"decision": "REPLACE_TRIGGER", "clause": "4", "trail": trail}
    note("none", True, "no clause fired: INCONCLUSIVE, write a BOUNDED-NULL-1 statement")
    return {"decision": "INCONCLUSIVE", "clause": None, "trail": trail}


def apply_vfull_rule(inp: dict[str, Any]) -> dict[str, Any]:
    """Secondary rule: V-full sensitivity beats cap-300 on split B (paired CI lower > 0) at
    equal or better specificity (point estimates) => land question_cap=1500 (C3), replay
    stages 3-4 on banked answers, re-apply the rule."""
    d = inp["delta_sensitivity_B"]
    spec_ok = (inp["spec_vfull_B"] is not None and inp["spec_cap300_B"] is not None
               and inp["spec_vfull_B"] >= inp["spec_cap300_B"])
    sens_ok = d["ci"][0] is not None and d["ci"][0] > 0
    land = bool(sens_ok and spec_ok and not inp.get("void_reasons"))
    return {"decision": "LAND_QUESTION_CAP_1500" if land else "HOLD_CAP_300",
            "why": f"delta sensitivity (V-full - cap300) on B = {d['point']} CI {d['ci']} "
                   f"(lower > 0: {sens_ok}); specificity V-full {inp['spec_vfull_B']} vs cap300 "
                   f"{inp['spec_cap300_B']} (>=: {spec_ok})",
            "then": "replay stages 3-4 on the banked answers (no frontdoor inference) and "
                    "re-apply the primary rule" if land else None}


# ── noise ────────────────────────────────────────────────────────────────────


def noise_report(run: RunDir, rows_by_id: dict[str, dict[str, Any]], items_by_id: dict[str, Any]
                 ) -> dict[str, Any]:
    nv = run.records("noise_verdict")
    nr = run.records("noise_revise")
    rev = run.records("revise")
    ver = run.records("verdict")
    pairs_v = [(ver[i]["status"], r["status"]) for i, r in nv.items() if i in ver]
    agree_v = (sum(1 for a, b in pairs_v if a == b) / len(pairs_v)) if pairs_v else None
    pairs_r = [(rev[i], r) for i, r in nr.items() if i in rev]
    agree_correct = (sum(1 for a, b in pairs_r if a["correct_final"] == b["correct_final"])
                     / len(pairs_r)) if pairs_r else None
    agree_text = (sum(1 for a, b in pairs_r if a["final_text"] == b["final_text"]) / len(pairs_r)
                  ) if pairs_r else None
    return {"verdict": {"n": len(pairs_v), "self_agreement": agree_v},
            "revision": {"n": len(pairs_r), "correctness_agreement": agree_correct,
                         "text_identical": agree_text}}


def noise_bound_per100(ev: dict[str, Any], n_wrong_reviewed: int, noise: dict[str, Any]) -> float | None:
    """Heuristic: items whose outcome could flip on a re-sample, per 100 items of the subset."""
    av, ar = noise["verdict"]["self_agreement"], noise["revision"]["correctness_agreement"]
    if av is None or ar is None or not ev["n"]:
        return None
    return 100.0 * ((1 - av) * ev["n_reviewed"] + (1 - ar) * n_wrong_reviewed) / ev["n"]


# ── top level ────────────────────────────────────────────────────────────────


def score_table(table: dict[str, Any], *, resamples: int = DEFAULT_RESAMPLES,
                boot_seed: int = DEFAULT_BOOT_SEED, extra_void: list[str] | None = None,
                noise: dict[str, Any] | None = None) -> dict[str, Any]:
    rows_all = table["rows"]
    pop = [r for r in rows_all if r.get("in_pop")]
    subsets = {
        "pooled": pop,
        "S1": [r for r in pop if r["stratum"] == "S1"],
        "S2": [r for r in pop if r["stratum"] == "S2"],
        "A": [r for r in pop if r["split"] == "A"],
        "B": [r for r in pop if r["split"] == "B"],
    }
    void: list[str] = list(extra_void or []) + list(table.get("problems") or [])
    ra = reviewer_accuracy(pop, "vstatus")
    if ra["unavailable_rate"] is not None and ra["unavailable_rate"] > UNAVAILABLE_MAX:
        void.append(f"verdict unavailable rate {ra['unavailable_rate']:.4f} > {UNAVAILABLE_MAX}")
    if not pop:
        void.append("empty primary population")

    t_info = tune_t_star(subsets["A"])
    t_star = t_info["t_star"]
    names = ["pi0", "pi1"] + [f"piQ@{t:.2f}" for t in REPORT_GRID] + [
        f"piQq@{t:.2f}" for t in REPORT_GRID]
    policies: dict[str, Any] = {}
    boots: dict[str, Bootstrap] = {}
    for sname, rows in subsets.items():
        if not rows:
            policies[sname] = {}
            continue
        sets = policy_sets(rows)
        boot = boots[sname] = Bootstrap(rows, resamples, boot_seed)
        tab = {}
        for name in names:
            ev = evaluate_policy(rows, sets[name])
            if sname != "A":
                ev["net_per100_ci"] = boot.ci(boot.net_per100(sets[name]))
            tab[name] = ev
        policies[sname] = tab

    def pol(sname: str, name: str) -> dict[str, Any]:
        return policies.get(sname, {}).get(name) or {"net_per100": None, "net_per100_ci": [None, None],
                                                    "cost_ok": False}

    tq = f"piQ@{t_star:.2f}" if t_star is not None else None
    diff_b: dict[str, Any] = {}
    if subsets["B"]:
        sets_b = policy_sets(subsets["B"])
        bb = boots["B"]
        base = bb.net_per100(sets_b["piQ@0.60"])
        for key, name in (("piQt_minus_piQ06", tq), ("pi1_minus_piQ06", "pi1")):
            if name is None:
                diff_b[key] = {"point": None, "ci": [None, None]}
                continue
            point = pol("B", name)["net_per100"] - pol("B", "piQ@0.60")["net_per100"]
            diff_b[key] = {"point": point, "ci": bb.ci(bb.net_per100(sets_b[name]) - base)}
    else:
        diff_b = {k: {"point": None, "ci": [None, None]}
                  for k in ("piQt_minus_piQ06", "pi1_minus_piQ06")}

    auc = {"point": auroc(pop), "ci": boots["pooled"].ci(boots["pooled"].auroc())
           if pop else [None, None]}
    auc_q = {"point": auroc(pop, "avg_q_q"),
             "ci": boots["pooled"].ci(boots["pooled"].auroc("avg_q_q")) if pop else [None, None]}
    triggers_06 = sum(1 for r in pop if fires(r.get("avg_q"), PROD_T))

    rule_in = {
        "void_reasons": void,
        "pi1_full": {"net_per100": pol("pooled", "pi1")["net_per100"],
                     "ci": pol("pooled", "pi1").get("net_per100_ci")},
        "piQ06_full": {"net_per100": pol("pooled", "piQ@0.60")["net_per100"],
                       "ci": pol("pooled", "piQ@0.60").get("net_per100_ci")},
        "pi1_B": {"net_per100": pol("B", "pi1")["net_per100"],
                  "ci": pol("B", "pi1").get("net_per100_ci")},
        "piQt_B": {"net_per100": pol("B", tq)["net_per100"] if tq else None,
                   "ci": pol("B", tq).get("net_per100_ci") if tq else [None, None]},
        "diff_B": diff_b,
        "auroc": auc,
        "cost_ok_B": {"pi1": pol("B", "pi1")["cost_ok"],
                      "piQt": pol("B", tq)["cost_ok"] if tq else False},
        "cost_ok_full_pi1": pol("pooled", "pi1")["cost_ok"],
        "triggers_06": triggers_06,
        "t_star": t_star,
    }
    verdict = apply_rule(rule_in)
    verdict["inputs"] = rule_in

    # secondary V-full rule: S1 items of split B with both verdicts
    s1b = [r for r in subsets["B"] if r["stratum"] == "S1" and r.get("vfull_status")]
    vfull_in: dict[str, Any] = {"void_reasons": void, "delta_sensitivity_B":
                                {"point": None, "ci": [None, None]},
                                "spec_vfull_B": None, "spec_cap300_B": None, "n": len(s1b)}
    if s1b:
        bv = Bootstrap(s1b, resamples, boot_seed)
        wrong0 = [r["y0"] == 0 for r in s1b]
        sens_v = bv.rate([r["vfull_status"] == "wrong" and r["y0"] == 0 for r in s1b], wrong0)
        sens_c = bv.rate([r["vstatus"] == "wrong" and r["y0"] == 0 for r in s1b], wrong0)
        acc_v = reviewer_accuracy(s1b, "vfull_status")
        acc_c = reviewer_accuracy(s1b, "vstatus")
        point = (acc_v["sensitivity"] - acc_c["sensitivity"]
                 if acc_v["sensitivity"] is not None and acc_c["sensitivity"] is not None else None)
        vfull_in.update(delta_sensitivity_B={"point": point, "ci": bv.ci(sens_v - sens_c)},
                        spec_vfull_B=acc_v["specificity"], spec_cap300_B=acc_c["specificity"])
    vfull_rule = apply_vfull_rule(vfull_in)
    vfull_rule["inputs"] = vfull_in

    per_stratum = {}
    for sname, rows in subsets.items():
        if not rows:
            continue
        per_stratum[sname] = {
            "n": len(rows), "accuracy_original": sum(r["y0"] for r in rows) / len(rows),
            "coverage_by_skip_reason": dict(Counter(r["skip"] for r in rows)),
            "coverage_by_skip_reason_question_key": dict(Counter(r["skip_q"] for r in rows)),
            "gate_pr": gate_pr(rows), "gate_pr_question_key": gate_pr(rows, "avg_q_q"),
            "auroc_neg_avg_q": auroc(rows), "auroc_neg_avg_q_question": auroc(rows, "avg_q_q"),
            "reviewer": reviewer_accuracy(rows, "vstatus"),
            "reviewer_vfull": reviewer_accuracy([r for r in rows if r.get("vfull_status")],
                                                "vfull_status"),
            "revision": revision_effect(rows),
            "answer_chars_over_1500": sum(1 for r in rows if (r.get("answer_chars") or 0) > 1500),
            # S1 letter not extractable (scored wrong). Watch it on the first 20 items: the render
            # line shows `Answer: <letter>` in backticks, and "Answer: `B`" does not parse.
            "s1_unparsed_original": sum(1 for r in rows if r.get("unparsed")),
        }
    if noise is not None:
        for sname, tab in policies.items():
            rows = subsets[sname]
            sets = policy_sets(rows) if rows else {}
            for name, ev in tab.items():
                nw = sum(1 for r, on in zip(rows, sets[name]) if on and r.get("wrong"))
                ev["noise_bound_per100"] = noise_bound_per100(ev, nw, noise)

    infra = {
        "answer_status": dict(Counter(r["answer_status"] for r in rows_all)),
        "verdict_unavailable_by_reason": dict(Counter(
            r.get("unavailable_reason") or "?" for r in pop if r.get("vstatus") == "unavailable")),
        "segment_infra_events": table.get("infra_events") or {},
        "quality_detector_fired": sum(1 for r in rows_all if r.get("qd")),
        "verdict_unavailable": ra["status_counts"].get("unavailable", 0),
        "revision_failed": sum(1 for r in pop if r.get("rev_failed")),
        "missing_records": table.get("missing"),
    }
    qd_rows = [r for r in rows_all if r.get("qd")]
    return {
        "schema": SCORE_SCHEMA,
        "n_items": len(rows_all), "n_primary": len(pop),
        "n_by_subset": {k: len(v) for k, v in subsets.items()},
        "quality_detector_stratum": {"n": len(qd_rows), "accuracy_original": (
            sum(r["y0"] for r in qd_rows) / len(qd_rows) if qd_rows else None),
            "reasons": dict(Counter(r["qd"] for r in qd_rows))},
        "policies": policies,
        "t_star_tuning_split_A": t_info,
        "auroc_neg_avg_q": auc, "auroc_neg_avg_q_question": auc_q,
        "triggers_at_0_60_eligible": triggers_06,
        "per_stratum": per_stratum,
        "infra": infra,
        "noise": noise,
        "cost_cap_X": {"gpu_s_per_net_fix": X_GPU_S_PER_FIX, "cpu_s_per_net_fix": X_CPU_S_PER_FIX,
                       "latency": f"p50(added over reviewed) <= {X_LATENCY_P50_RATIO} * "
                                  "p50(frontdoor answer wall over the same reviewed requests)",
                       "source": "operator Q1, 2026-09-30"},
        "bootstrap": {"resamples": resamples, "seed": boot_seed, "stratified_by": "suite",
                      "interval": "percentile 2.5/97.5", "paired": True},
        "verdict": verdict,
        "vfull_rule": vfull_rule,
        "caveat": "External validity: both strata are closed-form scored tasks (MC with a brief "
                  "justification; olympiad math), not open chat; the gate's Q comes from "
                  "memories spanning all suites.",
    }


def scorer_sanity(run: RunDir, items: list[dict[str, Any]], seed: int = 18, n: int = 10
                  ) -> list[dict[str, Any]]:
    from answer_scoring import extract_letter_answer  # noqa: PLC0415 (sys.path set by scoring)

    ans = run.records("answer")
    s1 = [it for it in items if it["stratum"] == "S1" and it["id"] in ans]
    pick = random.Random(seed).sample(s1, min(n, len(s1)))
    return [{"id": it["id"], "expected": it["expected"],
             "extracted": extract_letter_answer(ans[it["id"]]["answer_text"] or ""),
             "tail": (ans[it["id"]]["answer_text"] or "")[-160:]} for it in pick]


def belief_rows(result: dict[str, Any], *, run_id: str, records_sha: str, run_dir: Path,
                scored_at: str, items_sha256: str) -> list[dict[str, Any]]:
    """ClaimTuple-projectable rows (mirrors ``thesis_ufh13/score.py::belief_rows``).

    ``protocol_id`` is EMPTY on purpose: until RI-18's protocol is codified every row grades as
    an observation. The producer writes; an adapter (not yet registered) projects.
    """
    common = {"schema": BELIEF_SCHEMA, "date": scored_at, "protocol_id": "",
              "reps_basis": "scored", "attestation_path": str((run_dir / "score.json").resolve()),
              "attestation_sha256": records_sha,
              "attestation_locator": f"run_id={run_id};items_sha256={items_sha256}",
              "source_kind": "ri18-review-gate-measurement"}
    extra_common = {"run_id": run_id, "items_sha256": items_sha256,
                    "decision": result["verdict"]["decision"],
                    "clause": result["verdict"]["clause"]}
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any, *, direction: str, unit: str, claim: str, reps: int,
            extra: dict[str, Any] | None = None, cat: str = "CANDIDATE") -> None:
        if value is None:
            return
        rows.append({**common, "measurement_id": f"ri18:{run_id}:{metric}", "metric": metric,
                     "value": value, "category": cat, "claim": claim,
                     "metric_direction": direction, "reps": reps, "unit": unit,
                     "extra": {**extra_common, **(extra or {})}})

    pol = result["policies"]
    n = result["n_primary"]
    nb = result["n_by_subset"].get("B", 0)
    for name, label in (("pi1", "review every item"), ("piQ@0.60", "production gate (avg_q<0.6)")):
        ev = pol.get("pooled", {}).get(name)
        if ev:
            add(f"ri18.net_per100.{name}", ev["net_per100"], direction="higher_better",
                unit="items/100", reps=n, cat="BASELINE" if name == "piQ@0.60" else "CANDIDATE",
                claim=f"RI-18 net fixes per 100 items, {label}, primary population",
                extra={"ci95": ev.get("net_per100_ci"), "fixed": ev["fixed"],
                       "broken": ev["broken"]})
            add(f"ri18.gpu_s_per_net_fix.{name}", ev["gpu_s_per_net_fix"],
                direction="lower_better", unit="s", reps=n,
                claim=f"RI-18 GPU (:8083) device-seconds per net fix, {label}")
            add(f"ri18.cpu_s_per_net_fix.{name}", ev["cpu_s_per_net_fix"],
                direction="lower_better", unit="s", reps=n,
                claim=f"RI-18 CPU (:8070) device-seconds per net fix, {label}")
    t_star = result["t_star_tuning_split_A"]["t_star"]
    if t_star is not None:
        ev = pol.get("B", {}).get(f"piQ@{t_star:.2f}")
        if ev:
            add("ri18.net_per100.piQ_tstar_splitB", ev["net_per100"], direction="higher_better",
                unit="items/100", reps=nb,
                claim=f"RI-18 net fixes per 100, gate at tuned t*={t_star}, confirm split B",
                extra={"ci95": ev.get("net_per100_ci"), "t_star": t_star})
    add("ri18.auroc.neg_avg_q", result["auroc_neg_avg_q"]["point"], direction="higher_better",
        unit="fraction", reps=n, claim="RI-18 AUROC of -avg_q for frontdoor-wrong (eligible items)",
        extra={"ci95": result["auroc_neg_avg_q"]["ci"]})
    rv = result["per_stratum"].get("pooled", {}).get("reviewer", {})
    add("ri18.verdict.unavailable_rate", rv.get("unavailable_rate"), direction="lower_better",
        unit="fraction", reps=n, claim="RI-18 production verdict unavailable rate")
    add("ri18.gate.triggers_at_0_60", result["triggers_at_0_60_eligible"],
        direction="higher_better", unit="count", reps=n,
        claim="RI-18 production gate triggers at 0.6 on the primary population")
    return rows


def score_run(run_dir: Path, *, allow_incomplete: bool = False, resamples: int = DEFAULT_RESAMPLES,
              boot_seed: int = DEFAULT_BOOT_SEED, items_path: Path | None = None,
              workload_manifest_path: Path | None = None) -> dict[str, Any]:
    run = RunDir(run_dir)
    manifest = run.manifest()
    items, workload = render.load_workload(items_path or render.ITEMS_PATH,
                                           workload_manifest_path or render.MANIFEST_PATH)
    if manifest["items_sha256"] != workload["items_sha256"]:
        raise SystemExit("run manifest items sha != frozen workload sha")
    if any(it["stratum"] == "S2" for it in items):
        require_sympy()
    table = table_from_records(run, items, workload, allow_incomplete=allow_incomplete)
    segs = run.segments()
    void_segments = {k: v["void_reasons"] for k, v in segs.items() if v["void"]}
    rows_by_id = {r["id"]: r for r in table["rows"]}
    noise = noise_report(run, rows_by_id, {it["id"]: it for it in items})
    result = score_table(table, resamples=resamples, boot_seed=boot_seed, noise=noise)
    files = sorted(p for p in run_dir.glob("*.jsonl") if p.name != BELIEF_SIDECAR)
    records_sha = hashlib.sha256(b"".join(_sha256(p).encode() for p in files)).hexdigest()
    scored_at = _now()
    result.update({
        "run_id": manifest["run_id"], "scored_at": scored_at, "records_sha256": records_sha,
        "record_files": {p.name: _sha256(p) for p in files},
        "scorer_path": str(SCORER_PATH), "scorer_sha256": scorer_sha256(),
        "items_sha256": workload["items_sha256"], "void_segments_superseded": void_segments,
        "segments": {k: {kk: v[kk] for kk in ("kind", "void", "canaries_ok", "exit")}
                     for k, v in segs.items()},
        "scorer_sanity_sample": scorer_sanity(run, items),
        "allow_incomplete": allow_incomplete,
        "dry_run": manifest.get("dry_run"),
    })
    (run_dir / "score.json").write_text(json.dumps(result, indent=2, sort_keys=True,
                                                   default=_json_default) + "\n")
    rows = belief_rows(result, run_id=manifest["run_id"], records_sha=records_sha,
                       run_dir=run_dir, scored_at=scored_at, items_sha256=workload["items_sha256"])
    tmp = run_dir / (BELIEF_SIDECAR + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True, default=_json_default) + "\n"
                           for r in rows))
    tmp.replace(run_dir / BELIEF_SIDECAR)
    return result


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(type(obj).__name__)


def summary(result: dict[str, Any]) -> dict[str, Any]:
    v = result["verdict"]
    pol = result["policies"].get("pooled", {})
    return {
        "decision": v["decision"], "clause": v["clause"],
        "trail": [f"{t['clause']}: {'FIRED' if t['fired'] else 'no'} - {t['why']}"
                  for t in v["trail"]],
        "vfull_rule": result["vfull_rule"]["decision"],
        "n_primary": result["n_primary"], "t_star": result["t_star_tuning_split_A"]["t_star"],
        "net_per100": {k: [pol[k]["net_per100"], pol[k].get("net_per100_ci")]
                       for k in ("pi1", "piQ@0.60") if k in pol},
        "auroc": result["auroc_neg_avg_q"],
    }
