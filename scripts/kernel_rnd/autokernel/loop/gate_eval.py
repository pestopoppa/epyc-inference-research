#!/usr/bin/env python3
"""Deterministic gate evaluator: RESULT.json -> verdict.json, no LLM (AK parallel loop P1).

Design: epyc-root docs/design/autokernel-parallel-loop.md §2/§4 (P1). Replay report:
docs/design/autokernel-parallel-loop-p1-replay.md.

Inputs, all plain data:
  * the row's RESULT.json (dict) — or None when the row wrote none;
  * the run-end record `grants/<row>.ended` parsed to a dict (rc, reason) — optional;
  * `graph_kld_policy.json` (27B-anchored KLD / |dPPL| / top-1 gate, in force from its ruling date);
  * the speed floors in `bench.MEASURED_FLOOR_PCT` (scaled exactly as run.noise_floor_pct does) and a
    row's own pre-registered speed floor (`rule.geomean_min`);
  * the PPL contract (`gates.PPL_REL_BAR`, `gates.PPL_CONTRACT_AGREEMENT_FLOOR`);
  * lifecycle facts (superseded_by / withdrawn) — optional.

Layers, first decisive one wins except that a policy-gate FAIL always overrides a declared PASS:
  L0 lifecycle   superseded -> SUPERSEDED, withdrawn -> WITHDRAWN
  L1 run end     no finished RESULT (no RESULT, rc 75/143, reason != complete, RESULT complete:false)
                 -> NO RESULT, NO CLAIM (operator ruling 2026-11-21): SUPERSEDED when a successor row
                 exists (lifecycle["successor"]), otherwise WITHDRAWN; never FUTILITY/inconclusive.
                 A declared FAIL on such a run is still a measured negative (FAIL).
  L2 policy      graph-KLD gate on model-keyed blocks (27B / Q38FN) for the row's backend, the PPL
                 contract, the bench speed floor -> PASS / FAIL / NOISE
  L3 declared    the row's own pre-registered verdict token (job.py evaluated its RULE) classified by
                 a fixed vocabulary: candidate-attributable negatives FAIL, harness/instrument
                 failures FUTILITY, a named answer on a complete run PASS

Inconclusive (FUTILITY) is reserved for rows that RAN to a finished RESULT but could not decide.

Polar tokens (row templates since 2026-11-21, row_scaffold.py): one PASS/FAIL token per row in RESULT["polar"]
beside the named answer in RESULT["verdict"] (polar wins over the legacy reading of the answer), and split futility tokens
FUTILITY_HARNESS (instrument/harness, -> FUTILITY) vs FUTILITY_CANDIDATE (candidate negative, -> FAIL).
The legacy vocabulary below is still accepted.

Verdict alphabet PASS / FAIL / NOISE / FUTILITY (+ SUPERSEDED / WITHDRAWN); `filed_verdict` maps it to
the filer's pass / fail / inconclusive (NOISE is a measured null: fail, the DROP convention of
runend.map_verdict). Every verdict names the deciding gate id, thresholds and observed values.

CLI:  python3 -m autokernel.loop.gate_eval RESULT.json [--ended grants/<row>.ended] [--row ID]
                                           [--backend gpu|cpu] [--out verdict.json] [--successor ID]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "epyc.autokernel.gate_verdict.v1"
POLICY_PATH = Path(__file__).with_name("graph_kld_policy.json")
#: The 27B-anchored policy is in force for rows run on/after its ruling (operator 2026-11-18).
POLICY_IN_FORCE_FROM = "2026-11-18"
#: Ended reasons that mean the job ran to its own end (direct runs bypass the keeper's `complete`).
COMPLETE_REASONS = frozenset({"complete", "direct-run"})
#: The no-result filing rule id (operator ruling 2026-11-21: "No result -> no claim").
NO_CLAIM_GATE = "filing.no_result_no_claim"
#: Verdicts the filer can apply with zero LLM turns: decided by a policy gate, a lifecycle fact, the
#: no-result rule, or a polar row token. A legacy-vocabulary L3 token still gets owner review.
POLICY_DECIDED_RULES = frozenset({NO_CLAIM_GATE, "lifecycle.superseded", "lifecycle.withdrawn",
                                  "declared.polar", "declared.futility_harness", "declared.futility_candidate"})
FILED = {"PASS": "pass", "FAIL": "fail", "NOISE": "fail", "FUTILITY": "inconclusive",
         "SUPERSEDED": "SUPERSEDED", "WITHDRAWN": "WITHDRAWN"}


def _constants() -> dict[str, Any]:
    """Speed floors and PPL contract from the loop's own modules (single source of truth)."""
    try:
        from . import bench, gates
        return {"floors": bench.MEASURED_FLOOR_PCT, "floor_model": bench.MEASURED_FLOOR_MODEL_STEM,
                "ppl_rel_bar": gates.PPL_REL_BAR, "agreement_floor": gates.PPL_CONTRACT_AGREEMENT_FLOOR}
    except Exception:      # noqa: BLE001 - standalone use (tests, replay from another tree)
        return {"floors": {"pp512": {1: 2.175, 3: 1.432, 5: 0.479, 9: 0.168, 20: 0.029},
                           "tg128": {1: 3.452, 3: 2.527, 5: 2.422, 9: 2.021, 20: 1.188}},
                "floor_model": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M",
                "ppl_rel_bar": 0.005, "agreement_floor": 0.98}


def load_policy(path: Path | str = POLICY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def speed_floor_pct(floors: Mapping[str, Mapping[int, float]], surface: str, pairs: int) -> float | None:
    """run.noise_floor_pct's rule: max(sigma/sqrt(n) from the k=1 row, measured row at or below n)."""
    table = floors.get(surface)
    if not table:
        return None
    n = max(1, int(pairs))
    rows = {int(k): float(v) for k, v in table.items()}
    measured = rows[max(k for k in rows if k <= n)]
    return round(max(rows[1] / math.sqrt(n), measured), 4)


# ---- L3 vocabulary -----------------------------------------------------------------------------

#: Harness / instrument / environment failures: the candidate was not judged (AK-FAILRULE-1 harness side).
_HARNESS = re.compile(r"FOREIGN|WINDOW|CONTENTION|RESIDENCY|INSTRUMENT|CARRIER|BUILD|WRONG_GRANT|INPUT|"
                      r"UNPARSED|PROOF|IDENTITY|INSENSITIVE|NOT_REGISTERED|NO_DEGENERATE|TBO|SHARE|"
                      r"CONTROL_BAD|NATIVE_FAIL|TESTS_FAIL|SURFACE_FUTILITY|NEAR_UNPARSED")
#: A FAIL(...) whose stated cause is an invalid / contended / unrun measurement, not the candidate.
_INVALID_RUN = re.compile(r"^FAIL\s*\(\s*(visits\s+)?(invalid|contended|missing)|invalid-run|\(invalid:", re.I)
#: FUTILITY sub-reasons that are measured negatives about the candidate itself.
_FUTILITY_IS_FAIL = re.compile(r"^FUTILITY_(BELOW_BASELINE|ORACLE)\b")
_FAIL = re.compile(r"^(FAIL|DROP|LOSS|NO_GAIN|NO_EFFECT|NO_WIN|NEUTRAL\b|REJECT|PERF_COST|COST_|NOT_EXACT|"
                   r"GAIN_NOT_IDENTICAL|IDENTITY_REPLAY_FAIL|QDEC_FAIL|PREFLIGHT_FAIL|KEEP_BAR_MISS|"
                   r"EXECUTOR_\w*MISSED|L4_BELOW|NO_CRASH_BUT_NONZERO_RC|LADDER_DONE)|[-_]LOSS|_FAIL$")
_FUTILE = re.compile(r"^(FUTILITY|INCONCLUSIVE|NOT_JUDGEABLE|STOPPED|PARTIAL|INVALID|VOID|UNDERFILL|"
                     r"CONTROL_NONDETERMINISTIC|FIXTURES_PARTIAL|ABSTAIN)|_NO_KEEP$|_PARTIAL$")


def classify_token(native: str) -> tuple[str | None, str]:
    """Declared verdict string -> (PASS|FAIL|FUTILITY|None, rule id). None = no declared verdict."""
    text = str(native or "").strip()
    if not text:
        return None, "declared.none"
    tok = re.split(r"[\s:;,(]", text, maxsplit=1)[0].upper()
    # Polar row-template tokens (2026-11-21) win over the legacy vocabulary.
    if tok.startswith("FUTILITY_HARNESS"):
        return "FUTILITY", "declared.futility_harness"
    if tok.startswith("FUTILITY_CANDIDATE"):
        return "FAIL", "declared.futility_candidate"
    if tok.startswith("DROP") and re.match(r"DROP\s*\(\s*diagnostic", text, re.I):
        # A diagnostic row localises a defect; it makes no claim about the candidate.
        return "FUTILITY", "declared.diagnostic"
    if tok == "FAIL" and _INVALID_RUN.search(text):
        return "FUTILITY", "declared.invalid_run"
    if _FUTILITY_IS_FAIL.match(tok):
        return "FAIL", "declared.futility_candidate_negative"
    if tok.startswith(("FAIL", "FUTILITY", "SURFACE_FUTILITY", "NATIVE_FAIL", "TESTS_FAIL")) and _HARNESS.search(tok):
        return "FUTILITY", "declared.harness_failure"
    if _FUTILE.search(tok):
        return "FUTILITY", "declared.futility"
    if _FAIL.search(tok):
        return "FAIL", "declared.negative"
    return "PASS", "declared.answered"


# ---- L2 policy gate ----------------------------------------------------------------------------

_MODEL_27B = re.compile(r"(^|[^0-9])(q38_?)?27b", re.I)
_MODEL_Q38FN = re.compile(r"q38fn|flash.?next", re.I)


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _first(block: Mapping[str, Any], *keys: str) -> float | None:
    for k in keys:
        v = _num(block.get(k))
        if v is not None:
            return v
    return None


def _metrics(block: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    kld = _first(block, "mean_kld", "kld_mean", "cand_kld")
    if kld is not None:
        out["mean_kld"] = abs(kld)
    rel = _first(block, "ppl_rel", "dppl_rel")
    if rel is not None:
        out["abs_dppl_pct"] = abs(rel) * 100.0
    else:
        pct = _first(block, "dppl_pct", "cand_dppl_pct")
        if pct is not None:
            out["abs_dppl_pct"] = abs(pct)
        else:
            q, b = _first(block, "ppl_q", "ppl_cand"), _first(block, "ppl_base", "ppl_ref")
            if q is not None and b:
                out["abs_dppl_pct"] = abs(q / b - 1.0) * 100.0
    top = _first(block, "same_top_p_pct", "same_top1_pct", "same_top1_pct_info", "top1_pct")
    if top is None:
        frac = _first(block, "top1_agree", "same_top1")
        if frac is not None:
            top = frac * 100.0 if frac <= 1.0 else frac
    if top is not None:
        out["same_top_pct"] = top
    agree = _first(block, "ppl_contract_agreement", "agreement")
    if agree is not None:
        out["agreement"] = agree
    return out


def model_blocks(result: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Model-keyed metric blocks under RESULT's gate containers (`gates`, `kld`, `correctness`,
    `quality`, top level), depth <= 2. Arms named anchor/base/aa are never the candidate."""
    found: dict[str, dict[str, float]] = {}
    containers = [result] + [result[k] for k in ("gates", "kld", "correctness", "quality", "graph_kld")
                             if isinstance(result.get(k), Mapping)]
    for cont in containers:
        for key, block in cont.items():
            if not isinstance(block, Mapping) or re.search(r"(^|_)(aa|anchor|base)(_|$)", str(key), re.I):
                continue
            model = ("q38fn" if _MODEL_Q38FN.search(str(key)) else
                     "q38_27b" if _MODEL_27B.search(str(key)) else None)
            if model is None:
                continue
            m = _metrics(block)
            if m and model not in found:
                found[model] = m
    return found


def _check(gate: str, metric: str, observed: float, op: str, threshold: float) -> dict[str, Any]:
    ok = observed <= threshold if op == "<=" else observed >= threshold
    return {"gate": gate, "metric": metric, "observed": round(observed, 6), "op": op,
            "threshold": threshold, "passed": bool(ok)}


def policy_checks(result: Mapping[str, Any], policy: Mapping[str, Any], backend: str | None,
                  consts: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    blocks = model_blocks(result)
    for spec in (policy.get("backends", {}).get(backend or "", []) if backend else []):
        m = blocks.get(spec.get("model"))
        if not m or not spec.get("decisive"):
            continue
        gid = f"graph_kld.{backend}.{spec['model']}"
        if "max_mean_kld" in spec and "mean_kld" in m:
            checks.append(_check(gid, "mean_kld", m["mean_kld"], "<=", spec["max_mean_kld"]))
        if "max_abs_dppl_pct" in spec and "abs_dppl_pct" in m:
            checks.append(_check(gid, "abs_dppl_pct", m["abs_dppl_pct"], "<=", spec["max_abs_dppl_pct"]))
        if "min_same_top_pct" in spec and "same_top_pct" in m:
            checks.append(_check(gid, "same_top_pct", m["same_top_pct"], ">=", spec["min_same_top_pct"]))
    for model, m in blocks.items():
        if "agreement" in m:
            checks.append(_check(f"ppl_contract.{model}", "agreement", m["agreement"], ">=",
                                 consts["agreement_floor"]))
    rule = result.get("rule") if isinstance(result.get("rule"), Mapping) else {}
    gm, gmin = _num(result.get("geomean")), _num(rule.get("geomean_min"))
    if gm is not None and gmin is not None:
        checks.append(_check("speed.row_geomean_floor", "geomean", gm, ">=", gmin))
    sp = result.get("speed") if isinstance(result.get("speed"), Mapping) else None
    if sp and sp.get("surface") in consts["floors"] and _num(sp.get("effect_pct")) is not None \
            and consts["floor_model"] in str(sp.get("model", "")):
        floor = speed_floor_pct(consts["floors"], sp["surface"], int(sp.get("pairs") or 1))
        eff = float(sp["effect_pct"])
        checks.append({"gate": f"speed.bench_floor.{sp['surface']}", "metric": "effect_pct",
                       "observed": eff, "op": "|x|>", "threshold": floor,
                       "passed": eff > floor, "noise": abs(eff) <= floor})
    return checks


# ---- evaluator ---------------------------------------------------------------------------------

def backend_of(row: str | None, result: Mapping[str, Any] | None) -> str | None:
    for src in ((result or {}).get("backend"), (result or {}).get("lane")):
        if src in ("gpu", "cpu"):
            return src
    m = re.search(r"(^|-)(gpu|cpu)(-|$)", str(row or ""))
    return m.group(2) if m else None


def _row_date(row: str | None, result: Mapping[str, Any] | None) -> str | None:
    for k in ("at", "updated_utc", "ts"):
        v = (result or {}).get(k)
        if isinstance(v, str) and re.match(r"\d{4}-\d\d-\d\d", v):
            return v[:10]
    m = re.search(r"-(20\d\d)(\d\d)(\d\d)", str(row or ""))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _verdict(v: str, gate: str, layer: str, why: str, checks=(), native=None, **extra) -> dict[str, Any]:
    return _finalize({"schema": SCHEMA, "verdict": v, "filed_verdict": FILED[v], "gate_id": gate, "layer": layer,
            "rationale": why, "checks": list(checks), "native_verdict": native, **extra})


def _finalize(v: dict[str, Any]) -> dict[str, Any]:
    v["policy_decided"] = policy_decided(v)
    return v


def evaluate(result: Mapping[str, Any] | None, *, ended: Mapping[str, str] | None = None,
             row: str | None = None, backend: str | None = None,
             policy: Mapping[str, Any] | None = None, lifecycle: Mapping[str, Any] | None = None,
             consts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Pure: same inputs, same verdict. Never raises on malformed RESULT content."""
    consts = consts or _constants()
    policy = policy if policy is not None else load_policy()
    lifecycle = lifecycle or {}
    ended = ended or {}
    if lifecycle.get("superseded_by"):
        return _verdict("SUPERSEDED", "lifecycle.superseded", "L0",
                        f"superseded by {lifecycle['superseded_by']} (1 successor row)")
    if lifecycle.get("withdrawn"):
        return _verdict("WITHDRAWN", "lifecycle.withdrawn", "L0",
                        f"withdrawn ({lifecycle.get('withdrawn')}); 0 verdict-bearing results")
    rc, reason = str(ended.get("rc", "0")), ended.get("reason", "complete")
    if reason in COMPLETE_REASONS:
        reason = "complete"

    def no_claim(why: str, native_s: str | None = None) -> dict[str, Any]:
        succ = lifecycle.get("successor")
        if succ:
            return _verdict("SUPERSEDED", NO_CLAIM_GATE, "L1", f"{why}; no result -> no claim; successor {succ}",
                            native=native_s, successor=succ)
        return _verdict("WITHDRAWN", NO_CLAIM_GATE, "L1", f"{why}; no result -> no claim; 0 successor rows",
                        native=native_s)

    if not isinstance(result, Mapping):
        return no_claim(f"no RESULT.json; ended rc={rc} reason={reason}")
    native = result.get("verdict")
    native_s = "" if native is None else (native if isinstance(native, str) else json.dumps(native))
    declared, rule_id = classify_token(native_s)
    polar = str(result.get("polar") or "").strip().upper()
    if polar in ("PASS", "FAIL") and rule_id not in ("declared.futility_harness", "declared.futility_candidate"):
        declared, rule_id = polar, "declared.polar"
    if rc != "0" or (reason and reason != "complete" and not ended.get("rc")):
        if declared == "FAIL":
            return _verdict("FAIL", rule_id, "L1+L3", f"ended rc={rc} reason={reason}; declared {native_s[:80]}",
                            native=native_s)
        return no_claim(f"run did not finish: ended rc={rc} reason={reason}", native_s)
    if result.get("complete") is False and declared != "FAIL":
        return no_claim(f"RESULT complete=false; declared {native_s[:80] or '(none)'}; rc={rc}", native_s)
    date = _row_date(row, result)
    in_force = date is None or date >= POLICY_IN_FORCE_FROM
    be = backend or backend_of(row, result)
    checks = policy_checks(result, policy, be, consts) if in_force else []
    failed = [c for c in checks if not c["passed"] and not c.get("noise")]
    noise = [c for c in checks if c.get("noise")]
    obs = "; ".join(f"{c['gate']} {c['metric']}={c['observed']} {c['op']} {c['threshold']}" for c in checks)
    if failed:
        return _verdict("FAIL", failed[0]["gate"], "L2",
                        f"policy gate failed: {obs}; declared {native_s[:60] or '(none)'}", checks, native_s)
    if declared is not None:
        if declared == "PASS" and noise:
            return _verdict("NOISE", noise[0]["gate"], "L2", f"effect inside the speed floor: {obs}",
                            checks, native_s)
        why = f"declared {native_s[:120]} ({rule_id}); rc={rc}"
        if obs:
            why += f"; policy checks passed: {obs}"
        return _verdict(declared, rule_id if not checks or declared != "PASS" else checks[0]["gate"],
                        "L3" if not checks else "L2+L3", why, checks, native_s)
    if checks:
        if noise:
            return _verdict("NOISE", noise[0]["gate"], "L2", f"effect inside the speed floor: {obs}", checks)
        return _verdict("PASS", checks[0]["gate"], "L2", f"all policy gates passed: {obs}", checks)
    for key in ("pass", "ok", "passed"):
        if isinstance(result.get(key), bool):
            v = "PASS" if result[key] else "FAIL"
            return _verdict(v, f"declared.bool_{key}", "L3", f"RESULT {key}={result[key]}; rc={rc} (1 flag)")
    if result.get("complete") is True or result.get("partial") is False:
        return _verdict("PASS", "declared.complete_no_verdict", "L3",
                        f"no declared verdict; RESULT complete (data-collection row); rc={rc} (0 gates applicable)")
    return _verdict("FUTILITY", "declared.none", "L3",
                    f"no declared verdict and no policy-gated metric; rc={rc} (0 gates applicable)")


def policy_decided(verdict: Mapping[str, Any]) -> bool:
    """True when the verdict may be filed with zero LLM turns (runend --gate-eval --file)."""
    return verdict.get("gate_id") in POLICY_DECIDED_RULES or str(verdict.get("layer", "")).startswith("L2") \
        and verdict.get("verdict") in ("PASS", "FAIL", "NOISE") and bool(verdict.get("checks"))


def parse_ended(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in re.split(r"\s+", text or ""):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out.setdefault(k, v)
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path)
    ap.add_argument("--ended", type=Path)
    ap.add_argument("--row")
    ap.add_argument("--backend", choices=["gpu", "cpu"])
    ap.add_argument("--policy", type=Path, default=POLICY_PATH)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--successor", help="successor row id (no-result rule: SUPERSEDED instead of WITHDRAWN)")
    a = ap.parse_args(list(argv) if argv is not None else None)
    try:
        result = json.loads(a.result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = None
    ended = parse_ended(a.ended.read_text(encoding="utf-8")) if a.ended and a.ended.exists() else {}
    v = evaluate(result, ended=ended, row=a.row, backend=a.backend, policy=load_policy(a.policy),
                 lifecycle={"successor": a.successor} if a.successor else None)
    text = json.dumps(v, indent=1, sort_keys=True)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
