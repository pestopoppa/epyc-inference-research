#!/usr/bin/env python3
"""KPF-18-S1.R4: command `kpf18r4` (item 187) of the KPF-18 gate (kpf18_gate.py).

handoffs/active/kv-prefix-fork-and-paged-attention.md, box "KPF-18-S1.R4" (filed 2026-10-24): localise the 60k fork
deviation that R3 found fork-specific. R3's cold arms fell into two bit-identical occupancy classes and none ran at the
child's own occupancy, so R3 cannot separate a fork effect from a third occupancy class.
  r4   one load, g5 P = trunk 60000 + suffix0, each arm teacher-forced over forced_seq_183p4.json (kpf18_s1r.tf_arm):
       source slot 0 (checkpoint_at [T]); child slot 1 fork (residents {0}); erase 1, coldE slot 1 cold (residents {0},
       the child's occupancy); erase 1, child2 slot 1 fork (the fork A/A); coldA slot 2 (residents {0,1}), coldB slot 3
       (residents {0,1,2}) = R3's two cold classes; erase 1-3, decoy on slot 1 (group-6 trunk cut to child2's length,
       no shared prefix with P, so it can never be a fork source), childX slot 2 fork (residents {0,1} = coldA's
       occupancy in sequence count and cells).
       floor = d(coldA, coldB) (the R3 floor re-measured on this load); cold_max = max d over the three cold pairs.
       dFF = d(child, child2), dCE = d(child, coldE): FORK_PATH_NOT_REPRODUCIBLE iff dFF > 0; OCCUPANCY_NUMERICS iff
       dCE <= floor; FORK_PATH_SPECIFIC iff dCE > cold_max; BETWEEN_FLOOR_AND_COLD_SPREAD otherwise.
The decision rules are the launcher's PREDECLARATION.md; its sha256 (--s1r-predeclaration) is stamped into every record
(`r4.predeclaration_sha256`) and verdict() refuses records made under another sha. RULES[187] is the short machine copy.

Reused unchanged: the S1R inputs and plumbing (--s1r-predeclaration / --s1r-forced-seq / --s1r-tf-positions /
--s1r-long-group / --s1r-budget-factor, kpf18_s1r._prepare, _need, tf_arm), kpf18_r3's _d / pair_key / missing-line
check / cross_load shape and its step open/close pattern. The pure evaluators and verdict() take no server.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from pathlib import Path

import kpf18_arms as ARMS
import kpf18_s1r as S1R
import kpf18_r3 as R3

ITEM = 187
CMD = "kpf18r4"
TASK = "KPF-18-S1.R4"

FORK_ARMS = ("child", "child2", "childX")
COLD_ARMS = ("coldA", "coldB", "coldE")
JUDGED = ("child", "child2", "coldA", "coldB", "coldE")             # the rule's five arms
CX = "childX"                                                       # zero weight
ALL_ARMS = JUDGED + (CX,)
COLD_PAIRS = tuple(itertools.combinations(COLD_ARMS, 2))           # cold_max pairs
FLOOR_PAIR = ("coldA", "coldB")                                    # the R3 floor (R3: A≡C, B≡D, max = the A-B pair)
JUDGED_PAIRS = tuple(itertools.combinations(JUDGED, 2))            # the 10 recorded pairs
R3_FLOOR_RAW = 0.027456215011625318                                # R3 VERDICT item3_r3.floor_raw (recorded only)
STEP_KEYS = ("r4",)
# (name, slot, first-request mode, slots erased before, residents expected just before the arm). PREDECLARATION order.
# childX: a fork child resident beside slot 0 would itself be the best fork source (KPF-11: largest checkpoint p over
# every other slot; on the fake server childX forked from child2's near-end checkpoint at 60508, not from slot 0 at T),
# so coldA's second resident is replaced for childX by a DECOY of the same length that shares no prefix with P.
_FORK = {"fork": True, "cache": True, "ckpt": None}
PLAN = (("child", 1, "fork", (), (0,)),
        ("coldE", 1, "cold", (1,), (0,)),
        ("child2", 1, "fork", (1,), (0,)),
        ("coldA", 2, "cold", (), (0, 1)),
        ("coldB", 3, "cold", (), (0, 1, 2)),
        ("childX", 2, "fork", (1, 2, 3), (0, 1)))
DECOY_GROUP = 6                                                   # trunk 72000 tokens; LCP with g5 = 7 (chat header)
DECOY_SLOT = 1
DECOY_BEFORE = "childX"
RESIDENTS = {name: res for name, _s, _m, _e, res in PLAN}
SLOTS = {name: s for name, s, _m, _e, _r in PLAN}
# Estimate calibration (the R3 module constants, measured by kpf18-s1-remainder on this build + argv: cold 60512-token
# prefill 113.9 / 115.2 s; R3 itself measured 114-130 s; 0.114 s per teacher-forced position).
COLD60K_PF_S = R3.COLD60K_PF_S
TF_POS_S = R3.TF_POS_S

ARMS.ARM_ITEMS = tuple(sorted(set(ARMS.ARM_ITEMS) | {ITEM}))
ARMS.ARM_CMDS[CMD] = [ITEM]
ARMS.TASK_OF[ITEM] = TASK

ARMS.RULES[ITEM] = {
    "common": ("Rules: PREDECLARATION.md of the launcher (sha256 in every record as r4.predeclaration_sha256; verdict() "
               "refuses another sha). One load of the argv file as given, one step r4. Counts and values are reported, "
               "never a pass word."),
    "r4": ("18.3 part 4 localisation, g5 P = trunk 60000 + suffix0: source slot 0 (checkpoint_at [T]); child slot 1 fork "
           "(residents {0}); erase 1, coldE slot 1 cache_prompt false checkpoint_at [T] (residents {0}); erase 1, child2 "
           "slot 1 fork (residents {0}); coldA slot 2 (residents {0,1}); coldB slot 3 (residents {0,1,2}); erase 1-3, "
           "decoy slot 1 = g6 trunk cut to len(P)+K-1 (LCP with P < fork-min), childX slot 2 fork (residents {0,1}). "
           "Position i = kpf18_s1r.tf_arm. d = mean |dlp| over positions valid in "
           "child, child2, coldA, coldB, coldE. floor = d(coldA,coldB) lifted to 1e-4; cold_max = max d over the 3 cold "
           "pairs; dFF = d(child,child2); dCE = d(child,coldE). INVALID iff a judged arm fails its validity (timings or "
           "residents); ABSTAIN iff < 16 common or len(P) < 50000; FORK_PATH_NOT_REPRODUCIBLE iff dFF > 0; "
           "OCCUPANCY_NUMERICS iff dCE <= floor; FORK_PATH_SPECIFIC iff dCE > cold_max; BETWEEN_FLOOR_AND_COLD_SPREAD "
           "otherwise. childX (2x2 reading) and the cross-load reading vs R3 have zero weight. Forced sequence sha256 "
           + S1R.FORCED_SHA256),
}

# PREDECLARATION.md "Result lines": a missing one makes the verdict FAIL_MISSING_RESULT_LINE.
RESULT_LINES = (
    ("item3_r4", "verdict"), ("item3_r4", "floor_used"), ("item3_r4", "cold_max"), ("item3_r4", "d_child_vs_coldE"),
    ("item3_r4", "d_child_vs_child2"), ("item3_r4", "n_common_valid"), ("item3_r4", "arm_valid"), ("item3_r4", "d"),
    ("item3_r4", "occupancy_before"), ("item3_r4", "factorial_reading"), ("item3_r4", "cross_load_vs_r3"),
    ("run_valid",),
)


# ================================================================ CLI
def add_arguments(ap) -> None:
    g = ap.add_argument_group("KPF-18-S1.R4 (command kpf18r4, item 187; also uses the --s1r-* inputs)")
    g.add_argument("--r4-steps", default=",".join(STEP_KEYS), help="subset of " + ",".join(STEP_KEYS))


# ================================================================ pure evaluators (no server)
pair_key = R3.pair_key
_d = R3._d


def residents(occ: dict | None, slot: int) -> list[int] | None:
    """Other slots with n_prompt_tokens > 0 in a /slots snapshot ({str(id): n}); None if there is no snapshot."""
    if not occ:
        return None
    return sorted(int(s) for s, n in occ.items() if n and int(s) != slot)


def arm_validity(name: str, first: dict, occ: dict | None, *, T: int, n_task: int, S0: int) -> list[str]:
    """Position-0 timings and residents of an r4 arm against its predeclared role (empty list = valid)."""
    f = first or {}
    nf = f.get("n_fork_tokens") or 0
    bad = []
    if name in FORK_ARMS:
        if not (nf == T and f.get("fork_src_kind") == "checkpoint" and f.get("fork_src_slot") == 0):
            bad.append(f"{name}_not_forked_at_T:{f.get('n_fork_tokens')},{f.get('fork_src_kind')},{f.get('fork_src_slot')}")
        if f.get("prompt_n") != S0:
            bad.append(f"{name}_prompt_n_{f.get('prompt_n')}_ne_{S0}")
    elif name in COLD_ARMS:
        if f.get("cache_n") != 0 or f.get("prompt_n") != n_task or nf:
            bad.append(f"{name}_not_cold:{f.get('cache_n')},{f.get('prompt_n')},{f.get('n_fork_tokens')}")
    else:
        bad.append(f"unknown_arm_{name}")
        return bad
    want = list(RESIDENTS[name])
    have = residents(occ, SLOTS[name])
    if have != want:
        bad.append(f"{name}_residents_{have}_ne_{want}")
    return bad


def factorial_reading(lp: dict, common: list[int], cx_valid: bool, floor: float) -> dict:
    """Zero-weight 2x2 (fork/recompute x child's/coldA's occupancy) over common positions also valid in childX."""
    cx = lp.get(CX) or []
    pos = [i for i in common if i < len(cx) and cx[i] is not None] if cx_valid else []
    if len(pos) < ARMS.MIN_COMMON_POSITIONS:
        return {"label": "CHILDX_NOT_ESTABLISHED", "n_positions": len(pos), "childX_valid": cx_valid,
                "weight": "zero (recorded only)"}
    cells = {"fork_vs_cold_at_child_occupancy": _d(lp, "child", "coldE", pos),
             "fork_vs_cold_at_coldA_occupancy": _d(lp, CX, "coldA", pos),
             "fork_across_occupancy": _d(lp, "child", CX, pos),
             "cold_across_occupancy": _d(lp, "coldE", "coldA", pos)}
    return {"label": "CHILDX_RECORDED", "n_positions": len(pos), "childX_valid": cx_valid, "cells": cells,
            "childX_vs_coldA_within_floor": cells["fork_vs_cold_at_coldA_occupancy"]["mean_abs_dlogprob"] <= floor,
            "weight": "zero (recorded only)"}


def decoy_validity(decoy: dict | None, occ_cx: dict | None, *, n_min: int) -> list[str]:
    """The decoy resident childX needs: cold prefill of its whole prompt, no fork, an LCP with P below the fork minimum,
    and still resident on DECOY_SLOT with its full length just before childX."""
    d = decoy or {}
    t = d.get("first") or {}
    bad = []
    if not d:
        return ["decoy_missing"]
    if d.get("lcp_with_P") is None or d["lcp_with_P"] >= n_min:
        bad.append(f"decoy_lcp_with_P_{d.get('lcp_with_P')}_ge_fork_min_{n_min}")
    if t.get("cache_n") != 0 or t.get("prompt_n") != d.get("n") or (t.get("n_fork_tokens") or 0):
        bad.append(f"decoy_not_cold:{t.get('cache_n')},{t.get('prompt_n')},{t.get('n_fork_tokens')}")
    have = (occ_cx or {}).get(str(DECOY_SLOT))
    if have != d.get("n"):
        bad.append(f"decoy_not_resident_before_childX:{have}_ne_{d.get('n')}")
    return bad


def eval_r4(lp: dict[str, list], firsts: dict[str, dict], occ: dict[str, dict], *, T: int, n_task: int, S0: int,
            decoy: dict | None = None, n_min: int = 1024) -> dict:
    """lp[arm] = per-position logprob of forced[i] (None = invalid position) for child, child2, childX, coldA, coldB, coldE.
    `decoy` = {n, lcp_with_P, first} of childX's second resident (its failures make childX invalid; zero weight)."""
    reasons = {a: arm_validity(a, firsts.get(a) or {}, (occ or {}).get(a), T=T, n_task=n_task, S0=S0) for a in ALL_ARMS}
    reasons[CX] = reasons[CX] + decoy_validity(decoy, (occ or {}).get(CX), n_min=n_min)
    valid = {a: not r for a, r in reasons.items()}
    have = all(lp.get(a) is not None for a in JUDGED)
    K = min(len(lp[a]) for a in JUDGED) if have else 0
    common = [i for i in range(K) if all(lp[a][i] is not None for a in JUDGED)]
    base = {"T": T, "n_task": n_task, "S0": S0, "prompt_ge_50k": n_task >= ARMS.MIN_LONG_PROMPT, "positions": K,
            "n_common_valid": len(common), "arm_valid": valid, "arm_invalid_reasons": reasons,
            "n_valid": {a: sum(1 for x in (lp.get(a) or []) if x is not None) for a in ALL_ARMS},
            "residents_expected": {a: list(r) for a, r in RESIDENTS.items()},
            "floor_pair": pair_key(*FLOOR_PAIR), "cold_pairs": [pair_key(*p) for p in COLD_PAIRS],
            "r3_floor_raw": R3_FLOOR_RAW}
    empty = {"d": None, "floor_used": None, "cold_max": None, "d_child_vs_coldE": None, "d_child_vs_child2": None,
             "predeclared_rule_satisfied": None}
    if not all(valid[a] for a in JUDGED):
        return {**base, **empty, "verdict": "INVALID",
                "factorial_reading": {"label": "CHILDX_NOT_ESTABLISHED", "why": "a judged arm is invalid", "weight": "zero"}}
    if len(common) < ARMS.MIN_COMMON_POSITIONS or not base["prompt_ge_50k"]:
        why = (f"{len(common)} positions valid in all five < {ARMS.MIN_COMMON_POSITIONS}"
               if len(common) < ARMS.MIN_COMMON_POSITIONS else f"len(P) {n_task} < {ARMS.MIN_LONG_PROMPT}")
        return {**base, **empty, "verdict": "ABSTAIN", "reason": why,
                "factorial_reading": {"label": "CHILDX_NOT_ESTABLISHED", "why": why, "weight": "zero"}}
    d = {pair_key(x, y): _d(lp, x, y, common) for x, y in JUDGED_PAIRS}
    m = lambda x, y: d[pair_key(x, y)]["mean_abs_dlogprob"]  # noqa: E731
    floor_raw = m(*FLOOR_PAIR)
    floor = max(floor_raw, ARMS.FLOOR_EPS)
    cold_max = max(m(*p) for p in COLD_PAIRS)
    dff, dce = m("child", "child2"), m("child", "coldE")
    if dff > 0.0:
        verdict = "FORK_PATH_NOT_REPRODUCIBLE"
    elif dce <= floor:
        verdict = "OCCUPANCY_NUMERICS"
    elif dce > cold_max:
        verdict = "FORK_PATH_SPECIFIC"
    else:
        verdict = "BETWEEN_FLOOR_AND_COLD_SPREAD"
    return {**base, "d": d, "floor_raw": floor_raw, "floor_used": floor, "floor_matches_r3": floor_raw == R3_FLOOR_RAW,
            "cold_max": cold_max, "cold_max_pair": pair_key(*max(COLD_PAIRS, key=lambda p: m(*p))),
            "d_child_vs_coldE": dce, "d_child_vs_child2": dff, "child_vs_child2_exact_equal": d[pair_key("child", "child2")]["n_exact_equal"],
            "ratio_dCE_over_floor": dce / floor, "verdict": verdict,
            "predeclared_rule_satisfied": verdict == "OCCUPANCY_NUMERICS",
            "factorial_reading": factorial_reading(lp, common, valid[CX], floor)}


def cross_load(lp_now: dict, lp_r3: dict) -> dict:
    """Zero-weight: this load's child/coldA/coldB vs R3's (graphs ON, same build/argv), d over positions valid in both."""
    out = {}
    for a in ("child", "coldA", "coldB"):
        x, y = lp_now.get(a) or [], lp_r3.get(a) or []
        pos = [i for i in range(min(len(x), len(y))) if x[i] is not None and y[i] is not None]
        if pos:
            ds = [abs(x[i] - y[i]) for i in pos]
            out[a] = {"mean_abs_dlogprob": sum(ds) / len(ds), "max_abs_dlogprob": max(ds), "n_positions": len(pos),
                      "n_exact_equal": sum(1 for v in ds if v == 0.0)}
        else:
            out[a] = {"n_positions": 0}
    reproduces = bool(out) and all(v.get("n_positions") and v.get("n_exact_equal") == v["n_positions"] for v in out.values())
    return {"pairs": out, "reproduces_r3_exactly": reproduces,
            "weight": "zero (recorded only; shows whether this load reproduces R3; never pooled into the floor)"}


# ---------------------------------------------------------------- verdict over a records directory
def load_records(out_dir) -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(str(out_dir), "records", "*.json"))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        if r.get("kpf18_item") == ITEM and isinstance(r.get("r4"), dict):
            recs.append({"record": p, **r})
    return recs


def _latest(recs: list[dict]) -> dict:
    out: dict = {}
    for r in sorted(recs, key=lambda x: x.get("end_utc") or ""):
        out[r["r4"]["step_id"]] = r
    return out


def r3_lp(path) -> dict | None:
    """R3's long6 record -> its lp series (for the zero-weight cross-load reading)."""
    if not path:
        return None
    try:
        r = json.load(open(path))
        return (r["r3"]["result"] or {}).get("lp")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def missing_lines(v: dict) -> list[str]:
    miss = []
    for path in RESULT_LINES:
        cur = v
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                miss.append(".".join(path))
                break
            cur = cur[k]
    return miss


def verdict(out_dir, predecl_sha: str | None, *, run_valid: dict | None = None, r3_record=None) -> dict:
    allr = load_records(out_dir)
    done = [r for r in allr if r["r4"].get("step_complete")]
    stale = [r["record"] for r in done if predecl_sha and r["r4"].get("predeclaration_sha256") != predecl_sha]
    use = _latest([r for r in done if r["record"] not in stale])
    rec = use.get("r4")
    lg = {**rec["r4"]["result"], "record": rec["record"], "fresh_load": rec["r4"].get("fresh_load")} if rec else None
    if lg:
        blp = r3_lp(r3_record)
        lg["cross_load_vs_r3"] = (cross_load(lg.get("lp") or {}, blp) if blp else
                                  {"pairs": None, "why": f"no R3 record lp ({r3_record})", "weight": "zero"})
    v = {"rule": "PREDECLARATION.md (KPF-18-S1.R4)", "predeclaration_sha256": predecl_sha, "records_seen": len(allr),
         "records_step_complete": len(done), "excluded_predeclaration_mismatch": stale,
         "item3_r4": lg if lg else {"verdict": "INCOMPLETE"}}
    if run_valid is not None:
        v["run_valid"] = run_valid
    items = {"item3_r4": v["item3_r4"].get("verdict")}
    v["item_verdicts"] = items
    complete = all(x not in (None, "INCOMPLETE") for x in items.values())
    miss = missing_lines(v) if complete else []
    v["missing_result_lines"] = miss
    if miss:
        v["verdict"] = "FAIL_MISSING_RESULT_LINE"
    elif not complete:
        v["verdict"] = "INCOMPLETE"
    elif run_valid is not None and not run_valid.get("valid"):
        v["verdict"] = "RUN_INVALID"
    else:
        v["verdict"] = "COMPLETE"
    return v


# ================================================================ step (takes the gate module as `gate`)
def decoy_ids(gate, ctx, n: int) -> list[int]:
    """childX's second resident: the group-6 trunk cut to n tokens (= child2's slot length after its positions)."""
    t = gate.G(ctx, DECOY_GROUP)["trunk"]
    if len(t) < n:
        raise gate.Abort(f"decoy group g{DECOY_GROUP} trunk has {len(t)} < {n} tokens: refusing")
    return list(t[:n])



def _step_open(env, step_id: str, arm: str, key: str, **extra):
    ctx = env.ctx
    R = ARMS.open_run(env, ITEM, arm, rule=ARMS.RULES[ITEM][key], arm_key=key)
    R.extra["r4"] = {"step_id": step_id, "step": key, "predeclaration": ctx.args.s1r_predeclaration,
                     "predeclaration_sha256": getattr(ctx, "_s1r_predecl_sha", None), "step_complete": False,
                     "fresh_load": getattr(env.srv, "_r4_steps_done", 0) == 0, **extra}
    return R


def _step_close(env, R, result: dict) -> None:
    R.extra["r4"].update(result=result, step_complete=True, finished_in_step=True)
    ARMS.close_counts(env, R)
    R.finish()
    env.srv._r4_steps_done = getattr(env.srv, "_r4_steps_done", 0) + 1


def run_r4(env) -> None:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    g = a.s1r_long_group
    grp = gate.G(ctx, g)
    T, trunk = grp["T"], grp["trunk"]
    suf = grp["suffixes"][0]
    P = trunk + suf
    doc = ctx._s1r_forced
    want_p = (doc.get("prompt") or {}).get("ids_sha256")
    if S1R.sha256_ids(P) != want_p:
        raise gate.Abort(f"18.3.4 prompt sha256 {S1R.sha256_ids(P)} != the forced sequence's prompt {want_p}: refusing")
    forced = list(doc["ids"][:a.s1r_tf_positions])
    cl = ctx.client
    R = _step_open(env, "r4", "r4_teacher_forced", "r4", group=g, T=T, n_task=len(P),
                   forced_ids_sha256=doc.get("ids_sha256"), positions=len(forced))
    env.erase_all()
    env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm="r4_teacher_forced", cell_id="src", fork=False,
             ckpt=[T], group=g)
    dids = decoy_ids(gate, ctx, len(P) + len(forced) - 1)
    lcp_dp = ARMS.lcp(dids, P)
    if lcp_dp >= a.fork_min:
        raise gate.Abort(f"decoy (g{DECOY_GROUP}) shares {lcp_dp} tokens with P >= fork-min {a.fork_min}: refusing")
    cold = {"fork": False, "cache": False, "ckpt": [T]}
    per: dict = {}
    occ: dict = {}
    decoy: dict = {}
    for name, slot, mode, erase_before, _res in PLAN:
        if erase_before:
            env.erase(*erase_before)
        if name == DECOY_BEFORE:
            ctx.check_boundary()
            r, _cell = env.send(R, "decoy", dids, DECOY_SLOT, n_predict=0, arm="r4_teacher_forced", cell_id="decoy",
                                fork=False, cache=False, group=DECOY_GROUP)
            t = r.get("timings") or {}
            decoy = {"group": DECOY_GROUP, "slot": DECOY_SLOT, "n": len(dids), "lcp_with_P": lcp_dp,
                     "ids_sha256": S1R.sha256_ids(dids),
                     "first": {x: t.get(x) for x in ("prompt_n", "cache_n", "n_fork_tokens", "prompt_ms")}}
        occ[name] = R3._occupancy(cl)
        per[name] = S1R.tf_arm(env, R, name, slot, P, forced, _FORK if mode == "fork" else cold)
    env.erase_all()
    lp = {k: v["lp"] for k, v in per.items()}
    firsts = {k: v["first"] for k, v in per.items()}
    res = eval_r4(lp, firsts, occ, T=T, n_task=len(P), S0=len(suf), decoy=decoy, n_min=a.fork_min)
    res.update(first=firsts, failfast_at={k: v["failfast_at"] for k, v in per.items()}, lp=lp, occupancy_before=occ,
               residents_seen={k: residents(occ.get(k), SLOTS[k]) for k in per}, slots=dict(SLOTS), decoy=decoy)
    _step_close(env, R, res)


# ================================================================ plan
def est_r4(gate, T: int, S0: int, K: int) -> float:
    """Measured calibration (R3 constants) for the five 60k prefills (source, coldE, coldA, coldB, decoy); the gate's
    model for the three fork prefills; six teacher-forced arms."""
    return (5 * COLD60K_PF_S + 3 * gate.pf_s(S0, T) + 6 * K * TF_POS_S + 11 * gate.REQ_OVERHEAD_S)


def body_187(ctx, gate, env) -> list:
    a = ctx.args
    Step = gate.Step
    want = [x.strip() for x in (a.r4_steps or "").split(",") if x.strip()]
    steps: list = []
    if "r4" in want:
        g = a.s1r_long_group
        T, S0 = gate.G(ctx, g)["T"], gate.G(ctx, g)["suffix_lens"][0]
        K = a.s1r_tf_positions
        est = est_r4(gate, T, S0, K)

        def fn(est=est):
            S1R._need(env, est)
            run_r4(env)
        steps.append(Step(f"187 (3) r4 18.3.4 localisation g{g} P={T + S0}: child / coldE / child2 on slot 1 (residents "
                          f"{{0}}), coldA slot 2, coldB slot 3, decoy slot 1 + childX slot 2 (residents {{0,1}}), {K} positions x 6 arms "
                          f"(rule r4)", est, fn))
    return steps


def build_187(ctx, gate) -> list:
    a = ctx.args
    name, argv = ARMS.get_configs(ctx, gate)[0]
    srv = ARMS.server_cls(gate)(ctx, f"kpf18r4-{name}-{gate.utc_tag()}", Path(a.fork_bin), a.fork_min, argv)
    env = ARMS.Env(ctx, gate, srv, name)
    S1R._prepare(ctx, gate)
    return gate.start_steps(ctx, srv) + body_187(ctx, gate, env) + [gate.stop_step(srv)]


def preflight_187(ctx, gate) -> tuple[list[str], list[str]]:
    a = ctx.args
    probs = S1R._prepare(ctx, gate)
    notes = [f"R4 predeclaration {a.s1r_predeclaration} sha256 {ctx._s1r_predecl_sha}",
             f"R4 forced sequence {a.s1r_forced_seq} ids sha256 {(ctx._s1r_forced or {}).get('ids_sha256')} "
             f"(predeclared {a.s1r_forced_sha256})",
             f"R4 steps {a.r4_steps}; arms in order {[p[0] for p in PLAN]} (slots {dict(SLOTS)}, residents "
             f"{ {k: list(v) for k, v in RESIDENTS.items()} }); teacher-forced positions {a.s1r_tf_positions}"]
    if not ctx.ps.spec_only:
        g = a.s1r_long_group
        P = ctx.ps.groups[g]["trunk"] + ctx.ps.groups[g]["suffixes"][0]
        want = (ctx._s1r_forced or {}).get("prompt", {}).get("ids_sha256")
        if S1R.sha256_ids(P) != want:
            probs.append(f"18.3.4 prompt (g{g} trunk+suffix0) sha256 {S1R.sha256_ids(P)} != forced sequence prompt {want}")
        else:
            notes.append(f"R4 18.3.4 prompt g{g} = {len(P)} tokens, sha256 matches the forced sequence's prompt")
        if len((ctx._s1r_forced or {}).get("ids") or []) < a.s1r_tf_positions:
            probs.append("forced sequence shorter than --s1r-tf-positions")
        dn = len(P) + a.s1r_tf_positions - 1
        dt = ctx.ps.groups[DECOY_GROUP]["trunk"]
        dl = ARMS.lcp(dt[:dn], P)
        if len(dt) < dn or dl >= a.fork_min:
            probs.append(f"decoy g{DECOY_GROUP}: trunk {len(dt)} tokens (need {dn}), LCP with P {dl} (need < {a.fork_min})")
        else:
            notes.append(f"R4 decoy g{DECOY_GROUP} trunk[:{dn}] shares {dl} tokens with P (< fork-min {a.fork_min})")
    else:
        notes.append("R4: spec-only prompt set (plan only): prompt sha and forced length are checked at execute")
    bad = [x for x in (a.r4_steps or "").split(",") if x.strip() and x.strip() not in STEP_KEYS]
    if bad:
        probs.append(f"--r4-steps: unknown {bad}")
    return probs, notes


ARMS.BUILDERS[ITEM] = build_187
ARMS.PREFLIGHTS[ITEM] = preflight_187


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="verdict over a kpf18r4 --out directory (pure; starts nothing)")
    ap.add_argument("verdict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--predeclaration", required=True)
    ap.add_argument("--r3-record", default=None)
    ap.add_argument("--run-valid-json", default=None)
    ap.add_argument("--write", default=None)
    x = ap.parse_args()
    rv = json.load(open(x.run_valid_json)) if x.run_valid_json else None
    v = verdict(x.out, S1R.sha256_path(x.predeclaration), run_valid=rv, r3_record=x.r3_record)
    s = json.dumps(v, indent=1, default=str)
    if x.write:
        tmp = x.write + ".tmp"
        Path(tmp).write_text(s + "\n")
        os.replace(tmp, x.write)
    print(json.dumps({"verdict": v["verdict"], "item_verdicts": v["item_verdicts"], "records_seen": v["records_seen"],
                      "records_step_complete": v["records_step_complete"],
                      "excluded_predeclaration_mismatch": len(v["excluded_predeclaration_mismatch"]),
                      "missing_result_lines": v["missing_result_lines"]}))
