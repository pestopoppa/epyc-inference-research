#!/usr/bin/env python3
"""KPF-18-S1.R3: command `kpf18r3` (item 186) of the KPF-18 gate (kpf18_gate.py).

handoffs/active/kv-prefix-fork-and-paged-attention.md, box "KPF-18-S1.R3" (root 5d674dd53):
  long6   18.3 part 4 with a six-pair cold floor: on one load, child (fork at T) + coldA..coldD (cache_prompt false,
          checkpoint_at [T], slots and occupancy varied) + sameshape (the source's own state moved onto another slot
          through the --cache-ram prompt cache, no fork), each teacher-forced over forced_seq_183p4.json with
          kpf18_s1r.tf_arm. floor = max d over the 6 cold pairs; SATISFIED iff worst child-vs-cold <= floor. The
          child-vs-sameshape reading is recorded with zero weight.
  item2q  item 2 cell g3.s2 again, with three placement-varied Q repeats (Q1 beside F+O, Q2 beside O, Q3 alone), judged
          with kpf18_arms.judge_item2 against the cell's own Q-vs-O floor.
The decision rules are the launcher's PREDECLARATION.md; its sha256 (--s1r-predeclaration) is stamped into every record
(`r3.predeclaration_sha256`) and verdict() refuses records made under another sha. RULES[186] is the short machine copy.

The S1R plumbing is reused unchanged: --s1r-predeclaration / --s1r-forced-seq / --s1r-tf-positions / --s1r-long-group /
--s1r-budget-factor, kpf18_s1r._prepare (forced sequence, fillers), _need (gate boundary, KPF18_S1R_STOPFILE, budget),
tf_arm (teacher forcing) and item2_cell_validity. The pure evaluators and verdict() take no server.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from pathlib import Path

import kpf18_arms as ARMS
import kpf18_s1r as S1R

ITEM = 186
CMD = "kpf18r3"
TASK = "KPF-18-S1.R3"

COLD_ARMS = ("coldA", "coldB", "coldC", "coldD")
JUDGED = ("child",) + COLD_ARMS
SS = "sameshape"
COLD_PAIRS = tuple(itertools.combinations(COLD_ARMS, 2))           # the 6 floor pairs
CHILD_PAIRS = tuple(("child", c) for c in COLD_ARMS)                # the 4 judged pairs
ALL_PAIRS = CHILD_PAIRS + COLD_PAIRS
ITEM2Q_CELL = (3, 2)                                                # g3 (T 32000), suffix 2 (S 4000)
Q_REPEATS = (("Q1", (0, 1)), ("Q2", (0,)), ("Q3", ()))              # name, slots left resident while Q is built
ITEM2Q_MIN_Q = ARMS.ITEM2_MIN_AA                                    # 3
STEP_KEYS = ("long6", "item2q")
# Estimate calibration for long6, MEASURED on this build + argv by kpf18-s1-remainder (record kpf18-20261023T140648Z-9):
# cold 60512-token prefill 113.9 / 115.2 s, teacher-forced position 0.114 s (255 positions in 28.5-29.8 s per arm).
COLD60K_PF_S = 120.0
TF_POS_S = 0.12

ARMS.ARM_ITEMS = tuple(sorted(set(ARMS.ARM_ITEMS) | {ITEM}))
ARMS.ARM_CMDS[CMD] = [ITEM]
ARMS.TASK_OF[ITEM] = TASK

ARMS.RULES[ITEM] = {
    "common": ("Rules: PREDECLARATION.md of the launcher (sha256 in every record as r3.predeclaration_sha256; verdict() "
               "refuses another sha). One load of the argv file as given, steps in order long6, item2q. Counts and "
               "values are reported, never a pass word."),
    "long6": ("18.3 part 4, g5 P = trunk 60000 + suffix0: source slot 0 (checkpoint_at [T]); child slot 1 fork; coldA slot "
              "2, coldB slot 3; erase 2,3; coldC slot 3, coldD slot 2 (cache_prompt false, checkpoint_at [T]); erase 1-3, "
              "filler on slot 0 (source -> --cache-ram), filler on slot 2, sameshape = P on slot 2 slot_fork false. "
              "Position i = kpf18_s1r.tf_arm. d = mean |dlp| over positions valid in child+coldA..D. floor = max d over "
              "the 6 cold pairs (lifted to 1e-4); worst = max d(child, cold_k). SATISFIED iff the 5 judged arms are valid, "
              "forked at T, len(P) >= 50000, >= 16 common and worst <= floor; NOT_SATISFIED iff valid, >= 16 common and "
              "worst > floor; ABSTAIN iff < 16 common; INVALID iff a judged arm fails its validity. Child-vs-sameshape "
              "against cold-vs-sameshape is recorded with zero weight. Forced sequence sha256 " + S1R.FORCED_SHA256),
    "item2q": ("Item 2 cell g3.s2: F fork / O own (kpf18_s1r item 2); Q1 (F+O resident), Q2 (O resident), Q3 (alone), "
               "each = trunk+tail seeded on slot 3, filler-evicted into --cache-ram, loaded onto slot 2 (holding a filler) "
               "by trunk+suffix, slot_fork false. judge_item2([compare(F,O)], [compare(Qj,O) valid Qj]). "
               "G3S2_WITHIN_PLACEMENT_FLOOR / G3S2_OUTSIDE_PLACEMENT_FLOOR (>= 3 valid Q), G3S2_CONTROL_NOT_ESTABLISHED "
               "(< 3 valid Q), G3S2_INVALID (F or O invalid)."),
}

# PREDECLARATION.md "Result lines": a missing one makes the verdict FAIL_MISSING_RESULT_LINE.
RESULT_LINES = (
    ("item3_r3", "verdict"), ("item3_r3", "floor_used"), ("item3_r3", "worst_child_vs_cold"),
    ("item3_r3", "n_common_valid"), ("item3_r3", "arm_valid"), ("item3_r3", "d"), ("item3_r3", "pcache_reading"),
    ("item2_g3s2", "verdict"), ("item2_g3s2", "n_q_valid"), ("item2_g3s2", "q_vs_own_prefixes"),
    ("item2_g3s2", "fork_vs_own"), ("run_valid",),
)


# ================================================================ CLI
def add_arguments(ap) -> None:
    g = ap.add_argument_group("KPF-18-S1.R3 (command kpf18r3, item 186; also uses the --s1r-* inputs)")
    g.add_argument("--r3-steps", default=",".join(STEP_KEYS), help="subset of " + ",".join(STEP_KEYS))


# ================================================================ pure evaluators (no server)
def pair_key(x: str, y: str) -> str:
    return f"{x}|{y}"


def arm_validity(name: str, first: dict, *, T: int, n_task: int, S0: int) -> list[str]:
    """Position-0 timings of a long6 arm against its predeclared role (empty list = valid)."""
    f = first or {}
    nf = f.get("n_fork_tokens") or 0
    bad = []
    if name == "child":
        if not (nf == T and f.get("fork_src_kind") == "checkpoint" and f.get("fork_src_slot") == 0):
            bad.append(f"child_not_forked_at_T:{f.get('n_fork_tokens')},{f.get('fork_src_kind')},{f.get('fork_src_slot')}")
        if f.get("prompt_n") != S0:
            bad.append(f"child_prompt_n_{f.get('prompt_n')}_ne_{S0}")
    elif name in COLD_ARMS:
        if f.get("cache_n") != 0 or f.get("prompt_n") != n_task or nf:
            bad.append(f"{name}_not_cold:{f.get('cache_n')},{f.get('prompt_n')},{f.get('n_fork_tokens')}")
    elif name == SS:
        if f.get("cache_n") != T or f.get("prompt_n") != S0 or nf:
            bad.append(f"sameshape_not_prompt_cache_load_at_T:{f.get('cache_n')},{f.get('prompt_n')},{f.get('n_fork_tokens')}")
    return bad


def _d(lp: dict, x: str, y: str, pos: list[int]) -> dict:
    ds = [abs(lp[x][i] - lp[y][i]) for i in pos]
    return {"mean_abs_dlogprob": sum(ds) / len(ds), "max_abs_dlogprob": max(ds), "n_positions": len(ds),
            "pos0_abs_dlogprob": abs(lp[x][0] - lp[y][0]) if 0 in pos else None,
            "n_exact_equal": sum(1 for v in ds if v == 0.0)}


def pcache_reading(lp: dict, common: list[int], ss_valid: bool, floor: float) -> dict:
    """Zero-weight reading: child-vs-sameshape against cold-vs-sameshape (PREDECLARATION item 3)."""
    ss = lp.get(SS) or []
    pos = [i for i in common if i < len(ss) and ss[i] is not None] if ss_valid else []
    if len(pos) < ARMS.MIN_COMMON_POSITIONS:
        return {"label": "PCACHE_NOT_ESTABLISHED", "n_positions": len(pos), "sameshape_valid": ss_valid,
                "weight": "zero (recorded only)"}
    dc = _d(lp, "child", SS, pos)
    dk = {c: _d(lp, c, SS, pos) for c in COLD_ARMS}
    v = dc["mean_abs_dlogprob"]
    lo = min(x["mean_abs_dlogprob"] for x in dk.values())
    hi = max(x["mean_abs_dlogprob"] for x in dk.values())
    if v == 0.0:
        lab = "FORK_MATCHES_PCACHE_EXACTLY"
    elif v < lo:
        lab = "FORK_CLOSER_TO_PCACHE_THAN_EVERY_COLD"
    elif v <= hi:
        lab = "FORK_WITHIN_COLD_VS_PCACHE_RANGE"
    else:
        lab = "FORK_OUTSIDE_COLD_VS_PCACHE_RANGE"
    return {"label": lab, "n_positions": len(pos), "sameshape_valid": ss_valid, "d_child_vs_sameshape": dc,
            "d_cold_vs_sameshape": dk, "cold_vs_sameshape_min": lo, "cold_vs_sameshape_max": hi,
            "child_vs_sameshape_within_cold_floor": v <= floor, "n_exact_equal_child_sameshape": dc["n_exact_equal"],
            "weight": "zero (recorded only)"}


def eval_long6(lp: dict[str, list], firsts: dict[str, dict], *, T: int, n_task: int, S0: int) -> dict:
    """lp[arm] = per-position logprob of forced[i] (None = invalid position), arms child, coldA..D, sameshape."""
    reasons = {a: arm_validity(a, firsts.get(a) or {}, T=T, n_task=n_task, S0=S0) for a in JUDGED + (SS,)}
    valid = {a: not r for a, r in reasons.items()}
    have = all(lp.get(a) is not None for a in JUDGED)
    K = min(len(lp[a]) for a in JUDGED) if have else 0
    common = [i for i in range(K) if all(lp[a][i] is not None for a in JUDGED)]
    cf = firsts.get("child") or {}
    forked = cf.get("n_fork_tokens") == T and cf.get("fork_src_kind") == "checkpoint"
    base = {"T": T, "n_task": n_task, "S0": S0, "prompt_ge_50k": n_task >= ARMS.MIN_LONG_PROMPT, "forked_at_T": forked,
            "positions": K, "n_common_valid": len(common), "arm_valid": valid, "arm_invalid_reasons": reasons,
            "n_valid": {a: sum(1 for x in (lp.get(a) or []) if x is not None) for a in JUDGED + (SS,)},
            "floor_pairs": [pair_key(*p) for p in COLD_PAIRS], "judged_pairs": [pair_key(*p) for p in CHILD_PAIRS]}
    if not all(valid[a] for a in JUDGED):
        return {**base, "verdict": "INVALID", "predeclared_rule_satisfied": None, "d": None, "floor_used": None,
                "worst_child_vs_cold": None, "pcache_reading": {"label": "PCACHE_NOT_ESTABLISHED",
                                                                "why": "a judged arm is invalid", "weight": "zero"}}
    if len(common) < ARMS.MIN_COMMON_POSITIONS:
        return {**base, "verdict": "ABSTAIN", "reason": f"{len(common)} positions valid in all five < "
                f"{ARMS.MIN_COMMON_POSITIONS}", "predeclared_rule_satisfied": None, "d": None, "floor_used": None,
                "worst_child_vs_cold": None, "pcache_reading": {"label": "PCACHE_NOT_ESTABLISHED",
                                                                "why": "fewer than 16 common positions", "weight": "zero"}}
    d = {pair_key(x, y): _d(lp, x, y, common) for x, y in ALL_PAIRS}
    floor_raw = max(d[pair_key(*p)]["mean_abs_dlogprob"] for p in COLD_PAIRS)
    floor = max(floor_raw, ARMS.FLOOR_EPS)
    worst = max(d[pair_key(*p)]["mean_abs_dlogprob"] for p in CHILD_PAIRS)
    sat = bool(forked and base["prompt_ge_50k"] and worst <= floor)
    argmax = lambda ps: max(ps, key=lambda p: d[pair_key(*p)]["mean_abs_dlogprob"])  # noqa: E731
    return {**base, "d": d, "floor_raw": floor_raw, "floor_used": floor, "floor_pair": pair_key(*argmax(COLD_PAIRS)),
            "worst_child_vs_cold": worst, "worst_pair": pair_key(*argmax(CHILD_PAIRS)), "ratio_worst_over_floor": worst / floor,
            "within_floor": worst <= floor, "predeclared_rule_satisfied": sat,
            "verdict": "SATISFIED" if sat else "NOT_SATISFIED",
            "pcache_reading": pcache_reading(lp, common, valid[SS], floor)}


def cross_load(lp_now: dict, lp_base: dict) -> dict:
    """Zero-weight: this load's child/coldA/coldB vs the remainder's graphs-OFF ones, d over positions valid in both."""
    out = {}
    for a in ("child", "coldA", "coldB"):
        x, y = lp_now.get(a) or [], lp_base.get(a) or []
        pos = [i for i in range(min(len(x), len(y))) if x[i] is not None and y[i] is not None]
        if pos:
            ds = [abs(x[i] - y[i]) for i in pos]
            out[a] = {"mean_abs_dlogprob": sum(ds) / len(ds), "max_abs_dlogprob": max(ds), "n_positions": len(pos),
                      "n_exact_equal": sum(1 for v in ds if v == 0.0)}
        else:
            out[a] = {"n_positions": 0}
    return {"pairs": out, "weight": "zero (recorded only; confounded by load AND graphs; never pooled into the floor)"}


def q_validity(Q: dict, *, T: int, S: int, n: int) -> list[str]:
    bad = []
    if Q.get("http") != 200 or Q.get("error"):
        bad.append("Q_http")
    if len(Q.get("tokens") or []) != n:
        bad.append(f"Q_tokens_{len(Q.get('tokens') or [])}")
    t = Q.get("timings") or {}
    if t.get("cache_n") != T or t.get("prompt_n") != S or (t.get("n_fork_tokens") or 0):
        bad.append(f"Q_not_prompt_cache_load_at_T:{t.get('cache_n')},{t.get('prompt_n')},{t.get('n_fork_tokens')}")
    return bad


def fo_validity(F: dict, O: dict, *, T: int, S: int, n: int) -> list[str]:
    """F and O exactly as kpf18_s1r.item2_cell_validity (its Q leg is fed a passing dummy and dropped)."""
    dummy = {"http": 200, "tokens": [0] * n, "timings": {"cache_n": T, "prompt_n": S, "n_fork_tokens": 0}}
    return [b for b in S1R.item2_cell_validity(F, O, dummy, T=T, S=S, n=n) if not b.startswith("Q")]


def eval_item2q(fo: dict, fo_bad: list[str], qs: list[dict], n_cmp: int) -> dict:
    """fo = compare(F, O); qs = [{name, valid, invalid_reasons, q_vs_own (compare(Q,O)), ...}]."""
    valid_q = [q for q in qs if q.get("valid")]
    out = {"fork_vs_own": fo, "fo_valid": not fo_bad, "fo_invalid_reasons": fo_bad, "n_q": len(qs),
           "n_q_valid": len(valid_q), "q_vs_own_prefixes": {q["name"]: (q.get("q_vs_own") or {}).get("prefix") for q in qs},
           "q_invalid": {q["name"]: q.get("invalid_reasons") for q in qs if not q.get("valid")}}
    if fo_bad:
        out.update(verdict="G3S2_INVALID", judge=None)
        return out
    if len(valid_q) < ITEM2Q_MIN_Q:
        out.update(verdict="G3S2_CONTROL_NOT_ESTABLISHED", judge=None)
        return out
    j = ARMS.judge_item2([fo], [q["q_vs_own"] for q in valid_q], n_cmp)
    out["judge"] = j
    if j.get("predeclared_rule_satisfied") is None:
        out["verdict"] = "G3S2_CONTROL_NOT_ESTABLISHED"
    else:
        out["verdict"] = "G3S2_WITHIN_PLACEMENT_FLOOR" if j["predeclared_rule_satisfied"] else "G3S2_OUTSIDE_PLACEMENT_FLOOR"
    return out


# ---------------------------------------------------------------- verdict over a records directory
def load_records(out_dir) -> list[dict]:
    recs = []
    for p in sorted(glob.glob(os.path.join(str(out_dir), "records", "*.json"))):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        if r.get("kpf18_item") == ITEM and isinstance(r.get("r3"), dict):
            recs.append({"record": p, **r})
    return recs


def _latest(recs: list[dict]) -> dict:
    out: dict = {}
    for r in sorted(recs, key=lambda x: x.get("end_utc") or ""):
        out[r["r3"]["step_id"]] = r
    return out


def base_long_lp(path) -> dict | None:
    """The remainder's long:tf record -> its lp series (for the zero-weight cross-load reading)."""
    if not path:
        return None
    try:
        r = json.load(open(path))
        return (r["s1r"]["result"] or {}).get("lp")
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


def verdict(out_dir, predecl_sha: str | None, *, run_valid: dict | None = None, base_long_record=None) -> dict:
    allr = load_records(out_dir)
    done = [r for r in allr if r["r3"].get("step_complete")]
    stale = [r["record"] for r in done if predecl_sha and r["r3"].get("predeclaration_sha256") != predecl_sha]
    use = _latest([r for r in done if r["record"] not in stale])

    def res(step_id):
        r = use.get(step_id)
        return {**r["r3"]["result"], "record": r["record"], "fresh_load": r["r3"].get("fresh_load")} if r else None
    lg = res("long6")
    if lg:
        blp = base_long_lp(base_long_record)
        lg["cross_load_graphs_off_vs_on"] = (cross_load(lg.get("lp") or {}, blp) if blp else
                                             {"pairs": None, "why": f"no base record lp ({base_long_record})", "weight": "zero"})
    i2 = res("item2q")
    v = {"rule": "PREDECLARATION.md (KPF-18-S1.R3, root 5d674dd53)", "predeclaration_sha256": predecl_sha,
         "records_seen": len(allr), "records_step_complete": len(done), "excluded_predeclaration_mismatch": stale,
         "item3_r3": lg if lg else {"verdict": "INCOMPLETE"}, "item2_g3s2": i2 if i2 else {"verdict": "INCOMPLETE"}}
    if run_valid is not None:
        v["run_valid"] = run_valid
    items = {"item3_r3": v["item3_r3"].get("verdict"), "item2_g3s2": v["item2_g3s2"].get("verdict")}
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


# ================================================================ steps (take the gate module as `gate`)
def _step_open(env, step_id: str, arm: str, key: str, **extra):
    ctx = env.ctx
    R = ARMS.open_run(env, ITEM, arm, rule=ARMS.RULES[ITEM][key], arm_key=key)
    R.extra["r3"] = {"step_id": step_id, "step": key, "predeclaration": ctx.args.s1r_predeclaration,
                     "predeclaration_sha256": getattr(ctx, "_s1r_predecl_sha", None), "step_complete": False,
                     "fresh_load": getattr(env.srv, "_r3_steps_done", 0) == 0, **extra}
    return R


def _step_close(env, R, result: dict) -> None:
    R.extra["r3"].update(result=result, step_complete=True, finished_in_step=True)
    ARMS.close_counts(env, R)
    R.finish()
    env.srv._r3_steps_done = getattr(env.srv, "_r3_steps_done", 0) + 1


def _occupancy(cl) -> dict:
    return {str(s.get("id")): s.get("n_prompt_tokens") for s in (cl.slots() or [])}


def run_long6(env) -> None:
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
    fil = ctx._s1r_fillers
    cl = ctx.client
    R = _step_open(env, "long6", "long6_teacher_forced", "long6", group=g, T=T, n_task=len(P),
                   forced_ids_sha256=doc.get("ids_sha256"), positions=len(forced))
    env.erase_all()
    env.send(R, "source", trunk + grp["src_tail"], 0, n_predict=0, arm="long6_teacher_forced", cell_id="src", fork=False,
             ckpt=[T], group=g)
    per: dict = {}
    occ: dict = {}
    cold = {"fork": False, "cache": False, "ckpt": [T]}
    plan = (("child", 1, {"fork": True, "cache": True, "ckpt": None}, ()),
            ("coldA", 2, cold, ()), ("coldB", 3, cold, ()),
            ("coldC", 3, cold, (2, 3)), ("coldD", 2, cold, ()))
    for name, slot, first, erase_before in plan:
        if erase_before:
            env.erase(*erase_before)
        occ[name] = _occupancy(cl)
        per[name] = S1R.tf_arm(env, R, name, slot, P, forced, first)
    # sameshape: the source's own state -> --cache-ram (filler on slot 0) -> slot 2 (holding a filler), no fork
    ctx.check_boundary()
    env.erase(1, 2, 3)
    ev = R.row("ss_evict_filler", cl.stream(gate.body_for(fil[0], 1, 0, fork=False, n_probs=0)), slot=0)
    R.row("ss_prime_filler", cl.complete(gate.body_for(fil[1], 1, 2, fork=False, n_probs=0)), slot=2)
    occ[SS] = _occupancy(cl)
    per[SS] = S1R.tf_arm(env, R, SS, 2, P, forced, {"fork": False, "cache": True, "ckpt": None})
    env.erase_all()
    lp = {k: v["lp"] for k, v in per.items()}
    firsts = {k: v["first"] for k, v in per.items()}
    res = eval_long6(lp, firsts, T=T, n_task=len(P), S0=len(suf))
    res.update(first=firsts, failfast_at={k: v["failfast_at"] for k, v in per.items()}, lp=lp, occupancy_before=occ,
               slots={"child": 1, "coldA": 2, "coldB": 3, "coldC": 3, "coldD": 2, SS: 2},
               ss_evict_stall_ms=S1R.stall_ms(ev))
    _step_close(env, R, res)


def run_item2q(env) -> None:
    gate, ctx, a = env.gate, env.ctx, env.ctx.args
    g, i = ITEM2Q_CELL
    grp = gate.G(ctx, g)
    T, trunk, n, npb = grp["T"], grp["trunk"], a.n_predict, a.n_probs
    suf = grp["suffixes"][i]
    P = trunk + suf
    fil = ctx._s1r_fillers
    cl = ctx.client
    R = _step_open(env, "item2q", f"item2q_g{g}s{i}", "item2q", group=g, suffix=i)
    gate.item2_seed_source(ctx, R, g, T)
    cl.erase(1)
    F = R.row("fork_child", cl.complete(gate.body_for(P, n, 1, fork=True, n_probs=npb)), group=g, suffix=i, slot=1)
    O = R.row("source_own", cl.complete(gate.body_for(P, n, 0, fork=False, cache=True, n_probs=npb)), group=g, suffix=i, slot=0)
    qs, Qr = [], {}
    fi = 2                                                         # fil[0..1] are long6's
    for name, keep in Q_REPEATS:
        ctx.check_boundary()
        gate.erase_all(ctx, tuple(s for s in (0, 1, 2, 3) if s not in keep))
        occ = _occupancy(cl)
        R.row(f"{name}_seed_source", cl.complete(gate.body_for(trunk + grp["src_tail"], 0, 3, fork=False, ckpt=[T], n_probs=0)),
              group=g, slot=3)
        ev = R.row(f"{name}_evict_filler", cl.stream(gate.body_for(fil[fi], 1, 3, fork=False, n_probs=0)), slot=3)
        R.row(f"{name}_prime_filler", cl.complete(gate.body_for(fil[fi + 1], 1, 2, fork=False, n_probs=0)), slot=2)
        fi += 2
        Q = R.row(name, cl.complete(gate.body_for(P, n, 2, fork=False, cache=True, n_probs=npb)), group=g, suffix=i, slot=2)
        gate.erase_all(ctx, (2, 3))
        bad = q_validity(Q, T=T, S=len(suf), n=n)
        qo = gate.compare(Q, O, n)
        R.eq.setdefault("item2q_pcache_vs_own", []).append(qo)
        Qr[name] = Q
        qs.append({"name": name, "resident_slots": list(keep), "occupancy_before": occ, "valid": not bad,
                   "invalid_reasons": bad, "q_vs_own": qo, "fork_vs_q": gate.compare(F, Q, n),
                   "timings": {x: (Q.get("timings") or {}).get(x) for x in ("prompt_n", "cache_n", "n_fork_tokens")},
                   "evict_stall_ms": S1R.stall_ms(ev)})
    gate.erase_all(ctx)
    fo = gate.compare(F, O, n)
    R.eq.setdefault("item2q_fork_vs_own", []).append(fo)
    res = eval_item2q(fo, fo_validity(F, O, T=T, S=len(suf), n=n), qs, n)
    names = [q[0] for q in Q_REPEATS]
    res.update(cell=f"g{g}.s{i}", T=T, S=len(suf), q=qs,
               q_vs_q={f"{x}|{y}": gate.compare(Qr[x], Qr[y], n) for x, y in itertools.combinations(names, 2)},
               timings={k: {x: (r.get("timings") or {}).get(x) for x in ("prompt_n", "cache_n", "n_fork_tokens",
                                                                          "fork_src_kind", "fork_src_slot")}
                        for k, r in (("F", F), ("O", O))})
    _step_close(env, R, res)


# ================================================================ plan
def est_long6(gate, T: int, S0: int, K: int) -> float:
    """Measured calibration (module constants) for the five 60k prefills; the gate's model for the short ones."""
    return (5 * COLD60K_PF_S + 2 * gate.pf_s(S0, T) + 2 * gate.pf_s(S1R.FILLER_LEN, 0) + 6 * K * TF_POS_S
            + 12 * gate.REQ_OVERHEAD_S)


def est_item2q(gate, T: int, S: int, n: int) -> float:
    q = (gate.pf_s(T + gate.SRC_TAIL, 2 * T) + gate.pf_s(S, 2 * T) + gate.dec_s(n, 2 * T, True)
         + 2 * gate.pf_s(S1R.FILLER_LEN, 0) + 4 * gate.REQ_OVERHEAD_S)
    return (gate.pf_s(T + gate.SRC_TAIL, 0) + 2 * gate.pf_s(S, T) + 2 * gate.dec_s(n, T, True)
            + len(Q_REPEATS) * q + 4 * gate.REQ_OVERHEAD_S)


def body_186(ctx, gate, env) -> list:
    a = ctx.args
    Step = gate.Step
    want = [x.strip() for x in (a.r3_steps or "").split(",") if x.strip()]
    steps: list = []
    if "long6" in want:
        g = a.s1r_long_group
        T, S0 = gate.G(ctx, g)["T"], gate.G(ctx, g)["suffix_lens"][0]
        K = a.s1r_tf_positions
        est = est_long6(gate, T, S0, K)

        def fn(est=est):
            S1R._need(env, est)
            run_long6(env)
        steps.append(Step(f"186 (3) long6 18.3.4 teacher-forced g{g} P={T + S0}: child fork + coldA..coldD (checkpoint_at "
                          f"[T], slots 2,3,3,2) + sameshape via --cache-ram, {K} positions x 6 arms (rule long6)", est, fn))
    if "item2q" in want:
        g, i = ITEM2Q_CELL
        T, S = gate.G(ctx, g)["T"], gate.G(ctx, g)["suffix_lens"][i]
        est = est_item2q(gate, T, S, a.n_predict)

        def fn(est=est):
            S1R._need(env, est)
            run_item2q(env)
        steps.append(Step(f"186 (2) item2q g{g}.s{i} T={T} S={S}: F fork / O own / Q1 (F+O resident) Q2 (O) Q3 (alone) via "
                          f"--cache-ram onto slot 2, {a.n_predict} tok each (rule item2q)", est, fn))
    return steps


def build_186(ctx, gate) -> list:
    a = ctx.args
    name, argv = ARMS.get_configs(ctx, gate)[0]
    srv = ARMS.server_cls(gate)(ctx, f"kpf18r3-{name}-{gate.utc_tag()}", Path(a.fork_bin), a.fork_min, argv)
    env = ARMS.Env(ctx, gate, srv, name)
    S1R._prepare(ctx, gate)
    return gate.start_steps(ctx, srv) + body_186(ctx, gate, env) + [gate.stop_step(srv)]


def preflight_186(ctx, gate) -> tuple[list[str], list[str]]:
    a = ctx.args
    probs = S1R._prepare(ctx, gate)
    notes = [f"R3 predeclaration {a.s1r_predeclaration} sha256 {ctx._s1r_predecl_sha}",
             f"R3 forced sequence {a.s1r_forced_seq} ids sha256 {(ctx._s1r_forced or {}).get('ids_sha256')} "
             f"(predeclared {a.s1r_forced_sha256})",
             f"R3 steps {a.r3_steps}; long6 arms {list(JUDGED) + [SS]}; item2q cell {ITEM2Q_CELL} with {[q[0] for q in Q_REPEATS]}; "
             f"teacher-forced positions {a.s1r_tf_positions}"]
    if not ctx.ps.spec_only:
        g = a.s1r_long_group
        P = ctx.ps.groups[g]["trunk"] + ctx.ps.groups[g]["suffixes"][0]
        want = (ctx._s1r_forced or {}).get("prompt", {}).get("ids_sha256")
        if S1R.sha256_ids(P) != want:
            probs.append(f"18.3.4 prompt (g{g} trunk+suffix0) sha256 {S1R.sha256_ids(P)} != forced sequence prompt {want}")
        else:
            notes.append(f"R3 18.3.4 prompt g{g} = {len(P)} tokens, sha256 matches the forced sequence's prompt")
        need = 2 + 2 * len(Q_REPEATS)
        if len(ctx._s1r_fillers) < need:
            probs.append(f"only {len(ctx._s1r_fillers)} fillers with a zero-LCP first token found (need {need})")
        if len(ctx._s1r_forced.get("ids") or []) < a.s1r_tf_positions:
            probs.append("forced sequence shorter than --s1r-tf-positions")
    else:
        notes.append("R3: spec-only prompt set (plan only): prompt sha, fillers and forced length are checked at execute")
    bad = [x for x in (a.r3_steps or "").split(",") if x.strip() and x.strip() not in STEP_KEYS]
    if bad:
        probs.append(f"--r3-steps: unknown {bad}")
    return probs, notes


ARMS.BUILDERS[ITEM] = build_186
ARMS.PREFLIGHTS[ITEM] = preflight_186


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="verdict over a kpf18r3 --out directory (pure; starts nothing)")
    ap.add_argument("verdict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--predeclaration", required=True)
    ap.add_argument("--base-long-record", default=None)
    ap.add_argument("--run-valid-json", default=None)
    ap.add_argument("--write", default=None)
    x = ap.parse_args()
    rv = json.load(open(x.run_valid_json)) if x.run_valid_json else None
    v = verdict(x.out, S1R.sha256_path(x.predeclaration), run_valid=rv, base_long_record=x.base_long_record)
    s = json.dumps(v, indent=1, default=str)
    if x.write:
        tmp = x.write + ".tmp"
        Path(tmp).write_text(s + "\n")
        os.replace(tmp, x.write)
    print(json.dumps({"verdict": v["verdict"], "item_verdicts": v["item_verdicts"], "records_seen": v["records_seen"],
                      "records_step_complete": v["records_step_complete"],
                      "excluded_predeclaration_mismatch": len(v["excluded_predeclaration_mismatch"]),
                      "missing_result_lines": v["missing_result_lines"]}))
