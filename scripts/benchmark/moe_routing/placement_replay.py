#!/usr/bin/env python3
"""RT-6: offline expert-placement replay on RT-1 routing traces (numpy only, no inference).

Question (DI-081): how unbalanced is the per-step expert load across NUMA nodes (4 bins) or CCDs (12 bins) when a decode step
of T tokens (T in {1, 3}) or a prefill chunk (512 tokens) is served from a given placement, and how much can placement /
replication recover? Score = per-step max-bin-load / mean-bin-load (1.0 = perfect), averaged over steps and layers, with the
null beside it and replication priced in redundant expert copies.

Arms (a "unit" is one (token, expert) selection; every selection is served by exactly one replica of its expert):
  contiguous      static, expert e -> bin e*B//E (no learning)
  null_random     mean over 20 random equal-cardinality placements (routing skew only; placement carries no information)
  null_uniform    Monte Carlo: T*K units thrown uniformly at B bins (the floor a pure multinomial gives; no skew at all)
  eplb_R{0,r}     EPLB-global (deepseek-ai/EPLB eplb.py logic: redundant slots to the highest load-per-replica expert one at a time,
                  then LPT packing with equal cardinality) from an AGGREGATE load vector of the calibration set
  deepinfer_R{..} DeepInfer window policy (described in intake-2134 diff (a)-(e)): input is a WINDOW [steps, experts]; the replica
                  chosen minimises the SUM OVER STEPS of the worst per-replica load; packing is greedy on the sum over steps of the
                  worst bin load. Re-implemented from the description, not a port of policy.py (not on this host).
  oracle_R{r}     zero-lag exact-load per-step planner in the UltraEP shape: binary search on the target bin load with a greedy
                  feasibility oracle (overloaded bins by descending excess, their experts by descending load, destination =
                  largest slack, no duplicate expert per bin), at most r new replicas per step and layer; quota floor 1 (the paper's
                  1024-token floor presumes thousands of tokens per step)
  moonep          rank-level greedy with capacity ceil(units/B) and unlimited replicas: the per-step INTEGRALITY FLOOR (reported as
                  its mean replicas per step, the price of reaching the floor)
  lp_floor@{arm}  fractional min-max over the replicas a static arm placed (LPLB's LP, solved exactly through Hall's condition:
                  t* = max over bin subsets S of load(experts whose replicas all lie in S) / |S|); a bound, not a policy
Static arms are scored two ways: name = IDEALIZED dispatch (each selection goes to the replica bin with the least load so far in the step,
which needs the step's other loads); name@random = REALIZABLE stateless dispatch (each selection picks a replica uniformly at random).
R0 arms have no replicas, so no dispatch choice. The DI-081 "practical" column uses only realizable arms.
Calibration data never overlaps evaluation windows: decode evaluation (windows 0-23) calibrates on the prefill rows of windows >= 24;
prefill evaluation (windows with id % 4 == 3) calibrates on the other windows. --calib cross calibrates on the other domains.

DI-081 trigger: baseline / (best max-over-mean) >= 1.10 on >= 2 workloads at T in {1, 3}; baseline defaults to the handoff's 1.65.
Usage: placement_replay.py RESULTS_DIR --out OUT_DIR [--domains ..] [--bins 4,12] [--redundant-frac 0.0625] [--calib same|cross]
CPU analysis only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_io as tio  # noqa: E402

BASELINE_RATIO = 1.65
WINDOW_STEPS = 128
MAX_PER_STEP_ARM_STEPS = 1000


# ----------------------------------------------------------------------------------------------- placements
def place_contiguous(E: int, B: int) -> list:
    return [[(e * B) // E] for e in range(E)]


def place_eplb(w: np.ndarray, B: int, R: int) -> list:
    """EPLB-global: replicate by hottest load/replica, LPT-pack replicas with equal cardinality."""
    E = w.size
    cnt = np.ones(E, int)
    wf = w.astype(np.float64)
    for _ in range(R):
        cnt[int(np.argmax(wf / cnt))] += 1
    items = sorted(((e, wf[e] / cnt[e]) for e in range(E) for _ in range(cnt[e])), key=lambda x: -x[1])
    cap = math.ceil((E + R) / B)
    load, size = np.zeros(B), np.zeros(B, int)
    bins_of: list = [[] for _ in range(E)]
    for e, wt in items:
        openb = np.flatnonzero(size < cap)
        b = int(openb[np.argmin(load[openb])])
        bins_of[e].append(b)
        load[b] += wt
        size[b] += 1
    return bins_of


def place_deepinfer(Wm: np.ndarray, B: int, R: int) -> list:
    """DeepInfer-style window policy (per-step peak objective). Wm: [steps, E] loads."""
    S, E = Wm.shape
    Wm = Wm.astype(np.float64)
    cnt = np.ones(E, int)
    tot = Wm.sum(0)
    ar = np.arange(E)
    for _ in range(R):
        ratio = Wm / cnt
        am = ratio.argmax(1)
        m1 = ratio[np.arange(S), am]
        r2 = ratio.copy()
        r2[np.arange(S), am] = -1.0
        m2 = r2.max(1)
        others = np.where(am[:, None] == ar[None, :], m2[:, None], m1[:, None])
        score = np.maximum(others, Wm / (cnt + 1)[None, :]).sum(0)
        e = int(np.lexsort((-tot, score))[0])
        cnt[e] += 1
    items = []
    for e in range(E):
        for _ in range(cnt[e]):
            items.append((e, Wm[:, e] / cnt[e]))
    items.sort(key=lambda x: -x[1].sum())
    cap = math.ceil((E + R) / B)
    load = np.zeros((S, B))
    size = np.zeros(B, int)
    bins_of: list = [[] for _ in range(E)]
    ar_s = np.arange(S)
    for e, x in items:
        openb = np.flatnonzero(size < cap)
        am = load.argmax(1)
        m1 = load[ar_s, am]
        l2 = load.copy()
        l2[ar_s, am] = -1.0
        m2 = l2.max(1)
        others = np.where(am[:, None] == np.arange(B)[None, :], m2[:, None], m1[:, None])
        score = np.maximum(others, load + x[:, None]).sum(0)
        b = int(openb[np.lexsort((load.sum(0)[openb], score[openb]))[0]])
        bins_of[e].append(b)
        load[:, b] += x
        size[b] += 1
    return bins_of


# ----------------------------------------------------------------------------------------------- evaluation
def waterfill(base: np.ndarray, c: int) -> np.ndarray:
    """Distribute c units over bins with current loads `base` to minimise the max; integer for small c, fractional otherwise."""
    n = base.size
    add = np.zeros(n)
    if c <= 16:
        cur = base.astype(np.float64).copy()
        for _ in range(c):
            j = int(np.argmin(cur))
            cur[j] += 1
            add[j] += 1
        return add
    order = np.argsort(base, kind="stable")
    bs = base[order].astype(np.float64)
    for i in range(1, n + 1):
        need = i * bs[i] - bs[:i].sum() if i < n else float("inf")
        if c <= need:
            level = (bs[:i].sum() + c) / i
            add[order[:i]] = level - bs[:i]
            return add
    return add


def eval_static(X: np.ndarray, bins_of: list, B: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-step max/mean over bins for a (possibly replicated) static placement. X: [S, E] loads.
    rng=None: IDEALIZED dispatch, each unit goes to the replica bin with the least load so far in the step (needs the step's other
    loads). rng given: REALIZABLE stateless dispatch, each unit picks one of the expert's replicas uniformly at random."""
    S, E = X.shape
    single = np.array([len(set(b)) == 1 for b in bins_of])
    oh = np.zeros((E, B))
    for e in np.flatnonzero(single):
        oh[e, bins_of[e][0]] = 1.0
    base = X @ oh
    multi = np.flatnonzero(~single)
    tot = X.sum(1)
    if multi.size:
        mb = {int(e): np.array(sorted(set(bins_of[e]))) for e in multi}
        pb = {int(e): np.array([bins_of[e].count(b) for b in mb[int(e)]], float) / len(bins_of[e]) for e in multi}
        Xm = X[:, multi]
        for s in np.flatnonzero(Xm.sum(1) > 0):
            order = np.argsort(-Xm[s], kind="stable")
            for j in order:
                c = int(Xm[s, j])
                if c <= 0:
                    break
                e = int(multi[j])
                bb = mb[e]
                if rng is None:
                    base[s, bb] += waterfill(base[s, bb], c)
                else:
                    base[s, bb] += rng.multinomial(c, pb[e])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, base.max(1) / (tot / B), np.nan)


def lp_floor(X: np.ndarray, bins_of: list, B: int) -> np.ndarray:
    """Exact fractional min-max (Hall) for the replicas in bins_of. Per-step max/mean."""
    S, E = X.shape
    nS = (1 << B) - 1
    subs = np.arange(1, nS + 1)
    inc = ((subs[None, :] >> np.arange(B)[:, None]) & 1).astype(np.float32)          # [B, nS]
    size = inc.sum(0)
    masks = np.array([sum(1 << b for b in set(bins_of[e])) for e in range(E)])
    single = np.array([len(set(b)) == 1 for b in bins_of])
    oh = np.zeros((E, B), np.float32)
    for e in np.flatnonzero(single):
        oh[e, bins_of[e][0]] = 1.0
    tot = np.zeros((S, nS), np.float32)
    tot += (X.astype(np.float32) @ oh) @ inc
    multi = np.flatnonzero(~single)
    if multi.size:
        M = ((masks[multi][:, None] & ~subs[None, :]) == 0).astype(np.float32)        # [nm, nS]
        tot += X[:, multi].astype(np.float32) @ M
    tstar = (tot / size[None, :]).max(1)
    total = X.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, tstar / (total / B), np.nan)


def oracle_step(c_act: np.ndarray, e_act: np.ndarray, home: np.ndarray, B: int, R: int) -> tuple:
    L = np.bincount(home[e_act], weights=c_act, minlength=B).astype(np.int64)
    total = int(L.sum())
    lo, hi = -(-total // B), int(L.max())
    if hi <= lo:
        return hi, 0
    by_bin = [[(int(c_act[i]), int(e_act[i])) for i in np.flatnonzero(home[e_act] == b)] for b in range(B)]
    for b in range(B):
        by_bin[b].sort(reverse=True)

    def feas(tau: int):
        load = L.copy()
        used = 0
        holders: dict = {}
        for b in sorted(range(B), key=lambda b: -(load[b] - tau)):
            if load[b] <= tau:
                continue
            for c, e in by_bin[b]:
                rem = c
                while load[b] > tau and rem > 0:
                    cand = [d for d in range(B) if d != b and load[d] < tau and d not in holders.get(e, ())]
                    if not cand or used >= R:
                        return False, used
                    d = min(cand, key=lambda d: load[d])
                    m = min(rem, int(load[b] - tau), int(tau - load[d]))
                    load[b] -= m
                    load[d] += m
                    rem -= m
                    used += 1
                    holders.setdefault(e, set()).add(d)
                if load[b] <= tau:
                    break
            if load[b] > tau:
                return False, used
        return True, used

    best_u = 0
    while lo < hi:
        mid = (lo + hi) // 2
        ok, u = feas(mid)
        if ok:
            hi, best_u = mid, u
        else:
            lo = mid + 1
    if hi == int(L.max()):
        return hi, 0
    return hi, feas(hi)[1]


def moonep_step(c_act: np.ndarray, e_act: np.ndarray, home: np.ndarray, B: int) -> int:
    """Replicas created by the surplus->deficit greedy that reaches capacity ceil(units/B)."""
    L = np.bincount(home[e_act], weights=c_act, minlength=B).astype(np.int64)
    cap = -(-int(L.sum()) // B)
    rem = {int(e): int(c) for e, c in zip(e_act, c_act)}
    reps = 0
    seen: set = set()
    for _ in range(4 * B * B):
        s, d = int(np.argmax(L)), int(np.argmin(L))
        m = min(int(L[s]) - cap, cap - int(L[d]))
        if L[s] <= cap or m <= 0:
            break
        left = m
        for e in sorted((e for e in rem if home[e] == s and rem[e] > 0), key=lambda e: -rem[e]):
            t = min(rem[e], left)
            rem[e] -= t
            left -= t
            if (e, d) not in seen:
                seen.add((e, d))
                reps += 1
            if left == 0:
                break
        L[s] -= m
        L[d] += m
    return reps


def null_uniform_mc(T: int, K: int, B: int, trials: int = 20000, seed: int = 11) -> float:
    rng = np.random.default_rng(seed)
    units = rng.integers(0, B, size=(trials, T * K))
    cnt = np.stack([(units == b).sum(1) for b in range(B)], 1)
    return float((cnt.max(1) / (T * K / B)).mean())


def integrality_floor(T: int, K: int, B: int) -> float:
    return math.ceil(T * K / B) / (T * K / B)


# ----------------------------------------------------------------------------------------------- driver
def _calib_windows(cal: tio.Trace, l: int, E: int, T: int, rng: np.random.Generator, nwin: int):
    blk, nb, _ = tio.blocks(cal, T)
    sel = rng.choice(nb, size=min(nwin, nb), replace=False)
    remap = -np.ones(nb, np.int64)
    remap[sel] = np.arange(sel.size)
    b2 = np.where(blk >= 0, remap[np.maximum(blk, 0)], -1)
    return tio.layer_loads(cal.layer_ids(l), b2, sel.size, E)


def ci_by_window(r: np.ndarray, win: np.ndarray, boots: int = 600, seed: int = 5) -> list:
    ok = ~np.isnan(r)
    r, win = r[ok], win[ok]
    u, inv = np.unique(win, return_inverse=True)
    s = np.bincount(inv, weights=r)
    n = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(boots):
        idx = rng.integers(0, u.size, u.size)
        vals.append(s[idx].sum() / n[idx].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def replay_domain(args: tuple) -> dict:
    results_dir, dom, bins_list, red_frac, calib, baseline, layers_limit, seed = args
    t = tio.load_domain(results_dir, dom)
    others = [tio.load_domain(results_dir, d, parts=("bulk", "decode")) for d in sorted(tio.discover(results_dir)) if d != dom] if calib == "cross" else []
    L = t.L if not layers_limit else min(t.L, layers_limit)
    E, K = int(t.n_experts[0]), int(t.top_k[0])
    rng = np.random.default_rng(seed)

    dec = t.select(t.phase == 1)
    pre_eval = t.select((t.phase == 0) & (t.win % 4 == 3))
    if calib == "cross":
        cal_dec = tio.concat([o.select(o.phase == 0) for o in others])
        cal_pre = cal_dec
    else:
        cal_dec = t.select((t.phase == 0) & (t.win >= 24))
        cal_pre = t.select((t.phase == 0) & (t.win % 4 != 3))
    cnts = np.unique(pre_eval.win, return_counts=True)[1]
    t_pre = int(np.median(cnts)) if cnts.size else 0
    configs = [("decode", 1, dec, cal_dec), ("decode", 3, dec, cal_dec), ("prefill", t_pre, pre_eval, cal_pre)]
    out: dict = {"domain": dom, "arch": t.arch, "E": E, "K": K, "layers": L, "calib": calib, "configs": {}}
    for B in bins_list:
        R1 = int(round(red_frac * E))
        for phase, T, ev, cal in configs:
            if T <= 0:
                continue
            blk, nb, win_of = tio.blocks(ev, T)
            if nb == 0:
                continue
            step_idx = np.arange(nb) if (nb <= MAX_PER_STEP_ARM_STEPS) else np.unique(np.linspace(0, nb - 1, MAX_PER_STEP_ARM_STEPS).astype(int))
            arms: dict = {}

            def acc(name, per_step, idx=None):
                a = arms.setdefault(name, {"sum": 0.0, "n": 0, "vec": None, "idx": idx})
                v = per_step if a["vec"] is None else a["vec"] + per_step
                a["vec"] = v
                a["n"] += 1

            extra: dict = {}
            for l in range(L):
                ids_l = ev.layer_ids(l)
                X = tio.layer_loads(ids_l, blk, nb, E).astype(np.float64)
                cal_ids = cal.layer_ids(l)
                w = tio.layer_counts(cal_ids, E).astype(np.float64)
                Wm = _calib_windows(cal, l, E, T, rng, WINDOW_STEPS).astype(np.float64)
                contiguous = place_contiguous(E, B)
                acc("contiguous", eval_static(X, contiguous, B))
                nulls = []
                for _ in range(20):
                    perm = rng.permutation(E)
                    nulls.append(eval_static(X, [[int((perm[e] * B) // E)] for e in range(E)], B))
                acc("null_random", np.mean(nulls, 0))
                statics = {}
                for R in (0, R1):
                    statics[f"eplb_R{R}"] = place_eplb(w, B, R)
                    statics[f"deepinfer_R{R}"] = place_deepinfer(Wm, B, R)
                for name, pl in statics.items():
                    acc(name, eval_static(X, pl, B))
                    if "_R0" not in name:
                        acc(name + "@random", eval_static(X, pl, B, np.random.default_rng(seed + l)))
                for name in (f"eplb_R{R1}", f"deepinfer_R{R1}"):
                    acc("lp_floor@" + name, lp_floor(X, statics[name], B))
                # per-step arms on a subsample of steps
                home = np.array([b[0] for b in contiguous])
                orc = np.full(step_idx.size, np.nan)
                mo = np.zeros(step_idx.size)
                for j, s in enumerate(step_idx):
                    act = np.flatnonzero(X[s] > 0)
                    c_act = X[s][act].astype(np.int64)
                    mx, _u = oracle_step(c_act, act, home, B, R1)
                    orc[j] = mx / (X[s].sum() / B)
                    mo[j] = moonep_step(c_act, act, home, B)
                    extra.setdefault("oracle_copies", []).append(_u)
                acc("oracle_R%d" % R1, orc, step_idx)
                extra.setdefault("moonep_replicas", []).append(float(mo.mean()))
            key = f"B{B}/{phase}/T{T}"
            res: dict = {"bins": B, "phase": phase, "T": T, "steps": int(nb), "per_step_arm_steps": int(step_idx.size),
                         "units_per_step": T * K, "R1": R1, "redundant_frac": R1 / E, "arms": {}}
            for name, a in arms.items():
                v = a["vec"] / a["n"]
                widx = win_of if a["idx"] is None else win_of[a["idx"]]
                res["arms"][name] = {"max_over_mean": float(np.nanmean(v)), "ci95": ci_by_window(v, widx)}
            res["arms"]["null_uniform_mc"] = {"max_over_mean": null_uniform_mc(T, K, B)}
            res["arms"]["integrality_floor"] = {"max_over_mean": integrality_floor(T, K, B)}
            res["arms"]["moonep_floor"] = {"max_over_mean": integrality_floor(T, K, B), "mean_replicas_per_step_layer": float(np.mean(extra["moonep_replicas"]))}
            res["arms"]["oracle_R%d" % R1]["mean_new_replicas_per_step_layer"] = float(np.mean(extra["oracle_copies"]))
            for nm in ("eplb_R%d", "deepinfer_R%d", "eplb_R%d@random", "deepinfer_R%d@random"):
                res["arms"][nm % R1]["extra_copies_per_layer"] = R1
            real = lambda n: n.startswith(("eplb_", "deepinfer_")) and (n.endswith("@random") or "_R0" in n)
            practical = {n: v["max_over_mean"] for n, v in res["arms"].items() if real(n)}
            idealized = {n: v["max_over_mean"] for n, v in res["arms"].items() if n.startswith(("eplb_", "deepinfer_")) and not real(n)}
            bound = {n: v["max_over_mean"] for n, v in res["arms"].items() if n.startswith(("oracle_", "lp_floor@"))}
            bp, bb, bi = min(practical, key=practical.get), min(bound, key=bound.get), min(idealized, key=idealized.get)
            res["decision"] = {"baseline_ratio": baseline, "best_practical": bp, "best_practical_max_over_mean": practical[bp],
                               "ratio_practical": baseline / practical[bp], "best_idealized_dispatch": bi,
                               "best_idealized_dispatch_max_over_mean": idealized[bi], "ratio_idealized_dispatch": baseline / idealized[bi], "best_bound": bb, "best_bound_max_over_mean": bound[bb],
                               "ratio_bound": baseline / bound[bb], "ratio_measured_contiguous": res["arms"]["contiguous"]["max_over_mean"] / practical[bp]}
            out["configs"][key] = res
    return out


def to_markdown(rs: list, title: str, baseline: float) -> str:
    L = ["# " + title, "", f"Per-step max-bin-load / mean-bin-load (lower is better, 1.0 perfect), mean over steps and layers, [95% CI over windows]. "
         f"Trigger: {baseline} / best >= 1.10 on >= 2 workloads at T in {{1,3}}. Per-step arms (oracle) use <= {MAX_PER_STEP_ARM_STEPS} evenly spaced steps.", ""]
    keys = sorted({k for r in rs for k in r["configs"]}, key=lambda k: (int(k.split("/")[0][1:]), k.split("/")[1], int(k.split("T")[1])))
    for k in keys:
        L += [f"## {k}", ""]
        arms = sorted({a for r in rs for a in r["configs"].get(k, {}).get("arms", {})}, key=lambda a: (a.split("_")[0].split("@")[0], a))
        L += ["| arm | " + " | ".join(r["domain"] for r in rs) + " |", "|---|" + "---|" * len(rs)]
        for a in arms:
            cells = []
            for r in rs:
                v = r["configs"].get(k, {}).get("arms", {}).get(a)
                if not v:
                    cells.append("")
                    continue
                s = f"{v['max_over_mean']:.3f}"
                if "ci95" in v:
                    s += f" [{v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}]"
                if "mean_replicas_per_step_layer" in v:
                    s += f" ({v['mean_replicas_per_step_layer']:.1f} copies/step)"
                if "mean_new_replicas_per_step_layer" in v:
                    s += f" ({v['mean_new_replicas_per_step_layer']:.1f} copies/step)"
                if "extra_copies_per_layer" in v:
                    s += f" (+{v['extra_copies_per_layer']} copies/layer)"
                cells.append(s)
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        cells = []
        for r in rs:
            d = r["configs"].get(k, {}).get("decision")
            cells.append(f"realizable {baseline}/{d['best_practical_max_over_mean']:.3f} = **{d['ratio_practical']:.3f}** ({d['best_practical']}); idealized-dispatch {d['ratio_idealized_dispatch']:.3f} ({d['best_idealized_dispatch']}); bound {d['best_bound_max_over_mean']:.3f} ({d['best_bound']}) -> {d['ratio_bound']:.3f}" if d else "")
        L += ["| *trigger ratio* | " + " | ".join(cells) + " |", ""]
    L += ["## DI-081 trigger", ""]
    for B in sorted({int(k.split("/")[0][1:]) for k in keys}):
        for T in (1, 3):
            k = f"B{B}/decode/T{T}"
            meet_p = [r["domain"] for r in rs if r["configs"].get(k, {}).get("decision", {}).get("ratio_practical", 0) >= 1.10]
            meet_i = [r["domain"] for r in rs if r["configs"].get(k, {}).get("decision", {}).get("ratio_idealized_dispatch", 0) >= 1.10]
            meet_b = [r["domain"] for r in rs if r["configs"].get(k, {}).get("decision", {}).get("ratio_bound", 0) >= 1.10]
            L.append(f"* {k}: realizable static arms meet 1.10 on {meet_p}; with idealized least-loaded dispatch on {meet_i}; bounds meet 1.10 on {meet_b}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="")
    ap.add_argument("--bins", default="4,12")
    ap.add_argument("--redundant-frac", type=float, default=0.0625)
    ap.add_argument("--calib", choices=("same", "cross"), default="same")
    ap.add_argument("--baseline-ratio", type=float, default=BASELINE_RATIO)
    ap.add_argument("--layers", type=int, default=0, help="limit to the first N layers (0 = all)")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--title", default="RT-6 offline expert-placement replay")
    a = ap.parse_args(argv)
    doms = [d for d in a.domains.split(",") if d] or sorted(tio.discover(a.results_dir))
    bins = [int(x) for x in a.bins.split(",")]
    jobs = [(a.results_dir, d, bins, a.redundant_frac, a.calib, a.baseline_ratio, a.layers, a.seed) for d in doms]
    if a.procs > 1 and len(jobs) > 1:
        with Pool(min(a.procs, len(jobs))) as p:
            rs = p.map(replay_domain, jobs)
    else:
        rs = [replay_domain(j) for j in jobs]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"placement_replay_{a.calib}.json").write_text(json.dumps(rs, indent=1) + "\n")
    md = to_markdown(rs, a.title + f" (calibration: {a.calib}-domain)", a.baseline_ratio)
    (out / f"placement_replay_{a.calib}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
