#!/usr/bin/env python3
"""PB-QCOST-1 CONFIRM + q-axis probe (copy of pb_qcost_1.py with -ub as an axis; the original is untouched, sha256 in ORIG_DRIVER.sha256).
Rules for THIS run: PREDECLARATION.md (same directory). Added vs the original: --ubs (default 512,256,1024; one server per ub with
-b = -ub, so q = ub - 8n), per-ub rep plan, raw interval ms (t_ms) and q (dn) kept for the q fit, analyse_q() (pooled
t = a_n + b*q + d*q*c per n) and a per-server graphs/HSA/own-pid-VRAM proof from /proc/<pid>/environ + KFD. The ub = 512 phase is the
unchanged original protocol (3 reps x 5 lengths x n in {1,4}); the original text follows.

PB-QCOST-1 re-validation probe — quadratic prefill-cost fit at n = 1 and n = 4 generating slots, plus held-out P99 residual.

Task (gpu-serving-tie-in-program.md PB-QCOST-1, refined by intake-2223): the offline fit t = 324 + 0.987 q + 2.41e-5 q c ms
(analytics-20261020/PB-QCOST-1/REPORT.md) was taken at ONE decode-batch point (gen = 3 slots, decode_rows 24). Re-validate the
quadratic prefill cost at n = 1 and n = 4 generating slots (decode_rows 8 and 32), report the per-n intercept with its noise
floor AND UNIT, and add the held-out P99 absolute residual beside rmse/MAPE (KVU-16g-2 sets target-ms = TPOT target minus it).

Model under test (frozen, kpf18_gate.py rate model): prefill ms/tok = (1.15 + 0.0235 * occupied_k) * 1.1,
occupied_k = (own cached cells + n/2) / 1000 (K.pf_s).

Server: ONE champion launch, production gate argv with -np 5 (4 feeder slots + 1 prefill slot: n = 4 generating slots needs a
5th slot to prefill in; -np 4 would leave no prefill slot).  Slot 0 = prefill under test (cache_prompt=false, erased between
requests); slots 1..n = feeders decoding (DFlash2) for the whole block.
Blocks (interleaved to cancel host drift, ABAB...): [n=1 r1, n=4 r1, n=1 r2, n=4 r2, n=1 r3, n=4 r3]; lengths per block
ascending (odd reps) / descending (even reps).  Prefill lengths: 4k 8k 16k 32k 64k tokens.
PRE-DECLARED split: TRAIN = {4k, 16k, 64k} (per-n refit), HELD-OUT = {8k, 32k} (never used to fit).
Measures: (a) request level, ms/tok = TTFT/L;  (b) interval level from the server log "prompt processing, n_tokens" lines
(consecutive-line delta, normalised to ms per 512-token iteration) so P99 has hundreds of samples.

PRE-DECLARED PASS RULE (the task text names no numeric bar; the 10% held-out MAPE bar is operator-accepted, derived from the offline result of held-out
MAPE 4.0-7.8%, noise 40-65 ms/iter):  per n, with >= 3 valid reps at every length:
  PASS_n  iff  frozen-model request-level MAPE on HELD-OUT lengths <= 10%  AND  refit-model request-level MAPE on HELD-OUT <= 10%.
  FAIL_n  otherwise (the refit coefficients, intercept, noise floor and P99 are still reported and are the replacement).
  Overall: PASS iff PASS at n=1 and n=4; INCOMPLETE if any n lacks valid data.  P99 residual is reported, not thresholded.

--dry-run starts nothing, sends nothing, takes no claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, "/mnt/raid0/llm/tmp/pb-hol-1")
import pb_hol_1 as H  # noqa: E402  (reuse: LogTail, port_owner_ok, parse_deadline, load_decoder)
K = H.K  # kpf18_gate

SCHEMA = "epyc.pbqcost1.run.v1"
CHAMPION = H.CHAMPION
ARGV_FILE = H.ARGV_FILE
PROMPT_SET = H.PROMPT_SET
NP, PORT = 5, 18391
UBS = [512, 256, 1024]              # phase order: 512 first (the confirm), then the q axis
BASE_UB = 512
Q_REPS = {4000: 1, 8000: 1, 16000: 3, 32000: 3, 64000: 1}   # reps per length in a q-axis phase (ub != 512)
VRAM_GUARD_GIB = 60.0               # own-pid KFD VRAM guard (GPU-LEAK-1)
BOOT_SEED, BOOT_N = 20261023, 2000  # cluster bootstrap for the b, d confidence intervals
C_REF = 32000                       # cached-token reference for turning a slope difference into ms
PREFILL_SLOT = 0
LENGTHS = [4000, 8000, 16000, 32000, 64000]
TRAIN = [4000, 16000, 64000]
HELD = [8000, 32000]
NS = [1, 4]
REPS = 3
TRUNK_GROUP = 7                      # 80k trunk; slices [:L] are the prefill prompts
N_PRED_PREFILL = 2
FEED_NPRED = 65536                   # feeder decodes for the whole block; aborted at block end
PASS_MAPE = 10.0                     # percent, see PASS RULE
IT_Q = 512                           # tokens per iteration the interval ms are normalised to (gate -b 512)
OLD21 = {1: {"A": 766.85, "B": 0.010794, "q0": 504}, 4: {"A": 871.40, "B": 0.013798, "q0": 480}}   # 2026-10-21T21:29Z refit
B_OFF = 0.9865                       # ms/token ASSUMED in the 2026-10-23 replay; the q axis identifies it
OFFLINE = (324.2, 0.9865, 2.408e-5)  # a ms, b ms/tok, d ms/(tok*cached tok): the fit being re-validated at gen = 3
LOAD_S, LINK_S, STOP_S = 90.0, 5.0, 25.0
CLAIM_WAIT_S = 900.0                 # wait up to 15 min for a held mi210_0 claim, then give up (rc 75)


# ------------------------------------------------------------------ model + statistics (pure)
def rate_ms_per_tok(mid_tokens: float) -> float:
    """Frozen gate rate model at a mean occupancy of mid_tokens cells."""
    return (K.PF_BASE_MS + K.PF_PER_K_MS * mid_tokens / 1000.0) * K.B512_FACTOR


def pred_request_ms(L: int) -> float:
    return K.pf_s(L, 0) * 1000.0


def pct(v: list[float], p: float) -> float | None:
    if not v:
        return None
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    x = (len(s) - 1) * p / 100.0
    lo = int(math.floor(x))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    if len(xs) < 2 or max(xs) == min(xs):
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def resid_stats(meas: list[float], pred: list[float]) -> dict:
    if not meas:
        return {"n": 0}
    r = [m - p for m, p in zip(meas, pred)]
    ar = [abs(x) for x in r]
    return {"n": len(r), "rmse": round(math.sqrt(statistics.fmean(x * x for x in r)), 3),
            "mape_pct": round(100 * statistics.fmean(abs(a / m) for a, m in zip(r, meas)), 2),
            "bias_mean": round(statistics.fmean(r), 3), "p50_abs": round(pct(ar, 50), 3), "p95_abs": round(pct(ar, 95), 3),
            "p99_abs": round(pct(ar, 99), 3), "max_abs": round(max(ar), 3),
            "p99_under_prediction": round(pct(r, 99), 3)}   # one-sided: measured - predicted, the target-ms margin


def intervals_from_log(lines: list[str], slot: int, t0: float, t1: float, ub: int = BASE_UB) -> list[dict]:
    """Consecutive 'prompt processing' lines of `slot` inside [t0, t1] -> (start tokens, dn, dt) intervals."""
    pts = []
    for ln in lines:
        m, t = H.PROG_RE.search(ln), H.LOGTS_RE.match(ln)
        if m and t and int(m.group(1)) == slot:
            ts = float(t.group(1))
            if t0 - 1 <= ts <= t1 + 2:
                pts.append((ts, int(m.group(3))))
    out = []
    for (ta, na), (tb, nb) in zip(pts, pts[1:]):
        dn, dt = nb - na, tb - ta
        lo = 256 if ub >= BASE_UB else 128      # ub 512/1024 keep the original 256 floor (old data re-analyses identically)
        if lo <= dn <= 8192 and dt > 0:
            out.append({"c": na, "dn": dn, "dt_s": round(dt, 4), "t_ms": round(dt * 1000.0, 2),
                        "iter_ms": round(dt * 1000.0 * IT_Q / dn, 2)})
    return out


def analyse(rows: list[dict]) -> dict:
    """rows = pb_qcost_1.jsonl records.  Returns the per-n verdicts and numbers (pure; also usable post hoc)."""
    runs = [r for r in rows if r.get("kind") == "run" and r.get("ttft_ms") and not r.get("skipped") and r.get("http") == 200
            and r.get("ub", BASE_UB) == BASE_UB]      # rows without "ub" are the 2026-10-21 run (ub 512)
    res: dict = {"per_n": {}}
    ok_all = True
    for n in NS:
        rn = [r for r in runs if r["n_gen"] == n]
        per_len = {}
        for L in LENGTHS:
            v = [r["ttft_ms"] for r in rn if r["length"] == L]
            per_len[L] = {"reps_valid": len(v), "ttft_ms": v,
                          "meas_ms_per_tok_median": round(statistics.median(v) / L, 4) if v else None,
                          "frozen_pred_ms_per_tok": round(pred_request_ms(L) / L, 4)}
        have = all(per_len[L]["reps_valid"] >= REPS for L in LENGTHS)
        # request level
        tr = [(r["length"] / 2.0, r["ttft_ms"] / r["length"]) for r in rn if r["length"] in TRAIN]
        fit = ols([x for x, _ in tr], [y for _, y in tr])    # ms/tok = a + b * mid_tokens
        held = [r for r in rn if r["length"] in HELD]
        hm = [r["ttft_ms"] / r["length"] for r in held]
        frozen_h = [pred_request_ms(r["length"]) / r["length"] for r in held]
        refit_h = [fit[0] + fit[1] * r["length"] / 2.0 for r in held] if fit else []
        # interval level (per-512-token iteration, ms)
        iv = []
        for r in rn:
            for i in r.get("intervals") or []:
                iv.append({**i, "length": r["length"], "mid": i["c"] + i["dn"] / 2.0})
        ivt = [i for i in iv if i["length"] in TRAIN]
        ivh = [i for i in iv if i["length"] in HELD]
        ifit = ols([i["c"] for i in ivt], [i["iter_ms"] for i in ivt])      # iter_ms = A + B * c  (q = 512)
        i_frozen_h = [IT_Q * rate_ms_per_tok(i["mid"]) for i in ivh]
        i_refit_h = [ifit[0] + ifit[1] * i["c"] for i in ivh] if ifit else []
        i_off_h = [OFFLINE[0] + OFFLINE[1] * i["dn"] + OFFLINE[2] * i["dn"] * i["c"] for i in ivh]
        noise = resid_stats([i["iter_ms"] for i in ivt], [ifit[0] + ifit[1] * i["c"] for i in ivt]) if ifit else {"n": 0}
        req = {"frozen_heldout": resid_stats(hm, frozen_h), "refit_heldout": resid_stats(hm, refit_h) if fit else {"n": 0},
               "refit_ms_per_tok": {"a_at_zero_occupancy": round(fit[0], 4), "b_per_cached_token": round(fit[1], 8)} if fit else None,
               "frozen_all_points": resid_stats([r["ttft_ms"] / r["length"] for r in rn],
                                                [pred_request_ms(r["length"]) / r["length"] for r in rn])}
        itv = {"unit": f"ms per server iteration (q = {IT_Q} tokens, interval = consecutive log lines)",
               "refit_intercept_ms_at_c0": round(ifit[0], 1) if ifit else None,
               "refit_slope_ms_per_cached_token": round(ifit[1], 6) if ifit else None,
               "noise_floor_train_rmse_ms_per_iter": noise.get("rmse"), "noise_floor_n_intervals": noise.get("n"),
               "frozen_model_heldout": resid_stats([i["iter_ms"] for i in ivh], i_frozen_h),
               "refit_model_heldout": resid_stats([i["iter_ms"] for i in ivh], i_refit_h) if ifit else {"n": 0},
               "offline_324_fit_heldout": resid_stats([i["iter_ms"] for i in ivh], i_off_h),
               "offline_324_fit_intercept_ms_at_q512": round(OFFLINE[0] + OFFLINE[1] * IT_Q, 1)}
        verdict = "INCOMPLETE"
        if have and fit:
            fh, rh = req["frozen_heldout"].get("mape_pct"), req["refit_heldout"].get("mape_pct")
            verdict = "PASS" if (fh is not None and rh is not None and fh <= PASS_MAPE and rh <= PASS_MAPE) else "FAIL"
        ok_all &= verdict == "PASS"
        res["per_n"][n] = {"verdict": verdict, "all_lengths_have_3_reps": have, "per_length": per_len,
                           "decode_rows": 8 * n, "request_level": req, "interval_level": itv,
                           "feeder_tok_s_median": (statistics.median([r["feeder_tok_s"] for r in rn if r.get("feeder_tok_s")])
                                                   if any(r.get("feeder_tok_s") for r in rn) else None)}
    ver = [res["per_n"][n]["verdict"] for n in NS]
    res["overall"] = "PASS" if all(v == "PASS" for v in ver) else ("INCOMPLETE" if "INCOMPLETE" in ver else "FAIL")
    return res


def usable(rows: list[dict]) -> list[dict]:
    """Keep run rows of COMPLETED phases only (a phase cut short by a deadline/abort is voided and redone on resume).
    Rows with no phase_id are legacy (the 2026-10-21 run) and are kept."""
    done = {r["phase_id"] for r in rows if r.get("kind") == "phase_done"}
    return [r for r in rows if r.get("kind") != "run" or "phase_id" not in r or r["phase_id"] in done]


def phase_estimate_s(ub: int, with_n0: bool = False) -> float:
    return plan_estimate([ub], with_n0)[0]


def _solve3(m: list[list[float]], v: list[float]) -> list[float] | None:
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-12:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(3):
            if r != c:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][3] / a[i][i] for i in range(3)]


def _feat(i: dict) -> list[float]:
    """Scaled features [1, q/512, q*c/(512*C_REF)] so the 3x3 normal equations are well conditioned."""
    return [1.0, i["dn"] / 512.0, i["dn"] * i["c"] / (512.0 * C_REF)]


def _suff(ivs: list[dict]) -> tuple[list[list[float]], list[float]]:
    xtx = [[0.0] * 3 for _ in range(3)]
    xty = [0.0] * 3
    for i in ivs:
        x, y = _feat(i), i["t_ms"]
        for a in range(3):
            xty[a] += x[a] * y
            for b in range(3):
                xtx[a][b] += x[a] * x[b]
    return xtx, xty


def _coef(xtx, xty) -> dict | None:
    k = _solve3(xtx, xty)
    if k is None:
        return None
    return {"a_ms": k[0], "b_ms_per_token": k[1] / 512.0, "d_ms_per_token_cached": k[2] / (512.0 * C_REF)}


def _pred(co: dict, i: dict) -> float:
    return co["a_ms"] + co["b_ms_per_token"] * i["dn"] + co["d_ms_per_token_cached"] * i["dn"] * i["c"]


def analyse_q(rows: list[dict]) -> dict:
    """Pooled fit over ALL ub cells, per n: t_ms = a_n + b*q + d*q*c on TRAIN-length intervals (q = actual dn, t_ms raw per iteration),
    held-out on HELD lengths.  Pure.  Verdict per n: INCOMPLETE (q not identified) / PASS (held-out interval MAPE <= PASS_MAPE) / FAIL."""
    import random
    runs = [r for r in rows if r.get("kind") == "run" and r.get("ttft_ms") and not r.get("skipped") and r.get("http") == 200]
    out: dict = {"per_n": {}, "model": "t_ms = a_n + b*q + d*q*c (per n), q = tokens in the iteration, c = cached prefix, unit ms/iteration"}
    for n in NS:
        rn = [r for r in runs if r["n_gen"] == n]
        ivs = []
        for k, r in enumerate(rn):
            for i in r.get("intervals") or []:
                ivs.append({**i, "length": r["length"], "ub": r.get("ub", BASE_UB), "req": k,
                            "t_ms": i.get("t_ms") if i.get("t_ms") is not None else i["dt_s"] * 1000.0})
        tr = [i for i in ivs if i["length"] in TRAIN]
        ho = [i for i in ivs if i["length"] in HELD]
        lev = {}
        for u in sorted({i["ub"] for i in ivs}):
            dn = [i["dn"] for i in ivs if i["ub"] == u]
            lev[u] = {"median_dn": statistics.median(dn), "n_intervals": len(dn),
                      "n_requests": len({i["req"] for i in ivs if i["ub"] == u})}
        spread = (max(v["median_dn"] for v in lev.values()) - min(v["median_dn"] for v in lev.values())) if lev else 0
        identified = len(lev) >= 3 and spread >= 500 and all(v["n_requests"] >= 3 for v in lev.values())
        co = _coef(*_suff(tr)) if len(tr) >= 3 else None
        res = {"q_levels_by_ub": lev, "q_spread_tokens": spread, "q_identified": bool(identified and co), "coef": None}
        if co:
            rmse = math.sqrt(statistics.fmean((i["t_ms"] - _pred(co, i)) ** 2 for i in tr))
            byreq: dict = {}
            for i in tr:
                byreq.setdefault(i["req"], []).append(i)
            suff = {k: _suff(v) for k, v in byreq.items()}
            keys, rng, bs = sorted(suff), random.Random(BOOT_SEED), []
            for _ in range(BOOT_N):
                xtx = [[0.0] * 3 for _ in range(3)]
                xty = [0.0] * 3
                for k in (rng.choice(keys) for _ in keys):
                    a, b = suff[k]
                    for p in range(3):
                        xty[p] += b[p]
                        for q in range(3):
                            xtx[p][q] += a[p][q]
                c2 = _coef(xtx, xty)
                if c2:
                    bs.append(c2)

            def ci(key):
                return [round(pct([x[key] for x in bs], 2.5), 8), round(pct([x[key] for x in bs], 97.5), 8)] if len(bs) > 50 else None
            held = resid_stats([i["t_ms"] for i in ho], [_pred(co, i) for i in ho]) if ho else {"n": 0}
            per_ub_held = {u: resid_stats([i["t_ms"] for i in ho if i["ub"] == u], [_pred(co, i) for i in ho if i["ub"] == u])
                           for u in lev if any(i["ub"] == u for i in ho)}
            louo = {}
            for u in lev:
                oth = [i for i in ivs if i["ub"] != u]
                cu = _coef(*_suff(oth)) if len(oth) >= 3 else None
                mine = [i for i in ivs if i["ub"] == u]
                louo[u] = resid_stats([i["t_ms"] for i in mine], [_pred(cu, i) for i in mine]) if cu and mine else {"n": 0}
            b_d = abs(co["b_ms_per_token"] - B_OFF) * spread
            d_old = OLD21[n]["B"] / 512.0      # raw-time d implied by the normalised 21:29Z slope B (ms per cached token per 512-token iteration)
            d_d = abs(co["d_ms_per_token_cached"] - d_old) * OLD21[n]["q0"] * C_REF
            res.update({"coef": {k: round(v, 8) for k, v in co.items()},
                        "implied_intercept_ms_at_q512_c0": round(co["a_ms"] + 512 * co["b_ms_per_token"], 2),
                        "ci95_b": ci("b_ms_per_token"), "ci95_d": ci("d_ms_per_token_cached"), "ci95_a": ci("a_ms"),
                        "noise_floor_train_rmse_ms_per_iter": round(rmse, 2), "n_train_intervals": len(tr),
                        "heldout_interval": held, "heldout_by_ub": per_ub_held, "leave_one_ub_out": louo,
                        "b_vs_assumed_0.9865": {"abs_diff_ms_per_token": round(abs(co["b_ms_per_token"] - B_OFF), 5),
                                                "times_q_span_ms": round(b_d, 1), "agrees_within_noise": b_d <= rmse},
                        "d_vs_2026_10_21_refit": {"d_old": round(d_old, 9), "diff_times_q0_times_Cref_ms": round(d_d, 1),
                                                  "agrees_within_noise": d_d <= rmse}})
        mp = res.get("heldout_interval", {}).get("mape_pct") if co else None
        res["verdict"] = "INCOMPLETE" if (not res["q_identified"] or mp is None) else ("PASS" if mp <= PASS_MAPE else "FAIL")
        out["per_n"][n] = res
    v = [out["per_n"][n]["verdict"] for n in NS]
    out["overall"] = "PASS" if all(x == "PASS" for x in v) else ("INCOMPLETE" if "INCOMPLETE" in v else "FAIL")
    return out


# ------------------------------------------------------------------ estimate
def est_iter_ms(n: int, ub: int, c: float) -> float:
    """Iteration time from the 2026-10-21 per-n refit with the offline q coefficient (estimate only, for the window plan)."""
    o = OLD21[n]        # A, B are per 512-NORMALISED iteration (iter_ms = dt*1000*512/dn); raw time at q0 = (A + B c) * q0 / 512
    q = ub - 8 * n
    return o["A"] * o["q0"] / 512.0 + B_OFF * (q - o["q0"]) + (o["B"] / 512.0) * q * c


def est_request_s(L: int, n: int, ub: int = BASE_UB) -> float:
    q = ub - 8 * n
    iters = max(1, math.ceil(L / q))
    return sum(est_iter_ms(n, ub, k * q) for k in range(iters)) / 1000.0 + 3.0


def reps_for(ub: int) -> dict:
    return {L: REPS for L in LENGTHS} if ub == BASE_UB else dict(Q_REPS)


def plan_estimate(ubs: list[int] | None = None, with_n0: bool = False) -> tuple[float, list[tuple[str, float]]]:
    steps = []
    for ub in (ubs or UBS):
        steps.append((f"[ub {ub}] server load + linkage + residency (-np 5, -b {ub} -ub {ub})", LOAD_S + LINK_S + 10))
        reps = reps_for(ub)
        for r in range(1, max(reps.values()) + 1):
            for n in NS:
                lens0 = [L for L in LENGTHS if reps[L] >= r]
                lens = lens0 if r % 2 else lens0[::-1]
                steps.append((f"[ub {ub}] block n={n} rep {r}: start {n} feeder(s) + warm", 15.0 + 8.0 * n))
                for L in lens:
                    steps.append((f"[ub {ub}]   prefill {L // 1000}k @ n={n} r{r}", est_request_s(L, n, ub) + 4))
                steps.append((f"[ub {ub}] block n={n} rep {r}: abort feeders + idle", 8.0))
        if with_n0 and ub == BASE_UB:
            steps.append(("[ub 512] n=0 control block (1 rep, no feeders)", 10.0 + sum(est_request_s(L, 1) * 0.8 + 4 for L in LENGTHS)))
        steps.append((f"[ub {ub}] stop by PID + verify dead", STOP_S))
    return sum(sec for _, sec in steps), steps


# ------------------------------------------------------------------ probe
class Probe:
    def __init__(self, args) -> None:
        self.args = args
        self.out = Path(args.out)
        self.rows = K.Jsonl(self.out / "pb_qcost_1.jsonl")
        self.srv = None
        self.client = None
        self.log = None
        self.deadline_hit = threading.Event()
        self.stop_lock = threading.Lock()
        self.trunk = self.decoders = None
        self.ub = BASE_UB
        self.phase_id = None
        self.servers: list[dict] = []     # one record per ub phase (server label, readback, linkage, vram, stop, graphs proof)
        self._load_prompts()

    def _load_prompts(self) -> None:
        trunk, dec = None, []
        for ln in Path(self.args.prompt_set).read_text().splitlines():
            r = json.loads(ln)
            if r.get("kind") == "trunk" and r["group"] == TRUNK_GROUP:
                trunk = r["ids"]
            elif r.get("kind") == "decoder":
                dec.append(r["ids"])
        if not trunk or len(trunk) < max(LENGTHS) or not dec:
            raise SystemExit(f"prompt set lacks group {TRUNK_GROUP} trunk >= {max(LENGTHS)} or decoder prompts")
        self.trunk, self.decoders = trunk, dec

    def stop_server(self) -> None:
        with self.stop_lock:
            srv = self.srv
            if srv is None or getattr(srv, "_stopped", False):
                return
            srv._stopped = True
            try:
                srv.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[pbq] stop failed: {exc}", flush=True)
                self.rows.add({"kind": "stop_error", "err": str(exc)})
            if srv.stop_rec:
                self.rows.add({"kind": "server_stop", **srv.stop_rec})

    def watchdog(self) -> None:
        while time.time() < self.args.deadline_epoch:
            if self.deadline_hit.wait(5):
                return
        print("[pbq] DEADLINE reached: stopping server by PID", flush=True)
        self.deadline_hit.set()
        self.stop_server()

    def check(self) -> None:
        if self.deadline_hit.is_set() or time.time() >= self.args.deadline_epoch:
            raise K.Abort("deadline reached")
        if self.srv and self.srv.sampler and self.srv.sampler.abort.is_set():
            raise K.Abort(f"VRAM: {self.srv.sampler.abort_reason}")
        if self.srv and self.srv.proc and self.srv.proc.poll() is not None:
            raise K.Abort(f"server died rc={self.srv.proc.returncode}")
        w = K.window_status()
        if w.get("preempt_requested_at"):
            raise K.Abort(f"GPU window preempt requested at {w['preempt_requested_at']}")

    def start_server(self, ub: int = BASE_UB) -> None:
        self.check()
        argv, _ = K.parse_argv_file(Path(self.args.argv_file))
        argv = K.set_flag(argv, ("-np", "--parallel"), str(NP))
        argv = K.set_flag(argv, ("-b", "--batch-size"), str(ub))      # -b = -ub: q per iteration = ub - decode rows
        argv = K.set_flag(argv, ("-ub", "--ubatch-size"), str(ub))
        ctx = K.Ctx(self.args, None, argv, self.args.launch_prefix.split(), {"source": str(self.args.argv_file)})
        ctx.out = self.out
        self.srv = K.Server(ctx, f"pbq1-ub{ub}-{K.utc_tag()}", Path(self.args.champion_bin), None)
        self.ub = ub
        self.srv.start()
        self.client, self.log = ctx.client, H.LogTail(self.srv.log)
        ok, why = H.port_owner_ok(self.srv.proc.pid, self.args.port)
        self.rows.add({"kind": "port_ownership", "ub": ub, "ok": ok, "detail": why, "pid": self.srv.proc.pid, "argv": self.srv.argv})
        if not ok:
            raise K.Abort(f"port {self.args.port} not owned by our pid {self.srv.proc.pid}: {why}")
        self.servers.append({"ub": ub, "srv": self.srv, "proof": self.prove(ub)})

    def prove(self, ub: int) -> dict:
        """Own-pid proof from /proc and KFD (never from startup-log text): HIP graphs ON/OFF, HSA knob, GPU residency."""
        pid = self.srv.proc.pid
        env = [x for x in Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace").split("\0") if x]
        gv = [x for x in env if x.startswith("GGML_CUDA_DISABLE_GRAPHS")]
        hv = [x for x in env if x.startswith("HSA_OVERRIDE_CPU_AFFINITY_DEBUG=")]
        vram = 0
        for f in Path(f"/sys/class/kfd/kfd/proc/{pid}").glob("vram_*"):
            try:
                vram += int(f.read_text().strip() or 0)
            except (OSError, ValueError):
                pass
        rb = self.srv.readback or {}
        proof = {"ub": ub, "server_pid": pid, "graphs": "OFF" if gv else "ON", "GGML_CUDA_DISABLE_GRAPHS_in_environ": gv or None,
                 "HSA_OVERRIDE_CPU_AFFINITY_DEBUG_in_environ": hv or None, "own_pid_kfd_vram_mib_at_ready": round(vram / 1048576, 1),
                 "exe": rb.get("exe"), "exe_matches_binary": rb.get("exe_matches_binary"),
                 "proc_cmdline": [x for x in Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if x],
                 "source": "/proc/<pid>/environ + /sys/class/kfd/kfd/proc/<pid>/vram_*", "time": K.now_iso()}
        K.write_json(self.out / f"graphs_proof_ub{ub}.json", proof)
        self.rows.add({"kind": "proof", **{k: v for k, v in proof.items() if k != "proc_cmdline"}})
        print(f"[pbq] ub {ub} proof: pid {pid} graphs {proof['graphs']} HSA {hv or 'ABSENT'} own-pid VRAM {proof['own_pid_kfd_vram_mib_at_ready']} MiB", flush=True)
        if gv:
            raise K.Abort(f"HIP graphs OFF in the server environ ({gv}); production condition is graphs ON")
        if vram <= 0:
            raise K.Abort(f"own pid {pid} holds 0 KFD VRAM at ready: GPU residency NOT proven")
        if self.args.require_hsa_knob and not hv:
            raise K.Abort("HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 absent from the server environ")
        return proof

    def wait_idle(self, tmo: float = 40.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < tmo:
            if not any(x.get("is_processing") for x in self.client.slots()):
                return True
            time.sleep(0.5)
        return False

    def erase_all(self) -> None:
        self.wait_idle()
        for s in range(NP):
            self.client.erase(s)
        busy = [x.get("id") for x in self.client.slots() if x.get("is_processing")]
        if busy:
            raise K.Abort(f"slots still busy after erase: {busy}")

    def start_feeders(self, n: int):
        stop = threading.Event()
        state = []
        for i in range(n):
            ids = self.decoders[i % len(self.decoders)]
            ids = ids[(i * 97) % len(ids):] + ids[:(i * 97) % len(ids)]    # distinct rotation per feeder
            body = K.body_for(ids, FEED_NPRED, 1 + i, fork=None, cache=False, n_probs=0)
            st = {"slot": 1 + i, "res": {}, "times": [], "n": 0}

            def _cb(cnt, st=st):
                st["n"] = cnt
                st["times"].append((time.time(), cnt))
            th = threading.Thread(target=lambda b=body, s=st: s["res"].update(self.client.stream(b, stop=stop, on_token=_cb, timeout=3600)),
                                  daemon=True)
            st["th"] = th
            th.start()
            state.append(st)
        t0 = time.time()
        while time.time() - t0 < 120:
            if all(s["n"] >= 8 for s in state):
                break
            if any(not s["th"].is_alive() for s in state):
                break
            time.sleep(0.2)
        bad = [s["slot"] for s in state if s["n"] < 8]
        if bad:
            stop.set()
            raise K.Abort(f"feeder(s) {bad} never started decoding ({[s['res'].get('error') for s in state]})")
        time.sleep(3.0)
        return stop, state

    def stop_feeders(self, stop, state) -> None:
        stop.set()
        for s in state:
            s["th"].join(timeout=60)

    def run_prefill(self, n: int, rep: int, L: int, state) -> dict:
        self.check()
        need = est_request_s(L, n, self.ub) * 1.3 + 30
        if self.args.deadline_epoch - time.time() < need:
            rec = {"kind": "run", "ub": self.ub, "phase_id": self.phase_id, "n_gen": n, "rep": rep, "length": L,
                   "skipped": f"insufficient time before deadline (< {need:.0f} s left)"}
            self.rows.add(rec)
            print(f"[pbq] n={n} r{rep} {L}: SKIPPED ({rec['skipped']})", flush=True)
            return rec
        dead = [s["slot"] for s in state if not s["th"].is_alive()]
        if dead:
            raise K.Abort(f"feeder slot(s) {dead} stopped decoding before the prefill ({[s['res'].get('error') for s in state]})")
        # prefill slot only: erase slot 0 (feeders keep decoding)
        self.client.erase(PREFILL_SLOT)
        off = self.log.size()
        f0 = [s["n"] for s in state]
        body = K.body_for(self.trunk[:L], N_PRED_PREFILL, PREFILL_SLOT, fork=None, cache=False, n_probs=0)
        r = self.client.stream(body, timeout=1800)
        t0 = r["t_send"]
        t1 = t0 + (r.get("wall_s") or 0)
        t_first = t0 + (r.get("ttft_ms") or 0) / 1000.0
        iv = intervals_from_log(self.log.lines_from(off), PREFILL_SLOT, t0, t_first, self.ub)
        fdur = max(t_first - t0, 1e-6)
        ftok = sum(max(0, s["n"] - a) for s, a in zip(state, f0))
        rec = {"kind": "run", "ub": self.ub, "phase_id": self.phase_id, "n_gen": n, "decode_rows": 8 * n, "rep": rep, "length": L, "slot": PREFILL_SLOT,
               "http": r.get("http"), "ttft_ms": r.get("ttft_ms"), "error": r.get("error"), "t_send": t0, "t_first": t_first,
               "frozen_pred_ms": round(pred_request_ms(L), 1), "intervals": iv, "n_intervals": len(iv),
               "feeder_tokens_during": ftok, "feeder_tok_s": round(ftok / fdur, 2), "timings": r.get("timings")}
        self.rows.add(rec)
        mt = (rec["ttft_ms"] or 0) / L
        print(f"[pbq] ub={self.ub} n={n} r{rep} {L // 1000}k: TTFT {rec['ttft_ms']} ms = {mt:.3f} ms/tok (frozen {pred_request_ms(L) / L:.3f}); "
              f"{len(iv)} intervals; feeders {rec['feeder_tok_s']} tok/s", flush=True)
        if r.get("http") != 200 or not rec["ttft_ms"]:
            print(f"[pbq] WARNING: invalid request {r.get('http')} {r.get('error')}", flush=True)
        return rec

    def run_all(self) -> None:
        for ub in self.args.ub_list:
            self.check()
            need = phase_estimate_s(ub, self.args.with_n0) * 1.1
            if self.args.deadline_epoch - time.time() < need:
                raise K.Abort(f"phase ub={ub} needs ~{K.fmt_s(need)}, only {K.fmt_s(self.args.deadline_epoch - time.time())} left: not starting a phase that cannot finish")
            self.phase_id = f"ub{ub}-{K.utc_tag()}"
            self.start_server(ub)
            try:
                reps = reps_for(ub)
                for rep in range(1, max(reps.values()) + 1):
                    lens0 = [L for L in LENGTHS if reps[L] >= rep]
                    lens = lens0 if rep % 2 else lens0[::-1]
                    for n in NS:
                        self.check()
                        self.erase_all()
                        stop, state = self.start_feeders(n)
                        self.rows.add({"kind": "block", "ub": ub, "n_gen": n, "rep": rep, "feeder_slots": [s_["slot"] for s_ in state], "t": time.time()})
                        try:
                            for L in lens:
                                self.run_prefill(n, rep, L, state)
                        finally:
                            self.stop_feeders(stop, state)
                if self.args.with_n0 and ub == BASE_UB:
                    self.check()
                    self.erase_all()
                    stop, state = self.start_feeders(0)
                    self.rows.add({"kind": "block", "ub": ub, "n_gen": 0, "rep": 1, "feeder_slots": [], "t": time.time()})
                    try:
                        for L in LENGTHS:
                            self.run_prefill(0, 1, L, state)
                    finally:
                        self.stop_feeders(stop, state)
                self.rows.add({"kind": "phase_done", "ub": ub, "phase_id": self.phase_id, "t": time.time()})
            finally:
                self.stop_server()      # one server per ub; stopped by PID and verified dead before the next phase
                self.srv = None         # (its record stays in self.servers for the summary)

    def summarize(self, status: str) -> dict:
        f = self.out / "pb_qcost_1.jsonl"
        allrows = [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []
        rows = usable(allrows)
        a = analyse(rows)
        s = {"schema": SCHEMA, "status": status, "overall": a["overall"], "pass_rule": __doc__.split("PRE-DECLARED PASS RULE")[1].split("--dry-run")[0].strip(),
             "train_lengths": TRAIN, "heldout_lengths": HELD, "reps_required": REPS, "np": NP, "analysis": a,
             "skipped_runs": [r for r in rows if r.get("skipped")], "phases_done": [r["ub"] for r in allrows if r.get("kind") == "phase_done"], "binary": str(self.args.champion_bin),
             "binary_sha256": K.sha256_file(Path(self.args.champion_bin)),
             "ubs": list(self.args.ub_list), "analysis_q": analyse_q(rows),
             "servers": [{"ub": x["ub"], "label": x["srv"].label, "readback": x["srv"].readback, "linkage": x["srv"].linkage,
                          "vram": x["srv"].sampler.summary() if x["srv"].sampler else None, "stop": x["srv"].stop_rec,
                          "proof": x["proof"]} for x in self.servers], "finished": K.now_iso()}
        K.write_json(self.out / "summary.json", s)
        md = [f"# PB-QCOST-1 re-validation ({s['finished']})", "", f"status: {status}  overall: **{a['overall']}**", "",
              "PREDECLARED RULE (operator-accepted 10% held-out MAPE bar): " + s["pass_rule"], "",
              f"Shape departure: -np {NP} (production :8083 is -np 4): 4 feeders + 1 prefill slot are needed for n=4 generating slots. Other flags = gate argv.", ""]
        for n in NS:
            p = a["per_n"][n]
            rl, il = p["request_level"], p["interval_level"]
            md += [f"## n = {n} generating slot(s) (decode_rows {p['decode_rows']}): {p['verdict']}", "",
                   f"feeder aggregate tok/s during prefill (median): {p['feeder_tok_s_median']}",
                   f"- request level held-out {HELD}: frozen MAPE {rl['frozen_heldout'].get('mape_pct')}% rmse {rl['frozen_heldout'].get('rmse')} "
                   f"ms/tok P99|r| {rl['frozen_heldout'].get('p99_abs')} ms/tok; refit {rl['refit_ms_per_tok']} MAPE {rl['refit_heldout'].get('mape_pct')}%",
                   f"- per-iteration intercept at c=0, q={IT_Q}: {il['refit_intercept_ms_at_c0']} ms/iter (offline fit at q=512: {il['offline_324_fit_intercept_ms_at_q512']}); "
                   f"noise floor {il['noise_floor_train_rmse_ms_per_iter']} ms/iter ({il['noise_floor_n_intervals']} intervals)",
                   f"- held-out interval residual, UNIT ms/iter: frozen {il['frozen_model_heldout']}",
                   f"  refit {il['refit_model_heldout']}",
                   f"  offline-324 fit {il['offline_324_fit_heldout']}", "",
                   "| L | reps | measured ms/tok (median) | frozen ms/tok |", "|---|---|---|---|"]
            md += [f"| {L} | {v['reps_valid']} | {v['meas_ms_per_tok_median']} | {v['frozen_pred_ms_per_tok']} |"
                   for L, v in p["per_length"].items()]
            md.append("")
        c0 = [r for r in rows if r.get("kind") == "run" and r.get("n_gen") == 0 and r.get("ttft_ms") and not r.get("skipped")]
        if c0:
            md += ["## n = 0 control (idle prefill, 1 rep, not in the verdict)", "", "| L | measured ms/tok | frozen ms/tok |", "|---|---|---|"]
            md += [f"| {r['length']} | {r['ttft_ms'] / r['length']:.4f} | {pred_request_ms(r['length']) / r['length']:.4f} |" for r in c0]
            md.append("")
        if s["skipped_runs"]:
            md.append("SKIPPED: " + "; ".join(f"n={r['n_gen']} r{r['rep']} {r['length']}" for r in s["skipped_runs"]))
        aq = s["analysis_q"]
        md += ["", "## q axis (pooled t_ms = a + b*q + d*q*c, per n; unit ms per server iteration)", "", f"overall (q-fit held-out MAPE <= {PASS_MAPE}%): **{aq['overall']}**", ""]
        for n in NS:
            r = aq["per_n"][n]
            md += [f"- n={n}: {r['verdict']}; q identified {r['q_identified']}; levels {r['q_levels_by_ub']}; coef {r['coef']}; "
                   f"ci95 b {r.get('ci95_b')} d {r.get('ci95_d')}; noise rmse {r.get('noise_floor_train_rmse_ms_per_iter')} ms/iter; "
                   f"held-out {r.get('heldout_interval')}; by ub {r.get('heldout_by_ub')}; b vs 0.9865 {r.get('b_vs_assumed_0.9865')}; "
                   f"d vs 21:29Z {r.get('d_vs_2026_10_21_refit')}"]
        md += ["", "The mechanical verdict is verdict_q.py (PREDECLARATION.md); this driver summary is not the verdict."]
        (self.out / "summary.md").write_text("\n".join(md) + "\n")
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deadline", required=True, help="hard stop, e.g. 2026-10-22T02:50:00Z or HH:MM (UTC)")
    ap.add_argument("--champion-bin", default=str(CHAMPION))
    ap.add_argument("--argv-file", default=str(ARGV_FILE))
    ap.add_argument("--prompt-set", default=str(PROMPT_SET))
    ap.add_argument("--launch-prefix", default="env HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 " + K.DEFAULT_PREFIX)   # GPS-PIN-1 knob rides in the prefix
    ap.add_argument("--ubs", default=",".join(str(u) for u in UBS), help="comma list of -ub values, one server each, in order (first must be 512)")
    ap.add_argument("--with-n0", action="store_true", help="add the n=0 control block to the ub=512 phase (off by default)")
    ap.add_argument("--skip-done", action="store_true", help="resume: drop ub phases already recorded as phase_done in --out/pb_qcost_1.jsonl")
    ap.add_argument("--require-hsa-knob", action="store_true", default=True)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu-window-id", default=os.environ.get("PBQ_GPU_WINDOW_ID"))
    ap.add_argument("--vram-ceiling-gib", type=float, default=VRAM_GUARD_GIB)
    ap.add_argument("--vram-free-max-gib", type=float, default=2.0)
    ap.add_argument("--load-timeout-s", type=float, default=600)
    ap.add_argument("--budget-min", type=float, default=45.0, help="plan target only (the hard stop is --deadline)")
    ap.add_argument("--card", default=str(K.DEFAULT_CARD))
    ap.add_argument("--no-device-claim", default=None, metavar="REASON")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--claim-wait-s", type=float, default=CLAIM_WAIT_S)
    ap.add_argument("--analyse-only", metavar="JSONL", default=None, help="print analyse() of an existing run file and exit")
    args = ap.parse_args()
    if args.analyse_only:
        rows = [json.loads(ln) for ln in Path(args.analyse_only).read_text().splitlines()]
        print(json.dumps(analyse(rows), indent=1))
        return 0
    args.ub_list = [int(x) for x in args.ubs.split(",") if x.strip()]
    if not args.ub_list or args.ub_list[0] != BASE_UB or len(set(args.ub_list)) != len(args.ub_list) or any(u % 8 or u < 128 for u in args.ub_list):
        raise SystemExit(f"--ubs must start with {BASE_UB}, be distinct, multiples of 8 and >= 128: {args.ubs}")
    args.deadline_epoch = H.parse_deadline(args.deadline)
    args.out_given = args.out
    args.out = args.out or str(HERE / "runs" / K.utc_tag())
    now = time.time()

    probs, notes = [], []
    cb = Path(args.champion_bin)
    if not (cb.is_file() and os.access(cb, os.X_OK)):
        probs.append(f"champion binary {cb} not an executable file")
    elif os.path.realpath(cb).startswith(str(K.FROZEN_TREE) + "/"):
        probs.append("champion binary resolves inside the frozen tree")
    if not (1024 <= args.port <= 65535) or 8000 <= args.port <= 8999:
        probs.append(f"--port {args.port} refused (8000-8999 are production ports)")
    if K.port_listening(args.port):
        probs.append(f"port {args.port} already listening")
    for f in (K.LINKAGE, Path(args.argv_file), Path(args.prompt_set)):
        if not f.exists():
            probs.append(f"missing {f}")
    if Path(args.prompt_set).exists():
        try:
            trunk = dec = 0
            for ln in Path(args.prompt_set).read_text().splitlines():
                r = json.loads(ln)
                if r.get("kind") == "trunk" and r["group"] == TRUNK_GROUP:
                    trunk = len(r["ids"])
                elif r.get("kind") == "decoder":
                    dec += 1
            if trunk < max(LENGTHS) or not dec:
                probs.append(f"prompt set: trunk group {TRUNK_GROUP} has {trunk} ids (< {max(LENGTHS)}) or no decoder prompts")
            notes.append(f"prompt set trunk g{TRUNK_GROUP} {trunk} ids, {dec} decoder prompts, sha256 {K.sha256_file(Path(args.prompt_set))}")
        except (ValueError, KeyError) as exc:
            probs.append(f"prompt set unreadable: {exc}")
    if Path(args.argv_file).exists():
        hard, drift = K.argv_problems(K.parse_argv_file(Path(args.argv_file))[0], True)
        probs += hard
        drift = [d for d in drift if not d.startswith("-np")]
        if drift:
            notes.append("argv shape drift vs gate: " + "; ".join(drift))
    if K.port_listening(8083):
        notes.append("production :8083 is LISTENING now (an execute run refuses until its owner stops it)")
    left = args.deadline_epoch - now
    if args.skip_done and args.out_given and (Path(args.out_given) / "pb_qcost_1.jsonl").exists():
        done_ubs = {r["ub"] for r in (json.loads(ln) for ln in (Path(args.out_given) / "pb_qcost_1.jsonl").read_text().splitlines())
                    if r.get("kind") == "phase_done"}
        args.ub_list = [u for u in args.ub_list if u not in done_ubs]
        notes.append(f"--skip-done: phases already complete {sorted(done_ubs)}; remaining {args.ub_list}")
    if not args.ub_list:
        print("nothing left to run: every ub phase is recorded phase_done")
        return 0
    total, steps = plan_estimate(args.ub_list, args.with_n0)
    notes.append(f"phases (one server each): {args.ub_list}; n=0 control {'INCLUDED' if args.with_n0 else 'off'}; estimate model = 2026-10-21 per-n refit + assumed b {B_OFF}")
    if args.vram_ceiling_gib > VRAM_GUARD_GIB:
        probs.append(f"--vram-ceiling-gib {args.vram_ceiling_gib} above the {VRAM_GUARD_GIB} GiB own-pid guard")
    first = phase_estimate_s(args.ub_list[0], args.with_n0) * 1.1
    if left < first:
        probs.append(f"only {K.fmt_s(left)} until the hard stop; the next phase (ub {args.ub_list[0]}) needs ~{K.fmt_s(first)} (x1.1 margin)")
    notes.append(f"per-phase estimate (s): " + ", ".join(f"ub{u}={phase_estimate_s(u, args.with_n0):.0f}" for u in args.ub_list)
                 + f"; all phases ~{K.fmt_s(total)}; a phase is started only if it fits before the hard stop (else the run ends cleanly, resume with --skip-done)")

    print("PB-QCOST-1 plan")
    print(f"  hard stop {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(args.deadline_epoch))} (in {K.fmt_s(left)}); out {args.out}")
    print(f"  binary {cb}\n  argv {args.argv_file} with -np {NP}; prefix: {args.launch_prefix}; port {args.port}")
    print(f"  model under test: ms/tok = ({K.PF_BASE_MS} + {K.PF_PER_K_MS} * occupied_k) * {K.B512_FACTOR}; offline fit being re-validated: "
          f"t = {OFFLINE[0]} + {OFFLINE[1]} q + {OFFLINE[2]} q c ms (gen = 3)")
    print(f"  ub phases {args.ub_list} (-b = -ub); reps/length: ub 512 {reps_for(512)}, other ub {Q_REPS}; own-pid VRAM guard {args.vram_ceiling_gib} GiB; HSA knob required in server environ")
    print(f"  lengths {LENGTHS}; TRAIN {TRAIN} (per-n refit); HELD-OUT {HELD}; n in {NS}; reps {REPS}; slot {PREFILL_SLOT} prefill, "
          f"feeders slots 1..n decoding (DFlash2, n_predict {FEED_NPRED}); blocks interleaved n=1,n=4 per rep")
    print("  rule: " + " ".join(__doc__.split("PRE-DECLARED PASS RULE")[1].split("--dry-run")[0].split()))
    for d, sec in steps:
        print(f"   - {d}: ~{K.fmt_s(sec)}")
    print(f"  ESTIMATE (all remaining phases) total ~{K.fmt_s(total)} ({total / 60:.1f} min) against {K.fmt_s(left)} to the hard stop")
    for nn in notes:
        print("  note:", nn)
    for p in probs:
        print("  PROBLEM:", p)
    if args.dry_run:
        print("PREFLIGHT:", "FAIL" if probs else "PASS")
        print("DRY RUN complete: no process was started, nothing was sent, no claim taken.")
        return 0
    if probs:
        print("REFUSED (preflight).")
        return 2
    if not args.gpu_window_id:
        print("REFUSED: no --gpu-window-id / PBQ_GPU_WINDOW_ID")
        return 64

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    probe = Probe(args)
    stack, rc, status = [], 0, "aborted-before-status"

    def on_sig(signum, _f):
        probe.deadline_hit.set()
        probe.stop_server()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    threading.Thread(target=probe.watchdog, daemon=True).start()
    try:
        if args.no_device_claim is None:
            sys.path.insert(0, str(K.CLAIM_MOD_DIR))
            from autokernel.resource import device_claim as dc  # noqa: E402
            journal = dc.ClaimJournal(out / "claim" / "device-claim-journal.jsonl")
            try:
                cm = dc.acquire_device_claim("mi210_0", purpose=f"PB-QCOST-1 revalidation on :{args.port}", campaign_id="PB-QCOST-1",
                                             journal=journal, holder_label="pbqcost1", timeout_s=args.claim_wait_s, max_hold_s=max(600, left + 600))
            except dc.DeviceClaimError as exc:
                print(f"[pbq] REFUSED: mi210_0 device claim not acquired: {exc}", flush=True)
                return 75
            claim = cm.__enter__()
            stack.append(cm)
            left2 = args.deadline_epoch - time.time()
            if left2 < first:
                print(f"[pbq] REFUSED: after the claim wait only {K.fmt_s(left2)} remain, the next phase needs ~{K.fmt_s(first)}", flush=True)
                return 75
            K.write_json(out / "claim" / "device-claim-receipt.json", claim.receipt().to_dict())
        K.write_json(out / "run_meta.json", {"args": vars(args), "started": K.now_iso(), "estimate_s": total, "notes": notes,
                                              "gpu_window_id": args.gpu_window_id,
                                              "device_claim": args.no_device_claim or "mi210_0 acquired"})
        probe.run_all()
        status = "complete"
    except K.Abort as exc:
        print(f"[pbq] ABORT: {exc}", flush=True)
        status, rc = f"abort: {exc}", 3
    except SystemExit:
        status, rc = "signalled", 5
    except Exception as exc:  # noqa: BLE001
        print(f"[pbq] ERROR: {type(exc).__name__}: {exc}", flush=True)
        status, rc = f"error: {type(exc).__name__}: {exc}", 4
    finally:
        probe.deadline_hit.set()
        probe.stop_server()
        try:
            s = probe.summarize(status)
            print(f"[pbq] overall {s['overall']}; {out}/summary.md")
        except Exception as exc:  # noqa: BLE001
            print(f"[pbq] summary failed: {exc}", flush=True)
        for cm in reversed(stack):
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                print(f"[pbq] claim release failed: {exc}", flush=True)
        dead = all((x["srv"].stop_rec.get("verified_dead") if x["srv"].stop_rec else x["srv"].proc is None) for x in probe.servers) \
            and (probe.srv is None or bool(probe.srv.stop_rec and probe.srv.stop_rec.get("verified_dead")))
        print(f"[pbq] server verified dead: {dead}")
        if not dead:
            rc = rc or 6
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
