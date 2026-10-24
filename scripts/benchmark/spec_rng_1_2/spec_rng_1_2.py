#!/usr/bin/env python3
"""SPEC-RNG.1.2 runner: is the position-keyed counter RNG (poskey) worth carrying into a champion fold?

Handoff: handoffs/active/speculative-decoding-mtp-refresh.md, SPEC-RNG.1.2 (parents SPEC-RNG.1, SPEC-RNG.1.1).
Task text: "decide whether poskey is worth carrying into a champion fold, given its effect is limited to the DFlash2 rejection
path: measure DFlash2 depth-invariance of the token *distribution* (not samples)."

What the runner does (rule written BEFORE any run in the launcher dir's PREDECLARATION.md; this file implements exactly it):

  MAIN phase   one server (the :8083 shape, DFlash2 drafter, np4, temperature > 0). Six arms = RNG {stateful, poskey} x draft depth
               {0, 3, 7}; depth and RNG are per-request fields ("speculative.n_max", "rng_poskey"), so every arm shares one load and
               one shape. For each of 3 fixed prompts and each arm, N seeds (DISJOINT per arm, so the samples are independent) each
               generate K tokens. The marginal token distribution at generation index j = 1..K-1 (j = 0 never goes through the
               rejection path: null control) is compared between arms by chi-square homogeneity.
  PROBE phase  secondary, the "is there a benefit" half: the same seeded request at depth 7 issued alone vs inside a batch of 4,
               stateful vs poskey, paired by seed. Counts identical outputs.

stdlib only (runs under /usr/bin/python3 and under the orchestrator venv). Subcommands:
  drive   --port P --out DIR --deadline EPOCH [--dry]     run both phases, append DIR/records.jsonl, resumable
  analyze --records F --out VERDICT.json [--predeclaration P] [--summary VERDICT.json] [--beliefs F | --no-beliefs]
                                                           apply the predeclared rule; the verdict carries the rule sha256
                                                           (PREDECLARATION.md) and an ISO-8601 UTC analyzed_at; --summary writes
                                                           the launcher's top-level VERDICT.json with the same fields; a
                                                           self-hashed belief_measurements.jsonl is written beside --out
  selftest                                                 analyzer on synthetic data: a null passes, an injected bias fails
  plan                                                     print the plan (arms, N, request bodies) and exit
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# ----------------------------------------------------------------------------------------------------------------- plan
K_TOKENS = 12                 # generated tokens per request; indices 1..K-1 are tested (index 0 is the null control)
SEEDS_PER_BLOCK = 50          # per (prompt, arm); a block = SEEDS_PER_BLOCK x 3 prompts x 6 arms = 900 requests
MAX_BLOCKS = 12               # 600 seeds per (prompt, arm) at most
MIN_N_CELL = 300              # validity: every (prompt, arm) cell needs at least this many ok requests
DEPTHS = (0, 3, 7)
RNGS = ("stateful", "poskey")
ARMS = [(("s" if r == "stateful" else "p") + str(d), r, d) for r in RNGS for d in DEPTHS]   # s0 s3 s7 p0 p3 p7
PROMPTS = [
    "The old lighthouse keeper looked out over the storm and said:",
    "Q: Name something you would find in a market in a coastal town.\nA:",
    "Once the engineers had finished the survey, the first thing they noticed was",
]
TEMPERATURE, TOP_K, TOP_P, MIN_P = 1.0, 40, 1.0, 0.0
PROBE_SEEDS = 48
PROBE_DEPTH = 7
PROBE_TOKENS = 24
SEED_BASE_MAIN = 1_000_000    # seed = base + arm_index * 100_000 + prompt_index * 10_000 + k  (disjoint per arm and prompt)
SEED_BASE_PROBE = 5_000_000
SEED_BASE_FILL = 7_000_000

# predeclared thresholds (PREDECLARATION.md)
ALPHA = 0.01                  # family-wise per comparison (Bonferroni over prompts x positions)
ALPHA_ACC = 0.01              # acceptance-rate Welch test, per depth
MIN_DRAFTED_FRAC = 0.95       # engagement: share of depth>0 requests that drafted at all
MIN_REJECT_FRAC = 0.05        # engagement: rejected share of drafted tokens (the rejection path must be exercised)
POOL_MIN = 10                 # chi-square: token ids with union count < POOL_MIN are pooled into one bin
BENEFIT_MIN_GAIN = 6          # probe: poskey-identical-and-stateful-not minus the reverse, of PROBE_SEEDS
BENEFIT_ALPHA = 0.05          # probe: one-sided exact sign test on the discordant pairs

# comparison families: (name, arm A, arm B, role)
COMPARISONS = [
    ("P3", "p3", "p0", "primary"), ("P7", "p7", "p0", "primary"),          # poskey depth-invariance
    ("C3", "s3", "s0", "control"), ("C7", "s7", "s0", "control"),          # stateful depth-invariance (theorem-guaranteed)
    ("X0", "p0", "s0", "control"),                                          # RNG swap at depth 0
]


def seed_of(arm_idx, prompt_idx, k):
    return SEED_BASE_MAIN + arm_idx * 100_000 + prompt_idx * 10_000 + k


def main_body(prompt_idx, arm, seed):
    name, rng, depth = arm
    return {"prompt": PROMPTS[prompt_idx], "n_predict": K_TOKENS, "temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P,
            "min_p": MIN_P, "seed": seed, "ignore_eos": True, "cache_prompt": True, "return_tokens": True, "stream": False,
            "speculative.n_max": depth, "rng_poskey": rng == "poskey"}


def probe_body(prompt_idx, rng, seed, n=PROBE_TOKENS, depth=PROBE_DEPTH):
    b = main_body(prompt_idx, ("probe", rng, depth), seed)
    b["n_predict"] = n
    return b


def schedule(blocks=MAX_BLOCKS):
    """Yield (block, [(key, body)...]) in round-robin order; key identifies a request for resume."""
    for blk in range(blocks):
        reqs = []
        for k in range(blk * SEEDS_PER_BLOCK, (blk + 1) * SEEDS_PER_BLOCK):
            for ai, arm in enumerate(ARMS):
                for pi in range(len(PROMPTS)):
                    s = seed_of(ai, pi, k)
                    reqs.append((f"main/{arm[0]}/{pi}/{s}", {"phase": "main", "block": blk, "arm": arm[0], "prompt": pi, "seed": s},
                                 main_body(pi, arm, s)))
        yield blk, reqs


# ------------------------------------------------------------------------------------------------------------------ HTTP
def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def one(port, meta, body, timeout):
    t0 = time.time()
    try:
        r = post(port, body, timeout)
        t = r.get("timings", {})
        toks = r.get("tokens")
        if not isinstance(toks, list):
            return dict(meta, ok=False, t0=t0, t1=time.time(), error="no tokens list in response")
        return dict(meta, ok=True, t0=t0, t1=time.time(), tokens=toks, draft_n=int(t.get("draft_n", 0) or 0),
                    draft_n_accepted=int(t.get("draft_n_accepted", 0) or 0), n_predicted=t.get("predicted_n"))
    except Exception as e:  # recorded, never raised: the analyzer decides validity
        return dict(meta, ok=False, t0=t0, t1=time.time(), error=repr(e)[:300])


ABORT_AFTER = 8               # consecutive failed requests => the server is gone (VRAM guard, crash): stop, exit rc 75, the job restarts it
EXIT_SERVER_GONE = 75


class Sink:
    def __init__(self, path):
        self.path, self.lock, self.done = path, threading.Lock(), set()
        self.consec_fail = 0
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        self.done.add(r["key"])
                except ValueError:
                    pass

    def put(self, rec):
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            if rec.get("ok"):
                self.done.add(rec["key"])
                self.consec_fail = 0
            else:
                self.consec_fail += 1

    @property
    def server_gone(self):
        return self.consec_fail >= ABORT_AFTER


def run_batch(port, sink, items, deadline, conc=4):
    """items: [(key, meta, body)]; skips done keys; stops submitting past the deadline. Returns the number submitted."""
    todo = [it for it in items if it[0] not in sink.done]
    n = 0

    def work(it):
        key, meta, body = it
        if time.time() > deadline or sink.server_gone:
            return
        sink.put(dict(one(port, meta, body, min(300, max(30, deadline - time.time()))), key=key))

    with ThreadPoolExecutor(max_workers=conc) as ex:
        for _ in ex.map(work, todo):
            n += 1
    return n


def run_probe(port, sink, deadline):
    """Seeded request alone (idle server) vs inside a batch of 4 (3 distinct fillers), both RNG modes, paired by seed."""
    for i in range(PROBE_SEEDS):
        if time.time() > deadline or sink.server_gone:
            return
        s = SEED_BASE_PROBE + i
        for rng in RNGS:
            ka, kb = f"probe/{rng}/alone/{s}", f"probe/{rng}/batched/{s}"
            if ka not in sink.done:
                sink.put(dict(one(port, {"phase": "probe", "rng": rng, "cond": "alone", "seed": s}, probe_body(0, rng, s), 120), key=ka))
            if kb not in sink.done:
                items = [(kb, {"phase": "probe", "rng": rng, "cond": "batched", "seed": s}, probe_body(0, rng, s))]
                for f in range(3):
                    fs = SEED_BASE_FILL + i * 10 + f
                    items.append((f"fill/{rng}/{s}/{f}", {"phase": "fill", "rng": rng, "seed": fs}, probe_body(1 + f % 2, rng, fs)))
                # the 4 requests are issued together; fillers are recorded but never analysed
                res = [None] * 4
                th = [threading.Thread(target=lambda j=j: res.__setitem__(j, one(port, items[j][1], items[j][2], 120))) for j in range(4)]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                for j in range(4):
                    sink.put(dict(res[j], key=items[j][0]))


def drive(port, out, deadline, dry=False):
    os.makedirs(out, exist_ok=True)
    if dry:
        n = sum(len(r) for _, r in schedule())
        print(f"DRY plan: probe {PROBE_SEEDS} seeds x 2 RNG modes x (alone + batch of 4) at depth {PROBE_DEPTH}, {PROBE_TOKENS} tokens;")
        print(f"DRY plan: main {MAX_BLOCKS} blocks x {SEEDS_PER_BLOCK} seeds x {len(PROMPTS)} prompts x {len(ARMS)} arms = {n} requests of {K_TOKENS} tokens")
        print("DRY arms: " + " ".join(f"{a[0]}=({a[1]},depth {a[2]})" for a in ARMS))
        print("DRY request (p7, prompt 0): " + json.dumps(main_body(0, ARMS[5], seed_of(5, 0, 0)))[:420])
        print(f"DRY seeds: main base {SEED_BASE_MAIN} + arm*100000 + prompt*10000 + k (disjoint per arm); probe {SEED_BASE_PROBE}.., filler {SEED_BASE_FILL}..")
        return 0
    sink = Sink(os.path.join(out, "records.jsonl"))
    t_probe = time.time()
    run_probe(port, sink, deadline)
    print(f"probe done {time.time() - t_probe:.0f}s", flush=True)
    if sink.server_gone:
        print("SERVER GONE during the probe", flush=True)
        return EXIT_SERVER_GONE
    for blk, reqs in schedule():
        # one block ~2-3 min; do not start a block that cannot finish
        if deadline - time.time() < 240:
            print(f"STOP before block {blk}: {deadline - time.time():.0f}s left", flush=True)
            break
        t0 = time.time()
        run_batch(port, sink, [(k, m, b) for k, m, b in reqs], deadline)
        print(f"block {blk} done {time.time() - t0:.0f}s ok-total={len(sink.done)}", flush=True)
        if sink.server_gone:
            print(f"SERVER GONE in block {blk}", flush=True)
            return EXIT_SERVER_GONE
    print(f"COMPLETE ok-total={len(sink.done)}" if all(k in sink.done for _, r in schedule() for k, _, _ in r) else "CUT at the deadline", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------- statistics
def _gammainc_q(a, x):
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes gser/gcf)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(stat, dof):
    return 1.0 if dof <= 0 else _gammainc_q(dof / 2.0, stat / 2.0)


def homogeneity(counts_a, counts_b, pool_min=POOL_MIN):
    """Two-sample chi-square homogeneity on token-id counts (dicts). Ids with union count < pool_min pool into one bin.
    Returns (stat, dof, p, tv) with tv the total-variation distance of the empirical distributions."""
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    if na == 0 or nb == 0:
        return 0.0, 0, 1.0, 0.0
    ids = set(counts_a) | set(counts_b)
    bins, other_a, other_b = [], 0, 0
    for t in ids:
        ca, cb = counts_a.get(t, 0), counts_b.get(t, 0)
        if ca + cb < pool_min:
            other_a += ca
            other_b += cb
        else:
            bins.append((ca, cb))
    if other_a + other_b > 0:
        bins.append((other_a, other_b))
    tv = 0.5 * sum(abs(counts_a.get(t, 0) / na - counts_b.get(t, 0) / nb) for t in ids)
    if len(bins) < 2:
        return 0.0, 0, 1.0, tv
    n = na + nb
    stat = 0.0
    for ca, cb in bins:
        tot = ca + cb
        ea, eb = tot * na / n, tot * nb / n
        stat += (ca - ea) ** 2 / ea + (cb - eb) ** 2 / eb
    dof = len(bins) - 1
    return stat, dof, chi2_sf(stat, dof), tv


def welch_p(xs, ys):
    """Two-sided Welch z-test p-value (normal approximation; n is in the hundreds)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 1.0
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return 1.0 if mx == my else 0.0
    return math.erfc(abs(mx - my) / se / math.sqrt(2.0))


def sign_test_one_sided(k, n):
    """P(X >= k), X ~ Binomial(n, 1/2). n = discordant pairs."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2.0 ** n


# ------------------------------------------------------------------------------------------------------------------ analysis
def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def analyze(records, k_tokens=K_TOKENS, alpha=ALPHA):
    main = [r for r in records if r.get("phase") == "main"]
    cells = {}                                     # (arm, prompt) -> list of ok records
    n_all = {}
    for r in main:
        key = (r["arm"], r["prompt"])
        n_all[key] = n_all.get(key, 0) + 1
        if r.get("ok") and len(r["tokens"]) == k_tokens:
            cells.setdefault(key, []).append(r)
    arms = [a[0] for a in ARMS]
    n_cell = {f"{a}/{p}": len(cells.get((a, p), [])) for a in arms for p in range(len(PROMPTS))}
    res = {"n_cell": n_cell, "gates": {}, "comparisons": {}, "acceptance": {}, "probe": {}}
    gates = res["gates"]

    # V1: every cell has enough ok requests; transport/length failures below 1%
    gates["V1_cells_n_ge_min"] = all(v >= MIN_N_CELL for v in n_cell.values())
    tot_all, tot_ok = sum(n_all.values()), sum(len(v) for v in cells.values())
    gates["V1b_ok_fraction_ge_99"] = tot_all > 0 and tot_ok / tot_all >= 0.99
    # V2: engagement of the rejection path at depth > 0 (both RNG modes)
    eng = {}
    ok_eng = True
    for a in arms:
        d = int(a[1:])
        recs = [r for p in range(len(PROMPTS)) for r in cells.get((a, p), [])]
        if d == 0:
            eng[a] = {"draft_n_total": sum(r["draft_n"] for r in recs), "n": len(recs)}
            ok_eng = ok_eng and eng[a]["draft_n_total"] == 0
            continue
        drafted = [r for r in recs if r["draft_n"] > 0]
        dn, da = sum(r["draft_n"] for r in recs), sum(r["draft_n_accepted"] for r in recs)
        eng[a] = {"n": len(recs), "drafted_frac": len(drafted) / len(recs) if recs else 0.0, "draft_n": dn, "accepted": da,
                  "reject_frac": (1 - da / dn) if dn else 0.0}
        ok_eng = ok_eng and eng[a]["drafted_frac"] >= MIN_DRAFTED_FRAC and eng[a]["reject_frac"] >= MIN_REJECT_FRAC
    res["engagement"] = eng
    gates["V2_rejection_path_engaged"] = ok_eng

    # distribution comparisons
    def counts(arm, p, j):
        c = {}
        for r in cells.get((arm, p), []):
            t = r["tokens"][j]
            c[t] = c.get(t, 0) + 1
        return c

    for name, A, B, role in COMPARISONS:
        tests, tvs = [], []
        for p in range(len(PROMPTS)):
            for j in range(1, k_tokens):
                stat, dof, pv, tv = homogeneity(counts(A, p, j), counts(B, p, j))
                tests.append({"prompt": p, "pos": j, "stat": stat, "dof": dof, "p": pv, "tv": tv})
                tvs.append(tv)
        m = len(tests)
        worst = min(tests, key=lambda t: t["p"]) if tests else None
        res["comparisons"][name] = {"A": A, "B": B, "role": role, "n_tests": m, "bonferroni_threshold": alpha / m if m else None,
                                    "min_p": worst["p"] if worst else None, "argmin": {k: worst[k] for k in ("prompt", "pos", "dof")} if worst else None,
                                    "mean_tv": sum(tvs) / len(tvs) if tvs else None,
                                    "pass": bool(worst and worst["p"] >= alpha / m)}
        # null control at index 0 (no rejection path for any arm)
        n0 = []
        for p in range(len(PROMPTS)):
            n0.append(homogeneity(counts(A, p, 0), counts(B, p, 0))[2])
        res["comparisons"][name]["pos0_null_min_p"] = min(n0) if n0 else None
        res["comparisons"][name]["pos0_null_pass"] = bool(n0 and min(n0) >= alpha / len(n0))

    # acceptance rate, poskey vs stateful at the same depth (per-request accepted/drafted fraction)
    acc_ok = True
    for d in (3, 7):
        def fr(arm):
            return [r["draft_n_accepted"] / r["draft_n"] for p in range(len(PROMPTS)) for r in cells.get((arm, p), []) if r["draft_n"] > 0]
        xs, ys = fr(f"p{d}"), fr(f"s{d}")
        pv = welch_p(xs, ys)
        res["acceptance"][f"depth{d}"] = {"poskey_mean": sum(xs) / len(xs) if xs else None, "stateful_mean": sum(ys) / len(ys) if ys else None,
                                          "n_poskey": len(xs), "n_stateful": len(ys), "p": pv, "pass": pv >= ALPHA_ACC}
        acc_ok = acc_ok and pv >= ALPHA_ACC
    gates["acceptance_unchanged"] = acc_ok

    controls_ok = all(c["pass"] and c["pos0_null_pass"] for c in res["comparisons"].values() if c["role"] == "control") and \
        all(c["pos0_null_pass"] for c in res["comparisons"].values() if c["role"] == "primary")
    gates["controls_pass"] = controls_ok
    primary_ok = all(c["pass"] for c in res["comparisons"].values() if c["role"] == "primary")
    gates["primary_poskey_invariant"] = primary_ok

    # probe (secondary): identical outputs, alone vs batched, per RNG mode, paired by seed
    pr = {}
    for r in records:
        if r.get("phase") == "probe" and r.get("ok"):
            pr.setdefault((r["rng"], r["seed"]), {})[r["cond"]] = r["tokens"]
    seeds = sorted({s for (_, s) in pr})
    ident = {"stateful": {}, "poskey": {}}
    for s in seeds:
        for rng in RNGS:
            d = pr.get((rng, s), {})
            if "alone" in d and "batched" in d:
                ident[rng][s] = d["alone"] == d["batched"]
    paired = [s for s in seeds if s in ident["stateful"] and s in ident["poskey"]]
    gain = sum(1 for s in paired if ident["poskey"][s] and not ident["stateful"][s])
    loss = sum(1 for s in paired if ident["stateful"][s] and not ident["poskey"][s])
    p_sign = sign_test_one_sided(gain, gain + loss)
    res["probe"] = {"paired_seeds": len(paired), "identical_poskey": sum(ident["poskey"][s] for s in paired),
                    "identical_stateful": sum(ident["stateful"][s] for s in paired), "poskey_only": gain, "stateful_only": loss,
                    "sign_p_one_sided": p_sign, "net_gain": gain - loss}
    gates["V3_probe_paired_ge_40"] = len(paired) >= 40
    benefit = (gain - loss) >= BENEFIT_MIN_GAIN and p_sign < BENEFIT_ALPHA

    valid = gates["V1_cells_n_ge_min"] and gates["V1b_ok_fraction_ge_99"] and gates["V2_rejection_path_engaged"] and controls_ok
    if not valid:
        verdict, reason = "INCONCLUSIVE", "validity gate failed (cells, ok-fraction, engagement or controls): " + \
            ", ".join(k for k in ("V1_cells_n_ge_min", "V1b_ok_fraction_ge_99", "V2_rejection_path_engaged", "controls_pass") if not gates[k])
    elif not (primary_ok and acc_ok):
        verdict, reason = "POSKEY-BIASED", "poskey changes the token distribution or the acceptance rate on the DFlash2 rejection path"
    elif not gates["V3_probe_paired_ge_40"]:
        verdict, reason = "INCONCLUSIVE", "distribution invariant, but the benefit probe has fewer than 40 paired seeds"
    elif benefit:
        verdict, reason = "CARRY", "distribution invariant, acceptance unchanged, and a demonstrated reproducibility gain"
    else:
        verdict, reason = "DO-NOT-CARRY", "distribution invariant but no demonstrated reproducibility gain: not worth a fold slot"
    res["verdict"], res["reason"] = verdict, reason
    return res


# ------------------------------------------------------------------------------------------------------- belief write side
# VB-SPEC-RNG-1-2-WIRE: at the end of each run (`analyze`), one self-hashed row per decided component, mirroring what the
# epyc-root strict reader `scripts/vidya/adapters/spec_rng_1_2.py` projects (it re-derives every number from records.jsonl and
# refuses a sidecar that differs). This decides nothing: `claim_tuple.grade()` in epyc-root grades; there is no ladder here.
BELIEF_SCHEMA = "epyc.spec_rng_1_2.belief_measurement.v1"
BELIEFS_NAME = "belief_measurements.jsonl"
PROJECTABLE = ("CARRY", "DO-NOT-CARRY", "POSKEY-BIASED")
COMPONENTS = [c[0] for c in COMPARISONS] + ["acceptance_depth3", "acceptance_depth7", "probe"]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_sha256(row):
    return hashlib.sha256(canonical({k: v for k, v in row.items() if k != "content_sha256"}).encode()).hexdigest()


def _g(x):
    return f"{x:.4g}"


def proposition(component, res, rule_sha):
    """What the predeclared rule decided for one component, naming the verdict word and the rule digest (the root reader
    builds the same sentence from its own re-derivation and refuses a row whose sentence differs)."""
    head = f"SPEC-RNG.1.2 rule PREDECLARATION.md sha256 {rule_sha}, run verdict {res['verdict']}: "
    if component.startswith("acceptance_depth"):
        a = res["acceptance"][component.split("_", 1)[1]]
        body = (f"at draft depth {component[-1]} the per-request acceptance fraction poskey vs stateful has Welch "
                f"p {_g(a['p'])} {'>=' if a['pass'] else '<'} {ALPHA_ACC} (acceptance "
                f"{'unchanged' if a['pass'] else 'changed'})")
    elif component == "probe":
        p = res["probe"]
        body = (f"the benefit probe has net_gain {p['net_gain']} of {p['paired_seeds']} paired seeds, one-sided sign "
                f"p {_g(p['sign_p_one_sided'])}; a gain needs net_gain >= {BENEFIT_MIN_GAIN} and p < {BENEFIT_ALPHA}")
    else:
        c = res["comparisons"][component]
        body = (f"{component} ({c['A']} vs {c['B']}, {c['role']}) has minimum chi-square p {_g(c['min_p'])} over "
                f"{c['n_tests']} tests {'>=' if c['pass'] else '<'} Bonferroni {ALPHA}/{c['n_tests']} "
                f"(token distribution {'invariant' if c['pass'] else 'differs'}), index-0 null "
                f"{'passes' if c['pos0_null_pass'] else 'fails'}")
    return head + body


def belief_rows(res, *, run_name, rule_sha, inputs, verdict_sha, analyzed_at, runner_sha):
    """Eight rows for a projectable verdict; [] for INCONCLUSIVE (void: a re-grant re-runs it, nothing is decided)."""
    if res["verdict"] not in PROJECTABLE:
        return []
    rows = []
    for comp in COMPONENTS:
        if comp.startswith("acceptance_depth"):
            a = res["acceptance"][comp.split("_", 1)[1]]
            spec = {"metric": "spec_rng_acceptance_fraction_poskey", "value": a["poskey_mean"], "unit": "fraction",
                    "reps": min(a["n_poskey"], a["n_stateful"]), "anchor_value": a["stateful_mean"],
                    "reps_basis": "scored:drafted requests per RNG arm at this depth, smaller arm", "category": "CANDIDATE",
                    "stats": a}
        elif comp == "probe":
            p = res["probe"]
            spec = {"metric": "spec_rng_probe_net_gain", "value": p["net_gain"], "unit": "seeds", "reps": p["paired_seeds"],
                    "anchor_value": None, "reps_basis": "scored:paired probe seeds (alone vs batched, both RNG modes)",
                    "category": "CANDIDATE", "stats": p}
        else:
            c = res["comparisons"][comp]
            ns = [res["n_cell"][f"{arm}/{pi}"] for arm in (c["A"], c["B"]) for pi in range(len(PROMPTS))]
            spec = {"metric": "spec_rng_chi2_min_p", "value": c["min_p"], "unit": "p", "reps": min(ns),
                    "anchor_value": c["bonferroni_threshold"],
                    "reps_basis": "scored:ok requests per (prompt, arm) cell, smallest of the compared cells",
                    "category": "CANDIDATE" if c["role"] == "primary" else "BASELINE", "stats": c}
        row = {"schema": BELIEF_SCHEMA, "run_name": run_name, "component": comp, **spec, "metric_direction": "higher_better",
               "decided_proposition": proposition(comp, res, rule_sha), "verdict": res["verdict"], "reason": res["reason"],
               "predeclaration_sha256": rule_sha, "records_sha256": res["records_sha256"], "verdict_sha256": verdict_sha,
               "inputs": inputs, "analyzed_at": analyzed_at, "emitted_at": utc_now(),
               "runner": {"file": "scripts/benchmark/spec_rng_1_2/spec_rng_1_2.py", "sha256": runner_sha},
               "claim_class": "INDICATIVE correctness (no throughput value)"}
        row["content_sha256"] = content_sha256(row)
        rows.append(row)
    return rows


def _atomic_write(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=d)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def default_predeclaration(records_path):
    """Launcher layout: <launcher>/out/records.jsonl beside <launcher>/PREDECLARATION.md."""
    cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(records_path))), "PREDECLARATION.md")
    return cand if os.path.isfile(cand) else None


def finish(records_path, out_path, *, predeclaration=None, summary=None, beliefs=None, write_beliefs=True, run_name=None):
    """`analyze` end to end: verdict.json (+ rule sha + analyzed_at), optional top-level VERDICT.json, belief rows."""
    res = analyze(load(records_path))
    res["records_sha256"] = file_sha256(records_path)
    pre = predeclaration if predeclaration is not None else default_predeclaration(records_path)
    rule_sha = file_sha256(pre) if pre else None
    res["predeclaration_sha256"] = rule_sha
    res["analyzed_at"] = utc_now()
    runner_sha = file_sha256(os.path.abspath(__file__))
    res["runner_sha256"] = runner_sha
    _atomic_write(out_path, json.dumps(res, indent=2))
    verdict_sha = file_sha256(out_path)
    if summary:
        _atomic_write(summary, json.dumps({
            "verdict": res["verdict"], "reason": res["reason"], "probe": res["probe"], "gates": res["gates"],
            "predeclaration_sha256": rule_sha, "analyzed_at": res["analyzed_at"], "records_sha256": res["records_sha256"],
            "verdict_sha256": verdict_sha}, indent=2))
    note = None
    if write_beliefs:
        bpath = beliefs or os.path.join(os.path.dirname(os.path.abspath(out_path)), BELIEFS_NAME)
        unit = os.path.dirname(os.path.dirname(os.path.abspath(records_path)))
        rel = os.path.relpath(os.path.abspath(records_path), unit)
        inputs = {rel: res["records_sha256"]}
        if pre:
            inputs[os.path.relpath(os.path.abspath(pre), unit)] = rule_sha
        rows = belief_rows(res, run_name=run_name or os.path.basename(unit), rule_sha=rule_sha, inputs=inputs,
                           verdict_sha=verdict_sha, analyzed_at=res["analyzed_at"], runner_sha=runner_sha)
        if not rule_sha:
            note = "belief rows NOT written: no PREDECLARATION.md to name the rule (pass --predeclaration)"
            rows = []
        elif not rows:
            note = f"belief rows NOT written: verdict {res['verdict']} is void (nothing decided)"
        if rows:
            _atomic_write(bpath, "".join(canonical(r) + "\n" for r in rows))
            note = f"belief rows: {len(rows)} -> {bpath}"
        elif os.path.exists(bpath):
            os.unlink(bpath)   # a sidecar from an earlier analysis describes other records
    return res, note


# ---------------------------------------------------------------------------------------------------------------- self test
def synthetic_records(bias=0.0, seed=7, n=400, engaged=True):
    """Null data: every arm draws each token from one fixed Zipf-ish law (poskey arms at depth > 0 get `bias` mass moved from the
    top id to the second id at index >= 1 when bias > 0). Used by `selftest` and the unit tests."""
    rng = random.Random(seed)
    vocab = list(range(40))
    w = [1.0 / (i + 1) for i in vocab]
    recs = []
    for ai, (name, r, d) in enumerate(ARMS):
        for pi in range(len(PROMPTS)):
            for k in range(n):
                toks = []
                for j in range(K_TOKENS):
                    ww = list(w)
                    if bias and r == "poskey" and d > 0 and j >= 1:
                        mv = ww[0] * bias
                        ww[0] -= mv
                        ww[1] += mv
                    toks.append(rng.choices(vocab, ww)[0])
                dn = 0 if d == 0 or not engaged else d
                recs.append({"phase": "main", "key": f"{name}/{pi}/{k}", "arm": name, "prompt": pi, "seed": k, "ok": True, "tokens": toks,
                             "draft_n": dn, "draft_n_accepted": int(dn * 0.6) if dn else 0})
    for i in range(PROBE_SEEDS):
        for rngm in RNGS:
            base = [rng.randrange(40) for _ in range(PROBE_TOKENS)]
            for cond in ("alone", "batched"):
                recs.append({"phase": "probe", "rng": rngm, "cond": cond, "seed": i, "ok": True, "tokens": base})
    return recs


def selftest():
    r0 = analyze(synthetic_records())
    r1 = analyze(synthetic_records(bias=0.5))
    ok = r0["gates"]["primary_poskey_invariant"] and not r1["gates"]["primary_poskey_invariant"]
    print(json.dumps({"null": {"primary": r0["gates"]["primary_poskey_invariant"], "verdict": r0["verdict"]},
                      "biased": {"primary": r1["gates"]["primary_poskey_invariant"], "verdict": r1["verdict"]}}))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drive"); d.add_argument("--port", type=int, required=True); d.add_argument("--out", required=True)
    d.add_argument("--deadline", type=float, required=True); d.add_argument("--dry", action="store_true")
    a = sub.add_parser("analyze"); a.add_argument("--records", required=True); a.add_argument("--out", required=True)
    a.add_argument("--predeclaration", default=None, help="the rule file (default: PREDECLARATION.md one level above --records' dir)")
    a.add_argument("--summary", default=None, help="also write the launcher's top-level VERDICT.json here")
    a.add_argument("--beliefs", default=None, help=f"belief rows path (default: {BELIEFS_NAME} beside --out)")
    a.add_argument("--no-beliefs", action="store_true")
    a.add_argument("--run-name", default=None, help="run name in the belief rows (default: the launcher dir name)")
    sub.add_parser("selftest")
    sub.add_parser("plan")
    args = ap.parse_args(argv)
    if args.cmd == "drive":
        return drive(args.port, args.out, args.deadline, args.dry)
    if args.cmd == "plan":
        return drive(0, "/dev/null", 0, True)
    if args.cmd == "selftest":
        return selftest()
    res, note = finish(args.records, args.out, predeclaration=args.predeclaration, summary=args.summary, beliefs=args.beliefs,
                       write_beliefs=not args.no_beliefs, run_name=args.run_name)
    if note:
        print(note, file=sys.stderr)
    print(f"verdict {res['verdict']}: {res['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
