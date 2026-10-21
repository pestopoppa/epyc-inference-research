#!/usr/bin/env python3
"""DS41-CONF.1 -- DSpark confidence-head reliability analysis (stdlib only; numpy not used).

Input : JSONL written by llama.cpp with LLAMA_DSPARK_CONF_LOG=<path>, one line per verify round:
        {"round","slot","n_past","block","conf":[..],"cum":[..],"accepted":[0/1..],
         "evaluated":[0/1..],"n_accepted"}
Output: JSON with ROC-AUC (tie-aware, rank based) of
        (a) per-position confidence, (b) cumulative product, (c) position alone (score = -position,
        so earlier = more likely accepted), per-position 10-bin reliability tables, an ECE-style
        batch calibration error, round-level bootstrap 95% CIs and a verdict.

Only 'evaluated' positions are rows (positions after the first rejection have no label).
Verdict rule: AUC_cum >= AUC_pos + 0.05 => "opens_cost_aware_truncation", else "decline_recorded";
fewer than --min-rounds rounds => "insufficient_rounds" (stats still computed). With --holdout-log the
verdict must hold on both logs (else "split_disagrees" with both verdicts reported).

usage: conf_reliability.py --log X.jsonl [--holdout-log Y.jsonl] --out out.json [--min-rounds 500]
                           [--binary-id S] [--trunk S] [--drafter S]
       conf_reliability.py --selftest
"""
import argparse
import json
import math
import random
import sys

SEED = 20261021
N_BOOT = 2000
N_BINS = 10
MARGIN = 0.05


# ----------------------------------------------------------------------------- loading
def load_rounds(path):
    """-> list of rounds; each round = list of (pos, conf, cum, accepted) for evaluated positions."""
    rounds = []
    n_bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                conf, cum, acc, ev = r["conf"], r["cum"], r["accepted"], r["evaluated"]
                k = len(conf)
                if not (len(cum) == len(acc) == len(ev) == k):
                    raise ValueError("length mismatch")
                rows = [(i, float(conf[i]), float(cum[i]), int(acc[i])) for i in range(k) if ev[i]]
            except (ValueError, KeyError, TypeError):
                n_bad += 1
                continue
            rounds.append((rows, int(r.get("block", len(conf)))))
    return rounds, n_bad


# ----------------------------------------------------------------------------- AUC
def auc(scores, labels):
    """ROC-AUC via average ranks (ties get the mean rank). None if one class is absent."""
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=scores.__getitem__)
    rank_sum_pos = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            if labels[order[t]]:
                rank_sum_pos += avg_rank
        i = j + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def flatten(rounds, idxs=None):
    conf, cum, negpos, lab = [], [], [], []
    it = rounds if idxs is None else (rounds[i] for i in idxs)
    for rows, _ in it:
        for pos, c, cu, a in rows:
            conf.append(c)
            cum.append(cu)
            negpos.append(-float(pos))
            lab.append(a)
    return conf, cum, negpos, lab


def three_aucs(rounds, idxs=None):
    conf, cum, negpos, lab = flatten(rounds, idxs)
    return auc(conf, lab), auc(cum, lab), auc(negpos, lab), len(lab)


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def bootstrap(rounds, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    n = len(rounds)
    out = {"auc_conf": [], "auc_cum": [], "auc_pos": [], "delta": []}
    if n == 0:
        return {k: None for k in out}
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        a, c, p, _n = three_aucs(rounds, idxs)
        if a is not None:
            out["auc_conf"].append(a)
        if c is not None:
            out["auc_cum"].append(c)
        if p is not None:
            out["auc_pos"].append(p)
        if c is not None and p is not None:
            out["delta"].append(c - p)
    res = {}
    for k, v in out.items():
        v.sort()
        res[k] = None if not v else {"lo": pct(v, 0.025), "hi": pct(v, 0.975), "n_valid": len(v)}
    return res


# ----------------------------------------------------------------------------- reliability
def reliability(rounds):
    """Per-position 10-bin tables + pooled ECE-style error."""
    per_pos = {}
    for rows, _ in rounds:
        for pos, c, _cu, a in rows:
            b = min(N_BINS - 1, max(0, int(c * N_BINS)))
            d = per_pos.setdefault(pos, [[0, 0.0, 0] for _ in range(N_BINS)])
            d[b][0] += 1
            d[b][1] += c
            d[b][2] += a
    tables, tot_n, tot_err = {}, 0, 0.0
    edges = [i / N_BINS for i in range(N_BINS + 1)]
    for pos in sorted(per_pos):
        bins = []
        for b, (n, sc, sa) in enumerate(per_pos[pos]):
            mc = sc / n if n else None
            ar = sa / n if n else None
            bins.append({"lo": edges[b], "hi": edges[b + 1], "count": n,
                         "mean_conf": mc, "accept_rate": ar})
            if n:
                tot_n += n
                tot_err += n * abs(mc - ar)
        tables[str(pos)] = bins
    ece = tot_err / tot_n if tot_n else None
    return tables, ece


# ----------------------------------------------------------------------------- one log
def analyse(path, min_rounds):
    rounds, n_bad = load_rounds(path)
    a_conf, a_cum, a_pos, n_rows = three_aucs(rounds)
    tables, ece = reliability(rounds)
    ci = bootstrap(rounds)
    delta = None if (a_cum is None or a_pos is None) else a_cum - a_pos
    if len(rounds) < min_rounds:
        verdict = "insufficient_rounds"
    elif delta is not None and delta >= MARGIN:
        verdict = "opens_cost_aware_truncation"
    else:
        verdict = "decline_recorded"
    widths = sorted({b for _, b in rounds})
    # base accept rate per position (evaluated rows only)
    pos_rate = {}
    for rows, _ in rounds:
        for pos, _c, _cu, a in rows:
            n, s = pos_rate.get(pos, (0, 0))
            pos_rate[pos] = (n + 1, s + a)
    return {
        "log": path,
        "n_rounds": len(rounds),
        "n_malformed_lines": n_bad,
        "n_positions": n_rows,
        "block_widths": widths,
        "auc": {
            "confidence_vs_accepted": a_conf,
            "cumulative_vs_accepted": a_cum,
            "position_only": a_pos,
            "position_only_orientation": "score = -position (earlier position => predicted more likely accepted)",
            "delta_cum_minus_pos": delta,
        },
        "bootstrap_95ci": {
            "method": "resample ROUNDS with replacement; %d resamples; seed %d" % (N_BOOT, SEED),
            "auc_conf": ci["auc_conf"], "auc_cum": ci["auc_cum"],
            "auc_pos": ci["auc_pos"], "delta_cum_minus_pos": ci["delta"],
        },
        "batch_calibration_error_ece": ece,
        "batch_calibration_error_note": (
            "Our approximation of DeepInfer's batch calibration error: pooled over all positions and "
            "10 equal-width confidence bins, sum_bin n_bin*|mean_conf_bin - accept_rate_bin| / sum n_bin."),
        "accept_rate_by_position": {str(p): {"n": n, "rate": s / n} for p, (n, s) in sorted(pos_rate.items())},
        "reliability_by_position": tables,
        "rule": "AUC_cum >= AUC_pos + %.2f => opens_cost_aware_truncation else decline_recorded; "
                "rounds < %d => insufficient_rounds" % (MARGIN, min_rounds),
        "verdict": verdict,
    }


def combine_verdict(main, holdout):
    if holdout is None:
        return main["verdict"]
    if "insufficient_rounds" in (main["verdict"], holdout["verdict"]):
        return "insufficient_rounds"
    if main["verdict"] == holdout["verdict"]:
        return main["verdict"]
    return "split_disagrees"


def run(args):
    main = analyse(args.log, args.min_rounds)
    hold = analyse(args.holdout_log, args.min_rounds) if args.holdout_log else None
    result = {
        "claim_tuple_fields": {
            "n_rounds": main["n_rounds"] + (hold["n_rounds"] if hold else 0),
            "n_rounds_main": main["n_rounds"],
            "n_rounds_holdout": hold["n_rounds"] if hold else None,
            "n_positions": main["n_positions"] + (hold["n_positions"] if hold else 0),
            "block_widths": sorted(set(main["block_widths"]) | set(hold["block_widths"] if hold else [])),
            "binary_id": args.binary_id,
            "trunk": args.trunk,
            "drafter": args.drafter,
            "timing_claim": "none",
        },
        "main": main,
        "holdout": hold,
        "holdout_note": "holdout = later traffic window; verdict must hold on both" if hold else None,
        "verdict": combine_verdict(main, hold),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    fmt = lambda x: "n/a" if x is None else "%.4f" % x
    for name, r in (("main", main), ("holdout", hold)):
        if r is None:
            continue
        print("[%s] rounds=%d positions=%d widths=%s" % (name, r["n_rounds"], r["n_positions"], r["block_widths"]))
        print("[%s] AUC conf=%s cum=%s pos=%s delta=%s ECE=%s verdict=%s" % (
            name, fmt(r["auc"]["confidence_vs_accepted"]), fmt(r["auc"]["cumulative_vs_accepted"]),
            fmt(r["auc"]["position_only"]), fmt(r["auc"]["delta_cum_minus_pos"]),
            fmt(r["batch_calibration_error_ece"]), r["verdict"]))
    print("VERDICT: %s  (-> %s)" % (result["verdict"], args.out))
    return result


# ----------------------------------------------------------------------------- selftest
def _synth(path, n_rounds, informative, seed, block=8):
    """Informative: acceptance is Bernoulli(conf) with conf varying per position AND per round
    (so cumulative product carries per-round info that position alone does not).
    Uninformative: conf is noise independent of acceptance; acceptance depends on position only."""
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as fh:
        for rnd in range(n_rounds):
            quality = rng.random()  # latent round difficulty
            conf, accepted = [], []
            alive = True
            n_acc = 0
            for i in range(block):
                if informative:
                    c = min(0.999, max(0.001, quality * (1.0 - 0.03 * i) + rng.gauss(0, 0.05)))
                    a = 1 if rng.random() < c else 0
                else:
                    c = rng.random()
                    a = 1 if rng.random() < 0.85 ** (i + 1) else 0
                conf.append(c)
                if alive and a:
                    accepted.append(1)
                    n_acc += 1
                else:
                    alive = False
                    accepted.append(0)
            cum, p = [], 1.0
            for c in conf:
                p *= c
                cum.append(p)
            ev = [1 if (i < n_acc + 1 and i < block) else 0 for i in range(block)]
            fh.write(json.dumps({"round": rnd, "slot": 0, "n_past": 1000 + rnd * 4, "block": block,
                                 "conf": conf, "cum": cum, "accepted": accepted,
                                 "evaluated": ev, "n_accepted": n_acc}) + "\n")


def selftest():
    import os
    import tempfile
    ok = True
    # known-answer AUC checks (tie-aware)
    assert abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-12
    assert abs(auc([4, 3, 2, 1], [0, 0, 1, 1]) - 0.0) < 1e-12
    assert abs(auc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-12
    assert abs(auc([1, 2, 2, 3], [0, 0, 1, 1]) - 0.875) < 1e-12
    print("selftest: known-answer AUC checks pass")
    d = tempfile.mkdtemp(prefix="confrel_selftest_")
    try:
        for informative in (True, False):
            tag = "informative" if informative else "uninformative"
            lp, hp, op = (os.path.join(d, x) for x in ("main.jsonl", "hold.jsonl", "out.json"))
            _synth(lp, 600, informative, seed=11)
            _synth(hp, 600, informative, seed=12)
            ns = argparse.Namespace(log=lp, holdout_log=hp, out=op, min_rounds=500,
                                    binary_id="selftest", trunk="selftest", drafter="selftest")
            res = run(ns)
            m = res["main"]
            want = "opens_cost_aware_truncation" if informative else "decline_recorded"
            good = res["verdict"] == want
            if informative:
                good = good and m["auc"]["cumulative_vs_accepted"] > 0.7 and m["auc"]["confidence_vs_accepted"] > 0.7
            else:
                good = good and abs(m["auc"]["confidence_vs_accepted"] - 0.5) < 0.05 \
                    and m["auc"]["position_only"] > 0.55
            good = good and res["claim_tuple_fields"]["timing_claim"] == "none"
            print("selftest[%s]: expected verdict=%s got=%s -> %s" % (tag, want, res["verdict"], "PASS" if good else "FAIL"))
            ok = ok and good
        # insufficient rounds
        lp, op = os.path.join(d, "small.jsonl"), os.path.join(d, "small.json")
        _synth(lp, 100, True, seed=13)
        ns = argparse.Namespace(log=lp, holdout_log=None, out=op, min_rounds=500,
                                binary_id="", trunk="", drafter="")
        res = run(ns)
        good = res["verdict"] == "insufficient_rounds"
        print("selftest[insufficient]: %s" % ("PASS" if good else "FAIL"))
        ok = ok and good
    finally:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log")
    ap.add_argument("--holdout-log")
    ap.add_argument("--out")
    ap.add_argument("--min-rounds", type=int, default=500)
    ap.add_argument("--binary-id", default="")
    ap.add_argument("--trunk", default="")
    ap.add_argument("--drafter", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.log or not args.out:
        ap.error("--log and --out are required (or --selftest)")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
