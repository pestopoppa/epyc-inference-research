#!/usr/bin/env python3
"""KPF-64 analysis: realised vs ideal prefix-hit share per arm, bootstrap over sessions.

Input: one dir per arm (requests.jsonl + run_meta.json from replay_driver.py). Aggregates only.
  share_realised = sum(cache_n) / sum(prompt_n + cache_n)      (summed prompt tokens served from cache)
  share_ideal    = sum(ideal_tok) / sum(prompt_n + cache_n)    (never-evicted prefix bound, same requests)
  gap            = share_ideal - share_realised                (primary metric, in pp)
  uncached/request = sum(prompt_n) / requests                   (second primary metric)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MIN_EFFECT_PP = 0.5   # PREDECLARED.md decision rule
BOOT_B = 2000
BOOT_SEED = 6464


def load_arm(d):
    d = Path(d)
    allr = [json.loads(l) for l in open(d / "requests.jsonl") if l.strip()]
    bad = {r["session"] for r in allr if r.get("error")}   # a session with any error leaves the arm
    reqs = [r for r in allr if r["session"] not in bad]
    meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
    meta["excluded_sessions"] = sorted(bad)
    return reqs, meta


def per_session(reqs):
    s = defaultdict(lambda: {"n": 0, "total": 0, "cache": 0, "ideal": 0, "uncached": 0, "inferred": 0})
    for r in reqs:
        a = s[r["session"]]
        a["n"] += 1
        a["total"] += r["total_tok"]
        a["cache"] += r["cache_n"]
        a["ideal"] += r["ideal_tok"]
        a["uncached"] += r["prompt_n"]
        a["inferred"] += 1 if r.get("inferred_evict") else 0
    return dict(s)


def metrics(sess, ids):
    t = {k: sum(sess[i][k] for i in ids) for k in ("n", "total", "cache", "ideal", "uncached", "inferred")}
    if t["total"] == 0 or t["n"] == 0:
        return None
    re_, id_ = t["cache"] / t["total"], t["ideal"] / t["total"]
    return {"realised_share": re_, "ideal_share": id_, "gap_pp": 100 * (id_ - re_),
            "uncached_per_request": t["uncached"] / t["n"], "inferred_evictions": t["inferred"], **t}


def pct_ci(vals, lo=2.5, hi=97.5):
    v = sorted(vals)
    return [v[int(len(v) * lo / 100)], v[min(len(v) - 1, int(len(v) * hi / 100))]]


def bootstrap(sess_by_arm, B=BOOT_B, seed=BOOT_SEED):
    """Paired resamples of SESSIONS shared by all arms (sessions present in every arm)."""
    ids = sorted(set.intersection(*[set(s) for s in sess_by_arm.values()]))
    rng = random.Random(seed)
    draws = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(B)]
    out = {}
    for arm, s in sess_by_arm.items():
        ms = [metrics(s, d) for d in draws]
        out[arm] = {"gap_pp": [m["gap_pp"] for m in ms], "uncached": [m["uncached_per_request"] for m in ms],
                    "realised": [m["realised_share"] for m in ms]}
    return ids, out


def differs(delta_draws, min_pp=MIN_EFFECT_PP):
    lo, hi = pct_ci(delta_draws)
    mean = sum(delta_draws) / len(delta_draws)
    return {"mean": mean, "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0,
            "exceeds_min_effect": abs(mean) > min_pp, "real_effect": (lo > 0 or hi < 0) and abs(mean) > min_pp}


def analyse(arm_dirs, baseline=None, B=BOOT_B, seed=BOOT_SEED, min_pp=MIN_EFFECT_PP):
    data = {name: load_arm(d) for name, d in arm_dirs.items()}
    bad_any = set().union(*[set(m["excluded_sessions"]) for _, m in data.values()])
    sess = {n: per_session([r for r in rs if r["session"] not in bad_any]) for n, (rs, _) in data.items()}
    ids, boot = bootstrap(sess, B, seed)
    arms = {}
    for n, (reqs, meta) in data.items():
        m = metrics(sess[n], list(sess[n]))
        ev = (meta.get("events") or {})
        arms[n] = {**m, "sessions": len(sess[n]), "requests": sum(x["n"] for x in sess[n].values()),
                   "excluded_sessions": sorted(bad_any),
                   "observed_eviction_events": ev.get("observed_eviction_events"),
                   "log_present": ev.get("log_present"), "events_by_pattern": ev.get("by_pattern"),
                   "gap_pp_ci95": pct_ci(boot[n]["gap_pp"]), "uncached_per_request_ci95": pct_ci(boot[n]["uncached"]),
                   "realised_share_ci95": pct_ci(boot[n]["realised"])}
    base = baseline or sorted(arms)[0]
    contrasts = {}
    for n in arms:
        if n != base:
            contrasts[f"{n}_minus_{base}"] = {
                "gap_pp": differs([a - b for a, b in zip(boot[n]["gap_pp"], boot[base]["gap_pp"])], min_pp),
                "uncached_per_request": differs([a - b for a, b in zip(boot[n]["uncached"], boot[base]["uncached"])], float("inf"))}
    return {"schema": "kpf64.analysis.v1", "baseline": base, "paired_sessions": len(ids), "bootstrap_B": B,
            "seed": seed, "min_effect_pp": min_pp, "arms": arms, "contrasts_vs_baseline": contrasts}


def to_markdown(res):
    L = ["| arm | sess | req | realised % | ideal % | gap pp [95% CI] | uncached tok/req [95% CI] | evict observed | evict inferred |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, a in sorted(res["arms"].items()):
        L.append(f"| {n} | {a['sessions']} | {a['requests']} | {100*a['realised_share']:.2f} | {100*a['ideal_share']:.2f} | "
                 f"{a['gap_pp']:.2f} [{a['gap_pp_ci95'][0]:.2f}, {a['gap_pp_ci95'][1]:.2f}] | "
                 f"{a['uncached_per_request']:.0f} [{a['uncached_per_request_ci95'][0]:.0f}, {a['uncached_per_request_ci95'][1]:.0f}] | "
                 f"{a['observed_eviction_events'] if a['log_present'] else 'n/a (no log)'} | {a['inferred_evictions']} |")
    L += ["", f"Contrasts vs baseline {res['baseline']} (paired-by-session bootstrap, B={res['bootstrap_B']}, "
          f"real effect = CI excludes 0 AND |mean gap diff| > {res['min_effect_pp']} pp):", "",
          "| contrast | gap diff pp | 95% CI | real effect |", "|---|---|---|---|"]
    for k, c in res["contrasts_vs_baseline"].items():
        g = c["gap_pp"]
        L.append(f"| {k} | {g['mean']:.2f} | [{g['ci95'][0]:.2f}, {g['ci95'][1]:.2f}] | {'YES' if g['real_effect'] else 'no'} |")
    return "\n".join(L) + "\n"


# ---- belief-kernel hook (scripts/vidya/adapters/chunk_kv.py, kind=prefix_hit_share) -------------
def belief_measurements(res, served_pin, window, control_arm=None):
    """One chunk_kv `prefix_hit_share` measurement per arm: value = realised_tokens / ideal_tokens
    (both integers ride in the row; the strict reader re-derives it). served_pin needs
    role/model/kernel_ref; window needs window_id/region_lock_claim/start_utc/end_utc."""
    ctl = control_arm or res["baseline"]
    out = []
    for n, a in sorted(res["arms"].items()):
        out.append({"kind": "prefix_hit_share", "source_program": "KPF-64",
                    "subject": f"role={served_pin.get('role')} arm={n}",
                    "served_pin": dict(served_pin), "window": dict(window), "is_control": n == ctl,
                    "realised_tokens": int(a["cache"]), "ideal_tokens": int(a["ideal"])})
    return out


def write_belief_measurements(out_dir, *, run_id, producer, results_path, res, served_pin, window,
                              control_arm=None, chunk_kv=None):
    """Emit chunk_kv.beliefs.jsonl via the root adapter (import it from scripts/vidya/adapters on
    sys.path, or pass the module). results_path is the analysis JSON the rows hash."""
    if chunk_kv is None:
        from adapters import chunk_kv  # type: ignore  # noqa: needs <root>/scripts/vidya on sys.path
    return chunk_kv.write_belief_measurements(
        out_dir, run_id=run_id, producer=producer, results_path=results_path,
        measurements=belief_measurements(res, served_pin, window, control_arm))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR (repeatable)")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--boot", type=int, default=BOOT_B)
    a = ap.parse_args(argv)
    arms = dict(x.split("=", 1) for x in a.arm)
    res = analyse(arms, a.baseline, a.boot)
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    md = to_markdown(res)
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
