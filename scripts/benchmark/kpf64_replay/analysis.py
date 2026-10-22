#!/usr/bin/env python3
"""KPF-64 analysis: realised vs ideal prefix-hit share per arm, bootstrap over sessions.

Input: one dir per arm (requests.jsonl + run_meta.json from replay_driver.py). Aggregates only.
  share_realised = sum(cache_n) / sum(prompt_n + cache_n)      (summed prompt tokens served from cache)
  share_ideal    = sum(ideal_tok) / sum(prompt_n + cache_n)    (never-evicted prefix bound, same requests)
  gap            = share_ideal - share_realised                (primary metric, in pp)
  uncached/request = sum(prompt_n) / requests                   (second primary metric)

KPF-LB2 additions (secondary, never the predeclared decision metric):
  ideal_cross_head_share = sum(ideal_union_tok) / total          bound that also lets a request match any OTHER
                           session's earlier prompt head (block-granular lower bound)
  provider_cache_share   = recorded provider usage (manifest `reference`), what the commercial cache achieved
  fanout                 = child share of requests, realised peak in-flight requests from [t_start, t_end]
                           intervals, and run_meta's peak in-flight / active lineages
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
    s = defaultdict(lambda: {"n": 0, "total": 0, "cache": 0, "ideal": 0, "uncached": 0, "inferred": 0, "union": 0})
    for r in reqs:
        a = s[r["session"]]
        a["n"] += 1
        a["total"] += r["total_tok"]
        a["cache"] += r["cache_n"]
        a["ideal"] += r["ideal_tok"]
        a["union"] += r.get("ideal_union_tok", r["ideal_tok"])
        a["uncached"] += r["prompt_n"]
        a["inferred"] += 1 if r.get("inferred_evict") else 0
    return dict(s)


def metrics(sess, ids):
    t = {k: sum(sess[i][k] for i in ids) for k in ("n", "total", "cache", "ideal", "uncached", "inferred", "union")}
    if t["total"] == 0 or t["n"] == 0:
        return None
    re_, id_ = t["cache"] / t["total"], t["ideal"] / t["total"]
    un = t["union"] / t["total"]
    return {"realised_share": re_, "ideal_share": id_, "gap_pp": 100 * (id_ - re_),
            "ideal_cross_head_share": un, "gap_cross_head_pp": 100 * (un - re_),
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


def peak_overlap(reqs):
    """Realised peak of simultaneously in-flight requests from the [t_start, t_end] intervals."""
    ev = []
    for r in reqs:
        if r.get("t_start") is not None and r.get("t_end") is not None:
            ev += [(r["t_start"], 1), (r["t_end"], -1)]
    cur = best = 0
    for _, d in sorted(ev, key=lambda e: (e[0], e[1])):
        cur += d
        best = max(best, cur)
    return best


def ordering_report(reqs):
    """DS-LB1: compare the admission ordering keys offline. Pairs of requests that were in flight at
    the same time (intervals overlap) are ordered two ways: by arrival (FCFS: `arrival_seq`) and by
    least accumulated service (`svc_before_ms`, the session's served time before the request). A pair
    DISAGREES when the earlier arrival has the larger accumulated service. `las_disagreement_share` is
    the share of overlapping cross-session pairs on which a least-attained-service queue would have
    reordered an FCFS one; 0 means the two keys never differ on this load."""
    rows = [r for r in reqs if r.get("arrival_seq") is not None and r.get("t_start") is not None
            and r.get("t_end") is not None]
    rows.sort(key=lambda r: r["arrival_seq"])
    pairs = dis = 0
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if not (a["t_start"] < b["t_end"] and b["t_start"] < a["t_end"]):
                continue                       # intervals disjoint: never in flight together
            if a["session"] == b["session"]:
                continue
            pairs += 1
            if a["svc_before_ms"] > b["svc_before_ms"]:
                dis += 1
    return {"requests": len(rows), "overlapping_cross_session_pairs": pairs, "las_vs_fcfs_disagreements": dis,
            "las_disagreement_share": (dis / pairs) if pairs else None,
            "service_ms_total": sum(r["svc_ms"] for r in rows),
            "max_session_service_ms": max((r["svc_after_ms"] for r in rows), default=0.0)}


def fanout(reqs, meta):
    n = len(reqs)
    kids = sum(1 for r in reqs if r.get("is_child"))
    return {"requests": n, "child_requests": kids, "child_request_share": (kids / n) if n else 0.0,
            "peak_inflight_from_intervals": peak_overlap(reqs),
            "peak_inflight_requests": meta.get("peak_inflight_requests"),
            "peak_active_lineages": meta.get("peak_active_lineages"), "bursts": meta.get("bursts"),
            "oversubscribed": meta.get("oversubscribed"), "children": meta.get("children"),
            "think_scale": meta.get("think_scale"),
            "think_wait_s_total": sum(r.get("think_wait_s") or 0 for r in reqs)}


def reference_from_manifest(path):
    """Recorded-transcript reference numbers: the two ideal bounds, the provider-realised cache share
    and the recorded fan-out of the SAMPLE (not of the replay)."""
    m = json.loads(Path(path).read_text())
    t = m["totals"]
    return {"manifest": str(path), "schema": m.get("schema"), "sessions": t["sessions"],
            "ideal_within_session_share": t["ideal_share"],
            "ideal_cross_session_head_share": t.get("ideal_share_cross_head"),
            "provider_cache_share": t.get("provider_cache_share"),
            "by_store": t["by_store"], "recorded_fanout": m.get("fanout"), "child_links": m.get("child_links")}


def analyse(arm_dirs, baseline=None, B=BOOT_B, seed=BOOT_SEED, min_pp=MIN_EFFECT_PP, manifest=None):
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
                   "realised_share_ci95": pct_ci(boot[n]["realised"]),
                   "fanout": fanout([r for r in reqs if r["session"] not in bad_any], meta),
                   "ordering": ordering_report([r for r in reqs if r["session"] not in bad_any])}
    base = baseline or sorted(arms)[0]
    contrasts = {}
    for n in arms:
        if n != base:
            contrasts[f"{n}_minus_{base}"] = {
                "gap_pp": differs([a - b for a, b in zip(boot[n]["gap_pp"], boot[base]["gap_pp"])], min_pp),
                "uncached_per_request": differs([a - b for a, b in zip(boot[n]["uncached"], boot[base]["uncached"])], float("inf"))}
    out = {"schema": "kpf64.analysis.v2", "baseline": base, "paired_sessions": len(ids), "bootstrap_B": B,
           "seed": seed, "min_effect_pp": min_pp, "arms": arms, "contrasts_vs_baseline": contrasts}
    if manifest:
        out["reference"] = reference_from_manifest(manifest)
    return out


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
    L += ["", "Secondary bounds and fan-out (not decision metrics):", "",
          "| arm | ideal within % | ideal cross-head % | gap vs cross-head pp | child req share | peak in-flight (intervals / driver) | peak lineages |",
          "|---|---|---|---|---|---|---|"]
    for n, a in sorted(res["arms"].items()):
        f = a["fanout"]
        L.append(f"| {n} | {100*a['ideal_share']:.2f} | {100*a['ideal_cross_head_share']:.2f} | {a['gap_cross_head_pp']:.2f} | "
                 f"{100*f['child_request_share']:.1f}% | {f['peak_inflight_from_intervals']} / {f['peak_inflight_requests']} | "
                 f"{f['peak_active_lineages']} |")
    ref = res.get("reference")
    if ref:
        L += ["", f"Recorded-transcript reference ({ref['sessions']} sessions): ideal within-session "
              f"{100*ref['ideal_within_session_share']:.2f}%, cross-session-head "
              f"{100*(ref['ideal_cross_session_head_share'] or 0):.2f}%, provider-realised cache share "
              + ("n/a" if ref["provider_cache_share"] is None else f"{100*ref['provider_cache_share']:.2f}%")]
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
    ap.add_argument("--manifest", default=None, help="adapter manifest.json for the recorded-transcript reference")
    a = ap.parse_args(argv)
    arms = dict(x.split("=", 1) for x in a.arm)
    res = analyse(arms, a.baseline, a.boot, manifest=a.manifest)
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    md = to_markdown(res)
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
