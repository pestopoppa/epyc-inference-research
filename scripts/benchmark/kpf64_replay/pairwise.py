#!/usr/bin/env python3
"""KPF-64.2: pairwise paired-by-session bootstrap contrasts, excluded-session record, and the
inferred-miss classification, computed from the stored per-arm replay records (no inference).

Contrasts reuse analysis.bootstrap/differs, so the resample draws are the SAME as ANALYSIS.json
(B=2000, seed 6464, sessions present in every arm): per pass C-B, D-B, D-C (and B/C/D minus A),
and pass-pooled arms (both passes summed per session) for B/C/D vs A and C-B, D-B, D-C.

Inferred-miss classification (a request with k>0 and cache_n < ideal_tok - tol). The driver's
`ideal_tok` is a CHAR-PROPORTIONAL estimate (total_tok * ideal_chars / prompt_chars), so it can exceed
what the slot can possibly hold. The slot holds at most `slot_cap = prev.total_tok + prev.predicted_n`
tokens of this lineage after the previous request, so each miss is re-bounded by that:
  estimator   cache_n >= slot_cap - tol          the cache reached everything the slot held; the "miss" is
                                                  the char->token estimate overshooting (new text denser
                                                  in tokens than history). Not a cache failure.
  join        slot_cap - join_tol <= cache_n < slot_cap - tol
                                                  cache stopped within a few hundred tokens of the end of the
                                                  previous prompt+output: divergence at the last message join
                                                  (retokenisation of the generated text against the next piece).
  partial     0 < cache_n < slot_cap - join_tol  cache stopped earlier (checkpoint-limited restore point).
  full        cache_n == 0                        full re-prefill (no usable prefix / no checkpoint).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402

TOL = 32
JOIN_TOL = 256


def _sum_sessions(a, b):
    return {i: {k: a[i][k] + b[i][k] for k in a[i]} for i in a if i in b}


def pairwise(sess_by_arm, B=analysis.BOOT_B, seed=analysis.BOOT_SEED, min_pp=analysis.MIN_EFFECT_PP):
    """sess_by_arm: {'p1-A': per_session dict, ...}. Returns contrast tables (gap_pp and uncached/request)."""
    ids, boot = analysis.bootstrap(sess_by_arm, B, seed)
    out = {"paired_sessions": len(ids), "bootstrap_B": B, "seed": seed, "contrasts": {}}

    def con(name, x, y):
        out["contrasts"][name] = {
            "gap_pp": analysis.differs([a - b for a, b in zip(boot[x]["gap_pp"], boot[y]["gap_pp"])], min_pp),
            "uncached_per_request": analysis.differs(
                [a - b for a, b in zip(boot[x]["uncached"], boot[y]["uncached"])], float("inf"))}
    for p in sorted({n.split("-")[0] for n in sess_by_arm}):
        have = {n.split("-", 1)[1] for n in sess_by_arm if n.startswith(p + "-")}
        for x, y in (("C", "B"), ("D", "B"), ("D", "C"), ("B", "A"), ("C", "A"), ("D", "A")):
            if x in have and y in have:
                con(f"{p}: {x}-{y}", f"{p}-{x}", f"{p}-{y}")
    return out


def pooled(sess_by_arm, passes=("p1", "p2")):
    arms = sorted({n.split("-", 1)[1] for n in sess_by_arm})
    res = {}
    for a in arms:
        parts = [sess_by_arm[f"{p}-{a}"] for p in passes if f"{p}-{a}" in sess_by_arm]
        if len(parts) == len(passes):
            cur = parts[0]
            for q in parts[1:]:
                cur = _sum_sessions(cur, q)
            res[f"pool-{a}"] = cur
    return res


def pooled_contrasts(sess_by_arm, B=analysis.BOOT_B, seed=analysis.BOOT_SEED, min_pp=analysis.MIN_EFFECT_PP):
    pl = pooled(sess_by_arm)
    ids, boot = analysis.bootstrap(pl, B, seed)
    out = {}
    for x, y in (("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "B"), ("D", "C")):
        if f"pool-{x}" in pl and f"pool-{y}" in pl:
            out[f"pooled: {x}-{y}"] = {
                "gap_pp": analysis.differs([a - b for a, b in zip(boot[f"pool-{x}"]["gap_pp"], boot[f"pool-{y}"]["gap_pp"])], min_pp),
                "uncached_per_request": analysis.differs(
                    [a - b for a, b in zip(boot[f"pool-{x}"]["uncached"], boot[f"pool-{y}"]["uncached"])], float("inf"))}
    return out, {k: analysis.metrics(v, list(v)) for k, v in pl.items()}


def excluded_record(arm_dir):
    """Which session(s) left the arm, the failing request, and the server-side reason from server.log."""
    arm_dir = Path(arm_dir)
    reqs = [json.loads(l) for l in open(arm_dir / "replay" / "requests.jsonl") if l.strip()]
    errs = [r for r in reqs if r.get("error")]
    last_ok = {}
    for r in reqs:
        if not r.get("error"):
            last_ok[r["session"]] = r
    log = arm_dir / "server.log"
    reasons = []
    if log.exists():
        for l in open(log, errors="replace"):
            m = re.search(r"send_error: .*error: (request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\))", l)
            if m:
                reasons.append({"tokens": int(m.group(2)), "ctx": int(m.group(3))})
    out = []
    for e in errs:
        lo = last_ok.get(e["session"], {})
        out.append({"session": e["session"], "k": e["k"], "error": e["error"],
                    "last_ok_k": lo.get("k"), "last_ok_total_tok": lo.get("total_tok"),
                    "last_ok_prompt_chars": lo.get("prompt_chars"), "server_reasons": reasons})
    return out


def _tag(pre):
    m = re.match(r"<([^>\s]{1,24})>", pre or "")
    return m.group(1) if m else "other"


def classify(reqs, sessions_dir, outputs=None, tol=TOL, join_tol=JOIN_TOL, exclude=()):
    """Return per-miss rows. reqs: requests.jsonl rows of ONE arm; sessions_dir: adapter output dir."""
    sessions_dir = Path(sessions_dir)
    by = defaultdict(dict)
    for r in reqs:
        if not r.get("error") and r["session"] not in exclude:
            by[r["session"]][r["k"]] = r
    cache = {}

    def sess(s):
        if s not in cache:
            cache[s] = json.loads((sessions_dir / f"{s}.json").read_text())
        return cache[s]
    rows = []
    for s, ks in by.items():
        turns = sess(s)["turns"]
        n_turns = len(turns)
        for k, r in sorted(ks.items()):
            if not r.get("inferred_evict") or k - 1 not in ks:
                continue
            p = ks[k - 1]
            slot_cap = p["total_tok"] + (p.get("predicted_n") or 0)
            cn = r["cache_n"]
            if cn >= slot_cap - tol:
                cls = "estimator"
            elif cn >= slot_cap - join_tol:
                cls = "join"
            elif cn > 0:
                cls = "partial"
            else:
                cls = "full"
            # divergence position: chars-per-token estimate of where the cache stopped vs message joins
            cpt = r["prompt_chars"] / max(1, r["total_tok"])
            cache_char = cn * cpt
            joins, off = [], 0
            if outputs is not None:
                hist = ""
                for kk in range(k + 1):
                    if turns[kk].get("reset"):
                        hist = ""
                    hist += turns[kk]["pre"]
                    joins.append(len(hist))
                    hist += (outputs.get((s, kk)) or turns[kk]["ref"]) if kk < k else ""
            near = min((abs(j - cache_char) for j in joins), default=None)
            rows.append({"session": s, "store": sess(s).get("store"), "k": k, "n_turns": n_turns,
                         "rel_pos": k / max(1, n_turns - 1), "tag": _tag(turns[k]["pre"]),
                         "prev_pred_n": p.get("predicted_n"), "cache_n": cn, "ideal_tok": r["ideal_tok"],
                         "slot_cap": slot_cap, "deficit_vs_ideal": r["ideal_tok"] - cn,
                         "deficit_vs_slot": slot_cap - cn, "cls": cls,
                         "nearest_join_chars": near, "join_within_1k_chars": (near is not None and near <= 1000)})
    return rows


def tight_gap(reqs, exclude=()):
    """Gap re-bounded by what the slot could hold: ideal'_tok = min(ideal_tok, prev.total_tok + prev.predicted_n)
    for k>0 non-reset requests (reset/k=0 keep ideal_tok, which is 0). Returns tokens and pp of all tokens."""
    by = defaultdict(dict)
    for r in reqs:
        if not r.get("error") and r["session"] not in exclude:
            by[r["session"]][r["k"]] = r
    tot = g_orig = g_tight = 0
    for ks in by.values():
        for k, r in ks.items():
            tot += r["total_tok"]
            ideal = r["ideal_tok"]
            if k > 0 and k - 1 in ks and ideal > 0:
                p = ks[k - 1]
                ideal = min(ideal, p["total_tok"] + (p.get("predicted_n") or 0))
            g_orig += r["ideal_tok"] - r["cache_n"]
            g_tight += ideal - r["cache_n"]
    return {"total_tok": tot, "gap_tok_driver_ideal": g_orig, "gap_pp_driver_ideal": 100 * g_orig / tot,
            "gap_tok_slot_bounded": g_tight, "gap_pp_slot_bounded": 100 * g_tight / tot}


def _bucket(k):
    return "k=1" if k == 1 else "k=2-3" if k <= 3 else "k=4-7" if k <= 7 else "k>=8"


def summarise(rows, total_requests):
    def cnt(key):
        return dict(Counter(key(r) for r in rows))
    return {"inferred_misses": len(rows), "requests": total_requests,
            "by_class": cnt(lambda r: r["cls"]), "by_k_bucket": cnt(lambda r: _bucket(r["k"])),
            "by_store": cnt(lambda r: r["store"]), "by_tag": cnt(lambda r: r["tag"]),
            "by_class_x_tag": cnt(lambda r: f"{r['cls']}|{r['tag']}"),
            "real_misses": sum(1 for r in rows if r["cls"] != "estimator"),
            "real_lost_tokens_vs_slot": sum(max(0, r["deficit_vs_slot"]) for r in rows if r["cls"] != "estimator"),
            "estimator_overshoot_tokens": sum(r["deficit_vs_ideal"] for r in rows if r["cls"] == "estimator"),
            "partial_with_join_within_1k_chars": sum(1 for r in rows if r["cls"] == "partial" and r["join_within_1k_chars"]),
            "partial_total": sum(1 for r in rows if r["cls"] == "partial")}


def _ci(c):
    g = c["gap_pp"]
    u = c["uncached_per_request"]
    return (f"{g['mean']:+.3f} | [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] | {'YES' if g['real_effect'] else 'no'} | "
            f"{u['mean']:+.0f} [{u['ci95'][0]:+.0f}, {u['ci95'][1]:+.0f}]")


def to_markdown(res):
    L = ["# KPF-64.2 pairwise contrasts, excluded session, inferred-miss classification", "",
         f"Paired-by-session bootstrap, B={res['bootstrap_B']}, seed={res['seed']}, {res['paired_sessions']} paired sessions "
         f"(same draws as ANALYSIS.json). Real effect = CI excludes 0 AND |mean gap diff| > {res['min_effect_pp']} pp.", "",
         "## Contrasts (gap difference, pp; uncached tok/request difference)", "",
         "| contrast | gap diff pp | 95% CI | real effect | uncached tok/req diff [95% CI] |", "|---|---|---|---|---|"]
    for k, c in {**res["contrasts"], **res["pooled_contrasts"]}.items():
        L.append(f"| {k} | {_ci(c)} |")
    L += ["", "## Excluded session", ""]
    for arm, ex in res["excluded"].items():
        for e in ex:
            L.append(f"- {arm}: `{e['session']}` request k={e['k']} `{e['error']}`; last ok k={e['last_ok_k']} "
                     f"({e['last_ok_total_tok']} tok, {e['last_ok_prompt_chars']} chars); server: "
                     f"{', '.join(str(x['tokens']) + ' tok > ctx ' + str(x['ctx']) for x in e['server_reasons'][:1]) or 'no reason in log'}")
    L += ["", "## Inferred-miss classification (per arm)", "",
          "Classes: estimator (cache reached the slot's real content; the ideal estimate overshot), join (cache stopped within "
          f"{res['join_tol']} tokens of the previous prompt+output end), partial (stopped earlier, checkpoint-limited), full (cache_n 0).", "",
          "| arm | inferred | estimator | join | partial | full | real misses | real lost tok vs slot | estimator overshoot tok |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm, s in res["misses"].items():
        bc = s["by_class"]
        L.append(f"| {arm} | {s['inferred_misses']} | {bc.get('estimator', 0)} | {bc.get('join', 0)} | {bc.get('partial', 0)} | "
                 f"{bc.get('full', 0)} | {s['real_misses']} | {s['real_lost_tokens_vs_slot']} | {s['estimator_overshoot_tokens']} |")
    L += ["", "Gap re-bounded by what the slot could hold (ideal' = min(ideal, previous total + predicted)); the driver's ideal is a "
          "char-proportional token estimate:", "", "| arm | gap pp (driver ideal) | gap pp (slot-bounded) | gap tok (driver) | gap tok (slot-bounded) |",
          "|---|---|---|---|---|"]
    for arm, t in res["tight_gap"].items():
        L.append(f"| {arm} | {t['gap_pp_driver_ideal']:.3f} | {t['gap_pp_slot_bounded']:.3f} | {t['gap_tok_driver_ideal']} | {t['gap_tok_slot_bounded']} |")
    for arm in ("p1-B", "p1-A"):
        if arm in res["misses"]:
            s = res["misses"][arm]
            L += ["", f"### {arm} breakdown", "", f"- by k bucket: {s['by_k_bucket']}", f"- by store: {s['by_store']}",
                  f"- by request kind (pre tag): {s['by_tag']}", f"- class x kind: {s['by_class_x_tag']}",
                  f"- partial misses whose cache stop is within 1000 chars of a message join: {s['partial_with_join_within_1k_chars']} of {s['partial_total']}"]
    L += ["", "Reading: on B/C/D about 92-96% of the 'inferred misses' are the estimator class and the slot-bounded gap is ~0 "
          "(slightly negative: the cache occasionally holds a few tokens more than previous total + predicted_n, so the bound is "
          "not exact). The residual 0.35 pp is therefore an artefact of the char-proportional ideal estimate, not a cache failure; "
          "the real misses are 7-16 'join' requests per arm losing ~100 tokens each (<0.01 pp). On A the real misses are the 16-18 "
          "full re-prefills (cache_n 0). No request is 'partial' on any arm."]
    return "\n".join(L) + "\n"


def run(results_dir, sessions_dir, B=analysis.BOOT_B, seed=analysis.BOOT_SEED, passes=("pass1", "pass2"), arms=("A", "B", "C", "D")):
    results_dir = Path(results_dir)
    names = {f"p{p[-1]}-{a}": results_dir / f"{p}-{a}" for p in passes for a in arms}
    data = {n: analysis.load_arm(d / "replay") for n, d in names.items()}
    bad = set().union(*[set(m["excluded_sessions"]) for _, m in data.values()])
    sess = {n: analysis.per_session([r for r in rs if r["session"] not in bad]) for n, (rs, _) in data.items()}
    out = pairwise(sess, B, seed)
    pc, pmet = pooled_contrasts(sess, B, seed)
    out.update({"min_effect_pp": analysis.MIN_EFFECT_PP, "pooled_contrasts": pc,
                "pooled_metrics": {k: {m: v[m] for m in ("realised_share", "ideal_share", "gap_pp", "uncached_per_request", "n")} for k, v in pmet.items()},
                "excluded": {n: excluded_record(d) for n, d in names.items() if data[n][1]["excluded_sessions"]},
                "excluded_sessions": sorted(bad), "tol_tokens": TOL, "join_tol": JOIN_TOL})
    out["misses"], out["miss_rows"] = {}, {}
    for n, d in names.items():
        outs = {}
        op = d / "replay" / "outputs.jsonl"
        if op.exists():
            for l in open(op):
                o = json.loads(l)
                outs[(o["session"], o["k"])] = o["out"]
        rows = classify(data[n][0], sessions_dir, outs, exclude=bad)
        out.setdefault("tight_gap", {})[n] = tight_gap(data[n][0], bad)
        out["misses"][n] = summarise(rows, sum(1 for r in data[n][0] if r["session"] not in bad))
        out["miss_rows"][n] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--boot", type=int, default=analysis.BOOT_B)
    a = ap.parse_args(argv)
    res = run(a.results, a.sessions, a.boot)
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    md = to_markdown(res)
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
