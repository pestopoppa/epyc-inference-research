#!/usr/bin/env python3
"""RUS-2: trace-length and review-ratio signal probe on logged eval runs. ZERO inference, read-only.

Predeclared rules (fixed before the run):
* Data: benchmarks/results/eval/*.jsonl except seen_questions*. One sample = one role_results entry
  (key 'role:mode') of one question record. Samples missing tokens_generated, a bool `passed`, or
  with a non-null `error` are dropped and counted (reason table).
* Cell = (role, suite). Role/mode come from the entry. Duplicate samples (same suite, question_id,
  role-key, timestamp) are collapsed.
* Truncation: the records carry NO finish_reason field. Proxy, reported separately and EXCLUDED from
  both arms: a sample is `cap_proxy` if its tokens_generated equals the cell's maximum AND that
  maximum is reached by >=3 samples in the cell (a cap signature). Counts reported per cell.
* Temperature: recorded per sample if a `temperature` key exists on the entry; otherwise 'not_recorded'
  (counted). It is never inferred.
* Between-question AUROC: P(tokens_generated of correct > of incorrect), ties 0.5, pooled over all
  retained samples; one sample per question is NOT enforced (repeats are pooled), a bootstrap CI
  resamples QUESTIONS (200 draws, seed 20261021). <0.5 means correct samples are SHORTER.
* Within-question arm (question fixed effects): only questions with >=2 retained samples AND at least
  one correct and one incorrect sample (all-correct/all-wrong dropped, counted). Statistic = pooled
  over those questions of P(tokens correct > tokens incorrect) over correct/incorrect sample pairs
  (ties 0.5), questions weighted by their pair counts; also the question-level share where the mean
  correct length is shorter than the mean incorrect length, with N questions. Same bootstrap.
* Review ratio: deterministic keyword-chunk ratio on the ANSWER text (the only text the records carry;
  reasoning traces live in the tap logs, whose offsets do not resolve for these runs). Chunks = non-empty
  lines (split on blank lines or newlines); a chunk is a review chunk iff it matches REVIEW_RE.
  Reported only where the answer has >=4 chunks; coverage is stated. AUROC computed like tokens.
"""
import argparse, collections as C, glob, json, os, random, re, sys

REVIEW_RE = re.compile(r"\b(wait|hold on|let me (?:re-?check|re-?read|verify|double[- ]check|reconsider)|re-?check|double[- ]check|"
                       r"actually|on second thought|hmm|let me verify|verify that|to verify|i made (?:a|an) (?:mistake|error)|"
                       r"that(?:'s| is) (?:wrong|incorrect)|recompute|re-?evaluate|sanity check)\b", re.I)


def auroc_pairs(pos, neg):
    """(wins, pairs) for P(pos > neg), ties 0.5 — O(n log n) via sorted merge."""
    if not pos or not neg:
        return 0.0, 0
    ns = sorted(neg)
    import bisect
    w = 0.0
    for p in pos:
        lo = bisect.bisect_left(ns, p); hi = bisect.bisect_right(ns, p)
        w += lo + 0.5 * (hi - lo)
    return w, len(pos) * len(neg)


def stat_between(qs):
    pos, neg = [], []
    for samples in qs.values():
        for t, ok, _ in samples:
            (pos if ok else neg).append(t)
    w, n = auroc_pairs(pos, neg)
    return (w / n if n else None), n


def within_questions(qs):
    kept = {}
    for q, s in qs.items():
        if len(s) >= 2 and any(x[1] for x in s) and any(not x[1] for x in s):
            kept[q] = s
    return kept


def stat_within(qs):
    W = N = 0
    for s in within_questions(qs).values():
        w, n = auroc_pairs([t for t, ok, _ in s if ok], [t for t, ok, _ in s if not ok])
        W += w; N += n
    return (W / N if N else None), N


def boot(fn, qs, B=200, seed=20261021):
    keys = list(qs)
    if len(keys) < 2:
        return None
    rnd = random.Random(seed); vals = []
    for _ in range(B):
        draw = [rnd.choice(keys) for _ in keys]
        v, n = fn({i: qs[k] for i, k in enumerate(draw)})
        if v is not None:
            vals.append(v)
    if len(vals) < 20:
        return None
    vals.sort()
    return [round(vals[int(.025 * len(vals))], 4), round(vals[int(.975 * len(vals)) - 1], 4)]


def review_ratio(text):
    chunks = [c for c in re.split(r"\n+", text or "") if c.strip()]
    if len(chunks) < 4:
        return None
    return sum(1 for c in chunks if REVIEW_RE.search(c)) / len(chunks)


def load(results_dir):
    drops = C.Counter(); seen = set(); cells = C.defaultdict(lambda: C.defaultdict(list)); temp = C.Counter()
    nfiles = 0
    for f in sorted(glob.glob(os.path.join(results_dir, "eval", "*.jsonl"))):
        if os.path.basename(f).startswith("seen_questions"):
            continue
        nfiles += 1
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                drops["unparsable_line"] += 1; continue
            rr = r.get("role_results")
            if not isinstance(rr, dict):
                drops["no_role_results"] += 1; continue
            for key, e in rr.items():
                if not isinstance(e, dict):
                    continue
                suite, qid = r.get("suite"), r.get("question_id")
                dk = (suite, qid, key, r.get("timestamp"))
                if dk in seen:
                    drops["duplicate"] += 1; continue
                seen.add(dk)
                tok = e.get("tokens_generated")
                if not isinstance(tok, (int, float)) or isinstance(tok, bool) or tok <= 0:
                    drops["no_tokens_generated"] += 1; continue
                if not isinstance(e.get("passed"), bool):
                    drops["no_passed_bool"] += 1; continue
                if e.get("error"):
                    drops["error_set"] += 1; continue
                temp["recorded" if "temperature" in e else "not_recorded"] += 1
                rv = review_ratio(e.get("answer") if isinstance(e.get("answer"), str) else "")
                cells[(e.get("role", key.split(":")[0]), key.split(":")[-1], suite)][qid].append((float(tok), e["passed"], rv))
    return cells, drops, temp, nfiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="benchmarks/results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-questions", type=int, default=20)
    a = ap.parse_args()
    cells, drops, temp, nfiles = load(a.results_dir)
    rows = []
    for (role, mode, suite), qs in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        toks = [t for s in qs.values() for t, _, _ in s]
        mx = max(toks); nmx = sum(1 for t in toks if t == mx)
        cap = nmx >= 3
        kept = {q: [x for x in s if not (cap and x[0] == mx)] for q, s in qs.items()}
        kept = {q: s for q, s in kept.items() if s}
        allq = len(kept)
        pos = sum(1 for s in kept.values() for x in s if x[1]); tot = sum(len(s) for s in kept.values())
        b, bn = stat_between(kept); wv, wn = stat_within(kept)
        wq = within_questions(kept)
        shorter = sum(1 for s in wq.values()
                      if sum(t for t, ok, _ in s if ok) / sum(1 for _, ok, _ in s if ok) < sum(t for t, ok, _ in s if not ok) / sum(1 for _, ok, _ in s if not ok))
        rr = {q: [(rv, ok, None) for _, ok, rv in s if rv is not None] for q, s in kept.items()}
        rr = {q: s for q, s in rr.items() if s}
        rb, rn = stat_between(rr)
        rows.append({"role": role, "mode": mode, "suite": suite, "questions": allq, "samples": tot, "correct": pos,
                     "cap_proxy_excluded": (nmx if cap else 0), "cap_value": (mx if cap else None),
                     "between_auroc_tokens": None if b is None else round(b, 4), "between_pairs": bn,
                     "between_ci95": boot(stat_between, kept) if allq >= a.min_questions and b is not None else None,
                     "within_questions_mixed": len(wq), "within_pairs": wn,
                     "within_pooled_auroc": None if wv is None else round(wv, 4),
                     "within_ci95": boot(stat_within, kept) if len(wq) >= 10 else None,
                     "within_share_correct_shorter": (round(shorter / len(wq), 4) if wq else None),
                     "review_ratio_samples": sum(len(s) for s in rr.values()),
                     "review_ratio_auroc": None if rb is None else round(rb, 4)})
    out = {"files_scanned": nfiles, "drops": dict(drops), "temperature_field": dict(temp),
           "fields_absent": ["finish_reason", "per-sample temperature (unless temperature_field.recorded>0)", "reasoning trace text (answer only)"],
           "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print("files", nfiles, "drops", dict(drops), "temp", dict(temp), "cells", len(rows))
    print("role|mode|suite|Q|S|between_auroc[ci]|within(mixedQ,pairs)|within_auroc[ci]|share_shorter|cap_excl|review_n|review_auroc")
    for r in rows:
        if r["questions"] >= a.min_questions or r["within_questions_mixed"] >= 5:
            print("|".join(str(x) for x in [r["role"], r["mode"], r["suite"], r["questions"], r["samples"], r["between_auroc_tokens"], r["between_ci95"],
                  r["within_questions_mixed"], r["within_pairs"], r["within_pooled_auroc"], r["within_ci95"], r["within_share_correct_shorter"],
                  r["cap_proxy_excluded"], r["review_ratio_samples"], r["review_ratio_auroc"]]))


if __name__ == "__main__":
    main()
