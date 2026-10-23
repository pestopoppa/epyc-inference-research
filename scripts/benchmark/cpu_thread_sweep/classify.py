#!/usr/bin/env python3
"""Corrected degeneracy classifier for INF-70 gdn-rowexact runs.

Classes (by REASON, never a bare pass/fail):
  HTTP-ERROR  the request failed / no JSON
  EMPTY       zero tokens returned, no eos  (server produced nothing)
  EARLY-EOS   model emitted eos at/near token 0 -> empty or near-empty content.
              A REAL failure, but distinct from salad: the forward is not
              producing token noise, it is producing an immediate stop.
  SALAD       enough tokens to judge, and the token stream is degenerate
              (low unique ratio / one token dominating / long repeat run /
               non-linguistic content)
  COHERENT    otherwise
The original metric computed uniq_ratio/top_tok_share on n=1 (both 1.0) and the
`top < 0.25` clause flagged it DEGENERATE.  Statistics are only computed, and
only consulted, when n >= MIN_N (=16).
"""
import collections, json, os, sys

MIN_N = 16          # below this the token statistics are meaningless
EOS_MAX_N = 4       # <= this many tokens ending in eos == EARLY-EOS

def classify(d):
    if not isinstance(d, dict) or "error" in d:
        return dict(cls="HTTP-ERROR", n=0, note=str(d.get("error"))[:80] if isinstance(d, dict) else "not-a-dict")
    toks = d.get("tokens") or []
    n = len(toks)
    stop = d.get("stop_type")
    content = d.get("content", "") or ""
    st = dict(n=n, stop=stop, npred_req=(d.get("generation_settings") or {}).get("n_predict"))
    if n == 0:
        return dict(cls=("EARLY-EOS" if stop == "eos" else "EMPTY"), **st)
    if n <= EOS_MAX_N and stop == "eos":
        return dict(cls="EARLY-EOS", **st)
    if n < MIN_N:
        # short but not an eos stop: report as its own case rather than guessing
        return dict(cls=("EARLY-EOS" if stop == "eos" else "SHORT"), **st)
    uniq = len(set(toks)) / n
    c = collections.Counter(toks)
    top = c.most_common(1)[0][1] / n
    run = best = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    words = content.split()
    # linguistic sanity: fraction of chars that are ordinary text
    ok_chars = sum(ch.isascii() and (ch.isalnum() or ch in " .,;:'\"-()!?\n") for ch in content) / max(1, len(content))
    # word floor must scale with n: a 24-token completion legitimately has ~15 words.
    salad = (uniq < 0.35) or (top >= 0.25) or (best >= 6) or (len(words) < 0.25 * n) or (ok_chars < 0.85)
    st.update(uniq=round(uniq, 3), top=round(top, 3), run=best, words=len(words), ascii_ok=round(ok_chars, 3))
    return dict(cls=("SALAD" if salad else "COHERENT"), **st)

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/raid0/llm/tmp/inf70/agents/gdn-rowexact/runs"
    only = sys.argv[2] if len(sys.argv) > 2 else None
    for arm in sorted(os.listdir(root)):
        ad = os.path.join(root, arm)
        if not os.path.isdir(ad):
            continue
        if only and only not in arm:
            continue
        files = sorted(f for f in os.listdir(ad) if f.endswith(".json") and f != "probe.json")
        if not files and "probe.json" not in os.listdir(ad):
            continue
        print(f"\n### {arm}")
        for f in files:
            try:
                d = json.load(open(os.path.join(ad, f)))
            except Exception as e:
                print(f"  {f[:-5]:<14} UNREADABLE {e}"); continue
            r = classify(d)
            t = d.get("timings", {}) if isinstance(d, dict) else {}
            pp = t.get("prompt_per_second")
            tg = t.get("predicted_per_second")
            extra = " ".join(f"{k}={v}" for k, v in r.items() if k not in ("cls",))
            print(f"  {f[:-5]:<14} {r['cls']:<10} pp={pp if pp is None else round(pp,1)} tg={tg if tg is None else round(tg,2)} {extra}")
            if r["cls"] in ("SALAD", "COHERENT"):
                print(f"                 text={ (d.get('content','') or '')[:90]!r}")
        if "probe.json" in os.listdir(ad):
            rows = json.load(open(os.path.join(ad, "probe.json")))
            print(f"  -- probe: {len(rows)} lengths")
            for row in rows:
                rr = classify({"tokens": row.get("tokens"), "content": row.get("content", ""), "stop_type": row.get("stop_type")})
                print(f"     k={row['k']:<4} prompt_n={row['prompt_n']:<4} {rr['cls']:<10} {row.get('content','')[:60]!r}")
