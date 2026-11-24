#!/usr/bin/env python3
"""SPEC-COPY-COV-1: offline copy-match coverage of an EXACT longest-suffix index (no inference).

Question: would a training-free copy-proposal drafter (exact longest-suffix match, not our fixed-n hashed
ngram-mod) find usable continuations on archived agentic traffic often enough to be worth a GPU probe?

For every generated output position i of every request, the source is causal: the request's own prompt plus
its output prefix out[:i] (plus, per variant, other requests' outputs). We take the LONGEST suffix of the
causal context that occurs earlier in the source (suffix automaton: link(last) is exactly that state), propose
the 3 tokens that followed its first earlier occurrence, and count how many leading ones equal out[i:i+3].

Variants (corpus prepended before the request's prompt, sentinel-separated so no match spans documents):
  self      own prompt + own output prefix only (what a deployed per-request copy drafter sees)
  loo       self + the outputs of every OTHER request (leave-one-out; leak-free; THE DECISIVE VARIANT)
  insample  self + every output including this request's own (leaks the future; upper bound, reported only)
Classes: code = suite debugbench; prose = every other suite (hotpotqa, general).

Rule (box SPEC-COPY-COV-1, predeclared): hit = match length >= 5 AND 3 agreed tokens. On the loo variant,
if hit share < 5% on code AND < 2% on prose -> CLOSE_COPY_PATH, else FIRE_TRIGGER.

Usage:
  copy_cov_replay.py --results P.jsonl --prompts prompts_p1.jsonl --tokenize-bin <llama-tokenize> \
      --model <gguf> --out <dir>                # tokenizes (vocab-only load, CPU, seconds) then replays
  copy_cov_replay.py --tokens tokens.jsonl --out <dir>   # replay pre-tokenized {id,suite,prompt_ids,out_ids}
  copy_cov_replay.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

MIN_LEN = 5
AGREE = 3
CODE_SUITES = {"debugbench"}
THRESH = {"code": 0.05, "prose": 0.02}
VARIANTS = ("self", "loo", "insample")


class SAM:
    """Online suffix automaton over an integer alphabet; firstpos = end index of the first occurrence."""
    __slots__ = ("nxt", "link", "length", "fpos", "last")

    def __init__(self):
        self.nxt, self.link, self.length, self.fpos, self.last = [{}], [-1], [0], [-1], 0

    def add(self, c, pos):
        cur = len(self.length)
        self.nxt.append({}); self.length.append(self.length[self.last] + 1); self.link.append(0); self.fpos.append(pos)
        p = self.last
        while p != -1 and c not in self.nxt[p]:
            self.nxt[p][c] = cur
            p = self.link[p]
        if p != -1:
            q = self.nxt[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                cl = len(self.length)
                self.nxt.append(dict(self.nxt[q])); self.length.append(self.length[p] + 1)
                self.link.append(self.link[q]); self.fpos.append(self.fpos[q])
                while p != -1 and self.nxt[p].get(c) == q:
                    self.nxt[p][c] = cl
                    p = self.link[p]
                self.link[q] = self.link[cur] = cl
        self.last = cur

    def earlier_match(self):
        """(length, end index of first earlier occurrence) of the longest suffix seen before; (0, -1) if none."""
        s = self.link[self.last]
        return (self.length[s], self.fpos[s]) if s > 0 else (0, -1)


def replay_request(corpus, prompt, out):
    """Per output position: (match_len, agreed) for the causal source corpus + prompt + out[:i]."""
    sam, seq, rows = SAM(), [], []
    for t in list(corpus) + list(prompt):
        sam.add(t, len(seq)); seq.append(t)
    for i, tok in enumerate(out):
        if i > 0 or seq:
            L, fp = sam.earlier_match()
            agreed = 0
            if L > 0:
                for k in range(AGREE):
                    j = fp + 1 + k
                    if j >= len(seq) or i + k >= len(out) or seq[j] != out[i + k] or seq[j] < 0:
                        break
                    agreed += 1
            rows.append((L, agreed))
        sam.add(tok, len(seq)); seq.append(tok)
    return rows


def corpus_for(reqs, idx, variant):
    if variant == "self":
        return []
    c = []
    for k, r in enumerate(reqs):
        if variant == "loo" and k == idx:
            continue
        c.extend(r["out_ids"]); c.append(-(k + 1))  # negative sentinel: unique, never a real token
    return c


def summarize(per):
    n = len(per)
    hits = sum(1 for L, a in per if L >= MIN_LEN and a >= AGREE)
    hist = {}
    for L, _ in per:
        b = "0" if L == 0 else "1-4" if L < 5 else "5-9" if L < 10 else "10-31" if L < 32 else ">=32"
        hist[b] = hist.get(b, 0) + 1
    return {"positions": n, "hits": hits, "hit_share": hits / n if n else None,
            "agree_ge1_share_at_len_ge5": (sum(1 for L, a in per if L >= MIN_LEN and a >= 1) / n) if n else None,
            "len_hist": hist}


def run(reqs):
    out = {"rule": {"min_len": MIN_LEN, "agreed": AGREE, "thresholds": THRESH, "decisive_variant": "loo"},
           "n_requests": len(reqs), "variants": {}, "per_request": []}
    for v in VARIANTS:
        pooled = {"code": [], "prose": []}
        for i, r in enumerate(reqs):
            rows = replay_request(corpus_for(reqs, i, v), r["prompt_ids"], r["out_ids"])
            cls = "code" if r["suite"] in CODE_SUITES else "prose"
            pooled[cls].extend(rows)
            s = summarize(rows)
            out["per_request"].append({"variant": v, "id": r["id"], "class": cls, **{k: s[k] for k in ("positions", "hits", "hit_share")}})
        out["variants"][v] = {c: summarize(p) for c, p in pooled.items()}
    loo = out["variants"]["loo"]
    judgeable = all(loo[c]["positions"] for c in THRESH)
    close = judgeable and all(loo[c]["hit_share"] < THRESH[c] for c in THRESH)
    out["verdict"] = "NOT_JUDGEABLE_EMPTY_CLASS" if not judgeable else "CLOSE_COPY_PATH" if close else "FIRE_TRIGGER"
    return out


def tokenize(bin_path, model, text):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text); name = f.name
    try:
        p = subprocess.run([bin_path, "-m", model, "-f", name, "--ids", "--no-bos", "--log-disable"],
                           capture_output=True, text=True, timeout=600, check=True)
    finally:
        os.unlink(name)
    line = [l for l in p.stdout.splitlines() if l.strip().startswith("[")][-1]
    return json.loads(line)


def load_tokens(args):
    if args.tokens:
        return [json.loads(l) for l in open(args.tokens) if l.strip()]
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(args.prompts) if l.strip()}
    reqs = []
    for l in open(args.results):
        r = json.loads(l)
        if not r.get("ok") or not r.get("text"):
            continue
        p = prompts[r["id"]]["prompt"]
        reqs.append({"id": r["id"], "suite": r["suite"], "prompt_ids": tokenize(args.tokenize_bin, args.model, p),
                     "out_ids": tokenize(args.tokenize_bin, args.model, r["text"])})
    return reqs


def selftest():
    # 1. longest earlier suffix is exact (brute force check on a random sequence)
    import random
    rnd = random.Random(7)
    seq = [rnd.randrange(4) for _ in range(300)]
    sam = SAM()
    for i, t in enumerate(seq):
        sam.add(t, i)
        L, fp = sam.earlier_match()
        best = 0
        for n in range(1, i + 1):
            suf = seq[i + 1 - n:i + 1]
            if any(seq[s:s + n] == suf for s in range(0, i + 1 - n)):
                best = n
            else:
                break
        assert L == best, (i, L, best)
        if L:
            assert seq[fp + 1 - L:fp + 1] == seq[i + 1 - L:i + 1] and fp < i
    # 2. a verbatim copy of the prompt is a hit everywhere past the first MIN_LEN tokens
    prompt = list(range(100, 140))
    rows = replay_request([], prompt, prompt[:30])
    assert all(L >= MIN_LEN and a == AGREE for L, a in rows[MIN_LEN:27]), rows
    # 3. unrelated output gives no hits; sentinels never join documents
    rows = replay_request([1, 2, 3, 4, 5, 6, -1], [9, 9], [1, 2, 3, 4, 5, 6, 7])
    assert sum(1 for L, a in rows if L >= MIN_LEN and a >= AGREE) == 0
    # 4. loo excludes self, insample includes it (leak)
    reqs = [{"id": "a", "suite": "debugbench", "prompt_ids": [0], "out_ids": list(range(10, 40))},
            {"id": "b", "suite": "general", "prompt_ids": [0], "out_ids": list(range(50, 80))}]
    res = run(reqs)
    assert res["variants"]["loo"]["code"]["hits"] == 0 and res["variants"]["insample"]["code"]["hits"] > 0
    assert res["verdict"] == "CLOSE_COPY_PATH"
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results"); ap.add_argument("--prompts"); ap.add_argument("--tokens")
    ap.add_argument("--tokenize-bin"); ap.add_argument("--model"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.out or not (a.tokens or (a.results and a.prompts and a.tokenize_bin and a.model)):
        ap.error("need --out and either --tokens or --results/--prompts/--tokenize-bin/--model")
    os.makedirs(a.out, exist_ok=True)
    reqs = load_tokens(a)
    with open(os.path.join(a.out, "tokens.jsonl"), "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    res = run(reqs)
    with open(os.path.join(a.out, "RESULT.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"verdict": res["verdict"], **{v: {c: res["variants"][v][c]["hit_share"] for c in THRESH} for v in VARIANTS}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
