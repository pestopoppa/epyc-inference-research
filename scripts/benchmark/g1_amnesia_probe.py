#!/usr/bin/env python3
"""G1 — CoT-SFT amnesia probe (RULER S-NIAH-2) on the production frontdoor artifact.

Composition per handoff multiscreen-attention-evaluation.md (G1, 2026-08-22 spec):
  S-NIAH-2 = NeedleAdapter's essay haystack (Paul Graham corpus, ctx_len*4 chars,
             needle inserted at depth) x RULER's numeric needle (single 7-digit
             magic value, exact match on that value).

Grid: 4K / 32K / 64K / 128K x depths {0.1, 0.25, 0.5, 0.75, 0.9} x 10 samples
      = 50 trials per length, 200 total. Persist PER QUESTION.

Server contract (dedicated instance, free 83xx port):
  -np 1 -c 262144 --spec-type none --reasoning off terse.jinja -ub 8192
  --flash-attn on --mlock --no-mmap --device none
Request-level: cache_prompt=false, temperature 0, seed 42.

KV control: full grid at q8_0/q8_0, then 4K + shortest-drop length (or 128K if
no drop) at f16/f16.

Gate: >= 10pp below 4K control at any of 32K/64K/128K, paired 95% bootstrap CI
over needle placements excluding zero, surviving the f16-KV control.

Era E9 claim grammar: this script is an instrument; every persisted trial is a
drain point. Score = exact string match of the magic value in the generated text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any


ESSAYS_DIR = Path("/mnt/raid0/llm/data/eval/needle/repo/needlehaystack/PaulGrahamEssays")
CONTEXT_LENGTHS = [4096, 32768, 65536, 131072]
# 256K arm (INF-42 G1 decisive arm, 2026-10-23). Appended AFTER the base lengths so build_grid() draws the
# rng stream for the 200 base trials first: their keys/values/prompts are identical to the 2026-08-25..27
# run (test_base_grid_unchanged_by_extra_length). 262144*4 chars = ~224K tokens at 4 chars/token for this
# corpus, inside -c 262144. The essay corpus is 644K chars, so make_trial() repeats it (haystack*repeats).
EXTRA_LENGTHS = [262144]
ALL_LENGTHS = CONTEXT_LENGTHS + EXTRA_LENGTHS
DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]
SAMPLES = 10
SEED = 42
CHARS_PER_TOKEN = 4

# Canonical RULER S-NIAH-2 needle + template (essay haystack + numbers needle),
# from the RULER repo on disk (scripts/data/synthetic/constants.py 'niah' entry,
# singularised per niah.py for num_needle_q*num_needle_v == 1). The answer
# prefix is NOT part of the input prompt — RULER stores it separately as an
# eval-time extraction anchor (niah.py:277-279).
NEEDLE_TEMPLATE = "One of the special magic numbers for {key} is: {value}."
SINGLE_TEMPLATE = (
    "A special magic number is hidden within the following text. Make sure to memorize it. "
    "I will quiz you about the magic number afterwards.\n{context}\n"
    "What is the special magic number for {query} mentioned in the provided text?"
)


def load_essays() -> str:
    texts = []
    for f in sorted(ESSAYS_DIR.glob("*.txt")):
        texts.append(f.read_text(errors="replace"))
    return re.sub(r"\s+", " ", "\n\n".join(texts))


def make_magic_value(rng: random.Random) -> str:
    # canonical RULER magic number: 7 digits (generate_random_number in niah.py)
    return str(rng.randint(1_000_000, 9_999_999))


def make_trial(
    haystack_full: str,
    ctx_len: int,
    depth: float,
    rng: random.Random,
) -> dict[str, Any]:
    """One S-NIAH-2 trial: essay haystack truncated to ctx_len*4 chars, magic
    number needle inserted at depth, canonical single-needle question."""
    target_chars = ctx_len * CHARS_PER_TOKEN
    if target_chars > len(haystack_full):
        repeats = (target_chars + len(haystack_full) - 1) // len(haystack_full)
        haystack = (haystack_full * repeats)[:target_chars]
    else:
        haystack = haystack_full[:target_chars]

    key = make_magic_value(rng)
    value = make_magic_value(rng)
    # Guard: the magic value must not already occur in the haystack (collision
    # would make exact-match vacuous). 7-digit strings are astronomically rare
    # in the essay corpus; re-roll if it happens.
    guard = 0
    while value in haystack and guard < 50:
        value = make_magic_value(rng)
        guard += 1
    needle = NEEDLE_TEMPLATE.format(key=key, value=value)

    insert_pos = int(len(haystack) * depth)
    # snap to nearest whitespace boundary so the needle is not mid-word
    m = re.search(r"\s", haystack[insert_pos:])
    if m:
        insert_pos += m.start() + 1
    text_with_needle = haystack[:insert_pos] + needle + " " + haystack[insert_pos:]

    prompt = SINGLE_TEMPLATE.format(context=text_with_needle, query=key)
    return {
        "context_length": ctx_len,
        "depth": depth,
        "needle_key": key,
        "expected": value,
        "prompt": prompt,
        "prompt_chars": len(prompt),
    }


def score_response(text: str, expected: str) -> bool:
    return expected in text


def build_grid() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    haystack_full = load_essays()
    trials = []
    for ctx_len in ALL_LENGTHS:
        for depth in DEPTHS:
            for s in range(SAMPLES):
                t = make_trial(haystack_full, ctx_len, depth, rng)
                t["sample"] = s
                t["trial_id"] = f"l{ctx_len}_d{int(depth*100):02d}_s{s:02d}"
                trials.append(t)
    return trials


def call_server(
    port: int,
    prompt: str,
    *,
    max_tokens: int = 128,
    timeout: float = 7200.0,
) -> tuple[str, dict[str, Any]]:
    import httpx  # lazy: the unit tests and --summarize run without httpx installed

    t0 = time.monotonic()
    resp = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    elapsed = time.monotonic() - t0
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = dict(data.get("usage", {}))
    usage["elapsed_s"] = round(elapsed, 2)
    return text, usage


def bootstrap_paired_ci(
    control_scores: dict[float, float],
    long_scores: dict[float, float],
    n_iter: int = 100_000,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired bootstrap over needle placements (depths), excluding placements
    where the 4K control is zero (a placement missed even at short context
    cannot evidence a length-attributable drop) or missing on either side."""
    depths = [
        d for d in control_scores
        if d in long_scores
        and control_scores[d] is not None and long_scores[d] is not None
        and control_scores[d] > 0
    ]
    if not depths:
        return {"n_placements": 0}
    drops = [control_scores[d] - long_scores[d] for d in depths]
    rng = random.Random(seed)
    n = len(drops)
    means = []
    for _ in range(n_iter):
        sample = [drops[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "n_placements": n,
        "mean_drop_pp": round(sum(drops) / n * 100.0, 2),
        "ci95_low_pp": round(means[int(0.025 * n_iter)] * 100.0, 2),
        "ci95_high_pp": round(means[int(0.975 * n_iter)] * 100.0, 2),
    }


def analyze(trials: list[dict[str, Any]], out: Path | None = None) -> dict[str, Any]:
    """Per-length / per-depth recall and the paired bootstrap vs the 4K control.

    A trial with ``score is None`` (server died mid-request; persisted as a drain point) is UNSCORED:
    it is counted under ``unscored`` and is NOT a miss. (The 2026-08-27 summary counted the 36 unscored
    128K trials as misses and reported 14/50 = 28%; the honest figure was 14/14 scored.)
    """
    by_length: dict[int, list[dict]] = {}
    for t in trials:
        by_length.setdefault(int(t["context_length"]), []).append(t)

    report: dict[str, Any] = {"per_length": {}, "per_depth": {}, "bootstrap": {}}
    for L in ALL_LENGTHS:
        ts = by_length.get(L, [])
        if not ts:
            continue
        scored = [t for t in ts if t.get("score") is not None]
        unscored = len(ts) - len(scored)
        if not scored:
            report["per_length"][str(L)] = {"trials": 0, "hits": 0, "accuracy": None, "accuracy_pct": None,
                                            "unscored": unscored}
            continue
        n = len(scored)
        hits = sum(1 for t in scored if t["score"])
        backends = sorted({str(t.get("backend", "unrecorded")) for t in scored})
        report["per_length"][str(L)] = {
            "trials": n,
            "hits": hits,
            "accuracy": round(hits / n, 4),
            "accuracy_pct": round(100.0 * hits / n, 1),
            "unscored": unscored,
            "backends": backends,
        }

    for L in report["per_length"]:
        report["per_depth"][L] = {}
        for d in DEPTHS:
            ts = [t for t in by_length[int(L)] if t["depth"] == d and t.get("score") is not None]
            if not ts:
                continue
            hits = sum(1 for t in ts if t["score"])
            report["per_depth"][L][str(d)] = {
                "trials": len(ts),
                "hits": hits,
                "accuracy_pct": round(100.0 * hits / len(ts), 1),
            }

    ctrl_depths = report["per_depth"].get("4096", {})
    ctrl = {d: ctrl_depths[str(d)]["accuracy_pct"] / 100.0 for d in DEPTHS if str(d) in ctrl_depths}
    for L in (32768, 65536, 131072, 262144):
        if str(L) not in report["per_depth"] or not ctrl:
            continue
        long = {d: v["accuracy_pct"] / 100.0 for d in DEPTHS
                if (v := report["per_depth"][str(L)].get(str(d))) is not None}
        report["bootstrap"][str(L)] = bootstrap_paired_ci(ctrl, long)
        # 256K is a small DIAGNOSTIC arm (5-10 trials, operator 2026-08-27): record n so a reader cannot
        # mistake its interval for a 50-trial one.
        report["bootstrap"][str(L)]["n_trials_long"] = report["per_length"][str(L)]["trials"]
    return report


RC_DONE = 0
RC_SERVER_ERRORS = 3
RC_PENDING_MAX = 11      # stopped at --max-requests with work pending (caller restarts the server and calls again)
RC_PENDING_WINDOW = 12   # stopped at --deadline / --stopfile / not enough window for the next trial


def select_trials(trials, lengths=None, depths=None, samples_per_cell=None):
    """Filter the grid. ``samples_per_cell`` keeps samples 0..k-1 of every (length, depth) cell, so the
    256K diagnostic is a prefix of the full 256K grid and can be extended later without redoing trials."""
    out = []
    for t in trials:
        if lengths and t["context_length"] not in set(lengths):
            continue
        if depths and t["depth"] not in set(depths):
            continue
        if samples_per_cell is not None and t["sample"] >= samples_per_cell:
            continue
        out.append(t)
    return out


def is_pending(trials_dir: Path, t) -> bool:
    f = trials_dir / f"trial_{t['trial_id']}.json"
    if not f.exists():
        return True
    try:
        return json.loads(f.read_text()).get("score") is None
    except (OSError, ValueError):
        return True


def claim_trial(trials_dir: Path, trial_id: str, ttl_s: float, owner: str) -> bool:
    """O_EXCL claim so a CPU runner and a GPU runner never run the same trial. A claim older than ttl_s
    is stale (its owner died) and is taken over."""
    c = trials_dir / f".claim_{trial_id}"
    for _ in range(2):
        try:
            fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                if time.time() - c.stat().st_mtime > ttl_s:
                    c.unlink()
                    continue
            except FileNotFoundError:
                continue
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{owner} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        return True
    return False


def run_trials(trials, trials_dir: Path, call_fn, *, backend="unrecorded", meta=None, max_requests=None,
               deadline=None, stopfile=None, est_trial_s=0.0, abort_after=2, claim_ttl=4 * 3600.0,
               owner="g1", now=time.time, log=print):
    """Run pending trials; persist EVERY outcome as a drain point; return an RC_* code.

    Resume: a trial file with a non-null score is skipped; null-score files are re-run. Connection-level
    failures abort after ``abort_after`` consecutive errors (rc 3) instead of burning the whole grid into
    null-score files. No new trial starts after ``deadline`` (epoch s) or when ``stopfile`` exists, and
    none starts unless the remaining window covers 1.25x the slowest same-length trial seen so far
    (``est_trial_s`` before one has been seen).
    """
    trials_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    requests = 0
    consecutive_errors = 0
    slowest: dict[int, float] = {}
    pending_left = False
    for i, t in enumerate(trials):
        fname = trials_dir / f"trial_{t['trial_id']}.json"
        if not is_pending(trials_dir, t):
            log(f"[{i+1}/{len(trials)}] {t['trial_id']} RESUME (already scored)")
            continue
        if stopfile and os.path.exists(stopfile):
            log(f"stopfile present: not starting {t['trial_id']}")
            return RC_PENDING_WINDOW
        if max_requests is not None and requests >= max_requests:
            return RC_PENDING_MAX
        if deadline is not None:
            need = 1.25 * slowest.get(t["context_length"], est_trial_s)
            if now() + need > deadline:
                log(f"window too short for {t['trial_id']} (need ~{need:.0f}s, "
                    f"{deadline - now():.0f}s left): stopping")
                return RC_PENDING_WINDOW
        if not claim_trial(trials_dir, t["trial_id"], claim_ttl, owner):
            log(f"[{i+1}/{len(trials)}] {t['trial_id']} CLAIMED by another runner: skipping")
            pending_left = True
            continue
        claim = trials_dir / f".claim_{t['trial_id']}"
        try:
            requests += 1
            rec = dict(t)
            rec["backend"] = backend
            rec.update(meta)
            try:
                text, usage = call_fn(t["prompt"])
            except Exception as e:  # drain point: persist the failure, never lose the trial
                rec["response"] = None
                rec["usage"] = {"error": str(e)}
                rec["score"] = None
                rec["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                fname.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
                consecutive_errors += 1
                log(f"[{i+1}/{len(trials)}] {t['trial_id']} ERROR: {e}")
                if consecutive_errors >= abort_after:
                    log(f"{consecutive_errors} consecutive errors: server presumed dead, aborting (rc {RC_SERVER_ERRORS})")
                    return RC_SERVER_ERRORS
                continue
            consecutive_errors = 0
            rec["response"] = text
            rec["usage"] = usage
            rec["score"] = score_response(text, t["expected"])
            rec["scored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            tmp = fname.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
            os.replace(tmp, fname)
            slowest[t["context_length"]] = max(slowest.get(t["context_length"], 0.0), float(usage.get("elapsed_s", 0.0)))
            log(f"[{i+1}/{len(trials)}] {t['trial_id']} len={t['context_length']} depth={t['depth']:.2f} "
                f"score={int(rec['score'])} pt={usage.get('prompt_tokens')} elapsed={usage.get('elapsed_s')}s "
                f"backend={backend}")
        finally:
            try:
                claim.unlink()
            except FileNotFoundError:
                pass
    return RC_PENDING_MAX if pending_left else RC_DONE


def write_summary(out_dir: Path) -> dict[str, Any]:
    trials_dir = out_dir / "trials"
    files = sorted(trials_dir.glob("trial_*.json"))
    report = analyze([json.loads(f.read_text()) for f in files])
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    (out_dir / "SHA256SUMS").write_text(
        "\n".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("/mnt/raid0/llm/epyc-inference-research/data/frontdoor-amnesia-g1-2026-08-25"))
    ap.add_argument("--tag", default="q8")
    ap.add_argument("--lengths", type=int, nargs="*", default=None,
                    help="restrict grid lengths (f16 control, 256K diagnostic). Default: the 4 base lengths")
    ap.add_argument("--depths", type=float, nargs="*", default=None, help="restrict needle depths")
    ap.add_argument("--samples-per-cell", type=int, default=None,
                    help="keep samples 0..k-1 of each (length, depth) cell (256K diagnostic: 3)")
    ap.add_argument("--backend", default="unrecorded", help="cpu|gpu tag stored in every record and the summary")
    ap.add_argument("--meta", action="append", default=[], help="KEY=VALUE stored in every record (build, kv, ...)")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="stop after this many requests (rc 11 if work remains); the GPU job restarts the server in between")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: start no trial that cannot finish before it")
    ap.add_argument("--stopfile", default=None)
    ap.add_argument("--est-trial-s", type=float, default=0.0, help="window estimate for a trial before one has been timed")
    ap.add_argument("--abort-after", type=int, default=2, help="consecutive request errors before rc 3")
    ap.add_argument("--claim-ttl", type=float, default=4 * 3600.0)
    ap.add_argument("--pending", action="store_true", help="print the number of pending trials and exit (rc 0 none, 11 some)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze-only", type=Path, default=None, help="path to trials dir for analysis")
    ap.add_argument("--summarize", type=Path, default=None, help="consolidate persisted trials + write summary/SHA256SUMS")
    args = ap.parse_args()

    if args.analyze_only:
        trials = [json.loads(f.read_text()) for f in sorted(args.analyze_only.glob("trial_*.json"))]
        print(json.dumps(analyze(trials), indent=2))
        return 0

    if args.summarize:
        report = write_summary(args.summarize)
        print(f"summarized -> {args.summarize}")
        print(json.dumps(report, indent=2))
        return 0

    lengths = args.lengths or CONTEXT_LENGTHS
    trials = select_trials(build_grid(), lengths, args.depths, args.samples_per_cell)
    out_dir = args.out / f"{args.tag}"
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    if args.pending:
        n = sum(1 for t in trials if is_pending(trials_dir, t))
        print(n)
        return RC_PENDING_MAX if n else RC_DONE

    if args.dry_run:
        print(f"dry-run: {len(trials)} trials -> {trials_dir}; pending {sum(1 for t in trials if is_pending(trials_dir, t))}")
        for t in trials[:2]:
            print(t["trial_id"], t["context_length"], t["depth"], "prompt_chars", t["prompt_chars"])
        return 0

    if args.port is None:
        ap.error("--port is required to run trials")
    meta = dict(kv.split("=", 1) for kv in args.meta)
    rc = run_trials(trials, trials_dir, lambda prompt: call_server(args.port, prompt),
                    backend=args.backend, meta=meta, max_requests=args.max_requests, deadline=args.deadline,
                    stopfile=args.stopfile, est_trial_s=args.est_trial_s, abort_after=args.abort_after,
                    claim_ttl=args.claim_ttl, owner=f"{args.backend}:{os.getpid()}")
    report = write_summary(out_dir)
    print(json.dumps(report, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
