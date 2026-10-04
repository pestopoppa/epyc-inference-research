"""Versioned token-stream DEGENERACY detector (loops, stuck tokens, garbage bytes).

It is NOT a coherence or correctness judge. Every verdict carries `semantic: "unchecked"`;
"is the answer right / is the text sensible" comes from tier 0 (paired identity), tier 1(a)
(ground-truth graders) or tier 2 (an injected judge) of the gate, never from here.

Two rule sets live side by side, so a verdict always names the rules that produced it:

  degeneracy.v2        `classify`. Moved verbatim in logic from
                       /mnt/raid0/llm/tmp/gpu-block-27b-20261003/degeneracy.py (`inf70-degeneracy.v2`,
                       2026-10-04, q38t7-rescore). The id is now `degeneracy.v2`; `lineage` names the
                       origin so a row scored by either is recognisably the same rules.
                       Classes OK / DEGENERATE / SHORT / EARLY-EOS / EMPTY / HTTP-ERROR.
                       Triggers (each alone is sufficient):
                         top    one token >= 25% of the stream                (stuck token)
                         run    >= 6 identical consecutive tokens              (stuck token)
                         loop   a window of L = max(3p, 48) tokens in which >= 90% of tokens equal the
                                token p positions back, for some period 2 <= p <= 256 (>= 3 near-copies)
                         words  whitespace words < 0.25 n                       (garbage)
                         ascii  ordinary-text char share < 0.85                 (garbage; ascii-only is
                                `review`, never a failure on its own)
                       Corroborating only:
                         uniq   uniq < uniq_floor(n) = 0.30 * (max(n,200)/200) ** -0.5 (length-aware);
                                counts ONLY with top >= 0.10 or run >= 3. Alone it never fires.
                       1..15-token outputs are SHORT (not a failure). EARLY-EOS is n == 0 with eos.
  inf70-degeneracy.v1  `classify_v1`. FROZEN for continuity (`v1_cls` on every v2 verdict). It is the
                       lib_gpublock.classify reading of the INF-70 classify.py. Known defects: fixed
                       `uniq < 0.35` is length-insensitive (every 1000-1500-token reasoning trace is
                       SALAD), uniq fires alone, chat "stop" on a 1-token answer is EARLY-EOS.

The token ids MUST be real (see tokens.py); the classifier itself does not know where they came from,
so the gate (gate.py) owns provenance and records it next to each verdict.
"""
from __future__ import annotations

import collections

V1_ID = "inf70-degeneracy.v1"
V2_ID = "degeneracy.v2"
V2_LINEAGE = "inf70-degeneracy.v2 (gpu-block-27b-20261003/degeneracy.py, 2026-10-04)"
MIN_N = 16
EOS_MAX_N = 4
TOP_MAX = 0.25
RUN_MAX = 6
WORDS_MIN = 0.25
ASCII_MIN = 0.85
LOOP_PMAX = 256
LOOP_MIN_LEN = 48
LOOP_COPIES = 3
LOOP_MATCH = 0.90
UNIQ_A, UNIQ_N0, UNIQ_EXP = 0.30, 200, -0.5
SOFT_TOP, SOFT_RUN = 0.10, 3
OK_PUNCT = " .,;:'\"-()!?\n"
FAIL_CLASSES = ("DEGENERATE", "EMPTY", "EARLY-EOS", "HTTP-ERROR")


def uniq_floor(n: int) -> float:
    """Length-aware floor for |set(ids)|/n, ~30% under the lower envelope of 34 Qwen-27B reasoning
    traces (0.43 @200, 0.27 @500, 0.19 @1000, 0.17 @1500): 0.30 @200, 0.134 @1000, 0.110 @1500."""
    return UNIQ_A * (max(n, UNIQ_N0) / UNIQ_N0) ** UNIQ_EXP


def _stats(text: str, toks: list[int]) -> dict:
    n = len(toks)
    c = collections.Counter(toks)
    run = best = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    ok_chars = sum(ch.isascii() and (ch.isalnum() or ch in OK_PUNCT) for ch in text) / max(1, len(text))
    return {"uniq": len(c) / n, "top": c.most_common(1)[0][1] / n, "run": best,
            "words": len(text.split()), "ascii_ok": ok_chars}


def loop_evidence(toks: list[int], pmax: int = LOOP_PMAX) -> dict:
    """Smallest period p (2..pmax) with a window of L = max(3p, 48) tokens where >= 90% of tokens
    equal the token p back. Returns {period, window, match, at_token} or {} when there is none."""
    n = len(toks)
    for p in range(2, min(pmax, n // LOOP_COPIES) + 1):
        L = max(LOOP_COPIES * p, LOOP_MIN_LEN)
        if L + p > n:
            break
        m = [1 if toks[i] == toks[i - p] else 0 for i in range(p, n)]
        s = sum(m[:L])
        best, at = s, 0
        for j in range(L, len(m)):
            s += m[j] - m[j - L]
            if s > best:
                best, at = s, j - L + 1
        if best >= LOOP_MATCH * L:
            return {"period": p, "window": L, "match": round(best / L, 3), "at_token": at + p}
    return {}


def classify_v1(text: str, toks: list[int], finish: str | None, http_ok: bool = True) -> dict:
    """FROZEN inf70-degeneracy.v1 (identical logic to lib_gpublock.classify)."""
    if not http_ok:
        return {"classifier": V1_ID, "cls": "HTTP-ERROR", "n": 0}
    n = len(toks)
    eos = finish in ("stop", "eos")
    st = {"classifier": V1_ID, "n": n, "finish": finish}
    if n == 0:
        return {"cls": "EARLY-EOS" if eos else "EMPTY", **st}
    if n <= EOS_MAX_N and eos:
        return {"cls": "EARLY-EOS", **st}
    if n < MIN_N:
        return {"cls": "EARLY-EOS" if eos else "SHORT", **st}
    s = _stats(text, toks)
    reasons = [r for r, bad in (("uniq", s["uniq"] < 0.35), ("top", s["top"] >= TOP_MAX), ("run", s["run"] >= RUN_MAX),
                                ("words", s["words"] < WORDS_MIN * n), ("ascii", s["ascii_ok"] < ASCII_MIN)) if bad]
    st.update({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}, salad_reasons=reasons)
    return {"cls": "SALAD" if reasons else "COHERENT", "review": reasons == ["ascii"], **st}


def _v2_reasons(n: int, uniq: float, top: float, run: int, words: int, ascii_ok: float,
                loop: dict | None) -> tuple[list[str], list[str]]:
    hard = [r for r, bad in (("top", top >= TOP_MAX), ("run", run >= RUN_MAX), ("loop", bool(loop)),
                             ("words", words < WORDS_MIN * n), ("ascii", ascii_ok < ASCII_MIN)) if bad]
    soft = []
    if uniq < uniq_floor(n):
        soft.append("uniq_low")
        if top >= SOFT_TOP or run >= SOFT_RUN:
            hard.append("uniq+stuck")
    return hard, soft


def _v2_short(n: int, finish: str | None, http_ok: bool) -> dict | None:
    base = {"classifier": V2_ID, "semantic": "unchecked"}
    if not http_ok:
        return {**base, "cls": "HTTP-ERROR", "n": 0}
    eos = finish in ("stop", "eos")
    if n == 0:
        return {**base, "cls": "EARLY-EOS" if eos else "EMPTY", "n": 0, "finish": finish}
    if n < MIN_N:
        return {**base, "cls": "SHORT", "n": n, "finish": finish, "eos_short": eos and n <= EOS_MAX_N}
    return None


def classify(text: str, toks: list[int], finish: str | None, http_ok: bool = True) -> dict:
    """degeneracy.v2 from the full text and its (real) token ids."""
    n = len(toks) if http_ok else 0
    early = _v2_short(n, finish, http_ok)
    v1 = classify_v1(text, toks, finish, http_ok)["cls"]
    if early:
        return {**early, "v1_cls": v1}
    s = _stats(text, toks)
    loop = loop_evidence(toks)
    hard, soft = _v2_reasons(n, s["uniq"], s["top"], s["run"], s["words"], s["ascii_ok"], loop)
    return {"classifier": V2_ID, "semantic": "unchecked", "cls": "DEGENERATE" if hard else "OK",
            "review": hard == ["ascii"], "reasons": hard, "soft": soft, "n": n, "finish": finish,
            "uniq": round(s["uniq"], 3), "uniq_floor": round(uniq_floor(n), 3), "top": round(s["top"], 3),
            "run": s["run"], "words": s["words"], "ascii_ok": round(s["ascii_ok"], 3), "loop": loop or None,
            "v1_cls": v1}


def classify_stats(st: dict) -> dict:
    """degeneracy.v2 from STORED v1 stats (n, uniq, top, run, words, ascii_ok, finish/cls), for
    re-scoring banked runs. The loop trigger needs token ids, so it is reported as unchecked."""
    n = st.get("n") or 0
    v1 = st.get("cls")
    if v1 == "HTTP-ERROR":
        return {"classifier": V2_ID, "semantic": "unchecked", "cls": "HTTP-ERROR", "n": 0, "v1_cls": v1}
    early = _v2_short(n, st.get("finish"), True)
    if early:
        return {**early, "v1_cls": v1, "loop": "unchecked"}
    hard, soft = _v2_reasons(n, st["uniq"], st["top"], st["run"], st["words"], st["ascii_ok"], None)
    return {"classifier": V2_ID, "semantic": "unchecked", "cls": "DEGENERATE" if hard else "OK",
            "review": hard == ["ascii"], "reasons": hard, "soft": soft, "n": n,
            "uniq": st["uniq"], "uniq_floor": round(uniq_floor(n), 3), "top": st["top"], "run": st["run"],
            "loop": "unchecked (no token ids stored)", "v1_cls": v1}


def severity(c: dict) -> int:
    """Order for PAIRED comparison on the same prompt: 0 OK/SHORT, 1 ascii-only review,
    2 DEGENERATE, 3 EARLY-EOS/EMPTY/HTTP-ERROR."""
    if c["cls"] in ("OK", "SHORT"):
        return 0
    if c["cls"] == "DEGENERATE":
        return 1 if c.get("review") else 2
    return 3


def fails(c: dict) -> bool:
    """Absolute reading: DEGENERATE (not ascii-only review), EMPTY, EARLY-EOS or HTTP-ERROR.
    The gate itself is paired and uses `severity`; this is kept for single-arm callers."""
    return c["cls"] in FAIL_CLASSES and not c.get("review")
