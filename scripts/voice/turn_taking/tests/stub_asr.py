"""Deterministic stand-in for ASR-B on the synthetic (tone-burst) fake corpus: words = energy bursts, labelled in order
with the clip's own text (user prefix/full, suffix, or assistant). Exercises the gate LOGIC without a model."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tt_qa_gates as Q  # noqa: E402


def bursts(x, thr=300, min_gap=0.05, sr=16000):
    a = np.abs(x.astype(np.float64))
    n = len(a) // 160
    env = a[: n * 160].reshape(n, 160).mean(axis=1)
    on = env > thr
    out, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (j + int(min_gap * 100) < n and on[j: j + int(min_gap * 100) + 1].any())):
                j += 1
            out.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1
    return out


class StubAsr:
    def words(self, x, ctx=None):
        c = ctx["clip"]
        kind = ctx["kind"]
        if kind == "assistant":
            ws = Q.norm_words(c["text"]["assistant"])
        elif kind == "suffix":
            full = Q.norm_words(c["text"]["user_full"])
            ws = full[len(Q.norm_words(c["text"]["user_spoken"])):]
        else:
            ws = Q.norm_words(c["text"]["user_full"])
        bs = bursts(x)
        words = [{"w": ws[i] if i < len(ws) else "extra", "t0": b[0], "t1": b[1], "p": 0.9} for i, b in enumerate(bs)]
        return {"words": words, "text": " ".join(w["w"] for w in words)}
