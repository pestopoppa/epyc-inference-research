"""Acceptance-normalized decode metrics for GPU MTP A/B rows (import, never copy).

GPU greedy output under MTP is nondeterministic, so acceptance (and tok/s) swings
per visit.  The stable cost is ms per VERIFY STEP:

    step_ms = predicted_ms / (predicted_n - draft_n_accepted)

(box GPU-MTP-ACCEPTANCE-METRIC).  Higher ratio = arm B faster (ratio = A/B).
Pure Python, no I/O except load_jsonl.
"""
import json
import math


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def step_ms(q):
    """ms per verify step from one llama-server timings dict, or None.

    Falls back to predicted_n*1000/tok_s when predicted_ms is absent.  Returns
    None (never 0/inf) when acceptance accounts for every token or inputs are
    missing, so callers drop the sample instead of poisoning a mean.
    """
    n, acc, ms = q.get("predicted_n"), q.get("draft_n_accepted"), q.get("predicted_ms")
    if ms is None and q.get("tok_s") and n:
        ms = n * 1000.0 / q["tok_s"]
    if ms is None or n is None or acc is None or n - acc <= 0:
        return None
    return ms / (n - acc)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def geomean(xs):
    xs = list(xs)
    if not xs or any(x is None or x <= 0 for x in xs):
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def acceptance_rate(q):
    d, a = q.get("draft_n"), q.get("draft_n_accepted")
    return a / d if d else None


def ratio_stats(a_ms, b_ms):
    """Paired-cell ratios A/B (>1 means B faster) from per-visit step ms lists.

    mean ratio (the legacy gate), median ratio and, when lengths match, the
    geomean of per-pair ratios in list order (ABBA pairing by visit index within
    arm).  Samples that are None/<=0 are dropped first.
    """
    A = [x for x in a_ms if x]
    B = [x for x in b_ms if x]
    if not A or not B:
        return None
    out = {"n_a": len(A), "n_b": len(B), "mean_ratio": mean(A) / mean(B),
           "median_ratio": median(A) / median(B), "mean_a": mean(A), "mean_b": mean(B)}
    out["pair_geomean_ratio"] = geomean([a / b for a, b in zip(A, B)]) if len(A) == len(B) else None
    return out


def cells_from_visits(visits, arm_a, arm_b, prompts, metric=step_ms):
    """visits: dicts with arm, valid, prompts{pid: timings}.  Returns
    {pid: {"a": [...], "b": [...], **ratio_stats}} in visit order."""
    cells = {}
    for pid in prompts:
        a = [metric(v["prompts"][pid]) for v in visits if v["arm"] == arm_a and v.get("valid", True) and pid in v["prompts"]]
        b = [metric(v["prompts"][pid]) for v in visits if v["arm"] == arm_b and v.get("valid", True) and pid in v["prompts"]]
        st = ratio_stats(a, b)
        if st:
            cells[pid] = {"a": [x for x in a if x], "b": [x for x in b if x], **st}
    return cells


def cells_geomean(cells, key="mean_ratio"):
    return geomean([c[key] for c in cells.values()])
