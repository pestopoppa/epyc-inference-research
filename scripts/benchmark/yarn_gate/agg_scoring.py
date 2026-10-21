"""YARN-AGG-1: scoring / aggregation on top of the per-arm record streams. Pure, stdlib only, no model.

* Oolong records are the OOL-1 adapter's (``oolong_adapter.make_record``): ``score``, ``attempted``,
  ``no_answer``, ``failure_class``, ``context_window_id``, ``usage``. This module bins them by REAL prompt
  tokens and adds the attempted / no-answer rates and the HoloBench-style constant-info control.
* ``net_paired_mcnemar`` implements the ratified "Net, paired" rule in ``arms.MARGINS`` (no absolute slack).
* ``exact_match`` is the LongBioBench grader.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from scripts.benchmark.yarn_gate import retokenize
from scripts.benchmark.yarn_gate.arms import MARGINS

ALPHA = MARGINS["alpha"]


def record_real_n(rec: Mapping[str, Any]) -> int | None:
    u = rec.get("usage") or {}
    for k in ("prompt_tokens", "prompt_n"):
        if isinstance(u.get(k), int):
            return u[k] + int(u.get("cache_n") or 0) if k == "prompt_n" else u[k]
    return rec.get("real_n") if isinstance(rec.get("real_n"), int) else None


def _boot_mean(groups: list[list[float]], rng: random.Random) -> float:
    pick = [groups[rng.randrange(len(groups))] for _ in groups]
    return sum(sum(g) for g in pick) / max(1, sum(len(g) for g in pick))


def cluster_ci(records: Sequence[Mapping[str, Any]], n_boot: int = 2000, seed: int = 0) -> dict:
    by: dict[Any, list[float]] = defaultdict(list)
    for r in records:
        by[r.get("context_window_id", r.get("id"))].append(float(r["score"]))
    g = list(by.values())
    if not g:
        return {"n_items": 0, "n_contexts": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    b = sorted(_boot_mean(g, rng) for _ in range(n_boot))
    n = sum(len(x) for x in g)
    return {"n_items": n, "n_contexts": len(g), "mean": sum(map(sum, g)) / n,
            "lo": b[int(0.025 * (n_boot - 1))], "hi": b[int(math.ceil(0.975 * (n_boot - 1)))]}


def bucket_stats(rs: Sequence[Mapping[str, Any]]) -> dict:
    n = len(rs)
    return {"n": n, "attempted_rate": sum(bool(r.get("attempted")) for r in rs) / n if n else None,
            "no_answer_rate": sum(bool(r.get("no_answer")) for r in rs) / n if n else None,
            "ci": cluster_ci(rs)}


def oolong_by_real_length(records: Iterable[Mapping[str, Any]]) -> dict:
    """Per (task_group, REAL-length bin) stats. Records without a measured real_n are listed, not binned."""
    cells: dict[tuple, list] = defaultdict(list)
    unmeasured = []
    for r in records:
        if r.get("labels_given"):
            continue
        n = record_real_n(r)
        if n is None:
            unmeasured.append(r.get("id"))
            continue
        cells[(r.get("task_group"), retokenize.bin_label(n))].append({**r, "_zone": retokenize.label_zone(n, source="server_timings")})
    out = {}
    for (tg, b), rs in sorted(cells.items(), key=lambda kv: (str(kv[0][0]), int(kv[0][1].split("-")[0]))):
        zones = {r["_zone"] for r in rs}
        out[f"{tg}|{b}"] = {**bucket_stats(rs), "zone": zones.pop() if len(zones) == 1 else "mixed"}
    return {"cells": out, "unmeasured_ids": unmeasured}


def info_amount_control(records: Iterable[Mapping[str, Any]], num_labels_by_id: Mapping[Any, int],
                        band_width: int = 0) -> dict:
    """Constant-relevant-amount control. Score by real-length bin restricted to the num_labels value(s)
    present at EVERY length bin; ``band_width`` widens the match (0 = exact num_labels)."""
    by_len: dict[str, list] = defaultdict(list)
    for r in records:
        n = record_real_n(r)
        nl = num_labels_by_id.get(r.get("id"))
        if r.get("labels_given") or n is None or nl is None:
            continue
        by_len[retokenize.bin_label(n)].append((nl, r))
    if len(by_len) < 2:
        return {"available": False, "reason": "fewer than 2 real-length bins"}
    common = None
    for rs in by_len.values():
        s = {nl // (band_width + 1) for nl, _ in rs}
        common = s if common is None else common & s
    if not common:
        return {"available": False, "reason": "no num_labels band common to all lengths: drop NOT attributable"}
    res = {"available": True, "bands": sorted(common), "band_width": band_width, "by_length": {}}
    for b in sorted(by_len, key=lambda x: int(x.split("-")[0])):
        sel = [r for nl, r in by_len[b] if nl // (band_width + 1) in common]
        res["by_length"][b] = bucket_stats(sel)
    return res


def _binom_sf(b: int, n: int) -> float:
    """P(X >= b) for X ~ Binomial(n, 0.5), exact."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def net_paired_mcnemar(a0: Mapping[Any, float], a1: Mapping[Any, float], alpha: float = ALPHA,
                       right_at: float = 1.0) -> dict:
    """Net, paired: A1 FAILS iff the one-sided exact McNemar p (b = A0 right/A1 wrong discordants) <= alpha.

    Pairs are the item ids present in both arms. Also returns the point estimate (mean paired score
    difference, b, c). No absolute slack: a tiny but significant net loss FAILS; a large nonsignificant one
    does not, so the verdict string is always accompanied by the point estimate and the power note."""
    ids = sorted(set(a0) & set(a1), key=str)
    b = sum(1 for i in ids if a0[i] >= right_at and a1[i] < right_at)
    c = sum(1 for i in ids if a0[i] < right_at and a1[i] >= right_at)
    p = _binom_sf(b, b + c)
    n = len(ids)
    diff = sum(a1[i] - a0[i] for i in ids) / n if n else None
    return {"verdict": "FAIL" if (b + c) > 0 and p <= alpha else "NOT-FAIL", "n_pairs": n, "b_a0right_a1wrong": b,
            "c_a0wrong_a1right": c, "n_discordant": b + c, "p_one_sided": p, "alpha": alpha,
            "mean_score_diff_a1_minus_a0": diff, "rule": "Net, paired (one-sided exact McNemar), no absolute slack"}


_WS = re.compile(r"\s+")


def exact_match(pred: str, gold: str) -> bool:
    norm = lambda s: _WS.sub(" ", str(s).strip().lower()).strip(" .")
    return norm(pred) == norm(gold)
