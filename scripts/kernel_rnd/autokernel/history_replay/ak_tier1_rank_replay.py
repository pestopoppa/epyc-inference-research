#!/usr/bin/env python3
"""AK-TIER-1: does the fast (screen) tier rank candidates like the full (confirmed) tier?

Offline replay, zero compute. Reads closed experiments stores read-only
(`mode=ro&immutable=1`) and/or an AK-HR-1 `history_records.jsonl`.

Join rule (strict): two rows are a screen/confirmed pair only when they carry the SAME
`candidate_diff_sha256` and both carry a numeric effect. Within a group, the earliest
row (recorded_at, then rowid/attempt_id) is the screen and the latest row is the
confirmed one; a group with fewer than two distinct attempts is not a pair.

Report: n, Spearman rho (average ranks), sign agreement, bootstrap CI for rho, and the
pairwise ordering-flip rate grouped by duration gap. Stop-on-answer: the CI excludes 0.7
in either direction; otherwise the verdict is UNDERDETERMINED, never a pass.

Usage:
    python3 ak_tier1_rank_replay.py [--records history_records.jsonl] [--root /mnt/raid0/llm/autokernel] [--out out.json]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

SCHEMA = "epyc.autokernel.tier1_rank_replay.v1"
THRESHOLD = 0.7
GAP_BUCKETS = ((0.0, 600.0), (600.0, 3600.0), (3600.0, float("inf")))


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def rows_from_records(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cand = r.get("candidate") or {}
        o = r.get("outcome") or {}
        out.append({"diff": cand.get("candidate_diff_sha256"), "effect": o.get("effect_fraction"),
                    "ts": r.get("recorded_at") or "", "key": str(r.get("attempt_id") or r.get("rowid")),
                    "window": o.get("measure_window_s_sum"), "status": r.get("status")})
    return out


def rows_from_stores(root):
    out = []
    for db in sorted(glob.glob(str(Path(root) / "campaigns" / "*" / "store*" / "experiments.db"))):
        con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
        try:
            for aid, ts, st, eff, payload in con.execute(
                    "select attempt_id,recorded_at,status,effect_fraction,payload from experiments"):
                try:
                    d = json.loads(payload)
                except (TypeError, ValueError):
                    continue
                out.append({"diff": d.get("candidate_diff_sha256"), "effect": eff, "ts": ts or "",
                            "key": aid, "window": None, "status": st})
        finally:
            con.close()
    return out


def build_pairs(rows):
    groups = defaultdict(dict)
    for r in rows:
        if r["diff"] and _num(r["effect"]):
            groups[r["diff"]][r["key"]] = r  # dedupe the same attempt seen in two sources
    pairs = []
    for diff, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        ordered = sorted(members.values(), key=lambda r: (r["ts"], r["key"]))
        s, c = ordered[0], ordered[-1]
        pairs.append({"diff": diff, "screen": float(s["effect"]), "confirmed": float(c["effect"]),
                      "screen_status": s["status"], "confirmed_status": c["status"],
                      "gap_s": _gap(s["ts"], c["ts"])})
    return pairs


def _gap(a, b):
    from datetime import datetime
    try:
        f = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in a else "%Y-%m-%dT%H:%M:%SZ"
        g = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in b else "%Y-%m-%dT%H:%M:%SZ"
        return abs((datetime.strptime(b, g) - datetime.strptime(a, f)).total_seconds())
    except ValueError:
        return None


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def bootstrap_ci(x, y, n_boot=2000, seed=20261106):
    rng = random.Random(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        r = spearman([x[i] for i in idx], [y[i] for i in idx])
        if r is not None:
            vals.append(r)
    if len(vals) < 20:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]]


def flip_rates(pairs):
    cells = {f"{lo:g}-{hi:g}s": [0, 0] for lo, hi in GAP_BUCKETS}
    cells["unknown"] = [0, 0]
    for a, b in itertools.combinations(pairs, 2):
        ds, dc = a["screen"] - b["screen"], a["confirmed"] - b["confirmed"]
        if ds == 0 or dc == 0:
            continue
        gaps = [p["gap_s"] for p in (a, b) if p["gap_s"] is not None]
        label = "unknown"
        if gaps:
            g = max(gaps)
            for lo, hi in GAP_BUCKETS:
                if lo <= g < hi:
                    label = f"{lo:g}-{hi:g}s"
        cells[label][1] += 1
        cells[label][0] += (ds > 0) != (dc > 0)
    return {k: {"flips": f, "pairs": n, "rate": (f / n if n else None)} for k, (f, n) in cells.items()}


def replay(rows, n_boot=2000):
    pairs = build_pairs(rows)
    x = [p["screen"] for p in pairs]
    y = [p["confirmed"] for p in pairs]
    rho = spearman(x, y)
    ci = bootstrap_ci(x, y, n_boot) if rho is not None else None
    judged = [(a, b) for a, b in zip(x, y) if a != 0 and b != 0]
    agree = sum((a > 0) == (b > 0) for a, b in judged)
    if ci is None:
        verdict = "UNDERDETERMINED: fewer than 3 pairs or degenerate ranks"
    elif ci[1] < THRESHOLD:
        verdict = f"FAST TIER DOES NOT RANK LIKE FULL TIER: CI upper {ci[1]:.3f} < {THRESHOLD}"
    elif ci[0] > THRESHOLD:
        verdict = f"FAST TIER RANKS LIKE FULL TIER: CI lower {ci[0]:.3f} > {THRESHOLD}"
    else:
        verdict = f"UNDERDETERMINED: CI [{ci[0]:.3f}, {ci[1]:.3f}] contains {THRESHOLD}"
    return {"schema": SCHEMA, "n_pairs": len(pairs), "spearman_rho": rho, "rho_ci95": ci,
            "sign_agreement": {"agree": agree, "judgeable": len(judged),
                               "rate": (agree / len(judged) if judged else None)},
            "flip_rate_by_duration_gap": flip_rates(pairs), "threshold": THRESHOLD,
            "stop_on_answer_met": bool(ci and (ci[1] < THRESHOLD or ci[0] > THRESHOLD)),
            "verdict": verdict, "pairs": pairs}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", default=[])
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for r in a.records:
        rows += rows_from_records(r)
    if a.root:
        rows += rows_from_stores(a.root)
    res = replay(rows)
    text = json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
