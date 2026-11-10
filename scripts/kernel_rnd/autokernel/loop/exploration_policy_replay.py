#!/usr/bin/env python3
"""AK-WM-3: offline exploration-policy replay over recorded proposal archives (observe-only, stdlib only).

Replays alternative candidate-ordering and stop policies over the measured rows an AutoKernel
`experiments.db` already holds. It never opens a store for write (`mode=ro`), never reads the payload
column, never calls a model, never touches the live loop, and every report states
`observe_only=true, live_policy_changed=false`.

Objective (App. B.2 shape): `J = auc - lambda * parallel_penalty`.
  auc = sum over steps of best-so-far effect_fraction -- the SAME definition as
        `process_metrics.calculate` (`auc_best_so_far_effect_over_steps`); a stopped run holds its best to the
        common horizon so a stop policy is charged for the endpoint it gives up (AUC-frontier, not endpoint only).
  parallel_penalty = wasted parallel slots / n: with width W, each batch of W picks sees ONE prefix; a slot is
        wasted when it repeats an arm already picked in the same batch. W=1 -> 0. NOTE: no in-repo
        implementation of `parallel_penalty` exists (only the AUC half is implemented); this definition is the
        operational one chosen here and is reported with every result.

Prefix-only contract: a policy is `choose(prefix, candidates) -> index`. `prefix` holds ONLY the outcomes of rows
already selected before the current batch; each candidate exposes only pre-outcome features (`arm`, `pos`
= recorded order, `mechanism_id`). The harness strips outcomes from candidates; a test proves a policy that tries
to read one gets a KeyError.

Counterfactual validity: reordering a recorded pool assumes a row's effect is independent of when it is tried.
Rows were measured against different champion heads/epochs, so this is an approximation, stated in every report;
the random-order control gives the null distribution each policy is ranked against.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Callable, Sequence

SCHEMA = "epyc.autokernel.exploration_policy_replay.v1"
MEASURED = frozenset({"kept", "keep_candidate", "measured_null", "regression", "runtime_observed"})
DEFAULT_LAMBDA = 0.0


# ------------------------------------------------------------------ loading (read-only)
def load_pool(db: Path, arm_field: str = "target_symbol") -> list[dict]:
    """Measured rows with a finite numeric effect, chronological. Scalar columns only (never `payload`)."""
    con = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT attempt_id,recorded_at,campaign_id,mechanism_id,target_symbol,target_surface,status,"
            "effect_fraction FROM experiments ORDER BY recorded_at,rowid").fetchall()
    finally:
        con.close()
    pool = []
    for aid, at, camp, mech, sym, surf, st, eff in rows:
        if st not in MEASURED or not isinstance(eff, (int, float)) or eff != eff:
            continue
        arm = {"target_symbol": sym, "target_surface": surf, "mechanism_id": mech}[arm_field] or "?"
        pool.append({"id": aid, "campaign": camp, "mechanism_id": mech or "?", "arm": arm,
                     "status": st, "effect": float(eff), "pos": len(pool)})
    return pool


# ------------------------------------------------------------------ harness
class _Cand(dict):
    """Pre-outcome view of a pool row: no `effect`/`status` key exists, so a read raises KeyError."""


def _view(row: dict) -> _Cand:
    return _Cand(pos=row["pos"], arm=row["arm"], mechanism_id=row["mechanism_id"])


def replay_order(pool: Sequence[dict], choose: Callable, width: int = 1, stop: Callable | None = None) -> dict:
    """Run one policy to exhaustion (or until `stop(prefix)` is true). Returns the picked order + metrics."""
    remaining = list(pool)
    prefix: list[dict] = []          # outcomes of rows picked in COMPLETED batches only
    picked: list[dict] = []
    wasted = 0
    stopped_at = None
    while remaining:
        if stop is not None and stop(prefix):
            stopped_at = len(picked)
            break
        batch, batch_arms = [], set()
        for _ in range(min(width, len(remaining))):
            cands = [_view(r) for r in remaining]
            i = choose(prefix, cands)
            row = remaining.pop(i)
            if row["arm"] in batch_arms:
                wasted += 1
            batch_arms.add(row["arm"])
            batch.append(row)
        picked.extend(batch)
        prefix.extend({"arm": r["arm"], "effect": r["effect"], "pos": r["pos"]} for r in batch)
    return {"picked": picked, "wasted": wasted, "stopped_at": stopped_at}


def score(picked: Sequence[dict], horizon: int, wasted: int = 0, lam: float = DEFAULT_LAMBDA) -> dict:
    best, auc, curve = 0.0, 0.0, []
    first_imp = None
    for step in range(1, horizon + 1):
        if step <= len(picked):
            e = picked[step - 1]["effect"]
            best = max(best, e)
            if first_imp is None and e > 0:
                first_imp = step
        auc += best                      # held after a stop: the AUC-frontier charges the abandoned tail
        curve.append(best)
    pen = wasted / max(1, len(picked))
    return {"steps_used": len(picked), "auc": auc, "endpoint_best": best, "first_improvement_step": first_imp,
            "parallel_penalty": pen, "objective": auc - lam * pen,
            "frontier": {str(k): curve[min(k, horizon) - 1] for k in (5, 10, 20, 40, 80) if horizon >= 1}}


# ------------------------------------------------------------------ policies (prefix-only)
def p_recorded(prefix, cands):
    return min(range(len(cands)), key=lambda i: cands[i]["pos"])


def make_random(rng: random.Random):
    return lambda prefix, cands: rng.randrange(len(cands))


def p_diverse(prefix, cands):
    """Least-tried arm first (ties: recorded order)."""
    seen: dict = {}
    for r in prefix:
        seen[r["arm"]] = seen.get(r["arm"], 0) + 1
    return min(range(len(cands)), key=lambda i: (seen.get(cands[i]["arm"], 0), cands[i]["pos"]))


def make_ucb(c: float = 0.05):
    """UCB over arms: unseen arms score +inf (recorded order); seen: mean(best effect) + c*sqrt(ln t / n)."""
    import math

    def choose(prefix, cands):
        n: dict = {}
        s: dict = {}
        for r in prefix:
            n[r["arm"]] = n.get(r["arm"], 0) + 1
            s[r["arm"]] = s.get(r["arm"], 0.0) + r["effect"]
        t = max(1, len(prefix))

        def key(i):
            a = cands[i]["arm"]
            if a not in n:
                return (float("inf"), -cands[i]["pos"])
            return (s[a] / n[a] + c * math.sqrt(math.log(t + 1) / n[a]), -cands[i]["pos"])
        return max(range(len(cands)), key=key)
    return choose


def p_exploit(prefix, cands):
    """Greedy: arm with the best observed single effect so far; unseen arms last-resort-explored after."""
    best: dict = {}
    for r in prefix:
        best[r["arm"]] = max(best.get(r["arm"], float("-inf")), r["effect"])
    return max(range(len(cands)), key=lambda i: (best.get(cands[i]["arm"], 0.0), -cands[i]["pos"]))


POLICIES = {"recorded": lambda rng: p_recorded, "diverse_arm": lambda rng: p_diverse,
            "ucb_arm": lambda rng: make_ucb(), "exploit_arm": lambda rng: p_exploit}


def make_stop(k: int):
    """Plateau stop: no new best in the last k selected rows."""
    def stop(prefix):
        if len(prefix) < k:
            return False
        best, last = float("-inf"), -1
        for i, r in enumerate(prefix):
            if r["effect"] > best:
                best, last = r["effect"], i
        return len(prefix) - 1 - last >= k
    return stop


# ------------------------------------------------------------------ report
def _pct(sorted_vals, v):
    return sum(1 for x in sorted_vals if x < v) / len(sorted_vals) if sorted_vals else None


def replay_pool(pool: Sequence[dict], width: int = 1, lam: float = DEFAULT_LAMBDA, n_random: int = 200,
                seed: int = 0, stops: Sequence[int] = (5, 10, 20)) -> dict:
    n = len(pool)
    out = {"n_rows": n, "width": width, "lambda": lam, "policies": {}, "stop_policies": {}}
    if n == 0:
        return out
    rnd = []
    for i in range(n_random):
        rng = random.Random(f"{seed}:{i}")
        r = replay_order(pool, make_random(rng), width)
        rnd.append(score(r["picked"], n, r["wasted"], lam))
    aucs = sorted(x["auc"] for x in rnd)
    out["random_control"] = {"n": n_random, "auc_mean": sum(aucs) / len(aucs), "auc_p05": aucs[int(0.05 * (len(aucs) - 1))],
                             "auc_p95": aucs[int(0.95 * (len(aucs) - 1))],
                             "endpoint_best_mean": sum(x["endpoint_best"] for x in rnd) / len(rnd),
                             "objective_mean": sum(x["objective"] for x in rnd) / len(rnd)}
    for name, mk in POLICIES.items():
        r = replay_order(pool, mk(random.Random(seed)), width)
        sc = score(r["picked"], n, r["wasted"], lam)
        sc["auc_percentile_vs_random"] = _pct(aucs, sc["auc"])
        out["policies"][name] = sc
    base = out["policies"]["recorded"]
    for k in stops:
        r = replay_order(pool, p_recorded, width, stop=make_stop(k))
        sc = score(r["picked"], n, r["wasted"], lam)
        sc["steps_saved"] = n - sc["steps_used"]
        sc["endpoint_loss_vs_full_recorded"] = base["endpoint_best"] - sc["endpoint_best"]
        sc["auc_delta_vs_full_recorded"] = sc["auc"] - base["auc"]
        out["stop_policies"][f"plateau_k{k}"] = sc
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stores", nargs="+", type=Path)
    ap.add_argument("--arm", default="target_symbol", choices=("target_symbol", "target_surface", "mechanism_id"))
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rows", type=int, default=20)
    ap.add_argument("--pretty", action="store_true")
    a = ap.parse_args(argv)
    res = []
    for s in a.stores:
        pool = load_pool(s, a.arm)
        if len(pool) < a.min_rows:
            res.append({"path": str(s), "skipped": f"only {len(pool)} measured rows (< {a.min_rows})"})
            continue
        res.append({"path": str(s), "arm": a.arm, **replay_pool(pool, a.width, a.lam, a.random, a.seed)})
    print(json.dumps({"schema": SCHEMA, "observe_only": True, "live_policy_changed": False, "inference_executed": False,
                      "limitations": ["reorder assumes effect independent of try-order; rows span champion heads",
                                      "parallel_penalty is an operational definition (no in-repo App. B.2 implementation)",
                                      "AK-WM-2a matched archive is retired; inputs are raw experiments.db stores"],
                      "stores": res}, sort_keys=True, indent=2 if a.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
