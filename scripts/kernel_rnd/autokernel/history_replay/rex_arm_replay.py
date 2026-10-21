#!/usr/bin/env python3
"""AK-RX-1: REx arm + controls in the AK-adaptive policy panel offline replay (CPU, stdlib only).

Arms, all replayed over ONE recorded proposal archive (no live flip, no model call):
  fixed        recorded order; each proposal followed by its recorded repairs until kept or chain end.
  rex_C{C}     Thompson sampling, Beta(1 + C*h, 1 + C*(1-h) + N), C in {5,10,20,50}. Actions: draw the next
               fresh proposal (scored with h=0, N=0, the REx "empty program" root) or refine a revealed
               non-kept node (h = its rung fraction, N = repair attempts so far on its path).
  iid          equal-call control: draw recorded nodes uniformly with replacement, no tree.
  split_R{R}   fixed-total-calls sweep: n_proposals = budget/(1+R) style, each proposal gets up to R repairs.

Normalized archive schema (JSONL, one object per node; see adapt_* for real sources):
  proposal_id (str, unique), parent_id (str|null; null = fresh proposal; else the node this repairs),
  rungs (list of {"name": str, "frac": float in [0,1]} ORDERED: compile, op-test cases, NMSE; a rung
  never reached is simply absent), repair_attempts (int >= 0), kept (bool: cleared the keep gate),
  call_cost (int >= 1, default 1).
h = mean of rung fracs over the fixed ladder length LADDER_LEN (absent rungs count 0).

AK-RX-1.1: `adapt_experiments_db_chains` (CLI `--experiments-db ... --repair-chains`) builds the parent links
from the per-patch-round rows the loop already persists; no loop, run.py flag or row-schema change is involved.

Replay limitation (stated in every report): an outcome can only come from a record that exists. A refinement
selected on a node with no recorded child costs one call, yields NO outcome, kills that node, and is counted
as an unobserved selection; coverage = observed / (observed + unobserved). Fixed-sweep repairs beyond the end
of a recorded chain are not charged and counted as unobserved repairs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

LADDER_LEN = 3
C_VALUES = (5, 10, 20, 50)
SPLIT_R = (0, 1, 2, 3)
SCHEMA = "epyc.autokernel.rex_arm_replay.v1"


# ---------------------------------------------------------------- archive
class Archive:
    def __init__(self, nodes):
        self.nodes = {}
        for n in nodes:
            if n["proposal_id"] in self.nodes:
                raise ValueError("duplicate proposal_id " + n["proposal_id"])
            self.nodes[n["proposal_id"]] = {
                "id": n["proposal_id"], "parent": n.get("parent_id"),
                "h": ladder_h(n.get("rungs") or []), "N": int(n.get("repair_attempts", 0)),
                "kept": bool(n.get("kept")), "cost": max(1, int(n.get("call_cost", 1)))}
        self.children = {k: [] for k in self.nodes}
        self.roots = []
        for k, n in self.nodes.items():
            p = n["parent"]
            if p is None:
                self.roots.append(k)
            elif p in self.nodes:
                self.children[p].append(k)
            else:
                raise ValueError(f"{k}: parent {p} not in archive")
        for k in self.children:
            self.children[k].sort(key=lambda c: (self.nodes[c]["N"], c))

    def subtree(self, root):
        out, st = [], [root]
        while st:
            k = st.pop()
            out.append(k)
            st.extend(self.children[k])
        return out

    def total_cost(self):
        return sum(n["cost"] for n in self.nodes.values())

    def has_tree(self):
        return any(self.children[k] for k in self.children)

    def resample(self, rng):
        """Bootstrap over proposal trees (root + all its repairs), with replacement."""
        out = []
        for i in range(len(self.roots)):
            r = self.roots[rng.randrange(len(self.roots))]
            for k in self.subtree(r):
                n = self.nodes[k]
                out.append({"proposal_id": f"{k}#{i}", "parent_id": None if n["parent"] is None else f"{n['parent']}#{i}",
                            "rungs": [{"name": "h", "frac": n["h"] * LADDER_LEN}], "repair_attempts": n["N"],
                            "kept": n["kept"], "call_cost": n["cost"]})
        return Archive(out)


def ladder_h(rungs):
    s = sum(max(0.0, min(1.0, float(r["frac"]))) for r in rungs[:LADDER_LEN])
    return s / LADDER_LEN


def load_jsonl(path):
    return Archive([json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()])


# ---------------------------------------------------------------- adapter (real records)
_PASS = re.compile(r"(\d+)/(\d+) tests passed")


def rungs_from_gates(gates):
    """experiments.db payload['gates'] -> ordered rungs [compile, op-test, nmse(reference_comparison)].
    op-test frac = sum(passed)/sum(total) parsed from 'N/M tests passed' details, else gate pass fraction."""
    comp = ref = None
    ok = tot = 0
    gp = gt = 0
    for g in gates:
        name = g.get("gate")
        if name == "compile":
            comp = 1.0 if g.get("passed") else 0.0
        elif name == "correctness":
            gt += 1
            gp += 1 if g.get("passed") else 0
            m = _PASS.search(g.get("detail") or "")
            if m:
                ok += int(m.group(1)); tot += int(m.group(2))
        elif name == "reference_comparison":
            ref = 1.0 if g.get("passed") else 0.0
    out = []
    if comp is not None:
        out.append({"name": "compile", "frac": comp})
    if gt:
        out.append({"name": "op_test", "frac": (ok / tot) if tot else gp / gt})
    if ref is not None:
        out.append({"name": "nmse_reference_comparison", "frac": ref})
    return out


def adapt_experiments_db(db_paths):
    """Real adapter. One experiments row with payload['gates'] -> one ROOT node (no parent: the store keeps one
    final row per attempt, not the intermediate repair rounds). repair_attempts = max(patch_round-1, 0);
    kept = status=='kept'. Rows without a 'gates' list are skipped (cannot supply h)."""
    nodes, skipped = [], 0
    for db in db_paths:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for aid, cid, st, p in con.execute("select attempt_id,campaign_id,status,payload from experiments"):
            try:
                j = json.loads(p)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not isinstance(j.get("gates"), list):
                skipped += 1
                continue
            pr = j.get("patch_round")
            nodes.append({"proposal_id": f"{Path(db).parent.parent.name}:{cid}:{aid}", "parent_id": None,
                          "rungs": rungs_from_gates(j["gates"]),
                          "repair_attempts": max((pr or 1) - 1, 0), "kept": st == "kept", "call_cost": 1})
        con.close()
    return nodes, skipped


# ---------------------------------------------------------------- adapter v2 (AK-RX-1.1: repair chains)
#: Statuses whose row IS one authoring round's own attempt (one author call set, one patch). Each patch round that
#: ends before the final outcome is persisted as its own row by `loop.iterate -> dispose -> record_abandoned`
#: (patch_rejected = critic:patch refusal, gate_refused = pre-build gate refusal), and the round that reaches a
#: measurement ends the iteration with its outcome row. Summary rows of an exhausted attempt
#: (patch_rounds_exhausted, scope_blocked, hypothesis_retired) re-state the last disposed round and are EXCLUDED
#: so no round is counted twice; instrument outcomes (measurement_invalid, bench_failed, no_op_build) say nothing
#: about the repair and are excluded too. Every excluded status is counted in the report.
ATTEMPT_STATUSES = frozenset({"patch_rejected", "gate_refused", "authoring_failed", "kept", "keep_candidate",
                              "measured_null", "regression", "interaction_regression"})


def op_test_counts(gates):
    """(passed, total) summed over 'N/M tests passed' details of correctness gates; (None, None) if none parse."""
    ok = tot = 0
    seen = False
    for g in gates or []:
        if g.get("gate") == "correctness":
            m = _PASS.search(g.get("detail") or "")
            if m:
                ok += int(m.group(1)); tot += int(m.group(2)); seen = True
    return (ok, tot) if seen else (None, None)


def author_cost(payload, basis):
    """(calls, decoded_tokens|None) spent authoring this round. basis 'unit' -> 1; 'author_calls' -> sum of panel members' `calls`
    (floor 1: a round with no panel record still cost one authoring call)."""
    if basis == "unit":
        return 1, None
    calls = tokens = 0
    for panel in payload.get("author_panels") or []:
        for m in (panel or {}).get("members") or []:
            calls += int(m.get("calls") or 0)
            tokens += int(m.get("decoded_tokens") or 0)
    return max(1, calls), (tokens or None)


def adapt_experiments_db_chains(db_paths, cost_basis="author_calls"):
    """Repair-chain adapter. Unlike `adapt_experiments_db` (one root per gated row) this links the per-patch-round
    rows the loop already persists. Chain key within one store: (branch_id, mechanism_id, hypothesis_round), rows
    in recorded_at order; a row with patch_round r>=1 is the child of the immediately preceding row of the same
    key iff that row's patch_round == r-1 (otherwise it starts a new chain: a restart, a resume, or a missing
    round -- never guessed across a gap). Node fields: proposal_id, parent_id, rungs (ordered compile, op_test,
    nmse), repair_attempts (= patch_round-1), kept (status=='kept'), call_cost (see `author_cost`), plus
    non-replay extras (status, patch_round, hypothesis_round, op_test_passed/total, author_decoded_tokens).
    Returns (nodes, stats). Read-only: stores are opened mode=ro."""
    nodes = []
    stats = {"stores": 0, "rows_total": 0, "rows_included": 0, "excluded_status_counts": {}, "chains": 0,
             "chain_length_counts": {}, "cost_basis": cost_basis}
    for db in db_paths:
        db = Path(db)
        stats["stores"] += 1
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("select attempt_id,recorded_at,status,payload from experiments "
                           "order by recorded_at,attempt_id").fetchall()
        con.close()
        prefix = f"{db.parent.parent.name}/{db.parent.name}"
        last = {}      # chain key -> (node_id, patch_round)
        chain_len = {}  # node_id of chain root -> length
        root_of = {}
        for aid, _at, st, p in rows:
            stats["rows_total"] += 1
            try:
                j = json.loads(p)
            except (TypeError, ValueError):
                j = {}
            r = j.get("patch_round")
            if st not in ATTEMPT_STATUSES or not isinstance(r, int) or r < 1:
                stats["excluded_status_counts"][st] = stats["excluded_status_counts"].get(st, 0) + 1
                continue
            nid = f"{prefix}:{aid}"
            key = (j.get("branch_id"), j.get("mechanism_id"), j.get("hypothesis_round") or 0)
            prev = last.get(key)
            parent = prev[0] if prev is not None and prev[1] == r - 1 else None
            gates = j.get("gates") if isinstance(j.get("gates"), list) else []
            ok, tot = op_test_counts(gates)
            cost, toks = author_cost(j, cost_basis)
            node = {"proposal_id": nid, "parent_id": parent, "rungs": rungs_from_gates(gates),
                    "repair_attempts": r - 1, "kept": st == "kept", "call_cost": cost,
                    "status": st, "patch_round": r, "hypothesis_round": key[2],
                    "op_test_passed": ok, "op_test_total": tot, "author_decoded_tokens": toks}
            nodes.append(node)
            last[key] = (nid, r)
            root = nid if parent is None else root_of[parent]
            root_of[nid] = root
            chain_len[root] = chain_len.get(root, 0) + 1
            stats["rows_included"] += 1
        stats["chains"] += len(chain_len)
        for n in chain_len.values():
            stats["chain_length_counts"][str(n)] = stats["chain_length_counts"].get(str(n), 0) + 1
    return nodes, stats


# ---------------------------------------------------------------- arms
class Trace:
    def __init__(self, budget):
        self.budget, self.spent, self.first_kept = budget, 0, None
        self.observed = self.unobserved = 0

    def pay(self, cost):
        if self.spent + cost > self.budget:
            return False
        self.spent += cost
        return True

    def observe(self, kept):
        self.observed += 1
        if kept and self.first_kept is None:
            self.first_kept = self.spent

    def curve(self):
        return [1 if (self.first_kept is not None and self.first_kept <= t) else 0 for t in range(1, self.budget + 1)]


def run_fixed(a, budget, rng, order=None):
    tr = Trace(budget)
    for r in (order or a.roots):
        k = r
        while k is not None:
            n = a.nodes[k]
            if not tr.pay(n["cost"]):
                return tr
            tr.observe(n["kept"])
            k = None if n["kept"] or not a.children[k] else a.children[k][0]
    return tr


def run_rex(a, budget, rng, C):
    tr = Trace(budget)
    fresh = list(a.roots)
    rng.shuffle(fresh)
    revealed = []  # non-kept nodes still refinable
    while True:
        best, pick = -1.0, None
        if fresh:
            best, pick = rng.betavariate(1.0, 1.0 + C), "fresh"
        for k in revealed:
            n = a.nodes[k]
            s = rng.betavariate(1.0 + C * n["h"], 1.0 + C * (1.0 - n["h"]) + n["N"])
            if s > best:
                best, pick = s, k
        if pick is None:
            return tr
        if pick == "fresh":
            k = fresh.pop()
            n = a.nodes[k]
            if not tr.pay(n["cost"]):
                return tr
            tr.observe(n["kept"])
            if not n["kept"]:
                revealed.append(k)
        else:
            revealed.remove(pick)
            kids = a.children[pick]
            if not kids:  # refinement the archive never ran
                if not tr.pay(1):
                    return tr
                tr.unobserved += 1
                continue
            c = kids[0]
            n = a.nodes[c]
            if not tr.pay(n["cost"]):
                return tr
            tr.observe(n["kept"])
            if not n["kept"]:
                revealed.append(c)


def run_iid(a, budget, rng):
    tr = Trace(budget)
    # Each call is an independent fresh proposal (a root drawn with replacement); repairs of
    # earlier draws are never reused, so refinement gets no credit it did not earn.
    ids = sorted(a.roots)
    while ids:
        n = a.nodes[ids[rng.randrange(len(ids))]]
        if not tr.pay(n["cost"]):
            return tr
        tr.observe(n["kept"])
    return tr


def run_split(a, budget, rng, R):
    tr = Trace(budget)
    order = list(a.roots)
    rng.shuffle(order)
    for r in order:
        k, done = r, 0
        while True:
            n = a.nodes[k]
            if not tr.pay(n["cost"]):
                return tr
            tr.observe(n["kept"])
            if n["kept"]:
                break
            if done >= R:
                break
            if not a.children[k]:
                tr.unobserved += R - done
                break
            k = a.children[k][0]
            done += 1
    return tr


def arms():
    out = {"fixed": lambda a, b, r: run_fixed(a, b, r), "iid": run_iid}
    for C in C_VALUES:
        out[f"rex_C{C}"] = (lambda C: lambda a, b, r: run_rex(a, b, r, C))(C)
    for R in SPLIT_R:
        out[f"split_R{R}"] = (lambda R: lambda a, b, r: run_split(a, b, r, R))(R)
    return out


# ---------------------------------------------------------------- stats
def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def summarize(xs):
    return {"mean": sum(xs) / len(xs), "lo95": pct(xs, 0.025), "hi95": pct(xs, 0.975)}


def replay(a, budget, reps, seed):
    aset = arms()
    auc = {k: [] for k in aset}
    end = {k: [] for k in aset}
    cov = {k: [0, 0] for k in aset}
    for rep in range(reps):
        ba = a.resample(random.Random(f"{seed}/boot/{rep}"))
        for name, fn in aset.items():
            tr = fn(ba, budget, random.Random(f"{seed}/{name}/{rep}"))
            c = tr.curve()
            auc[name].append(sum(c) / budget)
            end[name].append(c[-1])
            cov[name][0] += tr.observed
            cov[name][1] += tr.unobserved
    res = {}
    for name in aset:
        d_auc = [x - y for x, y in zip(auc[name], auc["fixed"])]
        d_end = [x - y for x, y in zip(end[name], end["fixed"])]
        o, u = cov[name]
        res[name] = {"auc_frontier": summarize(auc[name]), "endpoint": summarize(end[name]),
                     "auc_minus_fixed": summarize(d_auc), "endpoint_minus_fixed": summarize(d_end),
                     "observed_calls": o, "unobserved_selections": u,
                     "coverage": (o / (o + u)) if (o + u) else None}
    return res


LIMITS = ("Replay uses only recorded outcomes. Refinements/repairs the archive never ran are unobserved: a "
          "refine of a node with no recorded child costs one call and yields no outcome (REx arms); sweep "
          "repairs past a recorded chain end are uncharged and counted unobserved. Bootstrap resamples whole "
          "proposal trees with replacement; intervals are 2.5/97.5 percentiles over replicates. Success = "
          "cleared the keep gate; AUC-frontier = mean over calls 1..budget of P(best-so-far success).")


def build_report(a, source, budget, reps, seed, extra=None):
    return {"schema": SCHEMA, "source": source, "seed": seed, "reps": reps, "budget_calls": budget,
            "archive": {"nodes": len(a.nodes), "roots": len(a.roots), "has_repair_tree": a.has_tree(),
                        "kept": sum(1 for n in a.nodes.values() if n["kept"]), "total_cost": a.total_cost()},
            "limitations": LIMITS, "arms": replay(a, budget, reps, seed), **(extra or {})}


def synthetic_fixture(n_ideas=40, seed=7):
    """Deterministic: ideas whose first attempt has h in {.33,.67}; repairs of high-h ideas are kept more often."""
    rng = random.Random(seed)
    nodes = []
    for i in range(n_ideas):
        good = i % 2 == 0
        h = 2 / 3 if good else 1 / 3
        nodes.append({"proposal_id": f"p{i}", "parent_id": None, "repair_attempts": 0, "kept": False,
                      "rungs": [{"name": "compile", "frac": 1.0}, {"name": "op_test", "frac": 1.0 if good else 0.0},
                                {"name": "nmse", "frac": 0.0}]})
        nodes.append({"proposal_id": f"p{i}r1", "parent_id": f"p{i}", "repair_attempts": 1,
                      "kept": rng.random() < (0.9 if good else 0.05),
                      "rungs": [{"name": "compile", "frac": 1.0}, {"name": "op_test", "frac": 1.0},
                                {"name": "nmse", "frac": 1.0}]})
    return nodes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", help="normalized archive JSONL")
    ap.add_argument("--experiments-db", action="append", default=[], help="real adapter input (repeatable)")
    ap.add_argument("--repair-chains", action="store_true",
                    help="with --experiments-db: link per-patch-round rows into repair chains (AK-RX-1.1)")
    ap.add_argument("--repair-cost", choices=("author_calls", "unit"), default="author_calls",
                    help="call_cost per node under --repair-chains (default: panel members' calls, floor 1)")
    ap.add_argument("--dump-archive", help="also write the normalized archive JSONL here (chains mode)")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--budget", type=int, default=0, help="calls (default: half the archive cost)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", default="rx1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    extra = {}
    if a.fixture:
        nodes, src = synthetic_fixture(), "synthetic_fixture"
    elif a.jsonl:
        nodes, src = [json.loads(l) for l in Path(a.jsonl).read_text().splitlines() if l.strip()], a.jsonl
    elif a.experiments_db and a.repair_chains:
        nodes, stats = adapt_experiments_db_chains(a.experiments_db, a.repair_cost)
        src, extra = "experiments.db(repair_chains):" + ",".join(a.experiments_db), {"repair_chain_stats": stats}
        if a.dump_archive:
            Path(a.dump_archive).write_text("".join(json.dumps(n, sort_keys=True) + "\n" for n in nodes))
    elif a.experiments_db:
        nodes, skipped = adapt_experiments_db(a.experiments_db)
        src, extra = "experiments.db:" + ",".join(a.experiments_db), {"rows_skipped_no_gates": skipped}
    else:
        ap.error("need --fixture, --jsonl or --experiments-db")
    arch = Archive(nodes)
    budget = a.budget or max(1, arch.total_cost() // 2)
    rep = build_report(arch, src, budget, a.reps, a.seed, extra)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
