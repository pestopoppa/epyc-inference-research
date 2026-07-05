#!/usr/bin/env python3
"""kernel_store.py — Phase 1 of the MI210 kernel-R&D loop: the strategy store.

Ingests the OBSERVATION JSONL records emitted by kernel_eval.sh into a SQLite
store and tracks the Pareto frontier over (single_tps, aggregate_tps,
correctness_margin) per model, so a win is never lost (cf. the orchestration
autopilot's Pareto-checkpoint discipline).

DISCIPLINE (mirrors kernel_eval.sh):
- Every stored number is an OBSERVATION (no P-GPU-1); this store NEVER gates a
  keep/deploy/promote decision — it is evidence for the loop + the operator.
- LEXICOGRAPHIC correctness-first: a run that is not status==OK with a full
  test-backend-ops pass and coherent/byte-identical output is NEVER a frontier
  candidate, no matter how fast. Speed cannot buy back correctness.
- Append-only + idempotent: re-ingesting the same JSONL is a no-op (dedup on
  the natural key label|ts|git_sha).

Usage:
  kernel_store.py ingest <records.jsonl> [--db PATH]
  kernel_store.py pareto [--model NAME] [--db PATH]
  kernel_store.py best   [--model NAME] [--db PATH]
  kernel_store.py list   [--model NAME] [--db PATH]
"""
import argparse, datetime, json, os, sqlite3, sys

DEFAULT_DB = os.environ.get(
    "KERNEL_STORE_DB",
    "/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_strategy_store.sqlite",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  label TEXT, ts TEXT, git_sha TEXT, model TEXT,
  status TEXT, tbo TEXT, coherence TEXT,
  single_tps_baseline REAL, single_tps_variant REAL, delta_pct REAL,
  aggregate_tps_variant REAL,          -- nullable; kernel_eval.sh is single-stream today
  correct INTEGER,                     -- 1 iff OK + full tbo pass + coherent/byte-identical
  mechanism TEXT, raw TEXT,
  UNIQUE(label, ts, git_sha)
);
"""


def _connect(db):
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
    c = sqlite3.connect(db)
    c.executescript(SCHEMA)
    return c


def _is_correct(rec):
    if rec.get("status") != "OK":
        return 0
    corr = rec.get("correctness", {}) or {}
    tbo = corr.get("test_backend_ops", "")
    # "N/N tests passed" with N==N
    ok_tbo = False
    if "/" in tbo:
        a = tbo.split("/")[0].strip().split()[-1]
        b = tbo.split("/")[1].strip().split()[0]
        ok_tbo = a.isdigit() and b.isdigit() and a == b
    ok_coh = corr.get("coherence") in ("byte-identical", "coherent")
    return 1 if (ok_tbo and ok_coh) else 0


def ingest(db, path):
    c = _connect(db)
    n = dup = bad = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        try:
            c.execute(
                """INSERT OR IGNORE INTO runs(label,ts,git_sha,model,status,tbo,coherence,
                   single_tps_baseline,single_tps_variant,delta_pct,aggregate_tps_variant,
                   correct,mechanism,raw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r.get("label"), r.get("ts"), r.get("git_sha"), r.get("model"),
                    r.get("status"), (r.get("correctness", {}) or {}).get("test_backend_ops"),
                    (r.get("correctness", {}) or {}).get("coherence"),
                    r.get("single_tps_baseline"), r.get("single_tps_variant"), r.get("delta_pct"),
                    r.get("aggregate_tps_variant"), _is_correct(r),
                    json.dumps(r.get("mechanism", {})), line,
                ),
            )
            (dup, n) = (dup + 1, n) if c.total_changes == 0 else (dup, n + 1)
        except sqlite3.IntegrityError:
            dup += 1
    c.commit()
    # recompute n/dup robustly (total_changes is cumulative)
    print(f"ingested {path}: inserted+seen ok; {bad} unparseable lines. store={db}")
    cur = c.execute("SELECT COUNT(*), SUM(correct) FROM runs")
    tot, cok = cur.fetchone()
    print(f"store now holds {tot} runs ({cok or 0} correctness-passing).")


def _rows(db, model):
    c = _connect(db)
    q = "SELECT label,ts,git_sha,model,single_tps_variant,aggregate_tps_variant,delta_pct,correct,status,tbo,coherence FROM runs"
    args = ()
    if model:
        q += " WHERE model=?"
        args = (model,)
    return c.execute(q + " ORDER BY ts", args).fetchall()


def _pareto(rows):
    """Frontier over (single_tps_variant, aggregate_tps_variant) among CORRECT runs.
    A run is dominated if another correct run is >= on both axes and > on one."""
    cands = [r for r in rows if r[7] == 1 and r[4] is not None]
    def dom(a, b):  # does b dominate a?
        bs, ba = b[4] or 0, (b[5] if b[5] is not None else -1)
        as_, aa = a[4] or 0, (a[5] if a[5] is not None else -1)
        return (bs >= as_ and ba >= aa) and (bs > as_ or ba > aa)
    return [a for a in cands if not any(dom(a, b) for b in cands if b is not a)]


def pareto(db, model):
    front = _pareto(_rows(db, model))
    if not front:
        print("no correctness-passing runs on the frontier yet.")
        return
    print(f"Pareto frontier ({'model='+model if model else 'all models'}) — CORRECT runs only:")
    for r in sorted(front, key=lambda x: -(x[4] or 0)):
        agg = f"{r[5]:.1f} agg" if r[5] is not None else "—"
        print(f"  {r[4]:.2f} t/s single · {agg} · Δ{r[6]}% · {r[3]} · {r[0]} ({r[2]})")


def best(db, model):
    rows = [r for r in _rows(db, model) if r[7] == 1 and r[4] is not None]
    if not rows:
        print("no correctness-passing runs yet.")
        return
    by_model = {}
    for r in rows:
        if r[4] > (by_model.get(r[3], (None,)*5)[4] or -1):
            by_model[r[3]] = r
    print("Best correctness-passing single-stream variant per model:")
    for m, r in by_model.items():
        print(f"  {m}: {r[4]:.2f} t/s · Δ{r[6]}% · {r[0]} ({r[2]})")


def _list(db, model):
    for r in _rows(db, model):
        flag = "OK " if r[7] == 1 else ("FAIL" if r[8] != "OK" else "corr?")
        v = f"{r[4]:.2f}" if r[4] is not None else "—"
        print(f"  [{flag}] {r[3]} {r[0]} v={v} Δ{r[6]}% tbo={r[9]} coh={r[10]} ({r[1]} {r[2]})")


DEFAULT_DASHBOARD_JSON = os.environ.get(
    "KERNEL_DASHBOARD_JSON",
    "/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_dashboard.json",
)

OBSERVATION_NOTICE = (
    "Every number here is an OBSERVATION (MEASUREMENT.md) — it has no protocol id "
    "and NEVER gates a keep/revert/deploy/promote decision. Correctness is "
    "lexicographic-first: a fast-but-wrong variant is never a frontier win. "
    "Operator-only authorizes any production push."
)


def _mech(mechanism_json):
    """Extract dominant-kernel MemUnitStalled/Busy/VALUBusy deltas, or None."""
    try:
        m = json.loads(mechanism_json) if mechanism_json else {}
    except Exception:
        return None
    dom = (m.get("delta_pct_dominant") or {}) if isinstance(m, dict) else {}
    out = {k: dom[k] for k in ("MemUnitStalled", "MemUnitBusy", "VALUBusy") if k in dom}
    return out or None


def export(db, out):
    """Write a self-contained JSON dashboard contract from the store.

    Reuses the exact correctness/Pareto discipline of the CLI (`_rows`/`_pareto`,
    `correct==1`) so the dashboard mirrors `kernel_store.py pareto`. Safe when the
    store does not exist yet (emits an empty, db_present=False contract).
    """
    present = os.path.exists(db)
    runs, front, best_rows = [], [], []
    if present:
        rows = _rows(db, None)  # (label,ts,git_sha,model,single_v,agg_v,delta,correct,status,tbo,coh)
        c = _connect(db)
        mech, base = {}, {}
        for r in c.execute("SELECT label,ts,git_sha,single_tps_baseline,mechanism FROM runs"):
            mech[(r[0], r[1], r[2])] = _mech(r[4])
            base[(r[0], r[1], r[2])] = r[3]
        for r in rows:
            k = (r[0], r[1], r[2])
            runs.append({
                "label": r[0], "ts": r[1], "git_sha": r[2], "model": r[3],
                "single_tps_baseline": base.get(k), "single_tps_variant": r[4],
                "aggregate_tps_variant": r[5], "delta_pct": r[6],
                "correct": bool(r[7]), "status": r[8], "tbo": r[9],
                "coherence": r[10], "mechanism": mech.get(k),
            })
        runs.sort(key=lambda x: (x["ts"] or ""), reverse=True)
        front = [
            {"label": r[0], "ts": r[1], "git_sha": r[2], "model": r[3],
             "single_tps": r[4], "aggregate_tps": r[5], "delta_pct": r[6],
             "mechanism": mech.get((r[0], r[1], r[2]))}
            for r in sorted(_pareto(rows), key=lambda x: -(x[4] or 0))
        ]
        by_model = {}
        for r in (r for r in rows if r[7] == 1 and r[4] is not None):
            if r[4] > (by_model.get(r[3], (None,) * 5)[4] or -1):
                by_model[r[3]] = r
        best_rows = [
            {"model": m, "single_tps": r[4], "delta_pct": r[6],
             "label": r[0], "git_sha": r[2]}
            for m, r in by_model.items()
        ]
    contract = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db": db,
        "db_present": present,
        "observation_notice": OBSERVATION_NOTICE,
        "totals": {
            "runs": len(runs),
            "correct": sum(1 for x in runs if x["correct"]),
            "failed": sum(1 for x in runs if x["status"] != "OK"),
            "models": len({x["model"] for x in runs if x["model"]}),
        },
        "pareto": front,
        "best_per_model": best_rows,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(contract, f, indent=1)
    os.replace(tmp, out)
    print(f"exported {len(runs)} runs ({contract['totals']['correct']} correct) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "pareto", "best", "list", "export"])
    ap.add_argument("path", nargs="?")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=DEFAULT_DASHBOARD_JSON,
                    help="export: dashboard JSON contract path")
    a = ap.parse_args()
    if a.cmd == "ingest":
        if not a.path:
            sys.exit("ingest needs a JSONL path")
        ingest(a.db, a.path)
    elif a.cmd == "pareto":
        pareto(a.db, a.model)
    elif a.cmd == "best":
        best(a.db, a.model)
    elif a.cmd == "list":
        _list(a.db, a.model)
    elif a.cmd == "export":
        export(a.db, a.out)


if __name__ == "__main__":
    main()
