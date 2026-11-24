"""census_reconcile.py -- converge the LIVE census with the COMMITTED one (RSL-R0, deterministic, no LLM).

Why: the live file (/mnt/raid0/llm/tmp/ak-bw-census) was seeded once from the committed census and then drifted; each side
held applied rows the other lacked, and `census_update.merge_static` never touches records/applied/seeds.

`reconcile(live, committed)` is a UNION, idempotent, and runs on every hook run (census_update --merge-static-from):
- applied     : union keyed by RESULT sha256 (live order kept, committed-only shas appended). A sha applied on either side
                is applied on both, so a re-filed RESULT is a no-op everywhere.
- records     : union keyed by record key. On the SAME key the record with the NEWER `measured_at` wins (ISO-8601 UTC
                strings compare lexically; a missing stamp loses; an exact tie keeps the live record).
                A record superseded by a classifier_split (`split_from`) is dropped when the split record is present.
- time_only   : same rule as records (it is written by the live updater too, so the committed copy no longer overwrites it).
- seeds       : union by id. On the same id `done` is terminal and wins; otherwise the live seed is kept.
- seed_stubs  : union keyed by the stub's `record`.
Callers re-run rerank/derive_surfaces afterwards (census_update.run does).
"""
from __future__ import annotations

import json


def CLONE(v):
    return json.loads(json.dumps(v))


def _newer(a: dict, b: dict) -> bool:
    """True when `a` (committed) is strictly newer than `b` (live)."""
    return str(a.get("measured_at") or "") > str(b.get("measured_at") or "")


def _union_keyed(live: dict, src: dict) -> list[str]:
    changed = []
    for k, v in src.items():
        if k not in live or (isinstance(v, dict) and isinstance(live[k], dict) and _newer(v, live[k])):
            live[k] = CLONE(v)
            changed.append(k)
    return changed


def reconcile(live: dict, committed: dict) -> list[str]:
    """Merge `committed` into `live` in place by the rules above. Returns the sections changed (sorted)."""
    changed = set()
    have = {a.get("sha256") for a in live.setdefault("applied", [])}
    for a in committed.get("applied") or []:
        if a.get("sha256") not in have:
            live["applied"].append(CLONE(a))
            have.add(a.get("sha256"))
            changed.add("applied")
    for sec in ("records", "time_only"):
        if committed.get(sec) and _union_keyed(live.setdefault(sec, {}), committed[sec]):
            changed.add(sec)
    recs = live.get("records", {})
    for r in [r for r in recs.values() if r.get("split_from")]:  # lumped record superseded by a classifier split
        for key, old in list(recs.items()):
            if all(old.get(f) == (r["split_from"] if f == "kernel_class" else r.get(f))
                   for f in ("device", "model", "kernel_class", "shape", "n")):
                del recs[key]
                changed.add("records")
    for sid, s in (committed.get("seeds") or {}).items():
        cur = live.setdefault("seeds", {}).get(sid)
        if cur is None or (s.get("status") == "done" and cur.get("status") != "done"):
            live["seeds"][sid] = CLONE(s)
            changed.add("seeds")
    stubs = live.setdefault("seed_stubs", [])
    have_s = {x.get("record") for x in stubs}
    for x in committed.get("seed_stubs") or []:
        if x.get("record") not in have_s:
            stubs.append(CLONE(x))
            have_s.add(x.get("record"))
            changed.add("seed_stubs")
    return sorted(changed)
