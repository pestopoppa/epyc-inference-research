#!/usr/bin/env python3
"""AGG-1.12 / AGG-1.11 follow-up: L3 period attribution (``l3_period.v2``), a no-model measurement.

No model call, no server, no process. The AGG-1.1 store is never written: it is copied (SQLite backup API) into a work
directory, the copy's L3 facts (``extractor LIKE 'l3%'``) are deleted and re-extracted from the stored document pages, tables
and chunks by ``src.aggregation.facts_det.extract_corpus_facts`` under a variant, and M1 is recomputed on the copy. The L4
facts of the copy are untouched (an L4 fact audited against the old L3 rows keeps its stored ``audit_status``).

Variants: ``control_v1`` (legacy extraction re-run; it must reproduce the stored L3 facts exactly), ``v2`` (the default components), each optional one added, and the leave-one-out variants. Each is scored over ALL 211 verified cells:

* raw M1  = ``phase1_arms.m1_hits`` over the ``facts`` table (the AGG-1.1 gate definition: any fact with a 2022 period and the
            gold value after unit scale);
* canonical = the AGG-1.8 ``cell_state`` rule on ``facts_canonical`` built by ``canonical.v1`` (what the planner reads),
            with ``canonical.v2`` + attributable_net_income reported alongside (the AGG-1.9 adopted guard).

fixes = cells not found before, found after; breaks = found before, not found after. A variant is ADOPTED only with 0 breaks
under BOTH readings (raw and canonical.v1) and at least one fix; otherwise the narrower variants are the candidates.
POST-HOC and IN-SAMPLE: the rule was written after reading the 11 L3-class cells of ``agg111_triage.json`` and is scored on
the same store; a held-out store does not exist for it.

CLI:  agg_l3period.py --orch-root DIR [--store PATH] [--work-dir DIR] [--out JSON]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import agg19 as G19  # noqa: E402
import phase1_arms as X  # noqa: E402

SCHEMA = "agg_l3period_probe.v1"
CORPUS_ID = "agg_phase0_fb10"
DEFAULT_STORE = Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/store.sqlite")
DEFAULT_WORK = Path("/mnt/raid0/llm/tmp/agg-l3period-20261112")
DEFAULT_OUT = HERE / "agg_l3period_probe.json"
# the one L3-class cell that is not on the AGG-1.1 store (AGG-1.4 backfill field)
EXTRA_STORE = Path("/mnt/raid0/llm/tmp/agg14-backfill-cpu-20261111-r3/out/store.sqlite")
EXPECTED = {"cells": 211, "l3_class": 11}


def sha256_file(p: Path) -> str:
    return X.sha256_file(p)


def import_orch(orch_root: Path) -> None:
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    if str(orch_root) not in sys.path:
        sys.path.insert(0, str(orch_root))


def fiscal_years(corpus: dict[str, Any]) -> dict[str, str]:
    """doc_name -> fiscal year label from the corpus manifest (``fiscal_year``, 'fiscal 2022 as each filing labels it')."""
    fy = str(corpus["fiscal_year"])
    return {d["doc_name"]: str(d.get("fiscal_year") or fy) for d in corpus["docs"]}


def git_head(root: Path) -> dict[str, Any]:
    """The orchestrator tree's commit and whether tracked files are clean (the probe binds its result to this)."""
    import subprocess  # noqa: PLC0415
    def run(*a: str) -> str:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, check=False).stdout.strip()
    return {"root": str(root), "commit": run("rev-parse", "HEAD"), "tracked_dirty": bool(run("status", "--porcelain", "-uno")),
            "facts_det_sha256": sha256_file(root / "src" / "aggregation" / "facts_det.py")}


def copy_store(src: Path, dst: Path) -> None:
    """SQLite backup of ``src`` (opened ``immutable=1``, never written) into a NEW file ``dst`` (refuses to overwrite: this
    module deletes nothing; clear a stale copy with ``scripts/safety/guarded_rm.sh``)."""
    if dst.exists():
        raise FileExistsError(f"{dst} exists; remove it with scripts/safety/guarded_rm.sh or pick another --work-dir")
    s = sqlite3.connect(f"file:{src}?immutable=1", uri=True)
    d = sqlite3.connect(str(dst))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


# --------------------------------------------------------------------------- variants
def variants() -> dict[str, dict[str, Any]]:
    """control_v1; ``v2`` (the default components); each optional component added to v2; v2 minus each default component."""
    from src.aggregation.facts_det import L3_PERIOD_COMPONENTS, L3_PERIOD_OPTIONAL, L3_PERIOD_V2  # noqa: PLC0415
    dflt = list(L3_PERIOD_COMPONENTS)
    out: dict[str, dict[str, Any]] = {"control_v1": {"period_version": None, "components": None},
                                      "v2": {"period_version": L3_PERIOD_V2, "components": dflt}}
    for c in L3_PERIOD_OPTIONAL:
        out[f"v2_plus_{c}"] = {"period_version": L3_PERIOD_V2, "components": dflt + [c]}
    out["v2_plus_all_optional"] = {"period_version": L3_PERIOD_V2, "components": dflt + list(L3_PERIOD_OPTIONAL)}
    for c in L3_PERIOD_COMPONENTS:
        out[f"v2_minus_{c}"] = {"period_version": L3_PERIOD_V2, "components": [x for x in dflt if x != c]}
    return out


def rerun_l3(store: Any, corpus_id: str, fy: dict[str, str], spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Delete the copy's L3 facts and write the facts the variant extracts (no model)."""
    from src.aggregation.facts_det import extract_corpus_facts  # noqa: PLC0415
    facts = extract_corpus_facts(store, corpus_id, period_version=spec["period_version"], fiscal_years=fy,
                                 components=spec["components"])
    store.conn.execute("DELETE FROM facts WHERE extractor LIKE 'l3%'")
    store.conn.commit()
    store.write_facts(facts)
    return facts


# --------------------------------------------------------------------------- measurement
def score(rows: list[dict[str, Any]], cells: list[tuple[str, str, float]]) -> dict[str, Any]:
    """Per-cell raw hit, canonical.v1 ok and canonical.v2(attributable) ok over ``rows`` (facts with ``corpus_id``)."""
    from src.aggregation.store import CANONICAL_RULE_V2, select_canonical  # noqa: PLC0415
    raw = G19.raw_hits(rows, cells)
    c1 = G19.canon_ok(select_canonical(rows)[0], cells)
    c2 = G19.canon_ok(select_canonical(rows, rule=CANONICAL_RULE_V2, guards=("attributable_net_income",))[0], cells)
    return {"raw": raw, "canon_v1": c1, "canon_v2_attrib": c2}


def diff(before: dict[str, bool], after: dict[str, bool]) -> dict[str, Any]:
    fixed = sorted(k for k in after if after[k] and not before[k])
    broken = sorted(k for k in before if before[k] and not after[k])
    return {"before": sum(before.values()), "after": sum(after.values()), "n": len(before), "fixed": fixed,
            "broken": broken, "n_fixed": len(fixed), "n_broken": len(broken)}


def l3_only_hits(rows: list[dict[str, Any]], cells: list[tuple[str, str, float]]) -> dict[str, bool]:
    return G19.raw_hits([r for r in rows if str(r.get("extractor") or "").startswith("l3")], cells)


def period_census(facts: Iterable[dict[str, Any]]) -> dict[str, int]:
    n = n_none = n_2022 = 0
    for f in facts:
        n += 1
        p = f.get("period")
        n_none += p is None
        n_2022 += p is not None and "2022" in str(p)
    return {"l3_facts": n, "period_none": n_none, "period_2022": n_2022}


def adoption(diff_raw: dict[str, Any], diff_canon: dict[str, Any],
             diff_canon2: dict[str, Any] | None = None) -> dict[str, Any]:
    """Adopted = 0 breaks under raw M1, ``canonical.v1`` and (when given) ``canonical.v2`` + attributable_net_income, and at
    least one fix under any of them."""
    diffs = [diff_raw, diff_canon] + ([diff_canon2] if diff_canon2 is not None else [])
    ok = all(d["n_broken"] == 0 for d in diffs) and any(d["n_fixed"] > 0 for d in diffs)
    return {"adopted": ok, "rule": "0 breaks under raw M1, canonical.v1 and canonical.v2(attributable_net_income); at least one fix"}


def l3_class_cells(triage: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"group": c["group"], "doc": c["doc"], "field": c["field"], "gold": c["gold"]}
            for c in triage["cells"] if c.get("primary") == "L3"]


# --------------------------------------------------------------------------- the probe
def run_probe(*, store_path: Path, work_dir: Path, orch_root: Path, set_dir: Path = HERE,
              extra_store: Path | None = EXTRA_STORE, log: Callable[[str], None] = print) -> dict[str, Any]:
    import_orch(orch_root)
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    inp = G19.load_inputs(set_dir)
    cells = inp["cells211"]
    keys = [f"{d}|{f}" for d, f, _g in cells]
    fy = fiscal_years(inp["corpus"])
    triage = json.loads((set_dir / "agg111_triage.json").read_text())
    l3c = l3_class_cells(triage)
    work_dir.mkdir(parents=True, exist_ok=True)
    src_sha = sha256_file(store_path)
    out: dict[str, Any] = {
        "schema": SCHEMA, "post_hoc": True, "in_sample": True, "model_calls": 0,
        "note": ("Written after reading the 11 L3-class cells of agg111_triage.json and scored on the store they come from: "
                 "POST-HOC and IN-SAMPLE. Not a registered gate; adoption here means 'safe to opt in', not 'validated out of sample'."),
        "orch": git_head(orch_root), "store": {"path": str(store_path), "sha256": src_sha}, "n_cells": len(cells), "corpus_id": CORPUS_ID,
        "fiscal_years": sorted(set(fy.values())), "variants": {}, "l3_class_cells": l3c,
    }
    stored = sqlite3.connect(f"file:{store_path}?immutable=1", uri=True)
    stored_rows = G19.fact_rows(stored, CORPUS_ID)
    stored_l3 = {r["fact_id"]: (r["period"], r["value"], r["field"], r["unit"]) for r in stored_rows
                 if str(r["extractor"]).startswith("l3")}
    stored.close()
    before = score(stored_rows, cells)
    out["before"] = {"raw_found": sum(before["raw"].values()), "m1_raw": sum(before["raw"].values()) / len(cells),
                     "canon_v1_ok": sum(before["canon_v1"].values()),
                     "canon_v2_attrib_ok": sum(before["canon_v2_attrib"].values()),
                     "l3_only_found": sum(l3_only_hits(stored_rows, cells).values())}
    log(f"before: raw {out['before']['raw_found']}/{len(cells)}, canonical.v1 {out['before']['canon_v1_ok']}")

    copy = work_dir / "store.copy.sqlite"
    copy_store(store_path, copy)
    store = CorpusStore(str(copy))
    try:
        for name, spec in variants().items():
            facts = rerun_l3(store, CORPUS_ID, fy, spec)
            rows = G19.fact_rows(store.conn, CORPUS_ID)
            after = score(rows, cells)
            rec: dict[str, Any] = {"spec": spec, "census": period_census(facts)}
            if name == "control_v1":
                new_l3 = {r["fact_id"]: (r["period"], r["value"], r["field"], r["unit"]) for r in rows
                          if str(r["extractor"]).startswith("l3")}
                rec["reproduces_stored_l3_exactly"] = new_l3 == stored_l3
                rec["n_stored_l3"], rec["n_rerun_l3"] = len(stored_l3), len(new_l3)
            rec["m1_raw"] = {"found": sum(after["raw"].values()), "m1": sum(after["raw"].values()) / len(cells)}
            rec["raw"] = diff(before["raw"], after["raw"])
            rec["canon_v1"] = diff(before["canon_v1"], after["canon_v1"])
            rec["canon_v2_attrib"] = diff(before["canon_v2_attrib"], after["canon_v2_attrib"])
            rec["l3_only"] = diff(l3_only_hits(stored_rows, cells), l3_only_hits(rows, cells))
            rec["adoption"] = adoption(rec["raw"], rec["canon_v1"], rec["canon_v2_attrib"])
            rec["l3_class_recovered"] = {
                "l3_only": [k for k in (f"{c['doc']}|{c['field']}" for c in l3c if c["group"] == "c")
                            if l3_only_hits(rows, cells)[k]] if False else None}
            rec["_after"] = after
            rec["_l3_only_after"] = l3_only_hits(rows, cells)
            out["variants"][name] = rec
            log(f"{name}: raw {rec['m1_raw']['found']}/{len(cells)} fixes {rec['raw']['n_fixed']} breaks {rec['raw']['n_broken']}; "
                f"canon.v1 fixes {rec['canon_v1']['n_fixed']} breaks {rec['canon_v1']['n_broken']}")
    finally:
        store.close()
    # the 11 L3-class cells: group c on this store, the group b cell on the AGG-1.4 r3 store copy
    c_keys = [f"{c['doc']}|{c['field']}" for c in l3c if c["group"] == "c"]
    for name, rec in out["variants"].items():
        after_raw, l3_after, after_c1 = rec.pop("_after")["raw"], rec.pop("_l3_only_after"), None
        rec["l3_class_group_c"] = {
            "cells": c_keys,
            "found_before_raw": [k for k in c_keys if before["raw"][k]],
            "found_after_raw": [k for k in c_keys if after_raw[k]],
            "l3_only_found_before": [k for k in c_keys if l3_only_hits(stored_rows, cells)[k]],
            "l3_only_found_after": [k for k in c_keys if l3_after[k]]}
        rec.pop("l3_class_recovered", None)
    if extra_store is not None and extra_store.exists():
        out["extra_store"] = extra_cell_probe(extra_store, work_dir, fy, l3c, inp, log)
    out["store_sha256_after"] = sha256_file(store_path)
    out["source_store_unchanged"] = out["store_sha256_after"] == src_sha
    best = [n for n, r in out["variants"].items() if n != "control_v1" and r["adoption"]["adopted"]]
    out["adopted_candidates"] = best
    v2 = out["variants"]["v2"]
    out["verdict"] = {"v2_default_adopted": v2["adoption"]["adopted"], "v2_raw": v2["raw"]["n_fixed"], "v2_breaks_raw": v2["raw"]["n_broken"],
                      "v2_breaks_canon_v1": v2["canon_v1"]["n_broken"], "m1_before": out["before"]["m1_raw"],
                      "m1_after": v2["m1_raw"]["m1"]}
    return out


def extra_cell_probe(extra: Path, work_dir: Path, fy: dict[str, str], l3c: list[dict[str, Any]], inp: dict[str, Any],
                     log: Callable[[str], None]) -> dict[str, Any]:
    """The group-b L3 cell (COCACOLA long_term_debt, FY2022) lives on the AGG-1.4 r3 store: copy it, re-run L3 per variant."""
    from src.aggregation.store import CorpusStore  # noqa: PLC0415
    gb = [c for c in l3c if c["group"] == "b"]
    cells = [(c["doc"], c["field"], c["gold"]) for c in gb]
    copy = work_dir / "store.agg14.copy.sqlite"
    copy_store(extra, copy)
    store = CorpusStore(str(copy))
    res: dict[str, Any] = {"path": str(extra), "sha256": sha256_file(extra), "cells": [f"{d}|{f}" for d, f, _ in cells],
                           "variants": {}}
    try:
        rows0 = G19.fact_rows(store.conn, CORPUS_ID)
        res["before_raw"] = G19.raw_hits(rows0, cells)
        res["before_l3_only"] = l3_only_hits(rows0, cells)
        for name, spec in variants().items():
            rerun_l3(store, CORPUS_ID, fy, spec)
            rows = G19.fact_rows(store.conn, CORPUS_ID)
            res["variants"][name] = {"raw": G19.raw_hits(rows, cells), "l3_only": l3_only_hits(rows, cells)}
            log(f"extra {name}: {res['variants'][name]}")
    finally:
        store.close()
    res["source_unchanged"] = sha256_file(extra) == res["sha256"]
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--extra-store", type=Path, default=EXTRA_STORE)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--set-dir", type=Path, default=HERE)
    a = ap.parse_args(argv)
    res = run_probe(store_path=a.store, work_dir=a.work_dir, orch_root=a.orch_root, set_dir=a.set_dir,
                    extra_store=a.extra_store)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
