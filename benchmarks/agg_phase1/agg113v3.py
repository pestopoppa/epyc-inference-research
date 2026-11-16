#!/usr/bin/env python3
"""AGG-1.16 (PREDECLARED_AGG113V3.md): ``agg_pipeline.v3`` out-of-sample non-regression on the AGG-1.13 set.

--phase replay   OFFLINE, no model, no server, no process signalling. From the saved AGG-1.13 r2 v2 ingest window replies: parity of the v2 typed finish
                 against the stored store, the harvest delta written into COPIES of the CPU v2 store and of the GPU row's post-audit store.v2, canonical.v3,
                 M1 raw / canonical vs v2 (pilot look first), and the arm C model-input identity proof over the 30 recorded answers.
--phase gpu-c    arm C ONLY (v3 store + agg110.v1) over the 30 questions on a private np1 server (--url). Run by the prepared row only when the replay's
                 identity proof fails for at least one question (PREDECLARED step 2).
--dry-run        validate inputs, print the plan; nothing written, no model.

    python3 agg113v3.py --phase replay --orch-root <orchestrator @ >= 92139534> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE), str(HERE / "agg113")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREDECLARED = HERE / "agg113" / "PREDECLARED_AGG113V3.md"
PREDECLARED_SHA = "9c99f8ffb0f2edd2753fd32ee9012f6777991da606dc96a02615dddec50310b8"
ORCH_PIN_MIN = "92139534"
V2_INGEST = Path("/mnt/raid0/llm/tmp/agg113-ingest-v2-cpu-20261114-r2/out")
V2_GPU = Path("/mnt/raid0/llm/tmp/agg113-arms-gpu-20261114-r2/out")
V2_M1 = (71, 67)          # the filed AGG-1.13 r2 v2 readings (raw, canonical) of 76
V2_C = 25                 # the filed AGG-1.13 r2 arm C P-large count of 30
RC_OK, RC_PRE, RC_GATE, RC_PARTIAL = 0, 2, 3, 75


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.open() if line.strip()]


def orch(root: Path) -> list[str]:
    sys.path.insert(0, str(root))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    from src.aggregation import facts_typed as ft, pipeline as pl  # noqa: PLC0415
    probs = []
    if getattr(pl, "PIPELINE_V3", None) != "agg_pipeline.v3":
        probs.append("orchestrator lacks agg_pipeline.v3 (need >= 92139534)")
    else:
        s = pl.pipeline_spec(pl.PIPELINE_V3)
        if s.l4_kwargs() != {"harvest": True} or not s.canonical_rule.startswith("canonical.v3"):
            probs.append(f"agg_pipeline.v3 spec is not v2 + harvest + canonical.v3: {s.as_record()}")
        if pl.pipeline_spec(pl.PIPELINE_V2).l4_kwargs() != {} or pl.pipeline_spec().version != pl.PIPELINE_V1:
            probs.append("v2 / default changed")
    if not hasattr(ft, "keep_slot_harvest"):
        probs.append("orchestrator lacks facts_typed.keep_slot_harvest")
    return probs


def inputs_problems() -> list[str]:
    p = []
    if not PREDECLARED.exists() or sha256(PREDECLARED) != PREDECLARED_SHA:
        p.append("PREDECLARED_AGG113V3.md missing or sha256 mismatch")
    for f in (V2_INGEST / "results.jsonl", V2_INGEST / "store.sqlite", V2_INGEST / "cells.json", V2_INGEST / "result.cpu.json",
              V2_GPU / "results.jsonl", V2_GPU / "store.v2.sqlite"):
        if not f.exists():
            p.append(f"missing input {f}")
    return p


# --------------------------------------------------------------------------- replay of the typed finish (agg115.py's, per document)
def replay_doc(ft: Any, recs: list[dict[str, Any]], planned: set[str], harvest: bool) -> list[dict[str, Any]]:
    from src.aggregation import facts_llm  # noqa: PLC0415
    kept_in = []
    for r in recs:
        chosen = set(r["chosen_by"])
        for f in r["facts"]:
            if (ft.keep_slot_harvest(f, chosen, planned) if harvest else ft.keep_slot(f, chosen)):
                kept_in.append(f)
    kept, _drop = ft.drop_unquoted(kept_in)
    return facts_llm.reconcile(list(kept))[0]


def plans_of(results: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in read_jsonl(results):
        if r.get("kind") == "t_window" and r.get("final"):
            out.setdefault(r["doc"], []).append(r)
    return out


def harvest_delta(ft: Any, plans: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    planned = set(ft.SCHEMA_FIELDS)
    base_all, delta = [], []
    missing = 0
    for doc in sorted(plans):
        base = replay_doc(ft, plans[doc], planned, False)
        harv = replay_doc(ft, plans[doc], planned, True)
        bids, hids = {f["fact_id"] for f in base}, {f["fact_id"] for f in harv}
        missing += len(bids - hids)
        base_all += base
        delta += [f for f in harv if f["fact_id"] not in bids]
    by_field: dict[str, int] = {}
    for f in delta:
        by_field[f["field"]] = by_field.get(f["field"], 0) + 1
    return base_all, {"added": len(delta), "added_by_field": by_field, "base_not_in_harvest": missing}, delta


def parity(st: Any, base: list[dict[str, Any]], corpus_id: str) -> dict[str, int]:
    stored = {r[0] for r in st.conn.execute("SELECT fact_id FROM facts WHERE extractor = 'l4_llm.v1'")}   # one corpus per store file
    rep = {f["fact_id"] for f in base}
    return {"stored_l4": len(stored), "replayed_l4": len(rep), "stored_not_replayed": len(stored - rep), "replayed_not_stored": len(rep - stored)}


def write_delta(P: Any, src: Path, dst: Path, delta: list[dict[str, Any]]) -> int:
    if dst.exists():
        raise SystemExit(f"{dst} exists; use a fresh --out (this script never removes files)")
    P.copy_store(src, dst)
    st = P.open_store(dst)
    try:
        have = {r[0] for r in st.conn.execute("SELECT fact_id FROM facts")}
        new = [f for f in delta if f["fact_id"] not in have]
        st.write_facts(new)
        st.conn.commit()
        return len(new)
    finally:
        st.close()


# --------------------------------------------------------------------------- arm C model-input identity
def arm_c_view(G110: Any, A: Any, st: Any, corpus_id: str) -> dict[str, Any]:
    from src.aggregation import sql_grammar as SG  # noqa: PLC0415
    canon = st.canonical_facts(corpus_id)
    ents = sorted({r["entity"] for r in canon if r.get("entity")})
    cat = SG.pivot_catalog(canon, names=ents, fields=tuple(A.METRICS))
    SG.create_pivot_view(st.conn, corpus_id, cat)
    fdesc = "\n".join(f"- {f}: {A.METRICS[f]}" for f in cat.fields)
    return {"cat": cat, "gbnf": SG.build_pivot_gbnf(cat), "fdesc": fdesc, "fields": list(cat.fields), "ents": ents,
            "docs_meta": json.dumps(st.corpus_docs(corpus_id), sort_keys=True, default=str), "built": sorted(st.built_layers(corpus_id)),
            "sql_present": bool(st.facts(corpus_id))}


def rows_of(st: Any, sql: str | None, cap: int) -> Any:
    if not sql:
        return None
    try:
        cur = st.conn.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchmany(cap)]
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


def identity(G110: Any, A: Any, P: Any, s2: Any, s3: Any, answers: list[dict[str, Any]], names: list[str], agg: list[dict[str, Any]],
             corpus_id: str) -> dict[str, Any]:
    v2, v3 = arm_c_view(G110, A, s2, corpus_id), arm_c_view(G110, A, s3, corpus_id)
    shared = {k: v2[k] == v3[k] for k in ("gbnf", "fdesc", "fields", "ents", "docs_meta", "built", "sql_present")}
    qtext = {q["id"]: q["question"] for q in agg}
    per: dict[str, Any] = {}
    for r in answers:
        q = r["qid"]
        same_prompt = (G110.sql_prompt_v1(qtext[q], v2["fdesc"], list(names), v2["fields"])
                       == G110.sql_prompt_v1(qtext[q], v3["fdesc"], list(names), v3["fields"]))
        ra, rb = rows_of(s2, r.get("sql_executed"), P.CFG.sql_row_cap), rows_of(s3, r.get("sql_executed"), P.CFG.sql_row_cap)
        per[q] = {"path": r.get("path"), "sql_prompt_same": same_prompt, "rows_same": ra == rb,
                  "n_rows_v2": len(ra) if isinstance(ra, list) else None, "n_rows_recorded": r.get("n_rows"),
                  "rows_match_recorded_count": (len(ra) if isinstance(ra, list) else 0) == (r.get("n_rows") or 0),
                  "identical": r.get("path") == "sql" and same_prompt and ra == rb and all(shared.values())}
    return {"shared_inputs_same": shared, "per_question": per, "n_identical": sum(v["identical"] for v in per.values()), "n": len(per)}


# --------------------------------------------------------------------------- phases
def run_replay(a: argparse.Namespace) -> int:
    import agg112 as G  # noqa: PLC0415
    import agg113 as H  # noqa: PLC0415
    import agg110 as G110  # noqa: PLC0415
    import agg19 as G19  # noqa: PLC0415
    import agg_phase0 as A  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    X.configure(H.CORPUS_ID)
    sym = G.orch_symbols()
    ft, pl = sym["ft"], sym["pl"]
    inp = H.load_inputs(H.D113)
    cells, cid = inp["cells"], H.CORPUS_ID
    keys = H.cell_keys(cells)
    pilot_keys = H.cell_keys(cells, H.PILOT_DOCS)
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"task": "AGG-1.16 offline replay (no model)", "predeclared_sha256": PREDECLARED_SHA, "orch_root": str(a.orch_root),
                           "orch_head": (a.orch_root / ".git").exists() and os.popen(f"git -C {a.orch_root} rev-parse HEAD").read().strip(),
                           "inputs_sha256": {str(p): sha256(p) for p in (V2_INGEST / "results.jsonl", V2_INGEST / "store.sqlite", V2_INGEST / "cells.json",
                                                                         V2_GPU / "results.jsonl", V2_GPU / "store.v2.sqlite")},
                           "model_calls": 0}
    plans = plans_of(V2_INGEST / "results.jsonl")
    base, dstats, delta = harvest_delta(ft, plans)
    res["harvest"] = dstats
    # 1.1 parity: the v2 finish replays to the stored L4 facts, and v2 re-canonicalisation reproduces the stored cell vector
    v2c = a.out / "store.v2check.sqlite"
    P.copy_store(V2_INGEST / "store.sqlite", v2c)
    st = P.open_store(v2c)
    try:
        res["parity_l4"] = parity(st, base, cid)
        pl.pipeline_spec(pl.PIPELINE_V2).refresh_canonical(st, cid)
        now2 = G.readings(G19.fact_rows(st.conn, cid), st.canonical_facts(cid), cells)
    finally:
        st.close()
    filed = json.loads((V2_INGEST / "cells.json").read_text())
    diff_raw = [k for k in keys if bool(now2["raw"].get(k)) != bool(filed["raw"].get(k))]
    diff_can = [k for k in keys if bool(now2["canonical"].get(k)) != bool(filed["canonical"].get(k))]
    pr = res["parity_l4"]
    res["parity_cells"] = {"raw": sum(bool(now2["raw"].get(k)) for k in keys), "canonical": sum(bool(now2["canonical"].get(k)) for k in keys),
                           "raw_cells_differ": len(diff_raw), "canonical_cells_differ": len(diff_can)}
    valid = (pr["stored_not_replayed"] == 0 and pr["replayed_not_stored"] == 0 and not diff_raw and not diff_can and dstats["base_not_in_harvest"] == 0)
    res["replay_valid"] = valid
    if not valid:
        res["verdict"] = "replay_invalid"
        (a.out / "result.replay.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(json.dumps(res, indent=1, default=str))
        return RC_PRE
    # 1.2 v3 CPU store: delta + canonical.v3; pilot look first
    v3p = a.out / "store.v3.sqlite"
    res["delta_written_cpu"] = write_delta(P, V2_INGEST / "store.sqlite", v3p, delta)
    st = P.open_store(v3p)
    try:
        res["canonical_v3"] = pl.pipeline_spec(pl.PIPELINE_V3).refresh_canonical(st, cid)
        now3 = G.readings(G19.fact_rows(st.conn, cid), st.canonical_facts(cid), cells)
    finally:
        st.close()
    r2, r3 = filed["raw"], now3["raw"]
    pilot = {"n": len(pilot_keys), "v2": sum(bool(r2.get(k)) for k in pilot_keys), "v3": sum(bool(r3.get(k)) for k in pilot_keys)}
    pilot["tripped"] = pilot["v3"] < pilot["v2"]
    res["pilot_look"] = pilot
    if pilot["tripped"]:
        res["verdict"] = "agg113v3_futility"
        (a.out / "result.replay.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(json.dumps(res, indent=1, default=str))
        return RC_GATE
    c2, c3 = filed["canonical"], now3["canonical"]
    res["m1"] = {"raw": {"v2": sum(bool(r2.get(k)) for k in keys), "v3": sum(bool(r3.get(k)) for k in keys), "n": len(keys),
                         "paired_v3_vs_v2": H.paired_113(r3, r2, keys),
                         "regressions": sorted(k for k in keys if r2.get(k) and not r3.get(k)), "fixes": sorted(k for k in keys if r3.get(k) and not r2.get(k))},
                 "canonical": {"v2": sum(bool(c2.get(k)) for k in keys), "v3": sum(bool(c3.get(k)) for k in keys),
                               "paired_v3_vs_v2": H.paired_113(c3, c2, keys),
                               "regressions": sorted(k for k in keys if c2.get(k) and not c3.get(k)),
                               "fixes": sorted(k for k in keys if c3.get(k) and not c2.get(k))}}
    (a.out / "cells.v3.json").write_text(json.dumps({"pipeline": "agg_pipeline.v3", "raw": {k: bool(v) for k, v in sorted(r3.items())},
                                                    "canonical": {k: bool(v) for k, v in sorted(c3.items())}}, indent=0) + "\n")
    # 1.3 v3 arm C store: delta into a copy of the GPU row's post-audit store.v2; 1.4 identity vs a v2 copy re-canonicalised under v2
    g3, g2 = a.out / "store.gpu.v3.sqlite", a.out / "store.gpu.v2check.sqlite"
    res["delta_written_gpu"] = write_delta(P, V2_GPU / "store.v2.sqlite", g3, delta)
    P.copy_store(V2_GPU / "store.v2.sqlite", g2)
    s2, s3 = P.open_store(g2), P.open_store(g3)
    try:
        before = s2.canonical_facts(cid)
        pl.pipeline_spec(pl.PIPELINE_V2).refresh_canonical(s2, cid)
        res["gpu_v2_canonical_reproduced"] = before == s2.canonical_facts(cid)
        pl.pipeline_spec(pl.PIPELINE_V3).refresh_canonical(s3, cid)
        recs = read_jsonl(V2_GPU / "results.jsonl")
        ans = [r for r in recs if r.get("kind") == "answer" and r.get("final") and r.get("arm") == "C"]
        agg = X.verified_agg(inp)
        names = sorted({d["company"] for d in inp["corpus"]["docs"]})
        idn = identity(G110, A, P, s2, s3, ans, names, agg, cid)
    finally:
        s2.close()
        s3.close()
    res["arm_c_identity"] = idn
    ids = [q["id"] for q in agg]
    corr2 = {r["qid"]: r.get("correct") is True for r in ans}
    all_id = idn["n_identical"] == len(ids) == idn["n"] and res["gpu_v2_canonical_reproduced"]
    res["p_large_c"] = {"v2": sum(corr2.get(q, False) for q in ids), "n": len(ids),
                        "v3": sum(corr2.get(q, False) for q in ids) if all_id else None,
                        "basis": "by_input_identity" if all_id else "identity_failed: the arm C GPU row is required (PREDECLARED step 2)",
                        "non_identical": sorted(q for q, v in idn["per_question"].items() if not v["identical"])}
    regress = res["m1"]["raw"]["regressions"] or res["m1"]["canonical"]["regressions"]
    if regress:
        res["verdict"] = "agg113v3_regresses"
    elif not all_id:
        res["verdict"] = "m1_holds_arm_c_pending_gpu"
    else:
        res["verdict"] = "agg113v3_holds"
        res["also"] = ["gain_untested_on_this_set"]
    (a.out / "result.replay.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "arm_c_identity"}, indent=1, default=str))
    print(json.dumps({"arm_c_identity": {k: v for k, v in idn.items() if k != "per_question"}}, indent=1))
    return RC_OK


def run_gpu_c(a: argparse.Namespace) -> int:
    """Arm C only on the replay's v3 store (copied once into --out), the AGG-1.13 r2 answer recipe (agg110.v1), units ``answer:C3:<qid>``."""
    import agg110 as G110  # noqa: PLC0415
    import agg113 as H  # noqa: PLC0415
    import phase0_arms as P  # noqa: PLC0415
    import phase1_arms as X  # noqa: PLC0415
    from src.aggregation import planner  # noqa: PLC0415
    from src.aggregation.llm_backend import LlamaServerBackend  # noqa: PLC0415
    from src.repl_environment.context_search import BM25Index  # noqa: PLC0415
    X.configure(H.CORPUS_ID)
    rp = json.loads((a.replay_dir / "result.replay.json").read_text())
    if not rp.get("replay_valid") or rp.get("verdict") not in ("m1_holds_arm_c_pending_gpu", "agg113v3_holds"):
        print(f"precondition: replay verdict {rp.get('verdict')!r} does not call for the arm C row")
        return 4
    inp = H.load_inputs(H.D113)
    agg = X.verified_agg(inp)
    names = sorted({d["company"] for d in inp["corpus"]["docs"]})
    a.out.mkdir(parents=True, exist_ok=True)
    rec = P.Recorder(a.out / "results.jsonl")
    smoke = os.environ.get("FILLER_SMOKE") == "1"
    sp = a.out / "store.v3.sqlite"
    if not sp.exists():
        P.copy_store(a.replay_dir / "store.gpu.v3.sqlite", sp)
        rec.emit("store_copied", src=str(a.replay_dir / "store.gpu.v3.sqlite"), sha256=sha256(sp), final=False)
    deadline = float(os.environ["FILLER_DEADLINE"]) if os.environ.get("FILLER_DEADLINE") else None
    guard = P.make_guard(deadline, os.environ.get("FILLER_STOPFILE"))
    rowstop = os.environ.get("AGG113V3_ROW_STOP")
    backend = LlamaServerBackend(a.url, model_id=a.model_id)
    st = P.open_store(sp)
    cid = H.CORPUS_ID
    try:
        chunks = st.chunks(cid)
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)
        bm25 = BM25Index([c["text"] for c in chunks])
        docs_meta, built, sql_present = st.corpus_docs(cid), sorted(st.built_layers(cid)), bool(st.facts(cid))
        scopes = P._scope_plan(planner, docs_meta, [("P-large", "agg", q) for q in agg])
        for q in (agg[:1] if smoke else agg):
            unit = f"answer:C3:{q['id']}"
            if unit in rec.done:
                continue
            guard()
            if rowstop and os.path.exists(rowstop):
                raise P.Stop("row stop")
            t, t0 = P.Tally(backend), time.time()
            text, stt, extra = G110.p_large_answer_v(
                backend, st, q, scope=scopes[f"answer:P-large:{q['id']}"], planner=planner, docs_meta=docs_meta, built=built,
                sql_present=sql_present, names=names, cfg=P.CFG, corpus_chunks=chunks, corpus_bm25=bm25,
                doc_chunks=lambda d, _b=by_doc: _b.get(d, []), planner_version=G110.V1, corpus_id=cid)
            ok = P.score_agg(q, text) if stt == "ok" else None
            rec.emit("answer", unit=unit, arm="C3", qid=q["id"], qclass="agg", status=stt, correct=ok, calls=t.calls, prefill_tokens=t.prefill,
                     answer=text[-600:], final=stt in ("ok", "out_of_window", "budget_exhausted"), wall_s=time.time() - t0,
                     pipeline="agg_pipeline.v3", **extra)
            if smoke:
                return RC_OK
    except P.Stop as e:
        rec.emit("stop", reason=str(e), final=False)
        return RC_PARTIAL
    finally:
        st.close()
    ids = [q["id"] for q in agg]
    mine = {r["qid"]: r.get("correct") is True for r in rec.records if r.get("kind") == "answer" and r.get("final") and r.get("arm") == "C3"}
    if not all(q in mine for q in ids):
        return RC_PARTIAL
    ref = {r["qid"]: r.get("correct") is True for r in read_jsonl(V2_GPU / "results.jsonl")
           if r.get("kind") == "answer" and r.get("final") and r.get("arm") == "C"}
    out = {"complete": True, "c_v3": sum(mine[q] for q in ids), "c_v2": sum(ref.get(q, False) for q in ids), "n": len(ids),
           "paired_v3_vs_v2": H.paired_113(mine, ref, ids), "regressions": sorted(q for q in ids if ref.get(q) and not mine[q]),
           "m1_from_replay": rp.get("m1")}
    out["verdict"] = "agg113v3_regresses" if out["regressions"] or rp["m1"]["raw"]["regressions"] or rp["m1"]["canonical"]["regressions"] \
        else "agg113v3_holds"
    (a.out / "result.gpu-c.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))
    return RC_OK


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("replay", "gpu-c"), required=True)
    ap.add_argument("--orch-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--replay-dir", type=Path, default=None, help="gpu-c: the replay's out dir (result.replay.json, store.gpu.v3.sqlite)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    probs = orch(a.orch_root) + inputs_problems()
    if a.phase == "gpu-c":
        if a.replay_dir is None or not (a.replay_dir / "result.replay.json").exists():
            probs.append(f"replay result missing under {a.replay_dir}")
        if not a.dry_run and not a.url:
            probs.append("--url required for gpu-c")
    if a.dry_run:
        print(json.dumps({"dry_run": True, "phase": a.phase, "model_calls": 0, "servers_started": 0, "wrote_files": 0, "ok": not probs, "problems": probs,
                          "plan": {"replay": "offline parity + harvest delta + canonical.v3 + M1 + arm C identity (ESTIMATE < 120 s, no model)",
                                   "gpu-c": "arm C only, 30 questions, agg110.v1, v3 store (ESTIMATE about 900 s incl. one cold 27B load)"}}, indent=1))
        return RC_PRE if probs else RC_OK
    if probs:
        print(json.dumps({"ok": False, "problems": probs}, indent=1))
        return RC_PRE
    if a.out is None:
        ap.error("--out is required")
    return run_replay(a) if a.phase == "replay" else run_gpu_c(a)


if __name__ == "__main__":
    sys.exit(main())
