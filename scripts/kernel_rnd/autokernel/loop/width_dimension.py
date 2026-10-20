"""AKX-ALL-27: width invariance as an OBSERVE-ONLY keep dimension (pure; no I/O beyond table loading).

Extends the AKX-ALL-17 ``keep_dimensions`` record (``surface_validation.keep_dimensions(width_invariance=...)``);
there is no second gate.  This module never refuses anything: refuse mode is monitor M7, and the champion keeps
knowingly width-variant routes, so the comparison is RELATIVE to the anchor's own AKX-ALL-26 table, never
absolute zero.

Op level: a keep whose footprint touches mmvq / mmq / fattn / GDN sources records its AKX-ALL-26 table against
the ANCHOR's table, cell by cell: ``new_failing`` (anchor PASS, keep FAIL), ``lower_first_fail_k`` (both FAIL,
keep fails at a smaller k), ``higher_first_fail_k``, ``fixed`` (anchor FAIL, keep PASS) or unchanged.  A keep for
which the probe did not run records ``not_measured``, never a pass.  Cells that are UNSUPPORTED, ERR or void on
either side are LISTED (``indeterminate``), never counted as passing.

Model level: the DF2-18 ``rowexact`` companion at widths 2, 4, 5, 8 (``MODEL_WIDTHS``) over a near-tie-dense
corpus of real code; see ``model_level_record``.  A zero-flip result on a confident prompt is recorded as
``uninformative``, and fixed-width perplexity is refused as evidence.

Table inputs: CPU cell records (``akx-all-26.cell.v1`` k-records, ``width_invariance.py``, jsonl or jsonl.gz) and
HIP/ROCm0 records (``record == "cell"`` rows of ``scripts/benchmark/akx_width_invariance``).
"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.width_invariance_dimension.v1"
MODEL_SCHEMA = "epyc.autokernel.width_invariance_model_level.v1"
MODEL_WIDTHS = (2, 4, 5, 8)
NEAR_TIE_GAPS = (1.0, 0.25)
#: Source families whose edit makes the op-level table applicable.
TOUCHED_FAMILIES = {
    "mmvq": re.compile(r"(^|/)mmvq[^/]*$"),
    "mmq": re.compile(r"(^|/)mmq[^/]*$"),
    "fattn": re.compile(r"(^|/)fattn[^/]*$|flash[-_]?attn", re.I),
    "gdn": re.compile(r"gated[-_]?delta[-_]?net|(^|/)gdn[^/]*$", re.I),
}
#: The DF2-18 recipe fields a model-level record must carry per width.
MODEL_WIDTH_FIELDS = ("near_tie_lt_1_0", "near_tie_lt_0_25", "top1_flips",
                      "row_mean_centred_max_delta", "max_dP")
#: Evidence the recipe rejects outright.
REFUSED_EVIDENCE_KEYS = ("perplexity", "ppl", "fixed_width_perplexity")


def touched_families(paths: Iterable[str]) -> dict[str, list[str]]:
    """Footprint paths grouped by the width-sensitive source family they hit (empty if none)."""
    hits: dict[str, list[str]] = {}
    for path in paths:
        for family, pattern in TOUCHED_FAMILIES.items():
            if pattern.search(str(path)):
                hits.setdefault(family, []).append(str(path))
    return hits


# ---------------------------------------------------------------------------------------------- tables
def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def _fold_cpu(per_k: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    from . import width_invariance as wi   # same fold the AKX-ALL-26 driver used
    pattern, first, status = wi.summarize(dict(per_k))
    return {"status": status, "first_fail_k": first, "pattern": pattern}


def load_table(path: str | Path) -> dict[str, dict[str, Any]]:
    """AKX-ALL-26 records -> ``{cell_key: {status, first_fail_k, ...}}`` with status in
    PASS | FAIL | UNSUPPORTED | VOID_AA | ERR.  CPU k-records and HIP ``cell`` records both load; a file
    holding neither raises (an empty table must not read as 'nothing failed')."""
    path = Path(path)
    cpu: dict[tuple, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    table: dict[str, dict[str, Any]] = {}
    with _open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("schema") == "akx-all-26.cell.v1":
                key = (rec["backend"], rec["route"], rec["nth"], rec["op"], str(rec["type"]).upper(),
                       rec["shape"], rec["mode"])
                cpu[key][int(rec["k"])] = rec
            elif rec.get("record") == "cell":
                status = {"OK": rec.get("verdict"), "UNSUPPORTED": "UNSUPPORTED"}.get(rec.get("status"), "ERR")
                if rec.get("unstable_ks"):
                    status = "VOID_AA"
                if rec.get("err_ks"):
                    status = "ERR"
                key = "|".join(str(x) for x in (rec.get("backend_bound") or rec.get("backend_row"),
                                                rec["id"], rec.get("sub", 0)))
                table[key] = {"status": status, "first_fail_k": rec.get("first_fail_k"),
                              "op": rec.get("op"), "type": rec.get("type")}
    for key, per_k in cpu.items():
        row = _fold_cpu(per_k)
        row.update(op=key[3], type=key[4])
        table["|".join(str(x) for x in key)] = row
    if not table:
        raise ValueError(f"{path}: no AKX-ALL-26 cell records")
    return table


# ---------------------------------------------------------------------------------------- op-level compare
def compare_tables(anchor: Mapping[str, Mapping[str, Any]],
                   keep: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Cell-by-cell RELATIVE comparison of a keep's table against the anchor's (pure)."""
    buckets: dict[str, list] = {n: [] for n in (
        "new_failing", "lower_first_fail_k", "higher_first_fail_k", "fixed", "unchanged_fail",
        "unchanged_pass", "indeterminate", "missing_in_keep", "absent_in_anchor")}
    for key in sorted(set(anchor) | set(keep)):
        a, k = anchor.get(key), keep.get(key)
        if a is None:
            buckets["absent_in_anchor"].append({"cell": key, "keep": k["status"]})
            continue
        if k is None:
            buckets["missing_in_keep"].append({"cell": key, "anchor": a["status"]})
            continue
        sa, sk = a["status"], k["status"]
        if sa not in ("PASS", "FAIL") or sk not in ("PASS", "FAIL"):
            if sa == sk == "UNSUPPORTED":
                continue  # identical structural absence: not a finding
            buckets["indeterminate"].append({"cell": key, "anchor": sa, "keep": sk})
        elif sa == "PASS" and sk == "FAIL":
            buckets["new_failing"].append({"cell": key, "keep_first_fail_k": k["first_fail_k"]})
        elif sa == "FAIL" and sk == "PASS":
            buckets["fixed"].append({"cell": key, "anchor_first_fail_k": a["first_fail_k"]})
        elif sa == "FAIL":
            fa, fk = a["first_fail_k"], k["first_fail_k"]
            if fk < fa:
                buckets["lower_first_fail_k"].append({"cell": key, "anchor": fa, "keep": fk})
            elif fk > fa:
                buckets["higher_first_fail_k"].append({"cell": key, "anchor": fa, "keep": fk})
            else:
                buckets["unchanged_fail"].append(key)
        else:
            buckets["unchanged_pass"].append(key)
    worse = bool(buckets["new_failing"] or buckets["lower_first_fail_k"])
    better = bool(buckets["fixed"] or buckets["higher_first_fail_k"])
    incomplete = bool(buckets["missing_in_keep"] or buckets["indeterminate"])
    verdict = ("mixed" if worse and better else "worse" if worse else "improved" if better else "unchanged")
    return {"verdict": verdict, "coverage_complete": not incomplete,
            "counts": {n: len(v) for n, v in buckets.items()}, **buckets}


# ------------------------------------------------------------------------------------- model-level record
def model_level_record(per_width: Mapping[int, Mapping[str, Any]] | None, *,
                       corpus: Mapping[str, Any] | None, build: str | None = None) -> dict[str, Any]:
    """The DF2-18 ``rowexact`` companion, as recorded (pure).

    ``per_width`` maps each of ``MODEL_WIDTHS`` to ``MODEL_WIDTH_FIELDS`` (reference-arm near-tie counts at
    top1-top2 gap < 1.0 and < 0.25, top-1 flips, row-mean-centred max delta, max dP).  ``corpus`` must say
    what was run (``kind`` = ``real_code_near_tie_dense``; ``name``; optional ``sha256``) -- repetitive text or an
    unlabelled corpus is recorded ``corpus_not_admissible``.  Zero flips with zero near-ties (a confident
    prompt) is ``uninformative``, never 'invariant'.  Perplexity keys are refused."""
    if not per_width:
        return {"schema": MODEL_SCHEMA, "disposition": "not_measured", "build": build,
                "reason": "DF2-18 rowexact runner did not run on the keep's build"}
    for width_row in per_width.values():
        bad = [k for k in REFUSED_EVIDENCE_KEYS if k in width_row]
        if bad:
            raise ValueError(f"fixed-width perplexity is not evidence of width invariance: {bad}")
    widths = {int(w): dict(r) for w, r in per_width.items()}
    missing = [w for w in MODEL_WIDTHS if w not in widths]
    incomplete = [w for w in widths if w in MODEL_WIDTHS
                  and any(widths[w].get(f) is None for f in MODEL_WIDTH_FIELDS)]
    corpus_ok = bool(corpus) and corpus.get("kind") == "real_code_near_tie_dense" and bool(corpus.get("name"))
    informative = {w: bool(widths[w].get("near_tie_lt_1_0")) for w in widths if w in MODEL_WIDTHS}
    if missing or incomplete:
        disposition, reason = "incomplete", f"missing widths {missing}; incomplete widths {incomplete}"
    elif not corpus_ok:
        disposition, reason = "corpus_not_admissible", "corpus must be real code, near-tie dense, and named"
    elif any(widths[w]["top1_flips"] for w in MODEL_WIDTHS):
        disposition, reason = "flips_observed", "top-1 flips recorded at " + str(
            [w for w in MODEL_WIDTHS if widths[w]["top1_flips"]])
    elif not all(informative.get(w) for w in MODEL_WIDTHS):
        disposition, reason = "uninformative", ("zero flips where the reference arm had no near-ties "
                                                "(confident prompt): not evidence of invariance")
    else:
        disposition, reason = "zero_flips_informative", "zero flips with near-ties present at every width"
    return {"schema": MODEL_SCHEMA, "disposition": disposition, "reason": reason, "build": build,
            "corpus": dict(corpus or {}), "widths": {str(w): widths[w] for w in sorted(widths)}}


# --------------------------------------------------------------------------------------------- dimension
def width_dimension(*, footprint: Sequence[str], anchor_table: Mapping[str, Any] | None,
                    keep_table: Mapping[str, Any] | None, model_level: Mapping[str, Any] | None = None,
                    anchor: str | None = None, keep: str | None = None) -> dict[str, Any]:
    """The record ``keep_dimensions(width_invariance=...)`` carries.  ``observe_only`` is always True and
    nothing here can refuse a keep.  Dispositions: ``not_applicable`` (footprint touches no width-sensitive
    source), ``not_measured`` (touched, but either table is absent), ``measured`` (relative comparison)."""
    touched = touched_families(footprint)
    base = {"schema": SCHEMA, "observe_only": True, "refuses": False, "anchor": anchor, "keep": keep,
            "touched": touched, "model_level": dict(model_level) if model_level else
            {"schema": MODEL_SCHEMA, "disposition": "not_measured", "reason": "no model-level record supplied"}}
    if not touched:
        return {**base, "disposition": "not_applicable",
                "reason": "footprint touches no mmvq/mmq/fattn/GDN source"}
    if not keep_table or not anchor_table:
        return {**base, "disposition": "not_measured",
                "reason": ("AKX-ALL-26 probe did not run on the keep's build" if not keep_table
                           else "no anchor AKX-ALL-26 table to compare against")}
    comparison = compare_tables(anchor_table, keep_table)
    return {**base, "disposition": "measured", "comparison": comparison,
            "reason": f"relative to anchor: {comparison['verdict']}"}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="AKX-ALL-27 observe-only width-invariance replay (no gate)")
    ap.add_argument("--anchor-table", required=True)
    ap.add_argument("--keep-table")
    ap.add_argument("--footprint", nargs="*", default=[], help="paths the keep touches")
    ap.add_argument("--anchor-label")
    ap.add_argument("--keep-label")
    args = ap.parse_args(argv)
    record = width_dimension(
        footprint=args.footprint, anchor_table=load_table(args.anchor_table),
        keep_table=load_table(args.keep_table) if args.keep_table else None,
        anchor=args.anchor_label, keep=args.keep_label)
    print(json.dumps(record, indent=1, sort_keys=True))   # redirect to a file; no scratch of its own
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
