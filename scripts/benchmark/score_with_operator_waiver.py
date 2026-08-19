#!/usr/bin/env python3
"""Write a SWE prediction artifact under an EXPLICIT, NAMED operator waiver.

`convert_sr_to_patch.py` fail-closes when a parseable SEARCH/REPLACE block is
parsed but cannot be applied: the resulting patch may under-represent the
model's intent, and that is not a judgement a converter may make silently.
That gate is a measurement trust boundary and is NOT modified by this tool.

This tool is the documented exception path. It refuses unless:
  * every ineligible instance is named on the command line, AND
  * each named instance's dropped blocks match the SHA-256 the operator
    supplied, so a waiver cannot silently widen as the capture changes, AND
  * the only ineligibility class present is skipped-parseable-blocks
    (capture-integrity failures and stopped-zero rows are NEVER waivable).

It writes predictions.json plus a waiver manifest recording exactly which
edits were dropped, so any downstream score carries its own provenance.
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

CONVERTER = Path("/mnt/raid0/llm/epyc-inference-research/artifacts/"
                 "architect-27b-finetunes-v8-20260726/expanded-six-arm-v4-tail-replay-20260727/"
                 "authority/convert_sr_to_patch.py")

def load_converter():
    spec = importlib.util.spec_from_file_location("csp", CONVERTER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["csp"] = mod
    spec.loader.exec_module(mod)
    return mod

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pq_path"); ap.add_argument("arm"); ap.add_argument("out")
    ap.add_argument("--runner-source", required=True)
    ap.add_argument("--waive", action="append", default=[], metavar="INSTANCE=SHA256",
                    help="instance whose dropped blocks are waived, pinned by "
                         "the SHA-256 of its concatenated skipped search texts")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    c = load_converter()
    expected = hashlib.sha256(Path(a.runner_source).read_bytes()).hexdigest()
    # Reuse the converter's own dataset rows (module-level global) verbatim.
    inst_rows = c.rows
    src = [json.loads(l) for l in open(a.pq_path) if l.strip()]

    waived = {}
    for w in a.waive:
        k, _, v = w.partition("=")
        if not v: print(f"FAIL malformed --waive {w!r}", file=sys.stderr); return 2
        waived[k] = v.lower()

    preds, diags, dropped = [], [], {}
    for x in src:
        inst = inst_rows[x["id"]]
        bd = []
        if x.get("finish_reason") == "length":
            patch = ""
        else:
            patch, _a, _s = c.apply_blocks(inst, x.get("response", ""), bd)
        skipped = [b for b in bd if str(b.get("outcome", "")).startswith("skipped")]
        if skipped:
            h = hashlib.sha256("".join(b.get("search_sha256", "") for b in skipped)
                               .encode()).hexdigest()
            dropped[x["id"]] = {"count": len(skipped), "sha256": h,
                                "blocks": [{"path": b.get("path"),
                                            "search_sha256": b.get("search_sha256"),
                                            "outcome": b.get("outcome")} for b in skipped]}
        preds.append({"instance_id": x["id"], "model_name_or_path": a.arm,
                      "model_patch": patch})
        diags.append(c.row_diagnostic(x, patch, bd, expected))

    status = c.summary_status(diags)
    hard = list(status.get("stopped_zero_parseable_instance_ids", []) or [])
    integrity_bad = status.get("capture_integrity_ineligible_instance_ids", []) or []
    if integrity_bad:
        print(f"REFUSING: capture-integrity failures are never waivable: {integrity_bad}",
              file=sys.stderr); return 3
    if hard:
        print(f"REFUSING: stopped-zero rows are never waivable: {hard}", file=sys.stderr)
        return 3

    unwaived = sorted(set(dropped) - set(waived))
    if unwaived:
        print("REFUSING: these instances dropped parseable edits but were not waived:",
              file=sys.stderr)
        for i in unwaived:
            print(f"  --waive {i}={dropped[i]['sha256']}   ({dropped[i]['count']} block(s))",
                  file=sys.stderr)
        return 4
    for i, sha in waived.items():
        if i not in dropped:
            print(f"REFUSING: {i} was waived but dropped nothing (stale waiver)",
                  file=sys.stderr); return 5
        if dropped[i]["sha256"] != sha:
            print(f"REFUSING: {i} waiver SHA mismatch\n  expected {sha}\n  actual   "
                  f"{dropped[i]['sha256']}", file=sys.stderr); return 6

    manifest = {"schema_version": 1, "arm": a.arm, "pq_path": str(a.pq_path),
                "pq_sha256": hashlib.sha256(Path(a.pq_path).read_bytes()).hexdigest(),
                "runner_source_sha256": expected,
                "gate": "convert_sr_to_patch.summary_status (UNMODIFIED)",
                "waiver_scope": "skipped_parseable_blocks_only",
                "waived_instances": {i: dropped[i] for i in waived},
                "prediction_count": len(preds),
                "conversion_status_without_waiver": status.get("conversion_status")}
    if a.dry_run:
        print(json.dumps(manifest, indent=1)); print("DRY RUN — nothing written")
        return 0
    c.atomic_write_json(Path(a.out), preds)
    Path(a.manifest).write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"WROTE {a.out} ({len(preds)} predictions) under waiver; manifest {a.manifest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
