#!/usr/bin/env python3
"""VB-KVU-PF.5 re-seal: rewrite record_sha256 of kpf13/kpf174 belief records so it re-derives on read.

Before the fix the drivers hashed the in-memory record; kpf174's geometry.pruned has INT keys, which serialize as strings
and sort differently, so record_sha256 never re-derived and the belief adapter (root scripts/vidya/adapters/
kv_prefix_fork.py) refused every kpf174 record ("record_sha256 does not re-derive"). This script re-seals each row with
kpf13_restore_ab.seal_record (sha over the JSON round-trip form, the adapter's own canonical form).

It NEVER touches the input: it writes a NEW jsonl (default: <input stem>.resealed-vbkvupf5.jsonl next to the input, so
the sibling rows/ dir still resolves) and a manifest beside it (old -> new sha per row, and whether the OLD sha
re-derives from the int-keyed form, i.e. the stated cause). Rows whose sha already re-derives are copied unchanged.

  /usr/bin/python3 reseal_records.py <belief_measurements.jsonl> [--out NEW.jsonl]
Exit: 0 written, 2 refused (output exists / input missing / a row that is neither clean nor explained by the cause).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
_SPEC = importlib.util.spec_from_file_location("kpf13_restore_ab",
                                               Path(__file__).resolve().parent.parent / "kpf13" / "kpf13_restore_ab.py")
k13 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(k13)


def rederives(row: dict) -> bool:
    return k13.sha256_json({k: v for k, v in row.items() if k != "record_sha256"}) == row.get("record_sha256")


def old_sha_explained(row: dict) -> bool:
    """The stated cause: the stored sha equals the hash of the body with geometry.pruned keys back as ints."""
    body = json.loads(json.dumps({k: v for k, v in row.items() if k != "record_sha256"}))
    pr = (body.get("geometry") or {}).get("pruned")
    if not isinstance(pr, dict):
        return False
    try:
        body["geometry"]["pruned"] = {int(k): v for k, v in pr.items()}
    except ValueError:
        return False
    return k13.sha256_json(body) == row.get("record_sha256")


def reseal(rows: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    out, man, probs = [], [], []
    for i, r in enumerate(rows):
        if rederives(r):
            out.append(r)
            man.append({"line": i + 1, "run_id": r.get("run_id"), "action": "unchanged (already re-derives)",
                        "record_sha256": r.get("record_sha256")})
            continue
        cause = old_sha_explained(r)
        if not cause:
            probs.append(f"line {i + 1} ({r.get('run_id')}): sha does not re-derive and is NOT explained by int-keyed "
                         "geometry.pruned: refusing to re-seal an unexplained record")
        new = k13.seal_record(r)
        out.append(new)
        man.append({"line": i + 1, "run_id": r.get("run_id"), "action": "resealed",
                    "old_record_sha256": r.get("record_sha256"), "new_record_sha256": new["record_sha256"],
                    "old_sha_explained_by_int_keyed_pruned": cause})
    return out, man, probs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecar")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    src = Path(a.sidecar)
    if not src.is_file():
        print(f"REFUSED: {src} missing")
        return 2
    dst = Path(a.out) if a.out else src.with_name(src.stem + ".resealed-vbkvupf5.jsonl")
    mpath = dst.with_name(dst.name + ".manifest.json")
    if dst.exists() or mpath.exists() or dst.resolve() == src.resolve():
        print(f"REFUSED: {dst} or its manifest exists (never overwrite; the input is kept as is)")
        return 2
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    out, man, probs = reseal(rows)
    if probs:
        print("REFUSED:\n  " + "\n  ".join(probs))
        return 2
    assert all(rederives(r) for r in out)
    dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    k13.write_json(mpath, {"task": "VB-KVU-PF.5", "at": k13.now_iso(), "input": str(src),
                           "input_sha256": k13.sha256_file(src), "output": str(dst),
                           "output_sha256": k13.sha256_file(dst), "rows": man,
                           "method": "kpf13_restore_ab.seal_record: sha256 over the JSON round-trip form"})
    print(f"wrote {dst} ({len(out)} rows, {sum(1 for x in man if x['action'] == 'resealed')} resealed) + {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
