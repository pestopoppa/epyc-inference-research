#!/usr/bin/env python3
"""INF-45 E5 v2 — qualify-or-discard every banked E5 cell manifest before any re-use.

Box: numa-placement-and-contention.md -> "RE-MEASURE E5 Stage-B on the corrected
placement" absorbs two banked-cell defects:

  * W0-mismatch: ``e5_manifests/`` holds cells whose own ``window`` says W0 but which
    were REGENERATED after 5d6a17f2 with a recipe W0 never ran. The W0 run of record is
    ``e5_manifests_w0/`` + the ``e5-w0-*`` run dirs. (Filed as 69 cells on 2026-08-11;
    this script counts the live set and reports any drift from that figure.)
  * missing-reasoning: manifests with no ``reasoning`` field ran at llama-server's
    ``--reasoning auto`` (flag omitted). For arch gemma4 that is ON, contradicting both
    registries -> discard. For the qwen arches auto is the stated intent (A11) -> the
    reasoning axis passes. (Filed as 172 on 2026-08-11.)

Every banked cell gets a tag on three axes plus an overall verdict. NO banked cell is
re-usable without a tag, and only ``QUALIFIED`` may be re-used (capacity_curve_fit.py
refuses anything else):

  reasoning  PASS | DISCARD
  placement  PASS (single-node first-touch) | UNRESOLVED (0-95 interleave=all, no runtime
             policy witness) | DISCARD (straddling cpuset, policy none, no declared defect
             control — requirement (b))
  cache      UNRESOLVED for every banked cell: none carries a per-cell cache-reset witness
             (BRF-T3: the harness dropped between model groups only) — requirement (c)
  verdict    DISCARD if any axis DISCARDs, QUALIFIED only if every axis PASSes, else
             UNRESOLVED.

``--stamp`` additionally writes ``superseded_by_run`` into each regenerated W0 cell in
``e5_manifests/`` (pointer to its run-of-record manifest and run dir). Idempotent.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from e5_v2_stageb_grid import nodes_of, policy_nodes  # noqa: E402

BD = REPO / "data" / "batched_decode"
FILED_W0_MISMATCH = 69
FILED_MISSING_REASONING = 172
W0_RUN_OF_RECORD = {  # handoff E5 W0 block: e5-w0-{qwen36-nothink,gemma,dense,80b}
    "qwen36_q8_0": "data/batched_decode/e5-w0-qwen36-nothink-20260723T194901Z",
    "gemma4_26b_a4b_q4km_mtp": "data/batched_decode/e5-w0-gemma-20260723T203422Z",
    "qwen36_27b_q8": "data/batched_decode/e5-w0-dense-20260723T204720Z",
    "qwen3_next_80b": "data/batched_decode/e5-w0-80b-20260723T221055Z",
}


def tag_cell(path: Path, manifest: dict, root: Path) -> dict:
    rel = str(path.relative_to(root))
    live_dir = rel.startswith("data/batched_decode/e5_manifests/")
    w0_mismatch = live_dir and manifest.get("window") == "W0"
    missing_reasoning = "reasoning" not in manifest
    reasons = []

    reasoning = "PASS"
    if missing_reasoning and manifest.get("architecture") == "gemma4":
        reasoning = "DISCARD"
        reasons.append("reasoning absent on gemma4 -> ran --reasoning auto = ON, registry says off")
    if w0_mismatch:
        reasoning = "DISCARD"
        reasons.append("regenerated W0 pre-registration: recipe W0 never ran (run of record is e5_manifests_w0/)")

    placement = "PASS"
    defect = manifest.get("defect_replication_control") is True
    for i, inst in enumerate(manifest.get("instances") or []):
        nodes = nodes_of(inst.get("cpu_list", ""))
        pol = inst.get("numactl_policy") or "none"
        if policy_nodes(pol) is None and len(nodes) > 1 and not defect:
            placement = "DISCARD"
            reasons.append(f"instances[{i}] straddles nodes {nodes} with policy none, no declared defect control")
        elif policy_nodes(pol) is not None and placement == "PASS":
            placement = "UNRESOLVED"
            reasons.append(f"instances[{i}] {pol}: no runtime policy witness")

    cache = "UNRESOLVED"
    reasons.append("no per-cell cache-reset witness (banked cells dropped per group only)")

    axes = (reasoning, placement, cache)
    verdict = "DISCARD" if "DISCARD" in axes else ("QUALIFIED" if set(axes) == {"PASS"} else "UNRESOLVED")
    return {
        "path": rel, "cell_id": manifest.get("cell_id"), "model_key": manifest.get("model_key"),
        "window": manifest.get("window"), "w0_mismatch": w0_mismatch, "missing_reasoning": missing_reasoning,
        "reasoning_axis": reasoning, "placement_axis": placement, "cache_axis": cache,
        "verdict": verdict, "reasons": reasons,
    }


def banked_paths(root: Path) -> list[Path]:
    pats = sorted(glob.glob(str(root / "data/batched_decode/e5_manifests*/*/*.json")))
    return [Path(p) for p in pats if "/e5_v2_manifests/" not in p]


def stamp(path: Path, manifest: dict, root: Path) -> bool:
    mk = manifest.get("model_key")
    run = W0_RUN_OF_RECORD.get(mk)
    record = root / "data/batched_decode/e5_manifests_w0" / mk / path.name
    ptr = {
        "run_dir": run,
        "run_of_record_manifest": str(record.relative_to(root)) if record.exists() else None,
        "stamped_by": "scripts/benchmark/e5_v2_qualify_banked.py --stamp (INF-45 E5 v2, 2026-11-11)",
        "meaning": "this file is a post-5d6a17f2 regeneration of a CLOSED W0 cell; it does not "
                   "describe what W0 executed — read the run of record instead",
    }
    if manifest.get("superseded_by_run") == ptr:
        return False
    manifest["superseded_by_run"] = ptr
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--output", default=None, help="tags JSONL (default data/batched_decode/e5_v2_manifests/banked_qualification.jsonl)")
    ap.add_argument("--stamp", action="store_true", help="write superseded_by_run into regenerated W0 cells")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.output) if a.output else root / "data/batched_decode/e5_v2_manifests/banked_qualification.jsonl"
    tags, stamped = [], 0
    for p in banked_paths(root):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(m, dict) or "cell_id" not in m:
            continue
        t = tag_cell(p, m, root)
        tags.append(t)
        if a.stamp and t["w0_mismatch"]:
            stamped += stamp(p, m, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(t, sort_keys=True) + "\n" for t in tags))
    n_w0 = sum(t["w0_mismatch"] for t in tags)
    n_nr = sum(t["missing_reasoning"] for t in tags)
    verdicts = collections.Counter(t["verdict"] for t in tags)
    print(f"tagged {len(tags)} banked manifests -> {out}")
    print(f"W0-mismatch {n_w0} (filed {FILED_W0_MISMATCH}); missing-reasoning {n_nr} (filed {FILED_MISSING_REASONING})")
    print(f"verdicts {dict(verdicts)}; stamped superseded_by_run on {stamped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
