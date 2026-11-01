#!/usr/bin/env python3
"""Exercise the AKX screen writer through the ROOT payload and browser renderer.

Pass the two checkout roots explicitly. This is a model-free synthetic contract
check and does not add a runtime dependency between the repositories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--root-root", type=Path, required=True)
    args = parser.parse_args()
    research = args.research_root.resolve(strict=True)
    root = args.root_root.resolve(strict=True)
    sys.path[:0] = [str(root), str(research / "scripts/kernel_rnd")]
    from autokernel.loop import keep_screen, status
    from dashboard import loop_status

    keep_id = "contract-keep"
    workload_names = ["w-t0", "w-t1", "w-touch", "w-unknown"]
    signatures = {name: digest({"fixture": name}) for name in workload_names}
    rows = []
    for name, classification, tier in (
            ("w-t0", "INERT", "T0"), ("w-t1", "INERT", "T1"),
            ("w-touch", "TOUCHED", None), ("w-unknown", "UNKNOWN", None)):
        witnesses = None
        if tier == "T0":
            witnesses = {"dso_digests_anchor": {"libggml.so": digest("dso")},
                         "dso_digests_candidate": {"libggml.so": digest("dso")}}
        elif tier == "T1":
            witnesses = {"coverage_profile_sha256": [digest("p0"), digest("p1")],
                         "shape_envelope_hash": digest("shape")}
        rows.append({
            "schema": keep_screen.ROW_SCHEMA, "keep_id": keep_id,
            "parent": "fixture-parent", "child": "fixture-child",
            "workload_id": name, "census_freshness_key": digest("census"),
            "footprint_sha256": digest("footprint"),
            "classification": classification,
            "decision_class": "INERT" if classification == "INERT" else "TOUCHED",
            "tier": tier, "witnesses": witnesses, "touched_detail": {},
            "unknown_reason": "native row not captured" if classification == "UNKNOWN" else None,
            "tool_versions": {}, "created_at": datetime.now(timezone.utc).isoformat(),
        })
    ids = sorted(signatures)
    by_id = {row["workload_id"]: row for row in rows}
    screen = {
        "schema": keep_screen.SCREEN_SCHEMA, "keep_id": keep_id,
        "workload_ids": ids,
        "rows_ref": [keep_screen._digest(by_id[key]) for key in ids],
        "touched_ids": ["w-touch"], "inert_ids": ["w-t0", "w-t1"],
        "unknown_ids": ["w-unknown"], "screen_digest": digest("screen"),
    }
    refusal = {
        "schema": keep_screen.REFUSAL_SCHEMA, "keep_id": keep_id,
        "workload_id": "w-touch", "statistic": "floor_interim", "delta_w": 0.5,
        "F_w": {"value": 0.5}, "effect_w": -1.0, "e_w": None, "blocks": 4,
        "verdict": "REGRESSION", "blast_radius_row_ref": keep_screen._digest(by_id["w-touch"]),
        "touched_detail": {}, "remedy": {"suggested_form": 1,
        "discriminator_hint": "fixture backend / shape"},
    }
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="akx-keep-screen-contract-") as tmp:
        store = Path(tmp)
        source_path = store / "keep.diff"
        source_path.write_text("synthetic source fixture\n", encoding="utf-8")
        keep_screen.write_record(store, keep_id=keep_id, campaign_id="fixture-campaign",
            campaign_manifest_sha256=digest("manifest"), source_path=str(source_path),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            workload_signatures=signatures, observed_at=now.isoformat().replace("+00:00", "Z"),
            screen=screen, rows=rows, refusals=[refusal])
        projection = keep_screen.project(store, keeps=[keep_id],
            resolved_targets=[{"target_ids": [name], "workload_signature": signatures[name],
                               "status": "ready"} for name in workload_names],
            campaign_id="fixture-campaign", manifest_digest=digest("manifest"),
            manifest_path=str(store / "resolved-campaign.json"), now=now)
        status.write(store, state="running", epoch=digest("epoch"), campaign_id="ak-loop",
            anchor_commit="a" * 40, surface="fixture", pairs=1, noise_floor_pct=0.0,
            accumulator={"keeps": [keep_id]}, keep_screen=projection)
        payload, _ = loop_status.snapshot(store, now=now.timestamp())
        matrix = {row["workload_id"]: row for row in payload["keep_screen"]["keeps"][0]["rows"]}
        expected = {"w-t0": ("INERT", "T0"), "w-t1": ("INERT", "T1"),
                    "w-touch": ("TOUCHED", None), "w-unknown": ("UNKNOWN", None)}
        actual = {key: (value["classification"], value["tier"])
                  for key, value in matrix.items()}
        if actual != expected or matrix["w-touch"]["verdict"] != "REGRESSION":
            raise AssertionError(f"producer→payload contract mismatch: {actual!r}")
        if matrix["w-touch"]["refusal_remedy"]["suggested_form"] != 1:
            raise AssertionError("bound refusal remedy did not reach ROOT payload")

        page = (root / "dashboard/static/loop.html").read_text(encoding="utf-8")
        start = page.index("function renderKeepScreen(d){")
        end = page.index("\nfunction renderImprovementTrajectory", start)
        renderer = page[start:end] + "\nglobalThis.renderKeepScreen = renderKeepScreen;"
        javascript = r'''const vm=require('vm'); let output='';
const host={set innerHTML(v){output=v;},get innerHTML(){return output;}};
const document={getElementById:()=>host};
const esc=v=>String(v??'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
 .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
const age=v=>String(v)+'s'; const context={document,esc,age}; vm.createContext(context);
vm.runInContext(JSON.parse(process.argv[1]),context);
context.renderKeepScreen(JSON.parse(require('fs').readFileSync(0,'utf8')));
process.stdout.write(host.innerHTML);'''
        rendered = subprocess.run(["node", "-e", javascript, json.dumps(renderer)],
            input=json.dumps({"keep_screen": payload["keep_screen"]}), text=True,
            capture_output=True, check=True).stdout
        for marker in ("INERT (T0)", "INERT (T1)", "TOUCHED", "UNKNOWN",
                       "REGRESSION", "form 1", "fixture backend / shape"):
            if marker not in rendered:
                raise AssertionError(f"browser output missing {marker!r}")
        print(json.dumps({"result": "PASS", "keep_id": keep_id,
                          "workloads": actual, "verdict": matrix["w-touch"]["verdict"],
                          "remedy": matrix["w-touch"]["refusal_remedy"],
                          "rendered_markers": ["INERT T0/T1", "TOUCHED", "UNKNOWN",
                                               "REGRESSION", "bound remedy"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
