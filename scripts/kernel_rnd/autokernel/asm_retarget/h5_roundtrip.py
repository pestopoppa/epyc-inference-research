#!/usr/bin/env python3
"""AK-ASM-T1-H5RT: the H5 (AITER gfx942 FA v3 fwd hd128) kernels round-trip bit-identically.

Retargets the two AITER gfx942 code objects with ``retarget_gfx942_gfx90a.py`` and byte-compares
(SHA-256 and size) against the shipped H5 gfx90a objects (``fa-asm-gfx90a/`` README digests).
CPU only: ROCm llvm-mc/ld.lld, no GPU, no build.

    h5_roundtrip.py AITER_FMHA_V3_FWD_MI300_DIR REFERENCE_DIR WORK_DIR [--json OUT]

Exit 0 only if every pair is bit-identical AND the gfx942 input digest equals the pinned one
(a different input could round-trip to a different, equally 'identical' output).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# name -> (gfx942 input sha256 pinned from AITER hsa/gfx942/fmha_v3_fwd/MI300, gfx90a reference sha256)
PINNED = {
    "fwd_hd128_bf16_rtne": (
        "379ccbd65ce645d7d61a018b56c096a541df4d91a9fddbe23b143adae4b889cb",
        "404c2bf2bc814f068f6604c48416e27819061f7e49ac796dae7fb820c22661b3"),
    "fwd_hd128_bf16_causal_rtne": (
        "ae0c7e5705cbde27a61dfcc71fb9be39dc48c7614098c4c2a3cf3f073add9bf5",
        "f4cc9fce6c6cbc4cad4a5fd355f5b641dfde02c2ba99c383e8f9e4708d42725c"),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def roundtrip(src_dir: Path, ref_dir: Path, work: Path) -> dict:
    out, audit = work / "out", work / "audit"
    out.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (src_sha, ref_sha) in PINNED.items():
        src, ref = src_dir / f"{name}.co", ref_dir / f"{name}.gfx90a.co"
        row = {"kernel": name, "pinned_input_sha256": src_sha, "pinned_reference_sha256": ref_sha}
        try:
            row["input_sha256"], row["reference_sha256"] = sha256(src), sha256(ref)
            proc = subprocess.run([sys.executable, str(HERE / "retarget_gfx942_gfx90a.py"), "one",
                                   str(out), str(audit), str(src)], capture_output=True, text=True)
            produced = out / f"{name}.gfx90a.co"
            row["retarget_rc"] = proc.returncode
            row["produced_sha256"] = sha256(produced) if produced.is_file() else None
        except OSError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["produced_sha256"] = None
        row["input_pinned"] = row.get("input_sha256") == src_sha
        row["reference_pinned"] = row.get("reference_sha256") == ref_sha
        row["bit_identical"] = (row["produced_sha256"] is not None
                                and row["produced_sha256"] == row.get("reference_sha256"))
        rows.append(row)
    ok = all(r["bit_identical"] and r["input_pinned"] and r["reference_pinned"] for r in rows)
    return {"schema": "epyc.autokernel.asm_t1_h5rt.v1", "ok": ok, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src_dir", type=Path)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    doc = roundtrip(a.src_dir, a.ref_dir, a.work_dir)
    if a.json:
        a.json.write_text(json.dumps(doc, indent=1) + "\n")
    for r in doc["rows"]:
        print(f"{r['kernel']}: bit_identical={r['bit_identical']} produced={r['produced_sha256']}")
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
