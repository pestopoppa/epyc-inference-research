#!/usr/bin/env python3
"""CS-45 instrument pin set: apply the scorer patches to a pinned checkout, and check every pin.

  python3 tt_pins.py check
  python3 tt_pins.py check --fdb-checkout DIR --tb-checkout DIR     # also re-derives the patched trees
  python3 tt_pins.py apply --kind fdb --checkout DIR --dest NEWDIR  # export HEAD (git archive) + patch

Refusals (non-zero exit): checkout HEAD != pinned commit; a base file whose SHA-256 differs from the
recorded base hash; a patch whose own hash differs from measurement_era_changes.json; a patched file
whose SHA-256 differs from the recorded patched hash; any remaining ``torch.hub`` / ``.cuda(`` /
``snakers4`` in a patched in-scope file; a Silero file that is not the pinned one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent.parent.parent / "data" / "turn-taking-cs45-v1"
FORBIDDEN = ("torch.hub", ".cuda(", "snakers4", "nemo_asr")
KIND = {"fdb": "fdb-3e799c4", "tb": "turnbench-38a6f87"}
COMMIT = {"fdb": "3e799c45a045256f47d5f1c9cda90157e2d2ec9e", "tb": "38a6f874322430cb3ca71d8a52aa1e636e88bad8"}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def era() -> dict:
    return json.loads((D / "measurement_era_changes.json").read_text())


def check_static() -> list[str]:
    bad = []
    e = era()
    for ch in e["changes"]:
        pf = D / ch["patch_file"]
        if not pf.exists() or sha(pf) != ch["patch_sha256"]:
            bad.append(f"{ch['id']}: patch digest")
    sil = json.loads((D / "pins.json").read_text())["silero_vad"]
    if sha(D / sil["repo_path"]) != sil["sha256"]:
        bad.append("silero: file vs pin")
    sys.path.insert(0, str(HERE))
    import tt_silero

    if tt_silero.PINNED_SHA256 != sil["sha256"]:
        bad.append("silero: tt_silero.PINNED_SHA256 vs pins.json")
    return bad


def apply(kind: str, checkout: Path, dest: Path) -> list[str]:
    bad = []
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT[kind]:
        return [f"{kind}: checkout HEAD {head} != pinned {COMMIT[kind]}"]
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
        subprocess.run(["git", "-C", str(checkout), "archive", "--format=tar", "-o", tf.name, "HEAD"], check=True)
        with tarfile.open(tf.name) as t:
            t.extractall(dest, filter="data")
    # group by file: several era entries can share one file/patch
    seen = set()
    for ch in era()["changes"]:
        if not ch["patch_file"].startswith(f"scorer_patches/{KIND[kind]}/") or ch["patch_file"] in seen:
            continue
        seen.add(ch["patch_file"])
        f = dest / ch["file"]
        if sha(f) != ch["base_sha256"]:
            bad.append(f"{ch['file']}: base sha256 {sha(f)} != recorded {ch['base_sha256']}")
            continue
        r = subprocess.run(["patch", "-p1", "--no-backup-if-mismatch", "-d", str(dest), "-i", str(D / ch["patch_file"])], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{ch['file']}: patch failed: {r.stdout[-200:]}{r.stderr[-200:]}")
            continue
        if sha(f) != ch["patched_sha256"]:
            bad.append(f"{ch['file']}: patched sha256 {sha(f)} != recorded {ch['patched_sha256']}")
        txt = f.read_text()
        for s in FORBIDDEN:
            if s in txt:
                bad.append(f"{ch['file']}: still contains {s!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "apply"])
    ap.add_argument("--kind", choices=list(KIND))
    ap.add_argument("--checkout")
    ap.add_argument("--dest")
    ap.add_argument("--fdb-checkout")
    ap.add_argument("--tb-checkout")
    a = ap.parse_args()
    bad = []
    if a.cmd == "apply":
        bad = apply(a.kind, Path(a.checkout), Path(a.dest))
    else:
        bad = check_static()
        for kind, co in (("fdb", a.fdb_checkout), ("tb", a.tb_checkout)):
            if co:
                with tempfile.TemporaryDirectory(dir="/mnt/raid0/llm/tmp") as td:
                    bad += apply(kind, Path(co), Path(td) / "x")
    for b in bad:
        print("REFUSED:", b)
    print("VERDICT", "PASS" if not bad else "REFUSED")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
