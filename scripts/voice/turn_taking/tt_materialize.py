#!/usr/bin/env python3
"""Materialise stereo clips (ch0 user, ch1 assistant) from the in-repo source recordings + recipes.

  python3 tt_materialize.py --corpus data/turn-taking-cs45-v1 --out /mnt/raid0/llm/datasets/turntaking-cs45-v1/clips [--pairs p1,p2]

Every written clip is re-hashed and compared with the manifest PCM-SHA-256; a mismatch aborts (exit 1).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_corpus as C  # noqa: E402


def materialize(D: Path, out: Path, pairs: set | None = None, include_reserve: bool = False) -> list[dict]:
    recs = {r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "recordings.jsonl").read_text().splitlines())}
    clips = [json.loads(x) for x in (D / "manifest" / "clips.jsonl").read_text().splitlines()]
    if include_reserve and (D / "manifest" / "reserve" / "clips.jsonl").exists():
        recs.update({r["recording_id"]: r for r in map(json.loads, (D / "manifest" / "reserve" / "recordings.jsonl").read_text().splitlines())})
        clips += [json.loads(x) for x in (D / "manifest" / "reserve" / "clips.jsonl").read_text().splitlines()]
    res = []
    cache = {}
    def rec(rid):
        if rid not in cache:
            cache[rid] = A.load_audio(D, recs[rid]["path"])[0].samples[:, 0]
        return cache[rid]
    out.mkdir(parents=True, exist_ok=True)
    for c in clips:
        if pairs and c["pair_id"] not in pairs:
            continue
        arr, lay = C.assemble_row(c, rec(c["user"]["recording_id"]), rec(c["assistant"]["recording_id"]))
        pc = A.Pcm(arr, A.SR)
        if pc.pcm_sha256() != c["pcm"]["pcm_sha256"]:
            raise SystemExit(f"{c['clip_id']}: PCM-SHA-256 mismatch")
        p = out / f"{c['clip_id']}.wav"
        A.write_wav(p, pc)
        res.append({"clip_id": c["clip_id"], "path": str(p)})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--include-reserve", action="store_true")
    a = ap.parse_args()
    r = materialize(Path(a.corpus), Path(a.out), set(a.pairs.split(",")) if a.pairs else None, a.include_reserve)
    print(f"materialised {len(r)} clips, all PCM-SHA-256 verified")
