"""CS-46 (e) secondary comparator: native parakeet-rs @6ad1501 run (scratch example ``eou_dump``, ORT CPU)
vs the NeMo CPU streaming reference, at 160 ms (parakeet-rs' own chunk).  parakeet-rs returns TEXT only
(no token ids), so the comparison is on the concatenated text (spaces removed) and on the call index of
the first ``[EOU]`` (reset_on_eou=1 run) vs NeMo's first <EOU> call.

usage: python -I compare_parakeet_rs.py --rust-out DIR --nemo-dir DIR --clips C --out FILE
"""
import argparse
import json
import re
from pathlib import Path


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9']", "", t.replace("[EOU]", "").replace("<EOU>", "").lower())


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust-out", type=Path, required=True)
    ap.add_argument("--nemo-dir", type=Path, required=True, help="c160_full-style dir of nemo_ref.py output")
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    clips = [json.loads(l) for l in a.clips.read_text().splitlines() if l.strip()]
    rows = []
    for c in clips:
        cid = c["clip_id"]
        r1 = [json.loads(l) for l in (a.rust_out / f"{cid}_r1.jsonl").read_text().splitlines() if l.strip()]
        r0 = [json.loads(l) for l in (a.rust_out / f"{cid}_r0.jsonl").read_text().splitlines() if l.strip()]
        n = json.loads((a.nemo_dir / f"{cid}.json").read_text())
        # r1 (reset) text up to and including its first [EOU]; NeMo up to its first <EOU>
        rust_eou = next((x["call"] for x in r1 if "[EOU]" in x.get("text", "")), None)
        rust_txt = "".join(x.get("text", "") for x in r1 if x.get("call", 10**9) <= (rust_eou if rust_eou is not None else 10**9))
        nemo_eou = n["eou_calls"][0] if n["eou_calls"] else None
        nemo_txt = "".join(s["text"] for s in n["steps"] if nemo_eou is None or s["call"] <= nemo_eou)
        a_, b_ = norm(rust_txt), norm(nemo_txt)
        rows.append({"clip_id": cid, "rust_eou_call": rust_eou, "nemo_eou_call": nemo_eou,
                     "eou_call_delta": None if rust_eou is None or nemo_eou is None else rust_eou - nemo_eou,
                     "rust_text": rust_txt.strip(), "nemo_text": nemo_txt.replace("<EOU>", "").strip(),
                     "norm_text_equal": a_ == b_, "char_edit_distance": lev(a_, b_), "n_chars_nemo": len(b_),
                     "rust_r0_has_eou_marker": any("[EOU]" in x.get("text", "") for x in r0),
                     "rust_r0_eou_surfaced": False})
    summ = {"n": len(rows), "eou_both": sum(r["rust_eou_call"] is not None and r["nemo_eou_call"] is not None for r in rows),
            "eou_call_equal": sum(r["eou_call_delta"] == 0 for r in rows), "norm_text_equal": sum(r["norm_text_equal"] for r in rows),
            "r0_eou_marker_clips": sum(r["rust_r0_has_eou_marker"] for r in rows),
            "deltas": sorted(r["eou_call_delta"] for r in rows if r["eou_call_delta"] is not None)}
    a.out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
