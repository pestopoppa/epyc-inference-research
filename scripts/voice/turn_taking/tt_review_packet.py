#!/usr/bin/env python3
"""Build the human-review packet for the pair splices (CS-45 acceptance: 'a sample of the pair splices is human-reviewed').

  python3 tt_review_packet.py --corpus data/turn-taking-cs45-v1 --out /mnt/raid0/llm/datasets/turntaking-cs45-v1/review_packet --n 20 --seed 20261020

Stratified-random sample (fixed seed). Writes per pair: both stereo clips (L = user, R = assistant), a mono
'user-only' listen copy of each, and REVIEW.md + review_form.csv with expected labels, cut points and a checklist.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tt_audio as A  # noqa: E402
import tt_materialize as M  # noqa: E402

QUOTA = {"CUT_IC": 10, "CUT_WAIT": 4, "CTX_BC": 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    D, out = Path(a.corpus), Path(a.out)
    pairs = [json.loads(x) for x in (D / "manifest" / "pairs.jsonl").read_text().splitlines()]
    clips = {c["clip_id"]: c for c in map(json.loads, (D / "manifest" / "clips.jsonl").read_text().splitlines())}
    rng = random.Random(a.seed)
    pick = []
    for fam, q in QUOTA.items():
        pool = sorted([p for p in pairs if p["family"] == fam], key=lambda p: p["pair_id"])
        # spread over both splits
        rng.shuffle(pool)
        pick += pool[:q]
    pick = pick[: a.n]
    pick.sort(key=lambda p: p["pair_id"])
    M.materialize(D, out / "stereo", {p["pair_id"] for p in pick})
    (out / "user_only").mkdir(parents=True, exist_ok=True)
    rows, md = [], []
    for p in pick:
        for cid in p["clips"]:
            pc = A.read_wav(out / "stereo" / f"{cid}.wav")
            A.write_wav(out / "user_only" / f"{cid}.wav", A.Pcm(pc.samples[:, :1], A.SR))
    for i, p in enumerate(pick, 1):
        ca, cb = (clips[c] for c in p["clips"])
        cut = cb["cut_decision"] if "cut_decision" in cb else None
        ctx = ""
        if cut:
            ctx = f"cut between '{cut['asr_prev_word']}' | '{cut['asr_next_word']}' at {cb['gold']['user_speech_end_s']:.3f} s (clip clock)"
        for c in (ca, cb):
            rows.append({"n": i, "pair_id": p["pair_id"], "family": p["family"], "split": p["split"], "clip_id": c["clip_id"], "expected_label": c["label"],
                         "user_text_spoken": c["text"]["user_spoken"], "assistant_text": c["text"]["assistant"], "user_speech_end_s": c["gold"]["user_speech_end_s"],
                         "stereo_wav": f"stereo/{c['clip_id']}.wav", "user_only_wav": f"user_only/{c['clip_id']}.wav",
                         "clean_cut_or_natural_end(Y/N)": "", "labels_correct(Y/N)": "", "identical_speech_in_pair(Y/N)": "", "notes": ""})
        md.append(f"### {i}. {p['pair_id']}  ({p['family']}, {p['split']}, speaker {p['speaker_id']})\n")
        md.append(f"- member a `{ca['clip_id']}` expected **{ca['label']}**; member b `{cb['clip_id']}` expected **{cb['label']}**")
        md.append(f"- user text (full): \"{ca['text']['user_full']}\"")
        if p["family"].startswith("CUT"):
            md.append(f"- b is cut after: \"{cb['text']['user_spoken']}\"  -> {ctx}")
        else:
            md.append(f"- a: assistant has FINISHED asking \"{ca['text']['assistant'][:90]}...\" then the user answers; b: assistant is still talking through the user's acknowledgement")
        md.append(f"- files: `stereo/{ca['clip_id']}.wav`, `stereo/{cb['clip_id']}.wav` (L user, R assistant); user-only copies in `user_only/`\n")
    (out / "review_form.csv").write_text("")
    with open(out / "review_form.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    head = f"""# CS-45 pair-splice review packet ({len(pick)} pairs, seed {a.seed})

Corpus: `{D}`. Packet generated from the in-repo source recordings; every clip's PCM-SHA-256 was re-verified against the manifest.
Stereo files: **left channel = user, right channel = assistant**. `user_only/` has the user channel alone.

## What to listen for (per pair, ~1 minute)

1. **Same recording.** Within a pair the user's voice, timbre, speed and room are identical up to the cut (cut pairs) or for the whole word (context pairs). Anything that sounds like a second take is a defect.
2. **Cut pairs (CUT_IC, CUT_WAIT):** member b stops at a word boundary. It must sound like a clean stop (no click, no half-word, no clipped consonant). Is member b's last word genuinely **unfinished / incomplete** (CUT_IC: you expect more words) or a **self-contained hold request** (CUT_WAIT: "hold on", "give me a second")? If the prefix could stand as a finished request, mark the label wrong.
3. **Member a** runs to a natural end and sounds finished.
4. **Matched trailing silence.** After the user stops there is exactly 2.0 s of silence in both members (nothing else on the user channel).
5. **Context pairs (CTX_BC):** the user's acknowledgement is the same sound in both; in a the assistant has finished a question and stopped, so the acknowledgement is an answer (complete); in b the assistant keeps talking over it, so it is a backchannel.
6. **Labels.** Do you agree with each expected label? If you would have labelled it differently, say so.
7. Audible TTS defects (garbled words, repeated syllables, wrong speaker) in either member.

Fill `review_form.csv` (Y/N columns + notes). A single `N` on columns 1-4 is a splice defect; labels you disagree with are label defects. Report counts and the pair ids.

## Pairs

"""
    (out / "REVIEW.md").write_text(head + "\n".join(md) + "\n")
    print(f"packet: {out} ({len(pick)} pairs, {len(rows)} clips)")


if __name__ == "__main__":
    main()
