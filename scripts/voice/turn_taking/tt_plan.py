#!/usr/bin/env python3
"""CS-45 deterministic plan: render jobs and pair definitions from the authored text inventory.

Pure data (no audio, no tools). The same plan feeds the renderer (tt_render_pipeline.py) and the
corpus builder (tt_build2.py), so a re-run reproduces the identical job list and pair assignment.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import tt_spec as S

PLAN_SEED = "cs45-turn-taking-v1"
LIBRI = Path("/mnt/raid0/llm/data/asr-eval/librispeech-test-clean-strat300")

# User voices: LibriSpeech test-clean speakers used ONLY as speaker-embedding references for the
# qwentts Base model (--ref-wav, no ref text). Genders were read off median F0 of the reference clip.
# selection and confirmation use DISJOINT speakers; the assistant voice is a ninth, separate speaker.
VOICES = [
    {"voice_id": "synth-u01", "libri_speaker": 1089, "ref": "1089-134686-0031", "gender": "M", "split": "selection"},
    {"voice_id": "synth-u02", "libri_speaker": 260, "ref": "260-123440-0010", "gender": "F", "split": "selection"},
    {"voice_id": "synth-u03", "libri_speaker": 5105, "ref": "5105-28233-0002", "gender": "M", "split": "selection"},
    {"voice_id": "synth-u04", "libri_speaker": 4507, "ref": "4507-16021-0025", "gender": "F", "split": "selection"},
    {"voice_id": "synth-u05", "libri_speaker": 2300, "ref": "2300-131720-0034", "gender": "M", "split": "confirmation"},
    {"voice_id": "synth-u06", "libri_speaker": 1580, "ref": "1580-141084-0026", "gender": "F", "split": "confirmation"},
    {"voice_id": "synth-u07", "libri_speaker": 4077, "ref": "4077-13754-0014", "gender": "M", "split": "confirmation"},
    {"voice_id": "synth-u08", "libri_speaker": 2094, "ref": "2094-142345-0005", "gender": "F", "split": "confirmation"},
]
ASSISTANT_VOICE = {"voice_id": "synth-asst", "libri_speaker": 3729, "ref": "3729-6852-0044", "gender": "n/a", "split": "shared-constant"}

SPLITS = ("selection", "confirmation")


def voice_ref_path(v: dict) -> str:
    return str(LIBRI / "wav" / f"{v['ref']}.wav")


def _seed(render_id: str) -> int:
    return int(hashlib.sha256((PLAN_SEED + "/" + render_id).encode()).hexdigest()[:8], 16) % 2_000_000_000


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("|", " ")).strip()


def norm_words(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()


def build_plan() -> dict:
    """Return {'jobs': [...], 'pairs': [...]} ; every id is a pure function of the spec order."""
    jobs: list[dict] = []
    pairs: list[dict] = []

    def split_voices(split: str):
        return [v for v in VOICES if v["split"] == split]

    # --- assistant recordings, partitioned by split so assistant contexts never cross the boundary
    def add_assist(kind: str, texts: list[str]) -> dict:
        pools = {"selection": [], "confirmation": []}
        for i, t in enumerate(texts):
            split = SPLITS[i % 2]
            rid = f"rec-a-{kind}-{i:02d}"
            jobs.append(
                {"render_id": rid, "role": "assistant", "kind": kind, "text": t, "voice_id": ASSISTANT_VOICE["voice_id"],
                 "ref": voice_ref_path(ASSISTANT_VOICE), "seed": _seed(rid), "split": split}
            )
            pools[split].append(rid)
        return pools

    pool_q = add_assist("question", S.ASSIST_QUESTION)
    pool_m = add_assist("monologue", S.ASSIST_MONOLOGUE)
    pool_p = add_assist("prompt", S.ASSIST_PROMPT)

    def add_user(rid: str, text: str, voice: dict, split: str, **extra) -> dict:
        j = {"render_id": rid, "role": "user", "text": clean_text(text), "voice_id": voice["voice_id"],
             "ref": voice_ref_path(voice), "seed": _seed(rid), "split": split}
        j.update(extra)
        jobs.append(j)
        return j

    counters = {"selection": 0, "confirmation": 0}
    pair_no = 0

    def next_voice(split: str) -> dict:
        vs = split_voices(split)
        v = vs[counters[split] % len(vs)]
        counters[split] += 1
        return v

    def next_pair_id() -> str:
        nonlocal pair_no
        pair_no += 1
        return f"tt1-p{pair_no:04d}"

    # --- CUT_IC and CUT_WAIT: one recording per pair; split alternates so each split gets half
    for fam, texts, a_label, b_label in (
        ("CUT_IC", S.CUT_IC, "complete", "incomplete"),
        ("CUT_WAIT", S.CUT_WAIT, "complete", "wait"),
    ):
        for i, raw in enumerate(texts):
            split = SPLITS[i % 2]
            voice = next_voice(split)
            pid = next_pair_id()
            pre, post = raw.split("|")
            n_prefix = len(norm_words(pre))
            rid = f"rec-u-{pid[-4:]}"
            add_user(rid, raw, voice, split, family=fam, pair_id=pid, prefix_words=n_prefix)
            pool = pool_p[split]
            asst = pool[(i // 2) % len(pool)]
            pairs.append({
                "pair_id": pid, "family": fam, "split": split, "voice_id": voice["voice_id"], "user_recording": rid,
                "assistant_a": asst, "assistant_b": asst, "assistant_mode_a": "ended", "assistant_mode_b": "ended",
                "label_a": a_label, "label_b": b_label, "contrast": "cut_point", "prefix_words": n_prefix,
                "text_full": clean_text(raw), "text_prefix": clean_text(pre),
            })

    # --- CTX_BC: same acknowledgement waveform; assistant context differs
    k = 0
    for rep in range(S.CTX_REPEATS):
        for ack in S.CTX_ACKS:
            split = SPLITS[(rep + S.CTX_ACKS.index(ack)) % 2]
            voice = next_voice(split)
            pid = next_pair_id()
            rid = f"rec-u-{pid[-4:]}"
            add_user(rid, ack, voice, split, family="CTX_BC", pair_id=pid, prefix_words=0)
            qs, ms = pool_q[split], pool_m[split]
            pairs.append({
                "pair_id": pid, "family": "CTX_BC", "split": split, "voice_id": voice["voice_id"], "user_recording": rid,
                "assistant_a": qs[(k // 2) % len(qs)], "assistant_b": ms[(k // 2) % len(ms)],
                "assistant_mode_a": "ended", "assistant_mode_b": "continuing",
                "label_a": "complete", "label_b": "backchannel", "contrast": "assistant_context", "prefix_words": 0,
                "text_full": clean_text(ack), "text_prefix": None,
            })
            k += 1
    # --- ALTERNATES: extra main-set cut pairs (see tt_spec.ALT_CUT_IC); ids continue the main numbering
    for i, raw in enumerate(S.ALT_CUT_IC):
        split = SPLITS[i % 2]
        voice = next_voice(split)
        pid = next_pair_id()
        pre, post = raw.split("|")
        n_prefix = len(norm_words(pre))
        rid = f"rec-u-{pid[-4:]}"
        add_user(rid, raw, voice, split, family="CUT_IC", pair_id=pid, prefix_words=n_prefix)
        pool = pool_p[split]
        asst = pool[(i // 2 + 2) % len(pool)]
        pairs.append({
            "pair_id": pid, "family": "CUT_IC", "split": split, "voice_id": voice["voice_id"], "user_recording": rid,
            "assistant_a": asst, "assistant_b": asst, "assistant_mode_a": "ended", "assistant_mode_b": "ended",
            "label_a": "complete", "label_b": "incomplete", "contrast": "cut_point", "prefix_words": n_prefix,
            "text_full": clean_text(raw), "text_prefix": clean_text(pre),
        })
    return {"plan_seed": PLAN_SEED, "voices": VOICES, "assistant_voice": ASSISTANT_VOICE, "jobs": jobs, "pairs": pairs}


def build_reserve_plan() -> dict:
    """Reserve stratum: same rules and voice partition as the main plan, NEW texts/recordings/dialogues, and the
    assistant contexts of the main set reused WITHIN the same split only (so no context crosses the boundary)."""
    main = build_plan()
    pools: dict[tuple[str, str], list[str]] = {}
    for j in main["jobs"]:
        if j["role"] == "assistant":
            pools.setdefault((j["kind"], j["split"]), []).append(j["render_id"])
    jobs: list[dict] = []
    pairs: list[dict] = []
    counters = {"selection": 1, "confirmation": 1}  # offset so reserve voices/contexts rotate differently from main
    no = 0

    def voice_for(split: str) -> dict:
        vs = [v for v in VOICES if v["split"] == split]
        v = vs[counters[split] % len(vs)]
        counters[split] += 1
        return v

    def add(fam, raw, a_label, b_label, contrast, ctxs, i):
        nonlocal no
        no += 1
        pid = f"tt1-r{no:04d}"
        split = SPLITS[i % 2]
        voice = voice_for(split)
        rid = f"rec-r-u-{no:04d}"
        pre = raw.split("|")[0] if "|" in raw else None
        n_prefix = len(norm_words(pre)) if pre else 0
        jobs.append({"render_id": rid, "role": "user", "text": clean_text(raw), "voice_id": voice["voice_id"], "ref": voice_ref_path(voice),
                     "seed": _seed(rid), "split": split, "family": fam, "pair_id": pid, "prefix_words": n_prefix})
        a_ctx, b_ctx, a_mode, b_mode = ctxs(split, i)
        pairs.append({"pair_id": pid, "family": fam, "split": split, "voice_id": voice["voice_id"], "user_recording": rid,
                      "assistant_a": a_ctx, "assistant_b": b_ctx, "assistant_mode_a": a_mode, "assistant_mode_b": b_mode,
                      "label_a": a_label, "label_b": b_label, "contrast": contrast, "prefix_words": n_prefix,
                      "text_full": clean_text(raw), "text_prefix": clean_text(pre) if pre else None})

    def prompt_ctx(split, i):
        p = pools[("prompt", split)]
        c = p[(i // 2 + 1) % len(p)]
        return c, c, "ended", "ended"

    for i, raw in enumerate(S.RESERVE_CUT_IC):
        add("CUT_IC", raw, "complete", "incomplete", "cut_point", prompt_ctx, i)
    for i, raw in enumerate(S.RESERVE_CUT_WAIT):
        add("CUT_WAIT", raw, "complete", "wait", "cut_point", prompt_ctx, i)

    def ctx_bc(split, i):
        q, m = pools[("question", split)], pools[("monologue", split)]
        return q[(i // 2 + 2) % len(q)], m[(i // 2 + 1) % len(m)], "ended", "continuing"

    for i, ack in enumerate(S.RESERVE_CTX_ACKS):
        add("CTX_BC", ack, "complete", "backchannel", "assistant_context", ctx_bc, i)
    return {"plan_seed": PLAN_SEED + "/reserve", "voices": VOICES, "assistant_voice": ASSISTANT_VOICE, "jobs": jobs, "pairs": pairs}


if __name__ == "__main__":
    p = build_plan()
    import collections

    print(len(p["jobs"]), "jobs;", len(p["pairs"]), "pairs")
    print(collections.Counter((x["family"], x["split"]) for x in p["pairs"]))
    print(collections.Counter(x["voice_id"] for x in p["pairs"]))
    print(json.dumps(p["jobs"][0], indent=1))
