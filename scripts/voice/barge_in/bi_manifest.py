#!/usr/bin/env python3
"""CS-48 scenario sources: FDB v1.5 sample folders and CS-45 manifest rows -> ``bi-scenario-v1`` dicts.

bi-scenario-v1: id, overlap_class in {user_interruption, backchannel, talking_to_others, background_speech},
stimulus_wav (user audio WITH the overlap event), clean_stimulus_wav (optional; same user audio WITHOUT it),
stimulus_offset_s (how long the cascade has already been speaking when the stimulus starts), user_speech_start_s /
user_speech_end_s (the overlap window inside the stimulus, seconds), stimulus_pcm_sha256 (checked when present).

FDB v1.5 data (Google Drive, https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3) is a MANUAL
operator fetch: this module only reads a local copy and never contacts Drive. Layout per FDB ``dataset/README.md``:
``<root>/<subset>/<sample>/{input.wav, clean_input.wav, metadata.json{timestamps:[start,end]}}`` with subsets
user_interruption, user_backchannel, talking_to_other, background_speech.
"""
from __future__ import annotations

import json
from pathlib import Path

import bi_audio as A

FDB15_SUBSETS = {
    "user_interruption": "user_interruption",
    "user_backchannel": "backchannel",
    "talking_to_other": "talking_to_others",
    "background_speech": "background_speech",
}
CLASSES = tuple(FDB15_SUBSETS.values())


class ManifestRefused(ValueError):
    pass


def scenarios_from_fdb15(root: str | Path, *, stimulus_offset_s: float = 3.0, limit: int | None = None) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        raise ManifestRefused(f"FDB v1.5 data not present at {root} (manual Google Drive fetch pending)")
    out = []
    for subset, cls in FDB15_SUBSETS.items():
        d = root / subset
        if not d.is_dir():
            continue
        for sd in sorted(p for p in d.iterdir() if p.is_dir())[:limit]:
            meta = json.loads((sd / "metadata.json").read_text())
            s, e = meta["timestamps"]
            sc = {"id": f"fdb15/{subset}/{sd.name}", "overlap_class": cls, "stimulus_wav": str(sd / "input.wav"),
                  "stimulus_offset_s": stimulus_offset_s, "user_speech_start_s": float(s), "user_speech_end_s": float(e),
                  "source": {"set": "full-duplex-bench-v1.5", "subset": subset, "sample": sd.name}}
            if (sd / "clean_input.wav").exists():
                sc["clean_stimulus_wav"] = str(sd / "clean_input.wav")
            out.append(sc)
    if not out:
        raise ManifestRefused(f"no FDB v1.5 samples under {root}")
    return out


def corpus_dir() -> Path:
    import bi_fdb
    return bi_fdb.cs45_data()


def scenarios_from_cs45(clips_dir: str | Path, out_dir: str | Path, *, corpus: str | Path | None = None,
                        stimulus_offset_s: float = 0.0, strata: tuple[str, ...] | None = None, clip_ids: set[str] | None = None) -> list[dict]:
    """Overlap scenarios from the real CS-45 manifest (``manifest/clips.jsonl`` + ``human_recorded.jsonl``).

    Admitted rows: label == "backchannel" (the user speaks a short acknowledgement while the assistant holds the floor;
    overlap class ``backchannel``) and any row carrying an explicit ``overlap_class`` in {user_interruption, backchannel,
    talking_to_others, background_speech} (the human-recorded stratum / FDB-style material; the synthesized stratum has
    no talking_to_others / background_speech rows by contract). Other labels (complete / incomplete / wait) are
    turn-end clips for CS-47, not overlap scenarios, and are skipped.

    Audio: the materialised stereo clip ``<clips_dir>/<clip_id>.wav`` (ch0 user, ch1 assistant; produce with
    ``scripts/voice/turn_taking/tt_materialize.py``). Its decoded PCM-SHA-256 is re-checked against the manifest row with
    CS-45's own ``tt_audio`` (mismatch -> ManifestRefused, never skipped). Only ch0 is the stimulus: the assistant channel
    is the recorded context and is replaced by the cascade under test. Clip time is kept (offset 0.0), so the user
    window is the row's ``gold.user_speech_start_s/end_s`` and the cascade has been speaking since t0 as in the clip.
    The clean condition is digital silence of the same length (no user at all)."""
    import bi_vad

    ta = bi_vad.tt("tt_audio")
    D = Path(corpus) if corpus else corpus_dir()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in ("clips.jsonl", "human_recorded.jsonl"):
        p = D / "manifest" / f
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    res = []
    for r in rows:
        cls = r.get("overlap_class") or ("backchannel" if r["label"] == "backchannel" else None)
        if cls not in CLASSES or (strata and r["stratum"] not in strata) or (clip_ids is not None and r["clip_id"] not in clip_ids):
            continue
        wav = Path(clips_dir) / f"{r['clip_id']}.wav"
        if not wav.exists():
            raise ManifestRefused(f"{r['clip_id']}: materialised clip {wav} missing (run tt_materialize.py)")
        pcm = ta.read_wav(wav)
        if pcm.pcm_sha256() != r["pcm"]["pcm_sha256"]:
            raise ManifestRefused(f"{r['clip_id']}: PCM-SHA-256 {pcm.pcm_sha256()} != manifest {r['pcm']['pcm_sha256']}")
        user = pcm.samples[:, 0].astype("<i2")
        if pcm.sample_rate != A.SR:
            raise ManifestRefused(f"{r['clip_id']}: sample rate {pcm.sample_rate}")
        sp = out / f"{r['clip_id']}.user.wav"
        A.write_wav(sp, user)
        g = r["gold"]
        res.append({"id": r["clip_id"], "overlap_class": cls, "stimulus_wav": str(sp), "stimulus_offset_s": stimulus_offset_s,
                    "user_speech_start_s": g["user_speech_start_s"], "user_speech_end_s": g["user_speech_end_s"],
                    "stimulus_pcm_sha256": A.pcm_sha256(user),
                    "source": {"set": "cs45-manifest", "stratum": r["stratum"], "split": r["split"], "label": r["label"],
                               "pair_id": r.get("pair_id"), "manifest_pcm_sha256": r["pcm"]["pcm_sha256"]}})
    return res
