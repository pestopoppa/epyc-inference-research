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


def scenarios_from_cs45(manifest_jsonl: str | Path, *, stimulus_offset_s: float = 3.0) -> list[dict]:
    """Rows of the CS-45 manifest that carry an overlap class. Each row must provide: id, overlap_class, user_wav,
    user_speech_start_s, user_speech_end_s, pcm_sha256 (of user_wav's int16 samples); optional clean_user_wav. The
    PCM-SHA-256 binding is enforced here (corpus rule): a row whose audio hash differs is refused, never skipped."""
    out = []
    for line in Path(manifest_jsonl).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("overlap_class") not in CLASSES:
            continue
        x, sr = A.read_wav(r["user_wav"])
        if A.pcm_sha256(x) != r["pcm_sha256"]:
            raise ManifestRefused(f"{r['id']}: PCM-SHA-256 {A.pcm_sha256(x)} != manifest {r['pcm_sha256']}")
        sc = {"id": r["id"], "overlap_class": r["overlap_class"], "stimulus_wav": r["user_wav"], "stimulus_offset_s": stimulus_offset_s,
              "user_speech_start_s": r["user_speech_start_s"], "user_speech_end_s": r["user_speech_end_s"],
              "stimulus_pcm_sha256": r["pcm_sha256"], "source": {"set": "cs45-manifest"}}
        if r.get("clean_user_wav"):
            sc["clean_stimulus_wav"] = r["clean_user_wav"]
        out.append(sc)
    return out
