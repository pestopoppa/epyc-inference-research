#!/usr/bin/env python3
"""CS-48 conformance fixtures: deterministic stimuli cut from pinned LibriSpeech test-clean clips (sha256-checked).

These exist ONLY to give the scorers real speech for Silero; the four overlap classes here are structural stand-ins
for the CS-45 manifest / FDB v1.5 material (acoustic content is irrelevant to a timing instrument). Annotations
(user_speech_start_s / user_speech_end_s) are the pinned Silero's first-segment start and last-segment end on the
built stimulus and are re-derived and compared on every conformance run, so they cannot drift silently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir
import bi_audio as A
import bi_vad

HERE = Path(__file__).resolve().parent
LS = Path("/mnt/raid0/llm/data/asr-eval/librispeech-test-clean-strat300/wav")
SRC_SHA = {  # sha256 of the source clips (checked before use)
    "1188-133604-0008": "850ed999201f",
    "121-121726-0006": "af8d05690b77",
    "1089-134686-0010": "a2f7cf27e284",
    "121-121726-0011": "580927cb8be3",
    "1320-122612-0014": "cec8fe56959f",
}
# (id, overlap_class, source, from_s, to_s, gain)
SPECS = [
    ("interruption-01", "user_interruption", "1089-134686-0010", 0.40, 4.41, 1.0),
    ("backchannel-01", "backchannel", "121-121726-0006", 0.40, 1.40, 1.0),
    ("talking-to-others-01", "talking_to_others", "121-121726-0011", 0.50, 4.04, 0.7),
    ("background-speech-01", "background_speech", "1320-122612-0014", 0.30, 3.30, 0.25),
]
AGENT_SRC = ("1188-133604-0008", 0.45, 18.0)  # response speech: clip, from_s, to_s (leading silence trimmed to ~0.1 s)
TAIL_S = 0.3


def _load(name: str) -> np.ndarray:
    p = LS / f"{name}.wav"
    if not A.file_sha256(p).startswith(SRC_SHA[name]):
        raise RuntimeError(f"fixture source {p} does not match its pinned sha256 prefix")
    x, sr = A.read_wav(p)
    assert sr == A.SR
    return x


def build_stimulus(spec) -> np.ndarray:
    _, _, src, a, b, gain = spec
    x = _load(src)[int(a * A.SR) : int(b * A.SR)].astype(np.float64) * gain
    x = np.clip(np.rint(x), -32768, 32767).astype("<i2")
    return np.concatenate([x, np.zeros(int(TAIL_S * A.SR), dtype="<i2")])


def agent_clip(model_sr: int = A.SR) -> np.ndarray:
    src, a, b = AGENT_SRC
    x = _load(src)[int(a * A.SR) : int(b * A.SR)]
    return A.resample_int16(x, A.SR, model_sr)


def annotate(x: np.ndarray) -> tuple[float, float]:
    segs = bi_vad.speech_segments_s(x)
    if not segs:
        raise RuntimeError("pinned Silero found no speech in the stimulus")
    return segs[0][0], segs[-1][1]


def materialise(outdir: str | Path, offset_s: float = 3.0) -> list[dict]:
    """Write the four stimulus wavs; return scenario dicts (bi-scenario-v1) using the committed annotations."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ann = json.loads((HERE / "fixtures" / "scenarios.json").read_text())["annotations"]
    res = []
    for spec in SPECS:
        x = build_stimulus(spec)
        p = out / f"{spec[0]}.wav"
        A.write_wav(p, x)
        a = ann[spec[0]]
        res.append({"id": spec[0], "overlap_class": spec[1], "stimulus_wav": str(p), "stimulus_offset_s": offset_s,
                    "user_speech_start_s": a["user_speech_start_s"], "user_speech_end_s": a["user_speech_end_s"],
                    "stimulus_pcm_sha256": A.pcm_sha256(x), "gain": 1.0,
                    "source": {"clip": spec[2], "from_s": spec[3], "to_s": spec[4], "gain": spec[5]}})
    return res


def main() -> None:
    ann = {}
    for spec in SPECS:
        x = build_stimulus(spec)
        s, e = annotate(x)
        ann[spec[0]] = {"user_speech_start_s": s, "user_speech_end_s": e, "stimulus_pcm_sha256": A.pcm_sha256(x)}
    (HERE / "fixtures").mkdir(exist_ok=True)
    (HERE / "fixtures" / "scenarios.json").write_text(json.dumps(
        {"schema": "bi-fixtures-v1", "specs": SPECS, "agent_clip": AGENT_SRC, "annotations": ann}, indent=1, sort_keys=True) + "\n")
    print(json.dumps(ann, indent=1))


if __name__ == "__main__":
    main()
