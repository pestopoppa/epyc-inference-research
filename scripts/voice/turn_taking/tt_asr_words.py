#!/usr/bin/env python3
"""CS-45 pinned word-timestamp ASR: production whisper.cpp (production-speech-v1) on CPU, CLI only.

Used by patched FDB ``get_transcript/asr.py`` (replacing NeMo parakeet-tdt-0.6b-v2 on CUDA) and by CS-48's
FDB v1 adapter. The binary, tree commit and model are pinned in data/turn-taking-cs45-v1/pins.json;
``PinnedWhisperCpp`` refuses to run if the binary or the model hash differs from the pin.

Word times are whisper.cpp token offsets grouped into words (``-ojf``); DTW token times are kept as an
extra field when the build supports ``--dtw large.v3.turbo``. Nothing here starts a server or a GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = HERE.parent.parent.parent / "data" / "turn-taking-cs45-v1" / "pins.json"
WHISPER_BIN_DIR = "/mnt/raid0/llm/whisper.cpp/build/bin"


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def parse_asr(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    words, cur, text = [], None, []
    for seg in d.get("transcription", []):
        text.append(seg.get("text", ""))
        for tk in seg.get("tokens", []):
            t = tk.get("text", "")
            if t.startswith("[_") or t.startswith("<|") or not t.strip():
                continue
            off = tk.get("offsets", {})
            dtw = tk.get("t_dtw", -1)
            dtw_ms = int(dtw) * 10 if isinstance(dtw, (int, float)) and dtw >= 0 else None
            if cur is None or t.startswith(" "):
                cur = {"w": t.strip(), "t0_ms": off.get("from"), "t1_ms": off.get("to"), "dtw_ms": dtw_ms, "p": tk.get("p")}
                words.append(cur)
            else:
                cur["w"] += t
                cur["t1_ms"] = off.get("to")
                if cur["dtw_ms"] is None:
                    cur["dtw_ms"] = dtw_ms
    return {"text": "".join(text).strip(), "words": words}


@dataclass
class _Result:
    text: str
    timestamp: dict


class PinnedWhisperCpp:
    def __init__(self, threads: int = 8, cpus: str | None = None, verify: bool = True):
        pins = json.loads(PINS.read_text())["asr_word_timestamps"]
        self.pins = pins
        self.binary = f"{WHISPER_BIN_DIR}/whisper-cli"
        self.model = pins["model_file"]
        if verify:
            if _sha(self.binary) != pins["binary_sha256"]:
                raise RuntimeError("whisper-cli binary differs from the CS-45 ASR pin (refused)")
            if _sha(self.model) != pins["model_sha256"]:
                raise RuntimeError("ASR model differs from the CS-45 ASR pin (refused)")
        self.threads, self.cpus = threads, cpus

    def transcribe(self, paths, timestamps: bool = True):
        out = []
        for p in paths:
            with tempfile.TemporaryDirectory() as td:
                base = os.path.join(td, "o")
                argv = [self.binary, "-m", self.model, "-f", p, "-t", str(self.threads), "-ng", "-l", "en", "-ojf", "-of", base, "-np"]
                if self.pins.get("dtw"):
                    argv += ["--dtw", self.pins["dtw"]]
                if self.cpus:
                    argv = ["taskset", "-c", self.cpus] + argv
                env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")}
                env.update({"HIP_VISIBLE_DEVICES": "-1", "LD_LIBRARY_PATH": f"{WHISPER_BIN_DIR}:/opt/rocm/lib"})
                subprocess.run(argv, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
                r = parse_asr(Path(base + ".json"))
            words = [{"word": w["w"], "start": (w["t0_ms"] or 0) / 1000.0, "end": (w["t1_ms"] or 0) / 1000.0} for w in r["words"]]
            out.append(_Result(r["text"], {"word": words}))
        return out
