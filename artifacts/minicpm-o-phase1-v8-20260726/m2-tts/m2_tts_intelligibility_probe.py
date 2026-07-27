#!/usr/bin/env python3
"""CPU-only, observation-grade ASR probe for sealed MiniCPM-o TTS chunks.

This consumes an already sealed Path-B TTS run.  It does not generate speech,
alter a serving stack, or make a perceptual-quality or lineup claim.  The ASR
model path must be an existing local CTranslate2 snapshot so this program has
no model-download code path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_wav(path: Path) -> tuple[dict[str, Any], bytes]:
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        payload = handle.readframes(params.nframes)
    if params.comptype != "NONE" or not payload:
        raise ValueError(f"unsupported or empty WAV: {path}")
    return ({
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "channels": params.nchannels,
        "sample_width_bytes": params.sampwidth,
        "sample_rate_hz": params.framerate,
        "frames": params.nframes,
        "duration_seconds": params.nframes / params.framerate,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }, payload)


def write_concat(chunks: list[Path], output: Path) -> dict[str, Any]:
    inspected = [inspect_wav(chunk) for chunk in chunks]
    first, _ = inspected[0]
    format_keys = ("channels", "sample_width_bytes", "sample_rate_hz")
    if any(any(item[key] != first[key] for key in format_keys) for item, _ in inspected[1:]):
        raise ValueError("WAV chunks do not share an identical PCM format")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite published concatenation: {output}")
    payload = b"".join(item_payload for _, item_payload in inspected)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(first["channels"])
        handle.setsampwidth(first["sample_width_bytes"])
        handle.setframerate(first["sample_rate_hz"])
        handle.writeframes(payload)
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite published concatenation: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    combined, _ = inspect_wav(output)
    return {"inputs": [item for item, _ in inspected], "combined": combined}


def transcript(model_path: Path, input_path: Path, threads: int) -> dict[str, Any]:
    # Import after local-path checks: importing the library has no network work;
    # passing a literal local directory prevents HF cache resolution/downloads.
    from faster_whisper import WhisperModel

    started = time.monotonic()
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=threads)
    load_seconds = time.monotonic() - started
    started = time.monotonic()
    segments, info = model.transcribe(
        str(input_path), language="en", task="transcribe", vad_filter=False,
        beam_size=5, temperature=0.0,
    )
    parsed = [{"start": item.start, "end": item.end, "text": item.text.strip()} for item in segments]
    elapsed = time.monotonic() - started
    return {
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "language": info.language,
        "language_probability": info.language_probability,
        "duration_seconds": info.duration,
        "load_seconds": load_seconds,
        "transcription_seconds": elapsed,
        "segments": parsed,
        "text": " ".join(item["text"] for item in parsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    wav_dir = run_dir / "server-output" / "round_000" / "tts_wav"
    chunks = [wav_dir / f"wav_{index}.wav" for index in range(3)]
    if not all(path.is_file() for path in chunks):
        raise FileNotFoundError("expected sealed wav_0.wav through wav_2.wav")
    if not (model_path / "model.bin").is_file():
        raise FileNotFoundError("model path is not a local CTranslate2 snapshot")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite observation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    concat = output_dir / "concatenated_pcm.wav"
    wavs = write_concat(chunks, concat)
    argv = [sys.executable, *sys.argv]
    transcription = {path.name: transcript(model_path, path, args.threads) for path in [*chunks, concat]}
    report = {
        "schema_version": 1,
        "classification": "observation_only_cpu_asr_intelligibility_probe",
        "claim_boundary": [
            "ASR agreement with the known generation text is an automated intelligibility signal only.",
            "This is not MOS, voice-quality, latency, production-readiness, or lineup evidence.",
        ],
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime": {
            "argv": argv,
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "cpu_only": True,
            "gpu_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
            "threads": args.threads,
        },
        "model": {
            "path": str(model_path),
            "model_bin_sha256": sha256(model_path / "model.bin"),
            "config_sha256": sha256(model_path / "config.json"),
            "compute_type": "int8",
            "network_disabled_by_design": True,
        },
        "source_run": str(run_dir),
        "wav_validation": wavs,
        "transcriptions": transcription,
        "expected_generation_text": "The MiniCPM audio path is working.",
    }
    report_path = output_dir / "asr_observation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
