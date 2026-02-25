#!/usr/bin/env python3
"""End-to-end validation of llama-tts-qwen3 codec token output.

Runs the C++ binary to generate codec tokens, then decodes them to audio
using the Qwen3-TTS Tokenizer Decoder (PyTorch).

Usage:
  python validate_tts_e2e.py \
    --text "Hello, this is a test." \
    --output /mnt/raid0/llm/tmp/tts_test.wav \
    [--reference]  # also generate PyTorch reference for comparison

Requires: qwen_tts package, torch, soundfile
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Paths
BINARY = "/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-tts-qwen3"
TALKER_GGUF = "/mnt/raid0/llm/models/Qwen3-TTS-12Hz-0.6B-Talker-Q4_K_M.gguf"
CP_GGUF = "/mnt/raid0/llm/models/Qwen3-TTS-12Hz-0.6B-CodePredictor-Q8_0.gguf"
SIDECAR = "/mnt/raid0/llm/models/qwen3-tts-sidecar.bin"
TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def run_cpp_binary(text: str, max_frames: int = 200, seed: int = 42,
                   temperature: float = 0.9, threads: int = 48) -> tuple[np.ndarray, float]:
    """Run the C++ binary and parse codec tokens from stdout."""
    cmd = [
        BINARY,
        "--model-talker", TALKER_GGUF,
        "--model-cp", CP_GGUF,
        "--sidecar", SIDECAR,
        "-p", text,
        "--max-frames", str(max_frames),
        "--temp", str(temperature),
        "--top-k", "50",
        "--rep-penalty", "1.05",
        "--seed", str(seed),
        "-t", str(threads),
    ]

    import os
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    print(f"Running: {' '.join(cmd[:4])} ... -p \"{text}\"")
    t0 = time.time()
    result = subprocess.run(
        ["numactl", "--interleave=all"] + cmd,
        capture_output=True, text=True, env=env, timeout=300
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"ERROR: Binary exited with code {result.returncode}")
        print(f"stderr: {result.stderr[-500:]}")
        sys.exit(1)

    # Parse codec tokens from stdout
    frames = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            codes = [int(x) for x in line.strip().split()]
            if len(codes) == 16:
                frames.append(codes)

    if not frames:
        print("ERROR: No frames generated")
        sys.exit(1)

    codes_array = np.array(frames, dtype=np.int64)  # shape: (n_frames, 16)
    print(f"C++ binary: {len(frames)} frames in {elapsed:.2f}s "
          f"({len(frames) / 12.5 / elapsed:.2f}x real-time)")

    # Extract timing from stderr
    for line in result.stderr.split("\n"):
        if "Generated" in line:
            print(f"  Internal timing: {line.strip()}")

    return codes_array, elapsed


def decode_to_audio(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Decode codec tokens to audio waveform using Qwen3-TTS Tokenizer Decoder."""
    print(f"\nLoading Tokenizer Decoder from {TOKENIZER_MODEL}...")
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        TOKENIZER_MODEL,
        device_map="cpu",
        dtype=torch.float32,
    )

    # Prepare codes: (n_frames, 16) -> (1, n_frames, 16) as torch.LongTensor
    codes_tensor = torch.from_numpy(codes).long().unsqueeze(0)  # (1, T, 16)
    codes_tensor = torch.clamp(codes_tensor, min=0, max=2047)

    print(f"Decoding {codes.shape[0]} frames ({codes.shape[0] / 12.5:.2f}s audio)...")
    t0 = time.time()

    # Create the encoded output dict expected by decode()
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2EncoderOutput,
    )
    encoded = Qwen3TTSTokenizerV2EncoderOutput(audio_codes=codes_tensor)

    with torch.no_grad():
        wavs, sr = tokenizer.decode(encoded)

    decode_time = time.time() - t0
    wav = wavs[0]  # first (only) batch element
    print(f"Decoded to {len(wav)} samples @ {sr}Hz ({len(wav)/sr:.2f}s) in {decode_time:.2f}s")

    return wav, sr


def generate_pytorch_reference(text: str, max_frames: int = 200) -> tuple[np.ndarray, int]:
    """Generate audio using full PyTorch pipeline for comparison."""
    print(f"\nGenerating PyTorch reference...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        TTS_MODEL,
        device_map="cpu",
        dtype=torch.float32,
    )

    t0 = time.time()
    wav, sr = model.generate_custom_voice(
        text=text,
        language="english",
    )
    elapsed = time.time() - t0
    print(f"PyTorch reference: {len(wav)/sr:.2f}s audio in {elapsed:.2f}s "
          f"({len(wav)/sr/elapsed:.2f}x real-time)")

    return wav, sr


def main():
    parser = argparse.ArgumentParser(description="End-to-end TTS validation")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the text to speech system.")
    parser.add_argument("--output", type=str, default="/mnt/raid0/llm/tmp/tts_cpp_output.wav")
    parser.add_argument("--reference", action="store_true", help="Also generate PyTorch reference")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=48)
    args = parser.parse_args()

    import soundfile as sf

    # Step 1: Run C++ binary
    print("=" * 60)
    print("Step 1: Generate codec tokens with C++ binary")
    print("=" * 60)
    codes, cpp_time = run_cpp_binary(args.text, args.max_frames, args.seed, threads=args.threads)
    print(f"Codec tokens shape: {codes.shape}")
    print(f"Token range: [{codes.min()}, {codes.max()}]")

    # Step 2: Decode to audio
    print("\n" + "=" * 60)
    print("Step 2: Decode codec tokens to audio")
    print("=" * 60)
    wav, sr = decode_to_audio(codes)

    # Step 3: Save WAV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav, sr)
    print(f"\nSaved: {output_path} ({len(wav)/sr:.2f}s @ {sr}Hz)")

    # Step 4: PyTorch reference (optional)
    if args.reference:
        print("\n" + "=" * 60)
        print("Step 3: Generate PyTorch reference")
        print("=" * 60)
        ref_wav, ref_sr = generate_pytorch_reference(args.text, args.max_frames)
        ref_path = str(output_path).replace(".wav", "_pytorch_ref.wav")
        sf.write(ref_path, ref_wav, ref_sr)
        print(f"Saved reference: {ref_path} ({len(ref_wav)/ref_sr:.2f}s @ {ref_sr}Hz)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    audio_duration = len(wav) / sr
    print(f"Text: \"{args.text}\"")
    print(f"Frames: {codes.shape[0]} ({codes.shape[0]/12.5:.2f}s @ 12.5Hz)")
    print(f"Audio: {audio_duration:.2f}s @ {sr}Hz")
    print(f"C++ generation time: {cpp_time:.2f}s")
    print(f"Real-time factor: {audio_duration / cpp_time:.2f}x")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
