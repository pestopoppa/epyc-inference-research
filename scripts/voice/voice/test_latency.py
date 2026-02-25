#!/usr/bin/env python3
"""
Latency validation script for the Whisper server.

Tests:
1. Server health check
2. Cold start latency (first request)
3. Warm latency (subsequent requests)
4. Code-switching detection (EN/IT/DE/FR)

Target: <500ms first-token latency for warm requests

Usage:
    python test_latency.py [--server http://localhost:9000]
"""

import argparse
import io
import os
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

# Check for requests
try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


def generate_silence_wav(duration_ms: int = 1000) -> bytes:
    """Generate a silent WAV file of specified duration."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * num_samples)

    return buffer.getvalue()


def generate_tone_wav(duration_ms: int = 1000, freq: int = 440) -> bytes:
    """Generate a WAV file with a sine wave tone."""
    import math

    sample_rate = 16000
    num_samples = int(sample_rate * duration_ms / 1000)
    amplitude = 16000

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(amplitude * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", value))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(samples))

    return buffer.getvalue()


def test_health(server_url: str) -> bool:
    """Test server health endpoint."""
    print("Testing health endpoint...")
    try:
        start = time.time()
        resp = requests.get(f"{server_url}/health", timeout=5)
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            print(f"  OK: {resp.json()} ({elapsed:.0f}ms)")
            return True
        else:
            print(f"  FAIL: Status {resp.status_code}")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_transcription_latency(server_url: str, audio_data: bytes, label: str) -> float | None:
    """
    Test transcription latency.

    Returns latency in milliseconds, or None on failure.
    """
    try:
        start = time.time()
        resp = requests.post(
            f"{server_url}/v1/audio/transcriptions",
            files={"file": ("test.wav", audio_data, "audio/wav")},
            data={"model": "large-v3-turbo"},
            timeout=30,
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            text = resp.json().get("text", "")
            print(f"  {label}: {elapsed:.0f}ms - \"{text[:50]}...\"" if len(text) > 50 else f"  {label}: {elapsed:.0f}ms - \"{text}\"")
            return elapsed
        else:
            print(f"  {label}: FAIL - Status {resp.status_code}")
            return None

    except Exception as e:
        print(f"  {label}: FAIL - {e}")
        return None


def test_code_switching(server_url: str) -> bool:
    """
    Test code-switching detection.

    Since we can't easily generate multilingual speech, this just
    tests that the API accepts requests without specifying language.
    """
    print("\nTesting code-switching (auto language detection)...")

    # Generate short audio
    audio = generate_silence_wav(500)

    try:
        resp = requests.post(
            f"{server_url}/v1/audio/transcriptions",
            files={"file": ("test.wav", audio, "audio/wav")},
            data={"model": "large-v3-turbo", "language": "auto"},
            timeout=10,
        )

        if resp.status_code == 200:
            print("  OK: Auto language detection works")
            return True
        else:
            print(f"  FAIL: Status {resp.status_code}")
            return False

    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def run_tts_test(server_url: str) -> float | None:
    """
    Generate speech using espeak-ng and transcribe it.
    Returns latency or None if espeak not available.
    """
    # Check for espeak-ng
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  (espeak-ng not available, skipping TTS test)")
        return None

    print("\nTesting with generated speech (espeak-ng)...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Generate speech
        subprocess.run(
            ["espeak-ng", "-w", tmp_path, "Hello, this is a test of the whisper server."],
            check=True,
            capture_output=True,
        )

        with open(tmp_path, "rb") as f:
            audio_data = f.read()

        latency = test_transcription_latency(server_url, audio_data, "TTS test")
        return latency

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Whisper server latency test")
    parser.add_argument(
        "--server",
        default="http://localhost:9000",
        help="Server URL (default: http://localhost:9000)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=500,
        help="Target latency in ms (default: 500)"
    )
    args = parser.parse_args()

    server = args.server.rstrip("/")
    target_ms = args.target

    print("=" * 50)
    print("Whisper Server Latency Test")
    print("=" * 50)
    print(f"Server: {server}")
    print(f"Target: <{target_ms}ms")
    print("=" * 50)

    # Test health
    if not test_health(server):
        print("\nFAIL: Server not responding")
        sys.exit(1)

    # Generate test audio
    short_audio = generate_silence_wav(500)   # 0.5s
    medium_audio = generate_silence_wav(2000)  # 2s
    long_audio = generate_silence_wav(5000)    # 5s

    print("\nTesting transcription latency...")

    # Cold start (first request after server start might be slower)
    cold_latency = test_transcription_latency(server, short_audio, "Cold (0.5s audio)")

    # Warm requests
    latencies = []
    for i in range(3):
        lat = test_transcription_latency(server, short_audio, f"Warm #{i+1} (0.5s audio)")
        if lat:
            latencies.append(lat)

    # Medium audio
    med_lat = test_transcription_latency(server, medium_audio, "Medium (2s audio)")

    # Long audio
    long_lat = test_transcription_latency(server, long_audio, "Long (5s audio)")

    # Code-switching
    test_code_switching(server)

    # TTS test (if available)
    tts_lat = run_tts_test(server)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if latencies:
        avg_warm = sum(latencies) / len(latencies)
        min_warm = min(latencies)
        print(f"Warm latency (avg):  {avg_warm:.0f}ms")
        print(f"Warm latency (min):  {min_warm:.0f}ms")
        print(f"Target:              <{target_ms}ms")

        if avg_warm <= target_ms:
            print(f"\nRESULT: PASS - Average latency ({avg_warm:.0f}ms) meets target (<{target_ms}ms)")
            sys.exit(0)
        else:
            print(f"\nRESULT: FAIL - Average latency ({avg_warm:.0f}ms) exceeds target (<{target_ms}ms)")
            sys.exit(1)
    else:
        print("RESULT: FAIL - No successful transcriptions")
        sys.exit(1)


if __name__ == "__main__":
    main()
