#!/usr/bin/env python3
"""
Batch transcription utility for audio/video files.

Uses faster-whisper directly (not the server) for maximum throughput.
Supports: WAV, MP3, M4A, FLAC, OGG, MP4, MKV, AVI, MOV, WEBM

Usage:
    python transcribe_batch.py input.mp4
    python transcribe_batch.py input.mp4 -o output.txt
    python transcribe_batch.py *.mp3 -o transcripts/
    python transcribe_batch.py input.mp4 --format srt
    python transcribe_batch.py input.mp4 --language en

Output formats:
    txt  - Plain text (default)
    srt  - SubRip subtitles
    vtt  - WebVTT subtitles
    json - Detailed JSON with timestamps
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

# Lazy import for help without dependencies
WhisperModel = None
model_instance = None


def get_model(model_name: str = "large-v3-turbo"):
    """Get or initialize the whisper model."""
    global model_instance, WhisperModel

    if model_instance is not None:
        return model_instance

    if WhisperModel is None:
        from faster_whisper import WhisperModel as WM
        WhisperModel = WM

    print(f"Loading model: {model_name}...", file=sys.stderr)
    start = time.time()

    model_instance = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=int(os.environ.get("OMP_NUM_THREADS", 64)),
    )

    print(f"Model loaded in {time.time() - start:.2f}s", file=sys.stderr)
    return model_instance


def format_timestamp(seconds: float, format_type: str = "srt") -> str:
    """Format seconds as timestamp for subtitles."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if format_type == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:  # srt
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def transcribe_file(
    input_path: Path,
    model_name: str,
    language: str | None,
    output_format: str,
) -> tuple[str, float, float]:
    """
    Transcribe a single audio/video file.

    Returns:
        (transcription_text, audio_duration, elapsed_time)
    """
    model = get_model(model_name)

    print(f"Transcribing: {input_path}", file=sys.stderr)
    start_time = time.time()

    # Handle language parameter
    lang = None if language in (None, "auto", "") else language

    segments, info = model.transcribe(
        str(input_path),
        language=lang,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 100,
        },
        word_timestamps=(output_format in ("srt", "vtt", "json")),
    )

    # Collect segments
    segment_list = list(segments)
    elapsed = time.time() - start_time
    duration = info.duration if info.duration else 0

    rtf = elapsed / duration if duration > 0 else 0
    print(
        f"  Duration: {duration:.1f}s, Time: {elapsed:.2f}s, RTF: {rtf:.3f}x, Lang: {info.language}",
        file=sys.stderr
    )

    # Format output
    if output_format == "txt":
        text = " ".join(seg.text.strip() for seg in segment_list)
        return text, duration, elapsed

    elif output_format == "srt":
        lines = []
        for i, seg in enumerate(segment_list, 1):
            lines.append(str(i))
            lines.append(f"{format_timestamp(seg.start, 'srt')} --> {format_timestamp(seg.end, 'srt')}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines), duration, elapsed

    elif output_format == "vtt":
        lines = ["WEBVTT", ""]
        for seg in segment_list:
            lines.append(f"{format_timestamp(seg.start, 'vtt')} --> {format_timestamp(seg.end, 'vtt')}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines), duration, elapsed

    elif output_format == "json":
        data = {
            "file": str(input_path),
            "language": info.language,
            "duration": duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in (seg.words or [])
                    ] if seg.words else None,
                }
                for seg in segment_list
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False), duration, elapsed

    else:
        raise ValueError(f"Unknown format: {output_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcription for audio/video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_batch.py podcast.mp3
    python transcribe_batch.py lecture.mp4 -o lecture.txt
    python transcribe_batch.py movie.mkv --format srt -o movie.srt
    python transcribe_batch.py *.wav -o transcripts/
    python transcribe_batch.py interview.m4a --language it
        """,
    )
    parser.add_argument("input", nargs="+", help="Input audio/video file(s)")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "-m", "--model",
        default="large-v3-turbo",
        help="Whisper model (default: large-v3-turbo)"
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Language code (e.g., en, it, de) or 'auto' for auto-detect"
    )
    args = parser.parse_args()

    # Collect input files
    input_files = []
    for pattern in args.input:
        path = Path(pattern)
        if path.exists():
            input_files.append(path)
        else:
            # Glob expansion
            from glob import glob
            matches = glob(pattern)
            input_files.extend(Path(m) for m in matches)

    if not input_files:
        print("ERROR: No input files found", file=sys.stderr)
        sys.exit(1)

    # Determine output handling
    output_path = Path(args.output) if args.output else None
    is_output_dir = output_path and (output_path.is_dir() or len(input_files) > 1)

    if is_output_dir and output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    # Process files
    total_duration = 0
    total_time = 0

    for input_file in input_files:
        try:
            text, duration, elapsed = transcribe_file(
                input_file,
                args.model,
                args.language,
                args.format,
            )
            total_duration += duration
            total_time += elapsed

            # Write output
            if is_output_dir and output_path:
                out_file = output_path / f"{input_file.stem}.{args.format}"
            elif output_path:
                out_file = output_path
            else:
                out_file = None

            if out_file:
                out_file.write_text(text, encoding="utf-8")
                print(f"  Output: {out_file}", file=sys.stderr)
            else:
                print(text)

        except Exception as e:
            print(f"ERROR processing {input_file}: {e}", file=sys.stderr)
            continue

    # Summary
    if len(input_files) > 1:
        overall_rtf = total_time / total_duration if total_duration > 0 else 0
        print(
            f"\nTotal: {total_duration:.1f}s audio in {total_time:.2f}s (RTF: {overall_rtf:.3f}x)",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()
