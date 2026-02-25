#!/usr/bin/env python3
"""Read a byte range from the inference tap log.

Used by the Claude debugger to examine raw inference output for
specific evaluation answers.

Usage:
    python scripts/benchmark/read_tap_section.py --offset 184320 --length 8192
    python scripts/benchmark/read_tap_section.py --offset 184320 --length 8192 --file /path/to/tap.log
"""

from __future__ import annotations

import argparse
import sys

DEFAULT_TAP_PATH = "/mnt/raid0/llm/tmp/inference_tap.log"


def read_tap_section(path: str, offset: int, length: int) -> str:
    """Read a byte section from the tap log and decode as UTF-8."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return f"[tap file not found: {path}]"
    except OSError as e:
        return f"[tap read error: {e}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a byte range from inference tap log")
    parser.add_argument("--offset", type=int, required=True, help="Byte offset to start reading")
    parser.add_argument("--length", type=int, required=True, help="Number of bytes to read")
    parser.add_argument("--file", default=DEFAULT_TAP_PATH, help=f"Tap file path (default: {DEFAULT_TAP_PATH})")
    args = parser.parse_args()

    text = read_tap_section(args.file, args.offset, args.length)
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
