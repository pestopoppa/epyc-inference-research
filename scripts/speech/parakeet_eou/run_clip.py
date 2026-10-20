"""Run the CS-46 arm over one WAV (channel 0) and print the feed() event stream as JSONL.

usage: python -I run_clip.py --wav W --model-dir DIR(vocab.txt, community fp32/int8 decoders) --export-dir DIR
       [--chunk-ms 160] [--reset none|decoder|full] [--pairing fp32_fp32|fp32_int8dec|fp32_selfdec|fp32_selfdec_int8]
       [--hold-s 1.0] [--frontend exact|nemo_va]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parakeet_eou.engine import OrtGraphs, ParakeetEOUArm, load_vocab  # noqa: E402
from parakeet_eou.parity import PAIRINGS, read_pcm16  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--chunk-ms", type=int, default=160, choices=(80, 160))
    ap.add_argument("--reset", default="none", choices=("none", "decoder", "full"))
    ap.add_argument("--pairing", default="fp32_fp32", choices=sorted(PAIRINGS))
    ap.add_argument("--hold-s", type=float, default=0.0)
    ap.add_argument("--frontend", default="exact", choices=("exact", "nemo_va"))
    a = ap.parse_args()
    dec = PAIRINGS[a.pairing]
    decp = a.export_dir / dec[8:] if dec.startswith("@export/") else a.model_dir / dec
    g = OrtGraphs(a.export_dir / f"encoder_stream_{a.chunk_ms}ms.onnx", decp)
    arm = ParakeetEOUArm(g, load_vocab(a.model_dir / "vocab.txt"), chunk_ms=a.chunk_ms, reset=a.reset,
                         hold_s=a.hold_s, frontend=a.frontend)
    print(json.dumps({"config": arm.config, "pairing": a.pairing, "wav": str(a.wav)}))
    x = read_pcm16(a.wav).astype(np.float32) / 32768.0
    blk = 1280          # arbitrary feed granularity (80 ms); exact frontend is block-size independent
    for i in range(0, len(x), blk):
        for e in arm.feed(x[i:i + blk]):
            print(json.dumps(e))
    for e in arm.finish():
        print(json.dumps(e))


if __name__ == "__main__":
    main()
