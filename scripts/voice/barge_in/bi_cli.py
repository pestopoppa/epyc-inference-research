#!/usr/bin/env python3
"""CS-48 command line. ``score`` is offline and cheap; ``play`` here only drives the scripted FAKE agent (the live
cascade run is CS-49 and is the inference-owning session's, never this task's).

  python -I bi_cli.py score <run_dir> [--out score.json] [--asr whisper-cpp]
  python -I bi_cli.py fixture-run <out_dir> --scenario interruption-01 --stop-after-s 0.4 [--real-time]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bi_clock
import bi_fake
import bi_fixtures
import bi_score
import bi_session as S


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("run_dir")
    sc.add_argument("--out")
    sc.add_argument("--asr", choices=["whisper-cpp"])
    sc.add_argument("--cache")
    fr = sub.add_parser("fixture-run")
    fr.add_argument("out_dir")
    fr.add_argument("--scenario", default="interruption-01")
    fr.add_argument("--stop-after-s", type=float, default=0.4)
    fr.add_argument("--real-time", action="store_true")
    a = ap.parse_args()
    if a.cmd == "score":
        asr = None
        if a.asr:
            import bi_vad
            asr = bi_vad.tt("tt_asr_words").PinnedWhisperCpp()
        res = bi_score.score_run(a.run_dir, asr=asr, cache=a.cache)
        b = bi_score.canonical_bytes(res)
        Path(a.out or Path(a.run_dir) / "score.json").write_bytes(b)
        sys.stdout.write(b.decode())
        return 0
    scen = {s["id"]: s for s in bi_fixtures.materialise(Path(a.out_dir) / "stimuli")}[a.scenario]
    ns = 1_000_000_000
    us = int(round((scen["stimulus_offset_s"] + scen["user_speech_start_s"] + a.stop_after_s) * ns))
    pcm = bi_fixtures.agent_clip(16000)
    S.run_pair(scen, bi_fake.make_factory(pcm, 16000),
               lambda cond: S.ScriptedPolicy([{"at_ns": us, "do": "cancel"}] if cond == "overlap" else []),
               Path(a.out_dir) / "run", clock_factory=bi_clock.RealClock if a.real_time else bi_clock.VirtualClock)
    print(json.dumps({"run_dir": str(Path(a.out_dir) / "run")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
