#!/usr/bin/env python3
"""Aggregate raw/*.jsonl into markdown tables (median, min-max) for README.md."""
import json, statistics as st, collections, glob, sys

RAW = "/mnt/raid0/llm/epyc-inference-research/artifacts/speech_cpu_realtime_20260924/raw"


def load(name):
    out = []
    for f in sorted(glob.glob(f"{RAW}/{name}")):
        for l in open(f):
            l = l.strip()
            if l:
                out.append(json.loads(l))
    return out


def fmt(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = st.median(xs)
    return f"{m:.{nd}f}" if len(xs) == 1 else f"{m:.{nd}f} ({min(xs):.{nd}f}-{max(xs):.{nd}f})"


print("## STT solo sweep (whisper-server -ng, large-v3-turbo)\n")
print("| threads | cores | clip | n | wall s | RTF |")
print("|---|---|---|---|---|---|")
g = collections.defaultdict(list)
for r in load("stt_sweep*.jsonl"):
    g[(r["threads"], r["cores"], r["clip"])].append(r)
for k in sorted(g, key=lambda k: (k[0], k[1], k[2] != "short_11s.wav")):
    rs = g[k]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | {fmt([r['wall_s'] for r in rs])} | {fmt([r['rtf'] for r in rs], 3)} |")

print("\n## TTS solo sweep (tts-server CPU, Qwen3-TTS 0.6B Q8_0)\n")
print("| threads | cores | text | format | n | first packet s | wall s | audio s | RTF |")
print("|---|---|---|---|---|---|---|---|---|")
g = collections.defaultdict(list)
for r in load("tts_sweep*.jsonl"):
    g[(r["threads"], r["cores"], r["text"], r["fmt"])].append(r)
for k in sorted(g, key=lambda k: (k[0], k[1], k[2] != "short", k[3])):
    rs = g[k]
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(rs)} | {fmt([r['first_packet_s'] for r in rs], 3)} | "
          f"{fmt([r['wall_s'] for r in rs])} | {rs[0]['audio_s']} | {fmt([r['rtf'] for r in rs], 3)} |")

conc = load("concurrent.jsonl")
if conc:
    print("\n## Concurrent (same layout; solo baselines taken with both servers resident)\n")
    print("| layout tag | mode | component | item | n | first packet / TTFT s | wall s | RTF | prebuffer s | LLM gen tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    g = collections.defaultdict(list)
    for r in conc:
        item = r.get("clip") or r.get("text") or "300-tok answer"
        g[(r["tag"], r["mode"], r["kind"], item)].append(r)
    order = {"solo-stt": 0, "solo-tts": 1, "solo-llm": 2, "stt+tts": 3, "stt+tts+llm": 4}
    for k in sorted(g, key=lambda k: (k[0], order.get(k[1], 9), k[2], k[3])):
        rs = g[k]
        if k[2] == "llm":
            tps = [r.get("llm_timings", {}).get("predicted_per_second") if isinstance(r.get("llm_timings"), dict) else None for r in rs]
            print(f"| {k[0]} | {k[1]} | frontdoor | {k[3]} | {len(rs)} | {fmt([r['llm']['ttft_s'] for r in rs], 3)} | "
                  f"{fmt([r['llm']['wall_s'] for r in rs])} | - | - | {fmt(tps, 1)} |")
        elif k[2] == "stt":
            print(f"| {k[0]} | {k[1]} | STT | {k[3]} | {len(rs)} | - | {fmt([r['wall_s'] for r in rs])} | "
                  f"{fmt([r['rtf'] for r in rs], 3)} | - | - |")
        else:
            print(f"| {k[0]} | {k[1]} | TTS | {k[3]} | {len(rs)} | {fmt([r['first_packet_s'] for r in rs], 3)} | "
                  f"{fmt([r['wall_s'] for r in rs])} | {fmt([r['rtf'] for r in rs], 3)} | "
                  f"{fmt([r.get('prebuffer_needed_s') for r in rs], 3)} | - |")
