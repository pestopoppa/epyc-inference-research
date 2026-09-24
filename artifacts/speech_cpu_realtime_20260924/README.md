# Production STT + TTS on CPU: can they run in real time? (2026-09-24)

**Question.** Could production STT (whisper.cpp) and TTS (qwentts.cpp) move off the MI210 onto the
EPYC 9655 CPU, freeing about 3+ GiB of VRAM (whisper :9000 holds **2.23 GB** VRAM right now per
`rocm-smi --showpids`; TTS :9002 is declared but not running), while staying real time,
running at the same time as each other, and running alongside the CPU frontdoor, for a
STT → frontdoor → TTS conversation loop? Operator granted the CPU time on 2026-09-24.

**Verdict.**

1. **Each one alone: yes, real time.** STT RTF is 0.12–0.20 at 24–32 threads (5–8× headroom).
   TTS RTF is 0.55–0.60 at 16 or more threads (about 1.7× headroom), and streaming starts in
   75–110 ms for a sentence and 180–330 ms for a paragraph.
2. **STT and TTS together, on separate cores (40 in total): yes, and neither slows down.**
3. **Either one alongside a frontdoor generation: no, and both sides collapse.** TTS RTF goes to
   7.8–9.4. STT RTF goes to 1.1–2.9. The frontdoor drops from 43 to 0.3 tok/s, and from 43 to
   7.4 tok/s even when speech uses only 16 cores. The cause is that the frontdoor runs `-t 96`
   across physical cores 0-95, so **no core in the allowed 0-79 set is free of it**. Both sides
   use spin-barrier threadpools, so whenever one of their threads is time-sliced, every other
   thread waits (mechanism inferred; the effect is measured).

**Recommendation.** Keep STT and TTS on the GPU for the conversation flow as the host is
partitioned today. Moving them to CPU needs an operator topology decision first (see the end of
this file). The CPU-only ASR latency floor also counts against the move: whisper pads every
input to a 30 s window, so an utterance of any length costs about 2 s at 24 threads (3.5 s clip:
1.92–2.03 s). The GPU does the 11 s clip in 0.21 s (freeze record, whisper `b3073792` commit
message).

## How the CPU binaries were obtained

**No build was needed.** The frozen production binaries were run in CPU mode, and nothing was
written inside `/mnt/raid0/llm/whisper.cpp` or `/mnt/raid0/llm/qwentts.cpp`.

| | STT | TTS |
|---|---|---|
| binary | `/mnt/raid0/llm/whisper.cpp/build/bin/whisper-server` (mtime 2026-07-31 11:34, ggml 0.18.0) | `/mnt/raid0/llm/qwentts.cpp/build/tts-server` (mtime 2026-07-31 10:14, ggml 0.17.0, reports `abab6b3`) |
| tree HEAD | `b3073792` production-speech-v1, clean | `2c1b518` production-speech-v1, clean |
| model | `ggml-large-v3-turbo.bin` | `qwen-talker-0.6b-base-Q8_0.gguf` + `qwen-tokenizer-12hz-Q8_0.gguf` |
| CPU switch | `-ng` + `HIP_VISIBLE_DEVICES=-1` | `GGML_BACKEND=CPU` + `HIP_VISIBLE_DEVICES=-1` |
| thread control | `-t N` | **no CLI flag.** `src/backend.h` hardcodes `hardware_concurrency()/2`, which ignores affinity and gives 96 here. An LD_PRELOAD shim ([`nprocs_shim.c`](nprocs_shim.c)) overrides `get_nprocs()` = `SHIM_NPROCS` = 2N. Each server log confirms `CPU threads: N`. |
| linkage | `raw/linkage_whisper.txt` PASS | `raw/linkage_tts.txt` PASS |
| proof of CPU residency | log: `failed to initialize ROCm: no ROCm-capable device`, `no GPU found` | same, and `rocm-smi --showpids` shows tts-server with **0 GPUs, 0 VRAM** |

Both binaries predate their freeze commits by about 5 h. Those commits carry only GPU patches:
whisper raises an FP8 HIP-version guard, and qwentts pins the ggml submodule that holds the
gfx90a patches. **Inferred:** the CPU code path is identical to what the frozen commits would
build. Both builds have `GGML_NATIVE=ON` (AVX-512/VNNI/BF16 detected) and
`GGML_OPENMP_ENABLED=OFF`, so ggml uses its own spin-polling threadpool, rebuilt for every
graph.

**Every measurement was pinned** with `taskset -c` inside 0-79. The harness client ran on 76-79.
Cores 88-95 and 184-191 were never used; the GPU sweep on :18072 was running throughout.
Every PID that was started and stopped is in `raw/pids_started.txt`, each stopped TERM→KILL
and confirmed dead. No running service was touched. The frontdoor :8070 received only the
allowed ~300-token generations, and at most one of mine was in flight at any time.

## Results (median, with (min-max); n in each row)

Full tables are in [`raw/summary_tables.md`](raw/summary_tables.md) (regenerate with
`python3 summarize.py`).

**STT solo** (`whisper-server` defaults: greedy, best-of 2, FA on). Wall time is the HTTP
round trip with the model already loaded.

| threads@cores | 11 s clip: wall / RTF | 86.5 s clip: wall / RTF |
|---|---|---|
| 8@0-7 | 4.44 s / 0.404 | 25.5 s / 0.295 (0.262-0.329) |
| 16@0-15 | 2.78 s / 0.253 | 16.3 s / 0.189 |
| 24@0-23 | 2.17 s / 0.197 | 14.4 s / 0.167 |
| 32@0-39 | 1.53 s / 0.140 | 10.5 s / 0.122 |
| **32@0-31** | **20.5 s / 1.86 (collapse)** | aborted after >8 min |
| 24@0-23, 3.5 s utterance | 1.97 s (1.92-2.03), RTF 0.56 | — |

On the long clip, whisper takes temperature fallbacks (`fallbacks = 6 p / 5 h` in the log),
which explains the spread between runs.

**TTS solo** (`seed=42`, and no `voice` field, as in production). `pcm` is the chunked stream;
`wav` is the one-shot file.

| threads@cores | sentence (4.24 s audio): first packet / RTF (pcm) | paragraph (35.7 s audio): first packet / RTF (pcm) | wav RTF (sentence / paragraph) |
|---|---|---|---|
| 8@0-7 | 0.113 s / 0.727 | 0.329 s / 0.759 | 0.698 / 0.806 |
| 16@0-15 | 0.084 s / 0.547 | 0.238 s / 0.595 | 0.476 / 0.571 |
| 24@0-23 | 0.075 s / 0.609 | 0.185 s / 0.589 | 0.469 / 0.491 |
| 32@0-39 | 0.085 s / 0.658 | 0.257 s / 0.551 | 0.438 / 0.532 |

TTS stops improving at about 16 threads, because frame-by-frame autoregressive decoding of a
0.6B model is bound by latency, not throughput. For comparison, GPU RTF at freeze was 0.169
(qwentts `2c1b518` commit message).

**Concurrent, at layout `main40`**: STT 24@0-23 and TTS 16@24-39, both servers resident.

| mode | STT 11 s RTF | STT 86.5 s RTF | TTS sentence: first pkt / RTF / prebuffer | TTS paragraph: first pkt / RTF / prebuffer | frontdoor 300 tok |
|---|---|---|---|---|---|
| solo (each alone) | 0.206 | 0.187 | 0.102 s / 0.692 / 0.33 s | 0.256 s / 0.600 / 0.05 s | 7.1 s, **43.3 tok/s** |
| STT + TTS | 0.199 | 0.243 | 0.091 s / 0.593 / 0.12 s | 0.311 s / 0.606 / 0.06 s | — |
| STT + TTS + frontdoor | **1.54** (1.13-1.87) | **2.91** | 1.83 s / **9.11** / 32.8 s | 1.52 s / **7.84** / 243 s | **906 s, 0.3 tok/s** (TTFT 29 s for 4 prompt tokens) |

**Concurrent, at layout `lean16`**: STT 8@64-71 and TTS 8@72-79. The frontdoor generation was
capped at 120 s so the production frontdoor would not be held.

| mode | STT 11 s RTF | TTS sentence: first pkt / RTF | frontdoor 300 tok |
|---|---|---|---|
| solo | 0.398 | 0.118 s / 0.729 | 7.0 s, 43.1 tok/s |
| STT + TTS + frontdoor | 0.799 | 0.503 s / **7.90** | 40.8 s, **7.4 tok/s** |

*Prebuffer* is the audio a player must buffer before starting playback so that the pcm stream
never underruns. Solo and STT+TTS need at most 0.47 s, and a 0.5 s jitter buffer covers every
run outside the frontdoor case. The `main40` STT+TTS+frontdoor phase was stopped after one of
its two planned repetitions, because it was holding the production frontdoor at 0.3 tok/s. One
repetition was enough to show the effect.

Frontdoor baseline note: 41–46 tok/s with **both speech servers resident but idle**. Idle
residency costs the frontdoor nothing, because the disposable ggml threadpools do not spin
between requests.

## Hazards found

1. **Layout collapse (whisper-server and whisper-bench, deterministic).** A single 30 s encode
   hangs for more than 60–300 s at `20@0-19`, `28@0-27`, `24@0-31`, `32@0-31` and `32@24-55`.
   The same thread counts are healthy at `16@0-15`, `16@0-23`, `24@0-23`, `32@0-39` and
   `32@0-47` (`raw/whisper_bench_encoder_matrix.txt`, three repetitions each). During the
   collapse, per-core sampling showed threads stacked two to three per core while cores in the
   mask sat idle, with nothing else runnable on those cores. **Inferred cause:** the per-graph
   thread spawn meets the scheduler's placement, and the spin barriers amplify every stacked
   thread. **Any CPU speech launcher must use a layout that has been measured**, or a build
   with a persistent, pinned threadpool.
2. **The TTS thread count cannot be controlled** without the preload shim. By default the
   server would start 96 spinning threads.

## What CPU speech would require (operator decision; not measured here)

- **Option A: stay on the GPU (recommended for now).** Costs about 2.2 GB of VRAM for whisper,
  plus TTS when it runs (**inferred** about 1–2 GB: 0.92 GB of weights and a 0.9 GB KV cache on
  CPU). It avoids the frontdoor collision entirely.
- **Option B: give speech its own cores.** Shrink the frontdoor, and any other CPU role
  running `-t 96` on 0-95 such as the architect :8074, to leave about 24–40 physical cores
  dedicated to speech. That means STT 16–24 and TTS 16. Frontdoor decode is bandwidth-bound,
  so the loss from giving up about 25% of its threads may be small, **but it has not been
  measured**. Measuring it requires relaunching the frontdoor, which is not this session's to
  do.
- **Option C: run speech on SMT siblings (96-175) of the frontdoor's cores.** This removes the
  time-slicing, so no thread gets stalled at a barrier, but it still shares execution ports.
  **Not tested**, because this run was restricted to 0-79.

## Exact commands

Harness: [`speech_cpu_bench.py`](speech_cpu_bench.py). Its server argv and env are logged per
start in `raw/pids_started.txt`.

```
taskset -c 76-79 python3 speech_cpu_bench.py stt_sweep 8,16,32,48        # 32@0-31 collapsed; interrupted
taskset -c 76-79 python3 speech_cpu_bench.py stt_sweep 24@0-23,32@0-39
taskset -c 76-79 python3 speech_cpu_bench.py tts_sweep 8@0-7,16@0-15,24@0-23,32@0-39
taskset -c 76-79 python3 speech_cpu_bench.py concurrent 24@0-23 16@24-39 2 main40
taskset -c 76-79 python3 speech_cpu_bench.py lean_llm 8@64-71 8@72-79 120 lean16
# server argv, as the harness launches them:
HIP_VISIBLE_DEVICES=-1 LD_LIBRARY_PATH=/mnt/raid0/llm/whisper.cpp/build/bin:/opt/rocm/lib taskset -c <cores> \
  /mnt/raid0/llm/whisper.cpp/build/bin/whisper-server -m /mnt/raid0/llm/models/whisper-ggml/ggml-large-v3-turbo.bin \
  --host 127.0.0.1 --port 19100 --inference-path /v1/audio/transcriptions -t <N> -ng
HIP_VISIBLE_DEVICES=-1 GGML_BACKEND=CPU SHIM_NPROCS=<2N> LD_PRELOAD=/mnt/raid0/llm/tmp/speech-cpu-20260924/nprocs_shim.so \
  LD_LIBRARY_PATH=/mnt/raid0/llm/qwentts.cpp/build:/opt/rocm/lib taskset -c <cores> \
  /mnt/raid0/llm/qwentts.cpp/build/tts-server --model .../qwen-talker-0.6b-base-Q8_0.gguf \
  --codec .../qwen-tokenizer-12hz-Q8_0.gguf --alias qwen3-tts-12hz-0.6b --host 127.0.0.1 --port 19102
```

Audio inputs are in `/mnt/raid0/llm/tmp/speech-cpu-20260924/`:

- `short_11s.wav` is `whisper.cpp/samples/jfk.wav`.
- `long_85s.wav` (86.5 s) concatenates `whisper-test/audio.wav` (56.8 s), jfk, and
  `qwentts.cpp/examples/freeman.wav` resampled nearest-neighbour to 16 kHz, with 0.5 s gaps.
- `utt_3p5s.wav` is the first 3.5 s of jfk.

Transcripts were checked for sanity only: jfk comes back verbatim. No WER was measured, and no
GPU-side accuracy comparison was made.

Host context: the production CPU llama-servers :8070 and :8074 both run `-t 96` on 0-95.
:8070's `/slots` showed all 4 slots idle before each LLM phase. :8074 was not seen generating
in `top` snapshots, but it was not monitored continuously. An orchestrator
Python job and megasync were active, the GPU sweep ran on 184-191, and the host was about 85%
idle outside the measurements.
