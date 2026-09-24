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

## Addendum: option C, speech on SMT siblings (measured 2026-09-24, 17:53–18:15 UTC)

The operator direction was to move speech to CPU only if it stays real time *while the
frontdoor generates*. This addendum measures option C.

**Sibling mapping.** Logical CPU `c+96` is the SMT sibling of physical core `c`. This was
checked with `lscpu -e` and `thread_siblings_list`, for example `cpu96: 0,96` and
`cpu135: 39,135`.

**Whisper layout on siblings.** Encoder bench, frontdoor idle, three repetitions each
(`raw/whisper_bench_encoder_matrix.txt`):

| layout | encode times | verdict |
|---|---|---|
| 16@96-111 | 2.46–2.54 s | stable |
| 24@96-119 | 3.65, 3.70, 1.93 s | jittery |
| 32@96-135 | 2.23–2.34 s | no gain |

**Layout used.** STT 16@96-111 (siblings of cores 0-15) and TTS 16@120-135 (siblings of cores
24-39).

**Frontdoor load.** Frontdoor generations were back to back with one in flight, each capped at
120 s. A guard would stop the run once throughput fell below 10 tok/s in more than one
repetition.

| mode | STT 11 s RTF | STT 86.5 s RTF | TTS sentence: first pkt / RTF | TTS paragraph: first pkt / RTF / prebuffer | frontdoor tok/s |
|---|---|---|---|---|---|
| solo (other servers resident, idle) | 0.448 (0.413-0.473) | 0.378 | 0.173 s / 0.820 | 0.308 s / 0.896 / 0.39 s | 26.7 (24.1-29.3), n=2 |
| STT + TTS, frontdoor idle | 0.383 (0.371-0.401) | 0.359 | 0.119 s / 0.791 | 0.312 s / 0.870 / 0.35 s | — |
| STT + TTS + frontdoor generating | **1.20** (0.34-1.74) | **3.90** (0.25-6.08) | 0.69 s / **18.8** (0.72-36.8) | 7.94 s / **21.6** / 731 s | **16.9** (6.3-37.0), n=51 |

**Result: option C is not real time.**

- **Speech with the frontdoor generating.** TTS runs 20–37× slower than real time, and STT runs
  1.2–6× slower. For about 16 minutes of repetition 0, 51 frontdoor generations ran back to back,
  and the speech requests crawled through them.
- **Spread.** The low ends of those ranges (TTS 0.72, STT 0.25) are requests that happened to
  land in gaps between frontdoor decode bursts. The wide spread therefore reflects when the
  collision happened, not a variable real-time margin.
- **Speech without the frontdoor.** Even with the frontdoor idle, speech on siblings is slower
  than on physical cores: STT RTF 0.36–0.45 against 0.19–0.25, and TTS RTF 0.79–0.90 against
  0.55–0.60. That leaves TTS only about 1.1–1.3× headroom.

**Frontdoor cost.**

- **During generation.** The frontdoor fell to 16.9 tok/s median. That is 37% below this
  window's baseline with speech idle (26.7 tok/s) and 61% below the earlier 41–46 tok/s. One
  generation went below 10 tok/s (6.3), so the guard counted one collapsed repetition.
- **Baseline confound.** This window's idle baseline (24–29 tok/s) was already below the earlier
  41–46. The cause is not identified. In the `main40` run, having speech resident but idle cost
  the frontdoor nothing on physical cores; whether that also holds on siblings was not isolated.
- **Early stop.** I stopped the run early, during repetition 1, by sending SIGINT to my harness.
  The guard had not tripped, but the verdict was already decisive and every further repetition
  held production at about 40% throughput loss for more than 15 minutes.
- **Cleanup.** Both servers stopped with rc=0 and were confirmed dead. The frontdoor slots were
  idle afterwards.

**Overall.** Neither disjoint physical cores inside 0-79 (option B-lite, `lean16`/`main40`) nor
SMT siblings (option C) keeps speech real time while the frontdoor generates. **Stay on the GPU
(option A).** The one path to CPU speech still untested is **real core partitioning**: reduce
the frontdoor below `-t 96` and take its cores out of its mask, so that speech has cores whose
siblings the frontdoor never touches. That needs a frontdoor relaunch, which is an operator
decision.

Command: `LLM_CAP_S=120 LLM_REPS=3 taskset -c 76-79 python3 speech_cpu_bench.py concurrent 16@96-111 16@120-135 2 smtC`.
Raw output is in `raw/concurrent_smtC.stdout`, `raw/concurrent.jsonl` (tag `smtC`) and
`raw/*_smtC.log`.

## Addendum: production layout vs CPU LLM roles (measured 2026-09-24, 19:19–19:43 UTC)

**Setup.** STT and TTS are now live on CPU:

- whisper on `:9000`: pid 2005824, `-t 24 -ng`, affinity 0-23.
- qwentts on `:9002`: pid 2006079, affinity 24-39, 16 threads via `/mnt/raid0/llm/cache/shims/nprocs_shim.so`.
- `rocm-smi` shows **0 VRAM for both**.

I only sent requests to the live services; I did not start, stop or restart them.

**How requests were run.** Speech requests went one at a time, as an urgent request would. The
LLM had one generation in flight, back to back, with `ignore_eos` and 1000 tokens. Every
speech request overlapped an in-flight generation (`llm_overlap` = 1.0).

**Safeguards.**

- Each LLM generation was hard-capped: 600 s in the first architect run, then 120–180 s.
- Each speech request was capped at 60–90 s after the first run.
- A guard stopped an arm after two degraded repetitions.

Harness: [`live_llm_contention.py`](live_llm_contention.py). Raw data: `raw/live_contention.{log,jsonl}`,
`raw/live_*.stdout`, `raw/llm_solo_baselines.txt`.

| condition | STT jfk 11 s RTF | STT 86.5 s RTF | TTS sentence: first pkt / RTF | TTS paragraph: first pkt / RTF | LLM decode |
|---|---|---|---|---|---|
| quiet (n=2) | 0.30, 0.22 | 0.22, 0.39 | 0.17 / 0.63; 0.09 / 0.55 | 0.23 / 0.59; 0.22 / 0.61 | — |
| **architect :8074** generating (live `-t 96` @0-95) | **≥ 58**: one request took about 10.7 min and finished only when the generation was aborted | not attempted | **13.6 s**; 0.56 s of audio in 164 s | not attempted | **0.59 tok/s** (solo 31.1–31.3) |
| **frontdoor :8070** generating (live `-t 96` @0-95) | not reached | not reached | **13.4 s**; 0.41 s of audio in 96 s | not reached | **1.05 tok/s** (solo 36.8) |
| **TEST Flash-Next `-t 56` @40-95** generating (partitioned) | 0.22, 0.44 | 0.25 | 1.00 s / **1.40**; 0.24 s / 0.98 | 0.24 s / **0.93** | 20.0–22.8 tok/s |

**Partitioned decode cost.**

- The TEST instance used the same binary, arguments and OMP/`GGML_IQK` environment as live
  `:8074`, with three changes: `-t 56`, `taskset -c 40-95`, and `--slot-save-path` pointed at a
  temporary directory. It was also launched under `numactl --interleave=all`, matching the live
  instance's memory policy.
- RAM was fine: 859 GB available. The instance loaded in 29 s with an RSS of 104 GB.
- Solo, with speech idle, it decoded 500 tokens at **23.0 and 24.5 tok/s**. The live `-t 96`
  instance did **31.1 and 31.3 tok/s**, so the partition costs **22–26%** of decode speed.
  While speech was running, the TEST instance reached 20.0–22.8 tok/s.
- Its PID was 2328222. I stopped it with TERM, confirmed it dead, and removed its temporary
  slot directory.

**Findings.**

1. **With `-t 96` roles on 0-95, the speech layout is not real time, and the collision is
   two-sided.**
   - Speech: STT slows by more than 50×, and TTS first packet goes to about 13 s.
   - LLM roles: the generating role collapses too, the architect to 0.6 tok/s and the frontdoor
     to 1 tok/s.
   - Cause: speech threads now sit permanently inside the LLMs' core masks, so the LLMs'
     barriers wait on the cores speech occupies. A speech request therefore stalls the LLM as
     badly as the LLM stalls it.
2. **Core partitioning fixes STT but leaves TTS marginal.**
   - STT is at RTF 0.22–0.44, about quiet-baseline levels.
   - TTS RTF is 0.93–1.40 against 0.55–0.63 quiet. First packet is 0.24–1.0 s, and the stream
     would need a playback buffer.
   - With disjoint cores, the remaining contention is **inferred** to be DRAM bandwidth: the
     MoE decode saturates memory, and TTS's per-frame autoregressive loop is sensitive to
     memory latency. This was not isolated.
   - The guard counted rep 0 as degraded (TTS RTF above 1), so rep 1 ran only the short items.

**Recommendation.**

- **Choosing between partitioning and a priority pause, partitioning is the one to adopt, and
  it has to happen now.** Right now any speech request that arrives during a `-t 96` generation
  wrecks both the speech and the LLM.
- **What partitioning means here:** relaunch the CPU LLM roles with their masks excluding 0-39,
  for example `-t 56` on 40-95. That costs about 22–26% of decode speed, measured on Flash-Next.
- **Why a priority pause alone won't do:** it cannot be applied mid-token, so a request is
  still stalled until the pause lands. Also, no pause mechanism exists today (inferred from the
  launch flags; not verified in code).
- **To make TTS real time under a partitioned LLM, add one of:**
  - (a) a pause or throttle of LLM decode while TTS streams, on top of the partition;
  - (b) move TTS back to the GPU (0.92 GB of weights). This is the cheaper choice: STT on CPU
    already frees the larger 2.2 GB.
- These are relaunch and topology decisions for the operator. **Not measured:** the frontdoor
  under the same partition (inferred to behave like Flash-Next, since both are
  bandwidth-bound MoE decode).

**Side observations.**

- The live speech services open `/dev/kfd`. They use 0 VRAM, so this is harmless.
  *Correction (2026-09-24, wrap-up):* this line first said the cause was an unset
  `HIP_VISIBLE_DEVICES`. `/proc/<pid>/environ` of whisper 2005824 and tts 2006079 shows
  `HIP_VISIBLE_DEVICES=-1` set on both (and `GGML_BACKEND=CPU`, `SHIM_NPROCS=32`, `LD_PRELOAD`
  on the TTS), so the HIP runtime opens `/dev/kfd` even with every device hidden.
- megasync, which is unpinned, was running at about 60–100% on core 16. That core is inside
  whisper's 0-23 mask and is a possible source of stragglers: quiet STT RTF spread 0.22–0.39.
