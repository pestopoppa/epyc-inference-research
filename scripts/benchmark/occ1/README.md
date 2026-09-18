# OCC-1 — bitmap frames vs raw text on a served local reader (GPU-only recipe)

epyc-root handoff: `handoffs/active/optical-context-compression.md` → OCC-1 (index row RTG-53).
Gates OCC-3. Scoped 2026-09-16 with zero inference.

## Claim this run can make (MEASUREMENT.md grammar)

- **Reader**: Qwen3-VL-30B-A3B-Instruct Q4_K_M + `mmproj-…-F16` (projector `qwen3vl_merger`).
  This is the production `worker_vision`/`vision_escalation` model. It runs on the MI210 (ROCm0) with
  the champion llama.cpp `ef81196d5` (`/mnt/raid0/llm/tmp/build-fold-ef81196d5`, build 10301, `GGML_HIP=ON`).
- **History fixture**: SQuAD v1.1 dev (sha256 `95aa6a52…72c9`). All passages are space-joined into one
  stream and cut into **39 identical 40,716-char chunks**. Each chunk gets 30 questions, seed 42, spread
  evenly across the chunk: **1,165 paired questions per arm**. Every arm gets the same chunk text and the
  same question bytes. Only the carrier differs.
- **Billed-token accounting**: the server's `usage.prompt_tokens` per request, with `cache_prompt:false`
  and `--cache-ram 0`. A cache hit voids the run. `timings.prompt_ms` and wall time are recorded as the
  local compute cost. The planner predicts image tokens by porting the champion's `smart_resize`, and
  the report checks the prediction against the server (`non_image_prompt_tokens_*` must be a small,
  near-constant overhead).
- **Recall metric**: official SQuAD F1 per question (higher = better), paired arm vs text. EM (exact
  McNemar) is secondary.

## Arms

All frames are 1568 px wide. Frame dimensions are multiples of 32 px, so Qwen3-VL consumes them with
**no resample**. The last frame of a chunk is trimmed to its printed rows but kept at ≥ 672 px, so the
production `--image-min-tokens 1024` floor never upsamples it. Bitmap fonts have no DPI: glyph size is
the cell in pixels.

| arm | font cell | frames/chunk | predicted prompt tokens/chunk (context only) | est. ratio vs text* | role |
|---|---|---|---|---|---|
| `text` | — | — | ~8,732 (Qwen3 BPE estimate) | 1.00 | paired baseline |
| `img-6x10-bw` | 6×10 misc-fixed | 1 | 2,401 | ~0.31 | primary candidate |
| `img-6x10-color` | 6×10, per-row hue bands | 1 | 2,401 | ~0.31 | upstream default variant |
| `img-8x8u-bw` | unscii-8 | 2 | 3,430 | ~0.42 | candidate |
| `img-8x13-bw` | 8×13 misc-fixed | 2 | 4,165 | ~0.50 | borderline candidate |
| `img-12x12u-bw` | unscii-8 Lanczos ×1.5 | 3 | 5,831 | ~0.68 | legibility control (cannot pass the cost bar) |

\* The ratio includes ~450 tokens of prompt and question overhead in both arms. The report computes
the real ratio from server counts.

## Pre-registered decision (in `run_occ1.py::PREREG`; do not edit after the first `run`)

- An arm is **POSITIVE** when its token ratio vs text is ≤ 0.50 **and** the lower bound of the 95% CI
  on its paired F1 delta is > −0.05. The CI is a percentile bootstrap that resamples whole chunks
  (10,000 iterations, seed 0).
- An arm is **NEGATIVE_COST** when its ratio is > 0.50, and **NOT_NONINFERIOR** otherwise.
- **OCC-1 is POSITIVE** if any arm is POSITIVE; that arm seeds OCC-3. Otherwise OCC-1 is NEGATIVE.
- **VOID** if any of these hold:
  - text-arm F1 < 0.60, or an incomplete run
  - an unrepaired transport error, or an unrepaired malformed 200 body (no `choices`/content/usage).
    Both are retried once, then stored as errors; `run` retries them on resume.
  - a prompt-cache hit, or a server identity mismatch
  - suite-fingerprint, frame-pixel or Pillow-version drift between `plan` and `run`
  - **pre-registration drift**. `report` grades with the `prereg` stored in `plan.json`, never with
    the code's `PREREG`. If the two differ, the run is VOID, and `summary.json` carries the plan's
    `prereg` plus the code's as `prereg_code`. So a threshold edit after the run can never change a
    verdict, and a VOID run writes no belief row.
  - **GPU residency not proven** (see the recipe)

## GPU runner recipe

Prerequisites (checked 2026-09-16, zero inference):

- **VRAM**: the MI210 had 27.6 GB used out of 64 GB in one sample. This server needs about 20–22 GB:
  17.7 GB weights, 1.0 GB mmproj, 0.8 GB q8_0 KV at 16k, about 0.8 GB overhead, plus the vision compute
  buffer. The prior 65k-ctx measurement was 21.0 GB.
- **Reader already in production**: the same model already serves `:8086` on the production kernel.
  This run uses a **separate** champion-binary server on test port `:18431`, so production is not
  touched. (`:8090` is the production embedder, so never use it.) The port is a parameter:
  `launch_reader.sh --port N` (or `OCC1_PORT=N`) plus `run_occ1.py --port N`. Before overriding it,
  check that the port is absent from `epyc-orchestrator/orchestration/launch_manifest.yaml` and
  from `ss -ltn`. The launcher refuses a port that is already listening. If
  VRAM is short, the session that owns the GPU decides whether to evict. Do not evict it yourself.
- **Region claim**: take the GPU region claim for the MI210 before launching and release it after.

```bash
RES=/mnt/raid0/llm/worktrees/sub-occ1          # or the research clone once merged
RUN=/mnt/raid0/llm/tmp/occ1-run-20260916        # ALREADY PLANNED: 234 requests, suite 261d8ac1eaed
PORT=18431                                       # free test port (checked 2026-09-16); never 8090
export OCC1_PORT=$PORT                           # report needs an epyc-root with the SC85 capture module ($EPYC_ROOT)
cd $RES

# 0. (already done 2026-09-16; re-run only if the plan dir is lost — deterministic, ~50 s CPU)
uv run --no-project --with pillow==12.3.0 --with tokenizers \
  python scripts/benchmark/occ1/run_occ1.py plan --out $RUN

# 1. launch the reader (foreground exec; capture YOUR pid, kill only it)
#    first record the VRAM baseline BEFORE the reader exists (run refuses without it)
cat /sys/class/drm/card2/device/mem_info_vram_used > $RUN/vram_baseline_bytes
cp $RUN/vram_baseline_bytes $RUN-pilot/vram_baseline_bytes
nohup scripts/benchmark/occ1/launch_reader.sh --port $PORT > $RUN/server.log 2>&1 &
SRV=$!                                           # taskset and the launcher exec, so this IS llama-server
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 5; done   # ~1–2 min load
# residency is proven BY `run` itself, DURING the run: it samples VRAM + /sys/class/kfd/kfd/proc
# before the first request, after every request, and every 2 s, into residency_samples.jsonl /
# residency.json. It is PROVEN only if >= 2 samples show VRAM >= 16 GiB above the baseline AND
# $SRV holds a KFD context. Otherwise report VOIDs the run. (ldd cannot prove a HIP run; ggml dlopens it.)
# If another tenant frees or grabs VRAM between baseline and launch, re-take the baseline.

# 2. pilot — 3 chunks, 18 requests, ~3–5 min; checks the plumbing before the full spend
uv run --no-project --with pillow==12.3.0 python scripts/benchmark/occ1/run_occ1.py plan --out $RUN-pilot --limit-chunks 3 --tokenizer ''
uv run --no-project --with pillow==12.3.0 python scripts/benchmark/occ1/run_occ1.py run  --out $RUN-pilot --limit-chunks 3 --tokenizer '' --server-pid $SRV
uv run --no-project python scripts/benchmark/occ1/run_occ1.py report --out $RUN-pilot
#   go/no-go: text F1 ≳ 0.8; non_image_prompt_tokens range narrow (≈ text-arm prompt − context tokens);
#   no truncation. If an image arm answers everything UNREADABLE, that is a RESULT, not a bug.

# 3. full run — 234 requests
uv run --no-project --with pillow==12.3.0 python scripts/benchmark/occ1/run_occ1.py run --out $RUN --server-pid $SRV
uv run --no-project python scripts/benchmark/occ1/run_occ1.py report --out $RUN   # summary.md / summary.json + belief_measurements.jsonl

# 4. teardown
kill $SRV; sleep 5; ps -p $SRV >/dev/null && kill -9 $SRV; ps -p $SRV || echo "server $SRV gone"
```

`run` can be resumed: completed request keys are skipped, and failed ones are retried. The `plan` and
`run` commands must use the same `--arms/--chunk-chars/--qpc/--seed/--limit-chunks` values.
Otherwise the fingerprint check refuses to run.

**Wall-clock estimate** (no OCC-specific measurement exists yet): the 2026-08-02 run of this model on
the MI210 fits latency ≈ 0.9 s + 11.6 ms × completion tokens for ~1.16k-token image prompts. OCC
requests carry 2.9k–9.3k prompt tokens (1–3 frames, each 2,401 tokens) and about 250 answer tokens.
That puts a request at roughly 5–15 s, and all **234 requests at 20–60 min**. With server load, the
pilot and the report, budget **about 1.5 h of MI210 time**.

## Tests (mocked server, no inference)

```bash
uv run --no-project --with pillow==12.3.0 python -m unittest scripts/benchmark/occ1/tests/test_occ1.py
```

## Belief-kernel wiring (SC85)

`report` writes `belief_measurements.jsonl` beside `summary.json` using epyc-root's
`scripts/vidya/adapters/occ1_optical_compression_capture.py`. The module is found through
`$EPYC_ROOT`, then `/mnt/raid0/llm/epyc-root`, then `/workspace`.

- **Rows per arm**: SQuAD F1 and EM. The text arm is BASELINE and the image arms are CANDIDATE.
  Each image arm also gets the paired F1 delta vs text, with its 95% CI, McNemar p and verdict, and
  the prompt-token ratio vs text.
- **Identity on every row**: the suite fingerprint, the `/props` serving identity (including the
  URL), and a digest of the pre-registration.
- **VOID runs**: a VOID run writes nothing, and removes a stale sidecar.
- **Protocol**: `--protocol-id` stays empty until an OCC protocol is codified under
  `measurement/protocols/`. Until then, every tuple grades `Judged/Located`, an observation.
- **Skipping or naming**: `--no-belief-measurements` skips the sidecar. `--run-id` names the run;
  the default is the `--out` directory name.
- **Ingest**: `python3 scripts/vidya/cli.py ingest occ1 --path $RUN`, run from epyc-root.

## Attribution

`render.py` (BDF/HEX parsing, glyph blitting, colour bands) and `fixture.py` (SQuAD flow, question
sampling, EM/F1) are adapted from `@oh-my-pi/snapcompact` `research/bdf.py` + `research/squad.py`
(MIT License, oh-my-pi @ `37eee7197`). Prompt wording is adapted from its `research/prompts/`. Fonts
are fetched at plan time and never committed: X.org misc-fixed 6x10/8x13 (public domain) and unscii-8
(public domain). SQuAD v1.1 is CC BY-SA 4.0 and is fetched, never committed.
