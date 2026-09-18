# M-12 long-context GPU serving recipe (MI210), prefix reuse, and the judge

> **M-12 GPU window deferred by operator 2026-09-16; smoke_prefix_reuse.sh ready (a6491b9e) for when it resumes.**
>
> Blocker status on branch `sub/m12-blockers-20260916` (handoff note; the root M-12 handoff is
> being ported by a wrap-up agent and is updated from `progress/2026-09/2026-09-16-sub-m12-blockers.md`):
>
> | Blocker | State |
> |---|---|
> | B1 | Fixed (`b69be5b2`). |
> | B2 | Fixed: BEAM `rag` / `trace` arms. |
> | B3 | Fixed: explicit, recorded generation parameters. |
> | B4 | Judge chosen: gemma-4-26B-A4B-it-ORIG-Q8_0, §6. |
> | B5 | Recipe and smoke written (`a6491b9e`). The smoke itself is inference, so it waits for the window. |
> | B6 | Merges: still open. |
> | B7 | pyarrow and pandas are both already declared in the `benchmark` extra and locked (pyarrow 24.0.0, pandas 3.0.3). The research `.venv` has pyarrow 25.0.1 and **no pandas**, so M-12a's Tulving adapter still needs pandas in that venv: `uv sync --extra benchmark`, or add pandas alone. |

**Status: NOT RUN.** Everything below comes from reading source code and from
planning estimates, not from measurement. The first 10 minutes of the M-12 window
are the smoke test in §4, and the plan stands or falls on it.

Owning handoff: `epyc-root:handoffs/active/episodic-memory-integrity.md` → M-12 (blocker B5, plus B3 and B4).
The token-length and fit analysis this doc builds on is
`epyc-root:progress/2026-09/2026-09-16-sub-op42-readiness.md`.

Who runs what:
- The operator runs everything here: server launches, `run_benchmark.py`, and the judge. Agents do not.
- This doc and its scripts start and stop nothing on their own.
- `smoke_prefix_reuse.sh` only sends requests to a server that is already running.

## 1. Binary and linkage

| Item | Value |
|---|---|
| Binary | `/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server` (autokernel champion `ef81196d5`, HIP build: `GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`) |
| Source | `/mnt/raid0/llm/tmp/fold-ef81196d5-src` |
| Device | `--device ROCm0 -ngl 99` (the MI210 is `/sys/class/drm/card2`, 64 GiB) |
| Loader env | `LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin`, `HSA_OVERRIDE_GFX_VERSION` unset (from `Recipe.server_env`) |

Before trusting any number, prove the binary loads its own ggml. The HIP backend
is dlopen'ed, so `ldd` alone proves nothing.

```bash
LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin \
  scripts/utils/verify_ggml_linkage.sh /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  /mnt/raid0/llm/tmp/build-fold-ef81196d5
```

GPU residency is proved separately: the smoke test samples MI210 VRAM
(`mem_info_vram_used`) during its cold request and records the maximum in its receipt.

## 2. Recipes

Recipes are imported, never transcribed. All three live in
`artifacts/serving-recipes/eval/`.

**Why a subdirectory.** They pass server flags (`--ctx-checkpoints`,
`--cache-ram`, `--reasoning`) as `extra_flags`, and the autokernel resolver
(`resolved_recipe`) reports those as `extra_flags_unsupported`. So these recipes
are eval-serving recipes, not autokernel-measurable ones. They have no serving
floor, and `test_resolved_recipe.py` does not scan them.

Print the exact launch line (the helper prints it and never launches anything):

```bash
python scripts/benchmark/m12_launch_argv.py artifacts/serving-recipes/eval/<recipe>.json --port <port>
```

**`-c` is the TOTAL context.** With `kv_unified=false` each slot gets `ctx / np`.
Both readers therefore give every slot **196,608** tokens:
- the longest BEAM 100K prompt is 191,456 Qwen tokens;
- the Tulving 200ch book prompt is 104,725 tokens;
- the rest of each slot covers the output budget.

| Recipe | Model | np | `-c` | per slot | spec-decode | extra | recipe_hash |
|---|---|---:|---:|---:|---|---|---|
| `qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4` **(reader, recommended)** | Qwen3.6-35B-A3B-MTP-Q8_0 | 4 | 786432 | 196608 | draft-mtp, n_max 4 | `--ctx-checkpoints 8 --cache-ram 16384` | `54674144…9c37` |
| `qwen3.8-27b-q8-gpu-dflash2-longctx196k-np1` (reader, fallback) | Qwen3.8-27B-Q8_0 + DFlash2 drafter | 1 | 196608 | 196608 | draft-dflash, n_max 8 | `--ctx-checkpoints 8 --cache-ram 32768` | `9285043f…d8d9` |
| `gemma-4-26b-a4b-orig-q8-gpu-judge-np4` **(judge, §6)** | gemma-4-26B-A4B-it-ORIG-Q8_0 | 4 | 32768 | 8192 | none | `--reasoning off` | `2fdeeef2…54eb` |

Everything else is the canonical GPU recipe, unchanged (`test_m12_eval_recipes.py`
enforces this):
- `-t/-tb 8`, `-b/-ub 2048`;
- f16 KV, `fa on`, `--no-kv-unified`;
- `taskset -c 184-191`;
- `--metrics --slots`.

Reader launch lines as printed by the helper (`--port 8199`):

```bash
# 35B-A3B-MTP, np=4
env -uHSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin \
  taskset -c 184-191 /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -np 4 -c 786432 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port 8199 --metrics --slots \
  --spec-type draft-mtp --spec-draft-n-max 4 --no-kv-unified --ctx-checkpoints 8 --cache-ram 16384

# 27B + DFlash2, np=1
env -uHSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin \
  taskset -c 184-191 /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  -m /mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf -np 1 -c 196608 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port 8199 --metrics --slots \
  -md /mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 --spec-type draft-dflash \
  --spec-draft-n-max 8 --no-kv-unified --ctx-checkpoints 8 --cache-ram 32768
```

### VRAM fit

These are estimates from the readiness package, not measurements. They were built
from the measured ctx-16384 residency peaks in `champion-max-performance-20260908`,
plus KV for the full-attention layers, plus about 1.5 GiB of compute buffer for a
long context (itself unmeasured).

| Config | Est. VRAM of 64 GiB |
|---|---|
| 35B-A3B-MTP np=4 @196K | ≈54–55 GiB |
| 27B + DFlash2 np=1 @196K | ≈46 GiB. The DFlash2 drafter context follows `-c`. |
| gemma-4 ORIG Q8_0 judge np=4 @8K | ≈29–30 GiB: 25.0 GiB of weights, 0.6 GiB global KV, and a small SWA cache. **Q8_0 fits, so Q4_K_M is not needed.** |

**The MI210 must be empty.** At 2026-09-16 10:xx UTC `card2` showed 40.6 GiB in
use by another load, so no reader fits until that load is gone.

## 3. Prefix reuse — how it works on this binary and what the recipe sets

The Qwen3.6/3.8 readers are hybrid models: GDN layers plus full attention every
4th layer. Their recurrent state cannot be rolled back to an arbitrary prefix, so
the server reuses a prefix **only by restoring a context checkpoint**
(`server-context.cpp`, "restored context checkpoint", about line 3469).

**When checkpoints are created** (`server-context.cpp:3640-3700`, completion tasks only):
1. at user-message starts;
2. **`4 + n_ubatch` = 2052 tokens before the prompt end;**
3. 4 tokens before the prompt end.

**Why checkpoint 2 does the work.** Every M-12 prompt is one user message: a
shared prefix (the book, or the conversation transcript) followed by a question
plus the chat-template tail, well under 2052 tokens. Checkpoint 2 therefore falls
inside the shared prefix. For the next question with the same prefix, the server:
- finds the longest common prefix;
- restores that checkpoint;
- re-decodes about 2.1K tokens instead of 105K–192K.

| Planning cost per question (bench-class estimates) | 35B | 27B |
|---|---:|---:|
| Warm question (restore + ~2.1K re-decode) | ≈2.8 s | ≈7.1 s |
| Cold 104.7K prefill | ≈94 s | ≈238 s |

**What the recipe sets, and why:**

| Setting | Value | Why |
|---|---|---|
| `--ctx-checkpoints` | 8 (default 32) | Each request creates at most about 3. Eight leaves margin and bounds host RAM: one checkpoint is the non-rollbackable state only (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`), about 60 MB (35B) or 150 MB (27B). |
| `--cache-ram` | 16384 (35B) / 32768 (27B), in MiB; default 8192 | When a slot switches prefix, its state (KV plus checkpoints) is saved to host RAM and can be restored later. One 192K prefix is about 4 GiB (35B) or 12.75 GiB (27B), so these sizes hold 2–4 prefixes. |
| `-ub` | 2048, unchanged | It sets the checkpoint stride above. **Changing `-ub` moves the checkpoint.** |
| `--checkpoint-min-step` | default 8192 | Irrelevant here: the near-end checkpoints are exempt from it. |
| `--slot-prompt-similarity` | default 0.10 | Routes a request to the slot that already holds its prefix. |

**What the client must do** (all enforced by the harness since B3, see §5):
- **`cache_prompt: true`.** The raw `/completion` path in `lib/executor.py`
  hard-coded `cache_prompt: false`, which would have defeated all of the above.
  The Tulving and BEAM suites now pin `cache_prompt=True`.
- **The chat-completions path.** The suites pin `enable_thinking`, and that forces
  it: Qwen3.6+ honours `chat_template_kwargs.enable_thinking` only there.
- **One prefix at a time.**
  - Tulving: every full-arm prompt shares the one book, so order does not matter.
  - BEAM: its 20 questions per conversation share one transcript. `BEAMAdapter`
    sets `preserve_order`, so the suite keeps conversation-contiguous order rather
    than `(-tier, id)`.

**np=4 does not speed up `run_benchmark`.** It sends one request at a time. What
np=4 buys is four warm prefixes, for example the Tulving book plus the current BEAM
transcript, without a RAM round-trip. Throughput from np>1 needs the row-exact
concurrent route (OP-39(C)).

## 4. The 10-minute prefix-reuse smoke (the window's first step)

```bash
# after the reader is up and the linkage check passed:
scripts/benchmark/smoke_prefix_reuse.sh 8199 35b   # or: ... 8199 27b
```

It prints one line per request and ends with
**`SMOKE_PREFIX_REUSE: PASS`** or **`SMOKE_PREFIX_REUSE: FAIL (<reasons>)`**. The
exit code is 0 on PASS. The receipt JSON is written to
`/mnt/raid0/llm/tmp/m12-smoke/`. Every request uses the M-12 parameters:
- `temperature 0`, `top_k 1`, `max_tokens 16`;
- `enable_thinking=false`, `cache_prompt=true`;
- `/v1/chat/completions`.

| Step | Request | What is checked |
|---|---|---|
| 0 | `GET /health`, `/props`, `/slots` | The served `model_path` is the expected GGUF, and **every slot has `n_ctx` ≥ 196608**. If either fails, nothing is sent. |
| A | Tulving 200ch book + Q1 (≈104.7K tok), cold | `timings.prompt_n` ≈ `usage.prompt_tokens` (a WARN if the slot was already warm). The MI210 VRAM maximum during the request goes into the receipt. |
| B | Same book + Q2 | **PASS needs both:** `timings.cache_n` ≥ 95% of `prompt_tokens`, and `timings.prompt_n` ≤ 2052 + 2048. A WARN if B's `prompt_ms` > 25% of A's. |
| C | Book + Q2 again | The same reuse checks. A WARN if the answer differs from B's. |
| D | *(35b only)* The longest BEAM 100K transcript (conv 11, ≈191.5K tok) + U1, cold | Cold accounting, as for A. |
| E | Same transcript + U2 | The same reuse checks as B. |
| F | Book + Q3 | **Reuse still holds after another prefix was served.** With np=4 the book is still in its own slot; with np=1 it must come back from `--cache-ram`. |

**Time**, from the bench-class prefill estimates:

| Reader | Legs | Tulving leg | BEAM leg | Total with load and preflight |
|---|---|---:|---:|---:|
| 35B | both | ≈1.7 min | ≈4.2 min | ≈6–7 min |
| 27B | Tulving only | ≈4.5 min | — (would take ≈10 min cold) | ≈6 min |

**What to look at by hand.**
- `GET /slots`: each slot's `n_ctx`, `n_prompt_tokens`, `n_prompt_tokens_cache`
  (reused) and `n_prompt_tokens_processed` for its last task.
- `timings.cache_n` / `timings.prompt_n` in each response. `cache_n` is
  `slot.n_prompt_tokens_cache = n_past` after the restore, so it is the server's
  own count of reused tokens.
- For the log lines as well ("created context checkpoint N of 8", "restored
  context checkpoint"), launch with `-lv 4`. Those messages are trace level
  (`LOG_TRC`, level 4) and the default threshold is 3. Doing so adds one flag and
  changes the launch line, so use it for diagnosis only.

**If it FAILs:**

| Symptom | Likely cause | Next check |
|---|---|---|
| `cache_n` ≈ 0 and `prompt_n` ≈ the full prompt | `cache_prompt` off, or no checkpoint was usable (the log says "forcing full prompt re-processing") | Relaunch with `-lv 4` and read the checkpoint lines |
| Slot `n_ctx` < 196608 | `-c` was not multiplied by np | Relaunch from the helper's line |
| Only F fails | The book was evicted and not restored | Raise `--cache-ram`, or keep np=4 |

A FAIL means the CEILING and VANILLA arms would take about 18 h and 16 h. Only
option B of OP-42 (M-12a, with a reduced full-arm n) is then bookable.

## 5. M-12 run parameters (B3) — explicit and recorded per row

Both adapters pin `inference_params`, and `run_benchmark` records what it
actually sent in every result row:
- `inference`: `max_tokens`, `temperature`, `timeout`, `enable_thinking` and its
  source, `cache_prompt`, endpoint;
- `finish_reason`;
- `provenance`: dataset or book identity, arm, retrieval budget.

| Parameter | Tulving (M-12a) | BEAM (M-12b) | Reason |
|---|---:|---:|---|
| `temperature` | 0.0 | 0.0 | Arms must be comparable. The harness default (0.6) would add sampling noise to a within-reader A/B. |
| `max_tokens` | **1024** | **2048** | M-12c(3): no synthesis cap below what the highest-nugget ability needs. See the note below. |
| `enable_thinking` | **false** | **false** | Qwen3.6+ reasoning otherwise spends the budget (MEMORY: thinking-off, 12/15 vs 7/15 on the frontdoor probe). Applied via `chat_template_kwargs` on `/v1/chat/completions`, which a pinned value forces. |
| `cache_prompt` | true | true | §3 |
| `timeout` (s) | 1800 | 1800 | A cold 192K prefill is about 10 min on the 27B |

**How the `max_tokens` values were chosen:**
- **Tulving:** the longest gold list in the 200ch set is 17 items and 317 chars
  (≈70–90 tokens), so 1024 is more than 10× that.
- **BEAM:** the longest reference answer is 1,584 chars (≈400 tokens), from
  summarization; event_ordering has up to 9 nuggets. 2048 is about 5× the longest
  reference.
- **Checking the cap:** any row with `finish_reason == "length"` hit the cap.
  Count those before citing a result.

**Arm selection** (environment variables, recorded in `provenance`, and cross-checked by the scorers):

| Arm | Tulving (`TULVING_CONTEXT_MODE`) | BEAM (`BEAM_CONTEXT_MODE`) |
|---|---|---|
| memory-off | `none` (question only) | `full` = BEAM's Vanilla column (whole transcript) |
| naive-memory control | — | `rag`: `pair_chunk` × BM25 (in-process), top `BEAM_RETRIEVAL_TOP_K` (default 10) |
| arm under test | `retrieved`: trace-FTS5 chapters, `TULVING_RETRIEVAL_TOP_K` (default 5) | `trace`: the same `pair_chunk`s indexed in a private trace store and searched via `navigation.search_records(order="relevance")`, with the same top-k |
| ceiling | `full` (whole book) | — |

Book selection is **`TULVING_CHAPTERS=200`**, and it is required for M-12a. Since B1,
the chapter set is part of every question id and every row's `provenance`, and the
scorer refuses a set mismatch.

## 6. Judge (B4): gemma-4-26B-A4B-it-ORIG-Q8_0, a separate launch after the reader

**What was decided (operator, 2026-09-16):**
- The M-12 judge is **gemma-4-26B-A4B-it-ORIG**.
- It is a different model family from the Qwen3.6-35B-A3B reader, satisfying
  M-12c(4): judge ≠ reader, held fixed across arms.
- It runs on the MI210 in a **separate server launch after the reader is unloaded
  (sequential, never co-resident)**.

**Quants on disk** (`/mnt/raid0/llm/models`):

| File | Size | Used? |
|---|---:|---|
| `gemma-4-26B-A4B-it-ORIG-Q8_0.gguf` | 26,859,859,872 B (25.0 GiB) | **Yes.** It fits alone. |
| `gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | 16,796,016,544 B (15.6 GiB) | Fallback only. |

The non-ORIG `gemma-4-26B-A4B-it-Q4_K_M*.gguf` files are **not** the judge.

**Recipe:** `artifacts/serving-recipes/eval/gemma-4-26b-a4b-orig-q8-gpu-judge-np4.json`. Its settings:
- np=4, `-c 32768` (8192 per slot; a judge prompt is question + answer ≤ 2048 + one nugget);
- no spec-decode (the gemma-4 MTP path has an assert-and-wedge history);
- `--reasoning off`, which also sets the server-side `enable_thinking=false`
  default, so replies land in `content`;
- everything else canonical.

```bash
env -uHSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin \
  taskset -c 184-191 /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q8_0.gguf -np 4 -c 32768 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port 8199 --metrics --slots \
  --no-kv-unified --reasoning off
```

**The judge is not known-good on the champion** (no canonical gemma GPU recipe
existed before this one). **Run a judge smoke first.**

1. Run `judge_beam_run.py` on a 5-question slice of the first reader result.
2. Confirm all of the following:
   - `unjudged` is empty;
   - every verdict parses to 0 / 0.5 / 1;
   - VRAM is non-zero during the calls.

Then judge each arm:

```bash
python scripts/benchmark/judge_beam_run.py <reader_result.json> --out-json <judged_<arm>.json> \
  --judge-model gemma-4-26B-A4B-it-ORIG-Q8_0 --judge-url http://127.0.0.1:8199
python scripts/benchmark/score_beam_run.py <judged_<arm>.json> --out-json <scored_<arm>.json> \
  --check-dataset --belief-measurements --arm <full|rag|trace>
```

`judge_beam_run` needs the orchestrator's `request_llm_judge_text` (`dae95a86`, blocker B6).

## 7. Window order

1. The MI210 is empty. Launch the reader from the helper's line, then run the linkage check (§1).
2. Run `smoke_prefix_reuse.sh <port> 35b`. **PASS is required** to continue with the long arms.
3. M-12a, one `run_benchmark.py` invocation per arm:
   - `TULVING_CHAPTERS=200 TULVING_CONTEXT_MODE=<none|retrieved|full>`;
   - `--server-mode --existing-server-port <port> --suite tulving_episodic --skip-moe-reduction --skip-speed-tests --new-run`.
4. Score M-12a: `score_tulving_run.py <result> --out-json … --belief-measurements --arm <arm>`.
   The chapter set now comes from the rows, so `--chapters` is optional and must agree if given.
5. M-12b, one invocation per arm:
   - `BEAM_CONTEXT_MODE=<full|rag|trace>`;
   - `--suite beam`, with the same run_benchmark flags.
6. Unload the reader. Launch the judge (§6), run the judge smoke, then judge and score each BEAM arm.
7. Unload the judge.

**Estimated wall-clock** (planning numbers):

| Reader | M-12a | M-12b | Judge | Total |
|---|---:|---:|---:|---:|
| 35B | ≈2.5 h | ≈4 h | ≈1.5–3 h | ≈8–10 h, plus load and the smokes |

## 8. Caveats

- **Nothing here has been measured.**
  - VRAM fits, prefill times and per-question costs are extrapolations from
    bench-class rows (MEASUREMENT.md INSTRUMENT-CLASS-1).
  - MTP and DFlash2 acceptance at a depth of 100K+ has never been measured.
  - Spec-decode leaves greedy output unchanged up to batch-split numerics, so it
    only affects wall-clock time.
- **The recipe hashes differ from the canonical recipes.** These are new serving
  conditions, and no serving floor applies.
- **A reader on the GPU answers "does memory help this reader".** It does not
  measure the production CPU `ingest_long_context` role, and it is not comparable
  to the June SRS (a different model and a 20ch book).
