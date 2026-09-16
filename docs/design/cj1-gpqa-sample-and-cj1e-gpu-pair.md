# CJ-1d / CJ-1e / CJ-3d preparation: GPQA sample, GPU ranking pair, BFCL tool path

**Date**: 2026-09-16 · **Owner handoff**: `epyc-root/handoffs/active/canonical-judge-suite-revamp.md`
(EVL-08) · **Status**: preparation only. No inference was run to write this; every number below is
read from an existing artifact whose path is given.

---

## 1. CJ-1d: the GPQA sample

### 1.1 How it is wired

The CJ instruments do **not** go into the 79-question YAML suite (`benchmarks/prompts/v1/*.yaml`),
because that suite is LLM-judged. GPQA is scored deterministically: canonical
`v7_quality_gate_runner.py` with canonical `answer_scoring.score_response`, which dispatches
`multiple_choice` to `extract_letter_answer` (CJ-1c). The runner already replays a pinned item set
through `--questions-in`, so wiring GPQA means producing that pin with a known sample identity:

```
python3 scripts/benchmark/cj_gpqa_sample.py --out <run_dir>/cj1_gpqa_manifest.json   # n=198, seed=42
```

- **Suite**: `gpqa_diamond_cot`. The letter-only `gpqa_diamond` prompt is refused, because it suppresses
  reasoning.
- **Population**: `artifacts/architect-bench-gpu-20260720/questions_gpqa_diamond_cot.json`, which is not
  tracked in git. It is byte-identical in ids, order, prompts and gold to the EVL-08 pin
  `data/gpqa-cj1-2026-08-25/pinned_questions.json` (verified 198/198). The code requires the id set to
  hash to `381ba365…fbb23194`, so a different population is refused rather than scored under the same key.
- **Selection**: the population is sorted by content-hash id, then drawn with `random.Random(seed).sample`.
  The result does not depend on file order or on the HF dataset revision. No `datasets` import is needed,
  so it runs under system `python3`.
- **Identity**: the manifest carries a `sample` block with these fields: schema, method, seed,
  `seed_effective`, n, `n_population`, the population file sha256, `population_ids_sha256`,
  `sample_ids_sha256`, tier counts and gold counts. For the full set: tiers 15/43/140; gold A38/B61/C62/D37.
- **GPQA canary**: the question text is never committed. The manifest belongs in the run's artifact
  directory.

### 1.2 Chosen n = 198 (the full Diamond set)

The cold-start rule says to use "the smallest slice that satisfies the four criteria". For GPQA-D, a
subset satisfies criterion 2 (paired significance) worse than the full set does, and it saves only a few
GPU-hours.

**Per-item wall time, measured on MI210** (read from the artifacts, not estimated):

| capture | model / serving | items | wall | s/item | mean completion tok |
|---|---|---:|---:|---:|---:|
| `architect-bench-gpu-20260814/gpqa_diamond_cot` | Qwen3.8-27B-Q8_0, v9 build-hip, 4-slot server, max_tokens 8192 | 198 | 14,902.6 s | **75.3** | 3,569 (18 truncated) |
| `architect-bench-gpu-20260720/full_gpqa_cot/…_ldhip` | Qwen3.6-27B-MTP-Q8_0, v9, np=1, max_tokens 8192 | 198 | 8,561.6 s | **43.2** | 1,831 (5 truncated) |
| `architect-bench-gpu-20260720/ablation_thinking/A4_frontdoor_35b_a3b_thinkoff` | Qwen3.6-35B-A3B-MTP-Q8_0, v9, np=1 | 50 | 1,034.5 s | **20.7** | 2,150 (0 truncated) |

Full n=198 therefore costs at most **≈4.1 h (27B) + ≈1.1 h (35B-A3B) ≈ 5.3 GPU-hours** on the v9 numbers.
These are upper bounds for the champion: DFlash2 runs at 79 tok/s at np=1 against the 48 tok/s median
decode rate in the 0814 capture.

**Statistics.** Take a model accuracy of p≈0.81 (the 0814 capture: 161/198). The binomial 95% CI
half-width is:

| n | CI half-width | sign-test power, discordance 18%, gap 7pp | discordance 18%, gap 5pp | discordance 12%, gap 3pp |
|---:|---:|---:|---:|---:|
| 50 | ±10.9pp | 0.12 | 0.07 | 0.03 |
| 100 | ±7.7pp | 0.30 | 0.16 | 0.08 |
| 150 | ±6.3pp | 0.45 | 0.23 | 0.13 |
| **198** | **±5.5pp** | **0.59** | 0.32 | 0.17 |

Power is for an exact two-sided sign test at α=0.05 over discordant pairs. The discordance rates come
from these same captures:
- Qwen3.8-27B vs Qwen3.6-27B, n=198: 25 vs 11 discordant, 18.2%, p=0.029.
- Qwen3.8-27B vs Qwen3.6-35B-A3B thinkoff, n=50: 6% discordant.
- Qwen3.8-27B vs the other 50-item arms: 6–16% discordant.

Even the full set has only a coin-flip chance of resolving a 7pp paired gap, and the vendor gap for the
sharpest pair is about 2pp. Subsetting would give up significance to save 2–3 GPU-hours, so **n=198** is
chosen. The seeded-subset path (`--n`) stays for reuse, but CJ-1e should not use it.

**What to expect from CJ-1e.** The ordering may come out *not resolvable*. Criterion 2 already allows
that outcome if the report includes the paired test. A null here is a finding about the instrument, not a
reason to rerun with more repeats. Extra repeats would reduce sampling noise within each item, but they
cannot add items.

### 1.3 Belief kernel

The write-side adapter already exists: `v7_quality_gate_beliefs.py` (SC32), which runs when the runner is
given `--belief-category`. It now copies the manifest's `sample` block **verbatim** into
`extra.prompt_set.sample`, keyed by suite name. A pin without a block, or a block that names another
suite, projects as `None` and is never reconstructed. The scored `reps` still come from the run summary.
Tests: `scripts/benchmark/test_cj_gpqa_sample.py`.

---

## 2. CJ-1e: the GPU ranking pair on champion `ef81196d5`

**Pair**: Qwen3.8-27B-Q8_0 with the DFlash2 drafter, and Qwen3.6-35B-A3B-MTP-Q8_0 with the MTP self-draft.
Both are proven on MI210 on this champion (`epyc-root/docs/design/champion-max-performance-20260908.md`,
and the 2026-09-08 35B-A3B sweep).

### 2.1 Build: verified present, not rebuilt

| field | value (checked 2026-09-16) |
|---|---|
| build dir | `/mnt/raid0/llm/tmp/build-fold-ef81196d5` (dir mtime 2026-09-08 09:24:39Z) |
| binary | `/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server`, mtime **2026-09-08 09:26:38Z**, sha256 `869effe5f5cda7f72bd78c8ee168a30f5878b3f32558cd0c02cd38a62a77db37` (**matches the champion doc**) |
| `libllama-common.so.0.0.10301` | mtime 2026-09-08 09:25:26Z |
| source | `/mnt/raid0/llm/tmp/fold-ef81196d5-src` HEAD `ef81196d5bdd4190b46dff4ae7eecc333a46c8ce`, committed 2026-09-08T08:53:12Z. The binary is newer than the commit, so it can contain it. |
| hazard | lives in `/mnt/raid0/llm/tmp`, so it can be reclaimed. Re-check the sha256 before launching. |

### 2.2 Model files (`ls -l`, 2026-09-16)

```
29047086048  2026-08-14  /mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf
 2056414752  2026-08-19  /mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
37801097504  2026-06-25  /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
```

### 2.3 Launch recipes

The source of truth is `artifacts/serving-recipes/{qwen3.8-27b-q8-gpu-dflash2-np4,qwen3.6-35b-a3b-q8-gpu-mtp}.json`,
rendered by `autokernel.loop.serving.Recipe.server_argv()` / `server_env()`. **Import them; do not retype
them.** The argv below is that renderer's output, with one declared deviation.

**Deviation: `-c 16384` becomes `-c 49152`.** The speed recipe uses `np=4` with `--no-kv-unified`, which
gives each slot 16384/4 = **4096 tokens**. GPQA-CoT needs about 250 prompt tokens plus
`max_tokens=8192`, and the 0814 capture already hit that cap on 18/198 items. At 4096 per slot, most items
would be truncated and the run would measure the context cap, not the model. The alternatives are
`-c 49152` (12288 per slot), or `np=1 -c 16384` with runner `--concurrency 1`, which is roughly 3× the wall
time. Changing `ctx` changes `recipe_hash`. That is correct here: this is a quality run, not a
serving-floor measurement, and it must not be quoted as the speed recipe. **VRAM at 49152 is unmeasured.**
The peak was 37.7 GiB at 16384, np=4, on a 64 GB card. The residency sample (§2.4) records the new peak.
If the server fails to load, fall back to `np=1`.

**Environment (both arms).** Start from the inherited environment, `unset HSA_OVERRIDE_GFX_VERSION`, and set
`LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin` (the loader-owned pin in
`Recipe.server_env`). Do **not** set `GGML_NOHUGEPAGE_PROCESS`, which is CPU-only per R23-58. No other
recipe env is required.

**Arm A: Qwen3.8-27B-Q8_0 + DFlash2** (port 18371; free on 2026-09-16, re-probe before launch)

```
env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin \
taskset -c 184-191 /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  -m /mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf -np 4 -c 49152 \
  -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on \
  --host 127.0.0.1 --port 18371 --metrics --slots \
  -md /mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf -ngld 99 \
  --spec-type draft-dflash --spec-draft-n-max 8 --no-kv-unified
```

**Arm B: Qwen3.6-35B-A3B-MTP-Q8_0, MTP self-draft** (port 18372). There is no `-md` and no `-ngld`: the
drafting head is part of the model weights.

```
env -u HSA_OVERRIDE_GFX_VERSION LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin \
taskset -c 184-191 /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -np 4 -c 49152 \
  -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on \
  --host 127.0.0.1 --port 18372 --metrics --slots \
  --spec-type draft-mtp --spec-draft-n-max 4 --no-kv-unified
```

Run the arms **one at a time**: one GPU and the standing GPU region claim. Use `taskset 184-191`, never
88-95, which is the CPU bench region.

**Runner (identical for both arms, only `--arm/--port/--models` differ).**

```
python3 scripts/benchmark/cj_gpqa_sample.py --out $RUN/cj1_gpqa_manifest.json
python3 scripts/benchmark/v7_quality_gate_runner.py --port 18371 --arm qwen3.8-27b-q8-dflash2 \
  --suites gpqa_diamond_cot --questions-in $RUN/cj1_gpqa_manifest.json --n 198 --seed 42 \
  --endpoint chat --no-enable-thinking --temperature 0.6 --top-p 0.95 --top-k 20 \
  --max-tokens 8192 --concurrency 4 --repeats 1 \
  --kernel champion-ef81196d5 --binary /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server \
  --models /mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf \
  --per-question-out $RUN/<arm>/per_question.jsonl --output $RUN/<arm>/result.json \
  --belief-category CANDIDATE --belief-config '{"recipe":"qwen3.8-27b-q8-gpu-dflash2-np4","ctx_override":49152}'
```

The sampling settings match the house GPQA-CoT precedent (the 0814 capture). With speculative decoding
and np>1, results are not bit-deterministic whatever the sampler, so do not claim a greedy replay.

**`enable_thinking` (Qwen3.x).** The runner sends `chat_template_kwargs={"enable_thinking": false}`, and
only on `--endpoint chat` (`/v1/chat/completions`). On `completion` the setting is inert, and the
effective request records that thinking was *not* sent. The server must apply the Jinja template: jinja is
on by default in this build. Confirm at launch that `GET /props` reports the Qwen template (a non-empty
`chat_template`). If you pass a template, use `--chat-template-file`, never `--chat-template <path>`: the
EVL-08 2026-08-26 incident turned 18 items into garbage that way. Two post-run proofs that thinking was
off: per-question `reasoning_chars == 0` for all rows, and no `<think>` blocks in the content. The
missing tags alone do not prove it.

### 2.4 Proof of linkage and residency, per arm

1. **Before launch**:
   `LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-fold-ef81196d5/bin scripts/utils/verify_ggml_linkage.sh /mnt/raid0/llm/tmp/build-fold-ef81196d5/bin/llama-server /mnt/raid0/llm/tmp/build-fold-ef81196d5`
   - Exit 0 is required. Exit 1 means the wrong tree. Exit 2 means nothing was inspected (a vacuous pass).
   - This check is necessary but not sufficient: `libggml-hip.so` is dlopened, so `ldd` cannot show it.
2. **During the run**, from the first request until the last:
   - Sample `/sys/class/drm/card2/device/mem_info_vram_used`, the KFD process count under
     `/sys/class/kfd/kfd/proc`, and sclk. The instrument is `autokernel.loop.residency.Sampler`.
   - Require **≥2 samples inside the request phase**, VRAM ≥ 1 GiB (`RESIDENT_FLOOR_BYTES`), and a KFD
     process count ≥ 1.
   - Record peak and median VRAM. A sample taken after the run is not evidence.
3. **After the run**: the llama-server pid you recorded (`ps -p`) is the process that served the run, and
   its `lstart` is later than the binary mtime. Kill only that pid, and confirm it is dead.

### 2.5 Emission

Both arms use the same key, slice and scaffold. For the analysis, reuse the EVL-08 statistic set:
accuracy, both-correct fraction, spread, and an exact sign test on discordant pairs. Emit the result as
`local_benchmarks.gpqa_diamond` **only if CJ-GATE adopts it**. This slice is the CoT framing under
`enable_thinking=false`, so the key name should be recorded with that framing in its definition.

---

## 3. CJ-3d: REPL protocol or native function calling?

**Verdict: this is not a pending operator decision at the wiring stage.**
- **Default**: measure the **native function-call path** under the key `bfcl_v3`, using the 2,760 checker-only rows.
- **Optional**: measure the production REPL path later under a *distinct* key, for example `bfcl_v3_repl_local`.

The CJ-3c note put this choice under CJ-GATE. What CJ-GATE actually owns is *adoption* ("which suites to
adopt … as a ranking prior"). Choosing which instrument to wire first is the executor's job, and the
handoff row says so ("Decide deliberately"). Why native is the default:

| | native function calling (`bfcl_v3`) | production REPL `TOOL()/CALL()/FINAL()` |
|---|---|---|
| what it measures | model tool-call competence, which is what BFCL was built for | model **plus** orchestrator scaffold, prompt and parser |
| vendor comparability (criterion 3) | the same kind of instrument as the vendor `bfcl_v3` key. This is the stated purpose of CJ-3e: the published keys are unrankable today (122B v4, Next-80B v3). | none: a different instrument, so it must use a different key |
| checker | the BFCL AST checker against `possible_answer/`, reusable as is | needs a new projection from REPL calls into BFCL's call schema: new scoring code, and scoring is a trust boundary |
| parse-failure risk | llama-server's own Qwen tool-call parser; tally parse failures as a separate outcome, never as wrong answers | the orchestrator parser is pinned (`22c476dd`) and leaves Qwen XML unparsed by design, so a parse failure must not look like a quality gap |
| cold-start consumer fit | `q_scorer` priors are per model, so a model-level instrument fits | measures the scaffold, which already shows up in autopilot's measured rewards |
| cost | lowest: direct HTTP to llama-server with `tools` | needs the orchestrator stack in the loop |

`ChatRequest.tools` being "accepted but never consumed" is a fact about the orchestrator API. It does not
affect a benchmark that calls llama-server directly. The REPL-path number is still worth having as a
*production-fidelity* check. It is additive and uses its own key, so it does not block the native wiring.
The operator's decision remains CJ-GATE: which key, if any, becomes the ranking prior.
