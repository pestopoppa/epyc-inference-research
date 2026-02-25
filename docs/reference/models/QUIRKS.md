# Model Runtime Quirks

Known issues and workarounds discovered during benchmarking. Check this before testing a new model.

## Critical Quirks

### Qwen3-Next (SSM Architecture)

**Issue**: State corruption when speculation is used

**Symptoms**:
- Garbage output after a few tokens
- Repetitive loops
- Model hangs

**Workaround**: Use expert reduction ONLY, never speculation or prompt lookup

```bash
# ✅ Safe
llama-cli -m Qwen3-Next-80B.gguf --override-kv qwen3next.expert_used_count=int:3

# ❌ Breaks model
llama-speculative -m Qwen3-Next-80B.gguf -md draft.gguf
```

### Qwen3-Coder-480B

**Issue**: BOS token is `,` (comma) instead of standard

**Symptoms**: 0% acceptance rate with standard draft models (BOS mismatch)

**Solution (2026-02-13)**: jukofyork vocab transplant draft (`Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf`) has matching BOS=comma. **Verified working**: 74-82% acceptance on code refactoring, 57% on novel generation.

**Recommended config (480B architect)**: Full experts + spec decode with jukofyork draft (K=16). Achieves 9.0 t/s (1.38x). Quality preserved — no MoE reduction for architect roles.
**Recommended config (30B frontdoor)**: MoE6 + spec + lookup = 47.11 t/s (2.58x). Lookup is net-positive on small MoE.

**Note on prompt lookup**: Works mechanically on MoE (18.4% acceptance on 480B) but net speed regression on large models. Net-positive on 30B (+27% on top of spec).

### Qwen3-235B-A22B Spec Decode

**Finding (2026-02-13)**: 0.6B Q8_0 draft dramatically outperforms 1.7B for 235B (55% vs 21% acceptance). Smaller draft wins on CPU due to faster proposal generation.

**Recommended config**: Full experts + 0.6B spec (K=16) = 6.08 t/s. Quality preserved for architect role.

**Note**: MoE4+spec was faster (8.21 t/s) but architect roles prioritize quality over speed.

### DeepSeek-R1-Distill-* Models

**Issue**: Vocab size mismatch between sizes (152,064 vs 151,936)

**Symptoms**: Token mismatch errors during speculation

**Workaround**: No speculation available. Use baseline or MoE reduction if applicable.

### Qwen3-Thinking-2507 Models (unsloth)

**Issue**: Missing BOS token metadata in GGUF

**Symptoms**: `draft model special tokens must match target model to use speculation`

**Root Cause**: The Qwen3-*-Thinking-2507 models from unsloth have `tokenizer.ggml.eos_token_id` but no `tokenizer.ggml.bos_token_id` in their GGUF metadata. All draft models have BOS tokens defined, causing a mismatch.

**Affected Models**:
- Qwen3-30B-A3B-Thinking-2507 (Q8_0 and Q4_K_S)
- Qwen3-4B-Thinking-2507

**Workaround**: Use MoE expert reduction only. Speculative decoding is incompatible.

```bash
# ✅ Safe - MoE reduction
llama-cli -m Qwen3-30B-A3B-Thinking-2507.gguf \
    --override-kv qwen3moe.expert_used_count=int:4

# ❌ Fails - spec decode (BOS mismatch)
llama-speculative -m Qwen3-30B-A3B-Thinking-2507.gguf \
    -md Qwen3-0.6B.gguf --draft-max 8
```

**Discovered**: 2026-01-12

### Gemma-3 Family (SWA Architecture)

**Speculative Decoding Issue**: ~~Sliding Window Attention (SWA) incompatible with speculative decoding in llama.cpp~~

**Status**: ✅ FIXED with PR #18720 (forward-looking SWA masking)

**Original Problem**: Gemma-3 uses Interleaved Sliding Window Attention (ISWA) with `sliding_window=1024`. The spec decode KV cache allocation failed because draft and target had incompatible cache structures.

**Solution**: PR #18720 adds forward-looking SWA masking in `find_slot()`, allowing cells that will be outside the attention window *after* batch insertion to be reused. This reduces SWA cache from 10240 MiB to 624 MiB (94% reduction).

```bash
# ✅ Now works (PR #18720 or upstream after merge)
llama-speculative -m gemma-3-27B.gguf -md gemma-3-1b.gguf --draft 4 -t 96

# Results: 42-81% acceptance rate, 12.26 t/s
```

**Prompt Lookup Issue**: ✅ FIXED (PRs #18729, #18730)

**Symptoms**: `GGML_ASSERT(batch.seq_id[...])` crash without `-c` flag (affects ALL models, not just SWA)

**Root Cause**: Two pre-existing bugs activated by upstream default changes:
- `lookup.cpp:109` - batch init with `params.n_ctx` (now defaults to 0)
- `lookahead.cpp:121` - same issue + n_seq_max validation

```bash
# ❌ Crashed without -c (before fix)
llama-lookup -m any-model.gguf -f prompt.txt --draft-max 4

# ✅ Now works (with PRs #18729 + #18730 or local cherry-pick)
llama-lookup -m gemma-3-27B.gguf -f prompt.txt --draft-max 4
```

**Test Result**: Prompt lookup works with SWA models (32.8% acceptance on Gemma-3-1b).

**Status**: PRs submitted to llama.cpp upstream, fixes cherry-picked to local fork.

**Note**: Vocab mismatch (1B=262144, 27B=262208) is safe - 64 token diff doesn't affect generation.

**Discovered**: 2026-01-09
**Spec Decode Fixed**: 2026-01-09 (PR #18720)
**Prompt Lookup**: Still broken as of 2026-01-10

### llama-lookup Binary (Large Context)

**Issue**: `llama-lookup` crashes with assertion failure on large context prompts

**Symptoms**:
```
GGML_ASSERT(src/llama-context.cpp:1008: n_tokens <= n_batch) failed
```

**Conditions**: Occurs with prompts >10K characters (e.g., document summarization)

**Workaround**: Use `llama-cli --lookup-ngram-min` instead of the dedicated binary:
```bash
# ✅ Works - llama-cli with lookup flag
numactl --interleave=all llama-cli \
    -m MODEL.gguf \
    --lookup-ngram-min 3 \
    -f large_prompt.txt \
    -n 500 --temp 0

# ❌ Crashes - llama-lookup binary
llama-lookup -m MODEL.gguf -f large_prompt.txt --draft-max 4
```

**Expected Speedup**: 12.7x on summarization tasks (per RESULTS.md)

**Discovered**: 2026-01-23

### Vision-Language (VL) Models

**Critical MTMD Fixes Required**: VL models crash in `llama_context::decode()` via `mtmd_helper_decode_image_chunk()` without these upstream commits:
- e047f9ee9: `mtmd: fix use_non_causal being reported incorrectly` (#18793)
- d98b54812: `Restore clip's cb() to its rightful glory` (#17914)
- c945aaaef: `mtmd: Fix ASR for LFM2.5-Audio-1.5B` (#18876)

**Cherry-picked**: 2026-01-27 to production-consolidated branch (commit 93eb39f39)

**Issue**: `llama-speculative` doesn't support VL models with mmproj files

**Symptoms**: Timeout or crash when running spec decode on Qwen2.5-VL, Qwen3-VL, or similar models

**Root Cause**: VL models require the mmproj (multimodal projector) file for vision processing. The `llama-speculative` binary doesn't support loading mmproj files, only the main model weights.

**Workaround**: Use `llama-mtmd-cli` for VL inference, or MoE expert reduction for MoE-VL models:

```bash
# ✅ Safe - llama-mtmd-cli for VL inference
llama-mtmd-cli -m Qwen2.5-VL-7B.gguf --mmproj mmproj-model-f16.gguf \
    --image image.png -p "prompt" -n 256

# ✅ Safe - MoE reduction for VL-MoE models
llama-mtmd-cli -m Qwen3-VL-30B-A3B.gguf --mmproj mmproj.gguf \
    --override-kv qwen3vlmoe.expert_used_count=int:4 \
    --image image.png -p "prompt"

# ❌ Broken - spec decode not supported for VL
llama-speculative -m Qwen2.5-VL-7B.gguf -md draft.gguf  # Times out
```

**Image Token Minimum**: Qwen-VL models require at minimum 1024 image tokens for grounding tasks. If accuracy issues occur, add `--image-min-tokens 1024`.

**Verified mmproj Paths** (2026-01-27):

| Model | Architecture | Model Path | mmproj Path |
|-------|--------------|------------|-------------|
| Qwen2.5-VL-7B | Dense | lmstudio-community/.../Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf | mmproj-model-f16.gguf |
| Qwen3-VL-4B | Dense | lmstudio-community/.../Qwen3-VL-4B-Instruct-Q4_K_M.gguf | mmproj-Qwen3-VL-4B-Instruct-F16.gguf |
| Qwen3-VL-8B | Dense | lmstudio-community/.../Qwen3-VL-8B-Instruct-Q4_K_M.gguf | mmproj-Qwen3-VL-8B-Instruct-F16.gguf |
| Qwen3-VL-30B-A3B | MoE | lmstudio-community/.../Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf | mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf |
| Qwen3-VL-235B-A22B | MoE | lmstudio-community/.../Qwen3-VL-235B-A22B-Instruct-Q4_K_M-*.gguf | mmproj-Qwen3-VL-235B-A22B-Instruct-F16.gguf |

**Performance Benchmarks** (2026-01-27, Twyne cover page):

| Model | Type | Prompt t/s | Gen t/s |
|-------|------|------------|---------|
| Qwen3-VL-4B | Dense | 104 | 14.8 |
| Qwen2.5-VL-7B | Dense | 97 | 10 |
| Qwen3-VL-30B-A3B (MoE4) | MoE | 29.6 | 5.3 |

**Discovered**: 2026-01-09
**MTMD fixes cherry-picked**: 2026-01-27

### VL Model Selection for Document Analysis (2026-01-28)

**Research Summary**: Comprehensive VL benchmark (12 questions) + Twyne whitepaper figure analysis

**Key Finding**: Smaller models outperform larger models on VL tasks.

| Model | VL Score | Speed | Recommendation |
|-------|----------|-------|----------------|
| **Qwen3-VL-4B** | **94%** | 18 t/s | ✅ **BEST for document figures** |
| Qwen3-VL-8B | 86% | 15 t/s | Good alternative |
| Qwen2.5-VL-7B | 81% | 17 t/s | ⚠️ Only option for agentic vision |
| Qwen3-VL-30B-A3B | 75% | 27 t/s | ❌ OCR errors ("Centric" instead of "Centre") |
| Qwen3-VL-235B | 56% | 4.6 t/s | ❌ Timeout truncation, slow |

**Why 4B beats 30B/235B:**
1. No timeout truncation (completes within limits)
2. Accurate OCR - correctly reads chart labels and axis values
3. Chart legend interpretation is accurate
4. Smaller context = faster processing

**Document Context Matters:**
- Without context: VL describes figures in isolation
- With summary (~8K chars): 2x quality improvement, figure descriptions tied to document meaning
- Full OCR (34K chars): Marginal improvement over summary, 2x slower

**Pipeline Integration** (implemented 2026-01-28):
- `DocumentPreprocessor._extract_summary_context()` extracts ~8K char summary
- Priority sections: abstract, summary, executive, introduction, overview
- `FigureAnalyzer` receives summary in `vl_prompt` parameter
- Logs: "Using {N} char summary context for figure analysis"

**Role Assignments:**

| Role | Model | Use Case |
|------|-------|----------|
| `worker_vision` (doc figures) | Qwen3-VL-4B | Document figure analysis (default) |
| `worker_vision` (agentic) | Qwen2.5-VL-7B | Tool-using vision tasks |
| `vision_escalation` | Qwen3-VL-30B-A3B | Manual request only (NOT auto-wired) |

**Agentic Warning**: All Qwen3-VL models score 0% on agentic tasks (empty tool calls). Use Qwen2.5-VL-7B for vision tasks requiring tool coordination.

### ColBERT Retrieval Models (NextPLAID Stack)

**LateOn-Code (130M, 128-dim) — Code Index (:8088)**

- **No query/document prefixes**: Model card explicitly states raw text input only. Do NOT add `search_query:` or `search_document:` prefixes — these are specific to models trained with them (e.g., GTE-ModernColBERT). Adding prefixes to LateOn-Code would degrade retrieval quality.
- MTEB-Code: 74.12
- ModernBERT backbone, ONNX INT8

**answerai-colbert-small-v1 (33M, 96-dim) — Docs Index (:8089)**

- **Dimension mismatch**: 96-dim output vs 128-dim code index. Not functionally broken (separate containers), but inconsistent.
- Unscored on BEIR. Small model with limited long-context generalization.
- ONNX INT8 available (official export).

**GTE-ModernColBERT-v1 (149M, 128-dim) — Candidate Docs Replacement**

- **Official ONNX INT8 available** (143MB on HuggingFace). No custom conversion needed.
- Uses `[Q] ` / `[D] ` prefixes (handled by NextPLAID container via `onnx_config.json`).
- BEIR avg 54.67, LongEmbed 88.39 (SOTA).
- 128-dim output (matches LateOn-Code). Hidden 768→128 via Dense projection (bundled in ONNX).
- ModernBERT backbone.
- Query latency: ~21ms (INT8, CPU).
- Local path: `/mnt/raid0/llm/models/gte-moderncolbert-v1-onnx/`
- **Discovered**: 2026-02-20

### PLAID Search Parameter Tuning (NextPLAID)

**Issue**: `n_ivf_probe` (default 8) and `n_full_scores` (default 4096) are never configured — both run at defaults.

**Opportunity**: Increasing `n_ivf_probe` from 8 to 16-32 could improve recall for code search. `n_full_scores` can be lowered to reduce reranking overhead.

**Status**: Noted for future tuning. No action taken yet.

**Discovered**: 2026-02-20

---

## Benchmarking Quirks

### Interactive Mode Hangs

**Issue**: `llama-cli` waits for user input if not configured correctly

**Symptoms**: Benchmark script hangs indefinitely

**Workaround**: Always use these flags:
```bash
llama-cli -m MODEL.gguf -f prompt.txt -n 128 \
    --no-display-prompt \
    --simple-io \
    --no-warmup \
    --temp 0
```

**Never use**: `-i` or `--interactive` in automated scripts

### Output Capture Issues

**Issue**: Some models output to stderr, breaking parsing

**Workaround**: Capture both streams
```bash
llama-cli ... 2>&1 | tee output.log
```

### `<think>` Tag Models

**Issue**: Thinking models emit `<think>...</think>` tags that inflate token counts

**Workaround**: Parse output to separate thinking from final response

## Performance Quirks

### Temperature and Speculation

**Issue**: Some models perform better with non-zero temperature during speculation

| Model | Best temp | Speed Impact |
|-------|-----------|--------------|
| Qwen2.5-VL-7B | 0.7 | 28.3 → 57.1 t/s |
| Qwen2.5-Math-72B | 0.5 | 6.0 → 7.5 t/s |
| Qwen2.5-Coder-32B | 0 | Best at temp=0 |

**Workaround**: If acceptance <50% at temp=0, try temp=0.3-0.7

### MoE Expert Count Sweet Spots

**Issue**: Below 4 experts causes instability (SIGSEGV, garbage output, UTF-8 decode errors)

| Model | Min Safe Experts | Issue at 2 Experts |
|-------|------------------|-------------------|
| Qwen3-VL-30B | 4 | ⚠️ Garbage output |
| Qwen3-Next-80B | 4 | ⚠️ Garbage output |
| Qwen3-235B | 4 | ⚠️ Garbage output |
| Any MoE | 4 | ⚠️ Unstable |

**Workaround**: Benchmark system starts MoE testing at 4 experts minimum

**Note**: Earlier reports of Qwen3-30B-A3B-Thinking crashes were caused by a stale build issue, not the model itself. Model works fine with moe2 and moe4 on clean builds.

## Memory Quirks

### Context Length Limits

| Model Family | Max Context | Notes |
|--------------|-------------|-------|
| Llama2 | 4K | Hard limit |
| Llama3 | 8K | Default, some support 128K |
| Qwen | 131K | But slower beyond 32K |
| DeepSeek-R1 | 65K | Official limit |

### VRAM/RAM Estimates

| Quantization | Size Formula |
|--------------|--------------|
| Q4_K_M | params × 0.5 GB |
| Q8_0 | params × 1.0 GB |
| F16 | params × 2.0 GB |

Example: 70B model at Q4_K_M ≈ 35GB

## REPL Tool Compliance

**Issue**: Models may use Python imports instead of REPL tools

**Symptoms**:
- `SecurityError: Dangerous operation not allowed: import os`
- Code uses `os.listdir()`, `pathlib.Path()`, `open()` instead of REPL tools
- Multiple failed turns before model adapts

**Affected Models**:
- Qwen3-Coder-30B-A3B (frontdoor) - Initially tried `pathlib` and `os.listdir`
- Other models may vary in instruction-following capability

**Workaround**: Add explicit NO IMPORTS warnings to system prompts:
```
## CRITICAL
1. **NO IMPORTS** - import/from are BLOCKED. Use ONLY the tools above.
2. **USE list_dir()** for files - NOT os.listdir or pathlib
3. **ALWAYS call FINAL(answer)** to complete the task

## Examples
List files: `result = list_dir('/path'); FINAL(result)`
Read file: `text = peek(1000, file_path='/path'); FINAL(text)`
```

**Tool → Python Equivalent Mapping**:

| REPL Tool | Forbidden Python Equivalent |
|-----------|---------------------------|
| `list_dir(path)` | `os.listdir()`, `pathlib.Path().iterdir()` |
| `peek(n, file_path)` | `open().read()`, `pathlib.Path().read_text()` |
| `grep(pattern)` | `re.findall()`, `grep` subprocess |
| `file_info(path)` | `os.stat()`, `pathlib.Path().stat()` |
| `run_shell(cmd)` | `subprocess.run()`, `os.system()` |
| `web_fetch(url)` | `requests.get()`, `urllib` |

**Testing**: Run `pytest tests/integration/test_model_tool_compliance.py -v`

**Discovered**: 2026-01-24

---

## Adding New Quirks

When discovering a new quirk:

1. Add to this file with:
   - Issue description
   - Symptoms
   - Workaround
   - Discovery date

2. Update `orchestration/model_registry.yaml`:
   ```yaml
   runtime_quirks:
     model_name:
       quirks:
         - issue: "Description"
           workaround: "Fix"
           discovered: YYYY-MM-DD
   ```

### llama-server /slots API

**Issue**: Field names differ between llama-server versions

**Symptoms**: Slot availability checks fail silently when code checks `state == 0` (legacy) but server returns `is_processing` (boolean, modern). Also, single-slot servers (`-np 1`) return a dict from `/slots`, not a list.

**Workaround**: Check both fields: `not s.get("is_processing", True) or s.get("state") == 0`. Wrap response with `isinstance(data, list)` guard.

**Discovered**: 2026-02-19 (in `EscalationPrewarmer._check_slot_available()`)

---

*See [MODELS.md](MODELS.md) for model configurations.*
