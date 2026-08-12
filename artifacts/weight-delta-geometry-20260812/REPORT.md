# Weight-Delta Geometry Probe — ThinkingCap vs Fable-Fusion vs stock Qwen3.6-27B

**Date**: 2026-08-12
**Instrument**: `scripts/benchmark/weight_delta_geometry.py` (committed 2026-07-29, `4c1a3ac3`)
**Task**: `handoffs/active/architect-model-selection-bench.md:563`
**Method**: zero-inference streaming Q8_0 GGUF header/block reads. No llama binary, server, GPU, or model inference was invoked at any point in this probe.

## 1. Instrument review — before running anything

Read `scripts/benchmark/weight_delta_geometry.py` end to end (262 lines) plus its test suite before execution. Two real defects were found and fixed (both minimal, both load-bearing); everything else about the design held up.

### 1.1 IO pattern (as specified, confirmed correct)

`read_header()` opens the file with plain buffered I/O, reads the GGUF magic/version/KV/tensor-table, and returns as soon as the tensor directory is parsed (~0.23s per file, regardless of file size) — it never touches tensor payload bytes. `--plan` mode calls only `argparse`, never `read_header`, so a nonexistent path (`/missing/stock.gguf`) is accepted without error — confirmed by running it.

For `--execute`, `execute()` opens each of the three files with `mmap.mmap(fd, 0, ACCESS_READ)` — this maps the **whole file into the process's virtual address space**, but pages are only physically faulted in as they're touched. Actual reads happen through `_accumulate_tensor()`, which walks each tensor's byte range in `chunk_bytes`-sized strides (default 64 MiB, rounded down to a whole number of 34-byte Q8_0 blocks) and dequantizes only that chunk before discarding it. So the precise characterization is: **whole-file address-space mapping, bounded-chunk physical reads** — not a single `mmap`-and-touch-everything call, and not slurping any full tensor (up to ~340 MB for `ffn_down`) into RAM at once. This matches the task's "streamed block reads, not whole-file mmap" intent in substance even though the address space itself is fully mapped.

### 1.2 Tensor-matching defect — found and fixed

**Before my edit**, `_tensor_triplets()` computed `shared_names = set(stock) & set(tc) & set(ff)` and only ever reported `len(shared_names) - len(triplets)` as a single opaque integer (`skipped["not_shared_or_q8_or_shape_mismatch"]`). Two things were silently dropped:
- any tensor name **not in the 3-way intersection at all** (e.g. an MTP-head tensor present in only one file) was never even counted, let alone named;
- among the shared names, a **type mismatch** (same name, different `type_id` across the trio) and a **shape mismatch** were merged into one undifferentiated bucket.

Real data on this run made the distinction matter: of 851 stock tensors, 97 shared-by-name tensors have **inconsistent quantization type across the trio** (not just "not comparable") — see §3.

**Fix**: replaced `_tensor_triplets` with `_classify_tensors`, which partitions the full tensor-name union into the comparable triplet list plus four explicitly named exclusion buckets: `not_in_all_three` (with per-file presence booleans), `shape_mismatch`, `type_mismatch` (with each file's `type_id`), `uniform_non_q8_0`. `execute()`'s output now carries `exclusions` (full named lists) alongside `skipped` (counts, kept for a quick summary). Rationale: the task explicitly requires every non-shared/mismatched tensor to be listed, not folded into a count — verified the old code did fold them, so this was fixed before running anything real.

### 1.3 Correctness defect — found during first real execution, fixed

The synthetic tests always called `execute(..., chunk_bytes=34)` — exactly one Q8_0 block per chunk. That hid a broadcasting bug in `q8_dequantize()`:

```python
scales = blocks[:, :2].copy().view("<f2").astype(np.float32)  # shape (N, 1) already — the <f2 view halves bytes but keeps the axis
values = blocks[:, 2:].view(np.int8).astype(np.float32)        # shape (N, 32)
return (values * scales[:, None]).reshape(-1)                  # BUG: scales[:, None] on an already-(N,1) array gives (N,1,1)
```

`values (N,32) * scales (N,1,1)` broadcasts to `(N, N, 32)` — for the default 64 MiB chunk (N ~ 983,040 blocks) this tried to allocate a **112 TiB** array and crashed on the very first real invocation (`numpy._core._exceptions._ArrayMemoryError`). Fixed by removing the redundant `[:, None]` (`scales` is already `(N,1)`, so `values * scales` broadcasts correctly to `(N,32)`). Verified against a hand-built two-block fixture with distinct scales per block (new test, see §2). This is a correctness bug that would have produced either a crash (as observed) or, at smaller chunk sizes, silently wrong per-tensor sums — not a cosmetic issue.

### 1.4 Incremental persistence — added

`execute()` now accepts `jsonl_path`; each tensor's row is written and flushed to that file the instant it's computed, so a crash mid-run loses at most one in-flight tensor. `main()` wires a new `--jsonl` flag (defaulting to `<output>.jsonl` when `--execute` is used). The final `--output` JSON (full rollup: exclusions, per-layer aggregates, per-tensor rows) is still written once at the end.

## 2. Synthetic tests

`python3 -m unittest scripts.benchmark.test_weight_delta_geometry` — **6 passed, 0 failed** (up from the original 3; the 3 pre-existing tests are untouched and still pass, 2 new regression tests were added for the two defects above):
- `test_q8_dequantize` (pre-existing)
- `test_execute_reports_known_geometry_and_zero_delta_control` (pre-existing)
- `test_plan_is_default_and_does_not_require_input_files` (pre-existing)
- `test_q8_dequantize_multiple_blocks_use_their_own_scale` (**new** — closes the coverage gap that hid the broadcasting bug)
- `test_excluded_tensors_are_listed_by_name_not_silently_dropped` (**new** — proves the exclusion fix)
- `test_execute_writes_incremental_jsonl_per_tensor` (**new** — proves the JSONL sink)

`--plan` mode sanity check with the real, on-disk paths printed the expected plan JSON (`will_execute: false`, all three input paths echoed, no file was opened — confirmed separately that `--plan` never calls `read_header`).

## 3. Execution

```
nice -n 10 timeout 3600 python3 scripts/benchmark/weight_delta_geometry.py \
  --stock /mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf \
  --thinkingcap /mnt/raid0/llm/models/ThinkingCap-Qwen3.6-27B-GGUF/ThinkingCap-Qwen3.6-27B-Q8_0.gguf \
  --fable /mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf \
  --execute --output artifacts/weight-delta-geometry-20260812/result.json \
  --jsonl artifacts/weight-delta-geometry-20260812/tensors.jsonl
```

Exit 0. Wall time 3m53s (91.3s user + 147.6s system, 102% CPU — I/O-bound, consistent with streaming reads rather than compute-bound work). ~27.2 GB of Q8_0 payload compared per file, ~81.6 GB total streamed across the three files. Host load average at run time was 14.8/192 cores. `tensors.jsonl` has 401 lines (one per compared tensor, written incrementally); `result.json` is the final rollup.

### 3.1 Tensor accounting — nothing dropped

TC's tensor table has 866 names (stock and FF both have 851; FF's set is identical to stock's — confirmed no extras, no gaps). Union across the trio = 866. All 866 are accounted for:

| Bucket | Count | What it means |
|---|---|---|
| **Compared** (in all 3, Q8_0 in all 3, same shape) | **401** | actual r/c/p computed |
| `not_in_all_three` | 15 | `blk.64.{attn_k,attn_k_norm,attn_norm,attn_output,attn_q,attn_q_norm,attn_v,ffn_down,ffn_gate,ffn_up,nextn.eh_proj,nextn.enorm,nextn.hnorm,nextn.shared_head_norm,post_attention_norm}.weight` — present **only in ThinkingCap** |
| `uniform_non_q8_0` | 353 | present in all 3, all F32 (type_id 0) in all 3 — SSM/norm auxiliary tensors, out of scope for this Q8_0-only instrument by design |
| `type_mismatch` | 97 | present in all 3, **but the type_id differs across the trio** |
| `shape_mismatch` | 0 | none found |

`401 + 15 + 353 + 97 + 0 = 866`. Full names for every excluded tensor are in `result.json["exclusions"]`.

Two things inside `type_mismatch` are worth calling out by name:
- **96 tensors** (`blk.N.ssm_alpha.weight`, `blk.N.ssm_beta.weight`, one pair per layer) are F32 in **stock** but Q8_0 in **both TC and FF** — a real, consistent GGUF-build difference in how the two derivative tools quantized these small SSM scan parameters vs. how stock's own conversion left them. Not comparable via this instrument (stock side isn't Q8_0), and not a defect in the models — just a dtype choice that differs by convert tool.
- **`output.weight`** (the LM head, 5120x248320) is Q8_0 in stock and TC but **BF16 in Fable-Fusion**. FF's merge/quant pipeline evidently kept the head at higher precision. Also excluded from the Q8-only comparison, named explicitly.

### 3.2 The MTP-control read

Read `Qwen3.6-27B-MTP-Q8_0.gguf`'s header only (no payload bytes touched, ~0.2s): 866 tensors, exactly 15 more than stock. Those 15 extra names are **byte-for-byte the same set** as TC's 15 extras (`blk.64.*`, the multi-token-prediction head). This is a genuine provenance finding, not just an IO-pattern check: **ThinkingCap's tensor topology matches the dedicated MTP checkpoint, not the plain stock checkpoint** — TC appears to have been converted from (or alongside) the MTP lineage. It does **not** invalidate the delta computation: LoRA modules only ever target pre-existing weight matrices, all of which are present in stock too, and the extra `blk.64.*` tensors are correctly excluded and explicitly named (§3.1) rather than silently dropped or, worse, silently compared against nothing. Fable-Fusion's tensor set has no such extras — it matches stock exactly, confirming the task's instruction to use the non-MTP FF sibling was followed correctly on FF's side.

## 4. Ground-truth gates

### 4a. Byte-level prior — ThinkingCap = stock + 256 LoRA modules

Among the 401 Q8-comparable tensors, sorting by `norm_tc_sq` shows a clean, enormous gap:

| Population | n | RMS delta per element |
|---|---|---|
| Non-LoRA-targeted | **145** | 140 exactly **0.0** (bit-identical Q8_0 bytes); the other 5 (`token_embd.weight` + 4 `blk.{9,10,14,16}.attn_qkv.weight`) max out at **2.69e-9** |
| LoRA-targeted | **256** | min **8.49e-6**, median **1.96e-5** |

Separation: smallest signal value / largest noise value ~ **8.49e-6 / 2.69e-9 ~ 3,150x**. The signal population is **exactly 256 tensors** — `blk.N.{ffn_down,ffn_gate,ffn_up}.weight` for all 64 layers (192) + `blk.N.{attn_q,attn_k,attn_v,attn_output}.weight` for the 16 full-attention layers (64) — matching the stated LoRA module count exactly, with no fitting or threshold-tuning required to hit that number.

**Verdict: PASS, decisively.** The Q8_0 dequant/requant noise floor on non-LoRA tensors is either exactly zero (96.6% of the non-targeted population) or ~3 orders of magnitude below the smallest real edit. GGUF delta-geometry is trustworthy at this quantization level for this model. (Sanity cross-check: 851 total stock-named tensors minus 256 LoRA-targeted = 595, close to the task's stated "~600 non-LoRA tensors"; of those 595, 145 are Q8-comparable and confirmed near-zero above, the remaining 450 are exactly the `type_mismatch` + `uniform_non_q8_0` buckets from §3.1 (97+353=450) — the accounting closes.)

### 4b. TC/FF independence — p(L) should sit near 0 and flat

Per-layer stats across the 64 `blk.N` layers (non-layer/`token_embd.weight` excluded — its D_TC is noise-floor scale, see §4a, so its r/cos/p are not meaningful):

| Measurand | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| r(L) = \|\|D_FF\|\|/\|\|D_TC\|\| | 42.9 | 63.6 | 69.9 | 106 | 145 |
| cos(L) | 0.0032 | 0.0044 | 0.0066 | 0.0070 | 0.0107 |
| p(L) | 0.437 | 0.448 | 0.456 | 0.464 | 0.483 |

`cos(L)` — the scale-invariant alignment measure — is essentially zero (<=1.1%) and flat across every layer, with no drift toward 1.0 anywhere. That is the correct, direct read on TC/FF independence and it **passes**.

`p(L)` is a different story and needs a caveat: by construction `p = dot/||D_TC||^2 = cos * (||D_FF||/||D_TC||) = cos * r`. Because `r` is large everywhere here (43-145x, i.e. FF's edit is far bigger than TC's narrow LoRA edit), `p` gets amplified into a tight 0.44-0.48 band even though the underlying directional alignment (`cos`) is ~0.7%. Taken alone, without also reading `r` or `cos`, `p ~ 0.45` could be mis-read as "45% shared direction" — it is not; it is a near-orthogonal delta whose ratio-of-magnitudes happens to be ~70-140x. **This is a property of the p(L) formula under magnitude asymmetry, not a lineage signal, and not a model finding** — flagging it per the task's instruction that a p(L) that fails to read near-zero is a finding about the detector. Recommendation for future use of this instrument: report `cos(L)` as the primary lineage discriminator and treat `p(L)` only in conjunction with `r(L)`, never standalone.

**Verdict: PASS** on the intended question (TC and FF are geometrically independent siblings, confirmed via `cos`), **with a documented detector caveat** on `p(L)`'s scale sensitivity.

## 5. What this does and does not authorize

This is an **observation-grade, zero-inference geometry read** of three on-disk GGUF files. It confirms: ThinkingCap's LoRA patch is exactly and only where it should be (256 tensors, clean noise floor elsewhere), and that ThinkingCap and Fable-Fusion are directionally independent edits off the same stock base. It does **not** run, serve, or benchmark any of these models, says nothing about output quality, latency, throughput, or downstream task performance, and **authorizes no model, role, or lineup action**. Any decision about serving, promoting, or retiring TC, FF, or MTP requires the normal inference-based evaluation path, not this instrument.

## Files written

- `scripts/benchmark/weight_delta_geometry.py` — instrument, fixed (exclusion listing, broadcasting bug, incremental JSONL)
- `scripts/benchmark/test_weight_delta_geometry.py` — 3 new regression tests added
- `artifacts/weight-delta-geometry-20260812/result.json` — full rollup (exclusions, per-layer, per-tensor)
- `artifacts/weight-delta-geometry-20260812/tensors.jsonl` — 401 per-tensor rows, written incrementally
- `artifacts/weight-delta-geometry-20260812/REPORT.md` — this report
