# DS41-W6 — DSpark drafter for DeepSeek-V4.1-Flash-1017 (2026-10-21)

Converter: `scripts/models/convert_ds41_dspark_1017.py` (gguf-py, GGUF->GGUF).

- Source: unsloth `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, sha256 `6ba6ac1b...3927a` (it matches the HF LFS oid).
- Output: `/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf`, 10,004,006,720 B, sha256
  `d9ddd4be6d57da6914786c5423450f02a62169d6df849e1f15fc7df411734ebd`.

## Conversion checks (the DS41-B13 gate, vs the Sept drafter)
- KVs: identical to the Sept drafter except `general.name` and `general.description`. Every architectural value was cross-checked against unsloth's `dflash.*` value, and none differ. The tokenizer KVs come from 1017 shard 1 (`add_bos_token=True`).
- Tensors: 80 vs 83. All names, shapes and dtypes match. The 3 missing tensors are `blk.{0,1,2}.exp_probs_b_vl.bias`, which unsloth does not ship. The loader marks them TENSOR_NOT_REQUIRED and no graph reads them.
- `token_embd` and `output` are byte-identical to the 1017 target's shard 1 (sha256 compared).
- The markov tables were cast BF16->F16 to match the Sept dtype. 347 and 324 of 33.1M entries flush to F16 subnormals, max abs err 3e-8, NMSE about 2e-20. So the loader's BF16 path is never used.
- Rename mapping, confirmed by correlation against the Sept tensors: each tensor correlates 0.84-0.96 with its counterpart and at most 0.4 off-diagonal.

## Load, smoke and alpha A/B
Setup:
- Port `llama.cpp-experimental-ds41-20261005/build-cpu`, 1017 MXFP4 target (mmap).
- `--spec-type draft-dspark`, `-t 24` on cpus 24-47, under region-lock build role.
- Temperature 0.7, top_p 1.0, top_k 0, fixed seeds, 128 tokens per prompt.
- Results are indicative only, with no quiet host.

The 16-prompt continuation set is in `driver/client.py`. Alpha below is accepted/drafted.

| arm | block 2 | block 3 |
|---|---|---|
| 1017 drafter (new), r2 | 0.5822 (878/1508) | 0.4764 (959/2013) |
| Sept drafter on 1017 target, r2 | 0.6094 (897/1472) | 0.4857 (967/1991) |
| 1017 drafter, r1 (6 prompts) | 0.5946 | 0.4986 |
| Sept drafter on 1017, r1 (6 prompts) | 0.6441 | 0.5189 |
| recorded Sept drafter + Sept Q4 (B13b, different prompts) | 0.516 | 0.444 |

Per-prompt pairs from r2:
- Target text was identical in all 16 prompts across both drafters.
- Block 2: mean -2.3 pp for the new drafter (sd 3.0), with 1 win, 9 losses and 6 ties.
- Block 3: mean +0.15 pp (sd 4.5), with 5 wins, 6 losses and 5 ties.
