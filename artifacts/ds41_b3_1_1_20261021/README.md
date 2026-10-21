# DS41-B3.1.1 - Engram-gate fix: real-model decode-logit bit-identity (2026-10-21)

Verdict: BIT-IDENTICAL on all 4 prompts. Fixed build vs base build, DeepSeek-V4.1-Flash-1017 MXFP4 target
(`/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf`).
No divergence anywhere, no NaN.

- Fix: llama.cpp `6b8291de5` (`experimental/intake-fixes-20261020`), `libllama.so.0.0.10349` built 2026-10-21T14:56:13Z.
- Base: `9154bb28f` (parent of the fix commit `abcd831fb`), `libllama.so.0.0.10346` built 2026-10-21T14:56:40Z.
- Fix tip also carries CH-19 (default-off, `GGML_CPU_FUSED_UPGATE` unset) and DSA-IDX.1 (header + test only).
- Build: Release, gcc-15/g++-15, GGML_NATIVE=ON, OpenMP, llamafile, shared libs, CPU only.
- Run: `-t 24`, cpuset 48-71 (region q2), mmap, n_ctx 8192, n_batch 2048, n_ubatch 512, flash-attn on, f16 KV,
  `GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active`.
- Method: `logit_dump.cpp` (links libllama only) does greedy argmax; per prompt 65 logit rows (prefill-last + 64 decode).
  Per row: FNV-64 of the full-vocab float logits, NaN count, top-10 (id, f32 bits hex, value).
- Compared: raw full-vocab logits (`cmp`, byte-for-byte), tokens, steps jsonl. The ~33 MB raw dumps are not committed;
  their sha256 are in `fix/logits.sha256` and `base/logits.sha256` (identical).
- Prompts (`prompt_ids.txt`): p00 = prompt-manifest ds41-decode p01 (raw BOS text); p01-p03 = BOS,`<User>`(128803),text,
  `<Assistant>`(128804),`</think>`(128822) from `q.txt`.
- The -0.0 case was decided separately: it does not matter (dot = sum*rstd*dim^-0.5 has positive scales).
