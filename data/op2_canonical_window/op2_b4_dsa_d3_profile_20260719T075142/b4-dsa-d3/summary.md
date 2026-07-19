# OP-2 B4 DSA-D3 Profile-First Summary - 2026-07-19

Status: completed, observation-grade. This is not a MEASUREMENT-gated promotion
number and does not admit GLM-5.2 to a production role.

Command shape:

- Runner: `scripts/benchmark/glm52_dsa_probe_runner.py --execute`
- Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin/llama-server`
- Library path: `/mnt/raid0/llm/llama.cpp-experimental/build-k24-cpu/bin`
- Model: `/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf`
- Stage: `kv_length_scaling`
- Context: `8192`
- Override: `glm-dsa.attention.indexer.top_k=int:2048`
- Threads / ubatch: `96` / `512`
- Device: CPU-only (`--device none -ngl 0`)
- Perf: `perf record -F 99 --call-graph dwarf`

Runtime result:

- Prompt tokens: `5906`
- Completion tokens: `1`
- Prompt time: `315917.603 ms`
- Prompt throughput: `18.6947 t/s`
- Server log confirms main DSA KV cache, indexer DSA KV cache, and
  `Lightning Indexer enabled`.
- Cleanup: no post-run `llama-server`, `llama-bench`, `perf`, or KFD PIDs were
  observed after the run.

Profile result:

- Raw profile: `perf.data`, `23619.821 MB`, `2935369` samples. This is local
  scratch evidence and intentionally not intended for git.
- Full callgraph `perf report` was interrupted because symbolization repeatedly
  emitted `addr2line ... could not send request`; the bounded symbol-only report
  completed at `perf_report_symbol_only.txt`.
- Total lost samples: `0`.
- Top symbol-only cycle samples:
  - `ggml_vec_dot_iq2_xxs_q8_K`: `18.74%`
  - `ggml_vec_dot_iq3_xxs_q8_K`: `14.58%`
  - `libgomp` scheduling/runtime frame: `14.53%`
  - `ggml_compute_forward_flash_attn_ext_tiled`: `10.78%`
  - `ggml_vec_dot_q5_K_q8_K`: `10.68%`
  - `tinyBLAS_Q0_AVX...gemm4xN`: `7.87%`
  - `ggml_compute_forward_lightning_indexer`: `1.08%`
  - `std::__adjust_heap<...cmp_top_k>`: `0.27%`

Decision:

D3.1 is closed as a no-go for immediate AVX-512BW Lightning Indexer kernel
work. The indexer path is active but not material in this profile; quantized
dot products, flash attention, OpenMP/runtime overhead, and related conversion
work dominate the sampled cycles. Do not start D3.2/D3.3 from this evidence.
Reopen D3 only if a future D2 real-sparse final-attention implementation or a
materially different GLM serving shape makes `GGML_OP_LIGHTNING_INDEXER`
cycle share large enough to matter.
