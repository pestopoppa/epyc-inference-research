# RT-2.1 C++ change spec for the routing tap (NOT BUILT; RT-1 is compute-gated, file only)

Applies to the `llama-moe-trace` port (RT-1) on a `llama.cpp-experimental` branch cut from current production.
The Python side (`routing_trace.schema.json`, `trace_shape.py`) is the contract the port must satisfy.

1. **Token order and sample boundaries.** Write one row per token in decode order, never per-batch aggregates.
   `sample_id` increments at each `llama_memory_clear` (RT-1 adjustment 2); `pos` is the token index within the
   sample. The manifest `samples[]` records `start_row`/`n_rows`.
2. **Phase tag.** `phase` = `prefill` while the sample's prompt is being ingested (every chunk of `-b`), `decode`
   once generation starts. Derive it from `batch.logits`/the prompt length the caller already knows; do not infer it
   from `n_tokens == 1` (a speculative verify block is decode with n_tokens > 1).
3. **Speculative block size.** `step` = forward-pass counter per sample; `spec_block` = `n_tokens` of that forward
   (chunk length for prefill, 1 plain decode, block size for DFlash2/MTP verify). `spec_accepted` is filled after
   verification (true/false per row) for `spec_block > 1` decode rows, null otherwise; if acceptance is unknown at
   tap time, back-patch the column when the verify result returns and never leave a guess.
4. **int16 ids, 128-expert DSpark layers.** Expert-index tensor is int16 `[rows, n_layers, top_k]`; layers with a
   smaller `top_k` pad with -1. `layers[]` carries a per-layer `n_experts` (256/512 main layers and the 128-expert
   DSpark layers differ); the tap must read `n_experts`/`top_k` per layer from the graph, not from one global.
5. **Stride hazard (RT-3 applies).** Read the router tensor row by row at `i * t->nb[1]`.
6. **Output.** `<run>/routing_trace.json` (manifest) + `routing_trace.rows.bin` + `routing_trace.ids.i16`, with
   `rows_sha256` and `ids.sha256` filled at close. Write-side belief wiring is RT-2's, not this rider's.
7. **On-disk layout as built (RT-1, llama.cpp `tools/moe-trace`, written by `moe-trace-core.h` `run_writer`; read by `trace_shape.load_run`).**
   One capture = a directory of shards `shard-<first_window>/`, each a complete run: `routing_trace.json` (manifest, last file written,
   absent on an interrupted or failed shard), `routing_trace.rows.bin` (16-byte rows, little endian: `u32 sample_id, u32 pos, u32 step, u16 spec_block,
   u8 phase 0=prefill 1=decode, u8 spec_accepted 0/1/255=null`), `routing_trace.ids.i16` (`int16 [rows][n_layers][max top_k]`, -1 pad).
   Manifest `layer` is the ORDINAL of the MoE layer; the graph layer index `il` is in `routing_trace.provenance.json` `layer_il` (they differ only on
   models with dense leading layers). Sidecars outside the schema: `routing_trace.eor.json` (EOR/IR_t per layer and phase with the K/N chance
   baseline), `routing_trace.provenance.json`, optional `routing_trace.argsort.i16` (SRP, full descending expert order). `spec_block` is the forward's
   `n_tokens`; the port never speculates, so `spec_accepted` is always null.
8. **RT-3 guard as built.** Startup selftest (stride-safe reader on a synthetic strided tensor, guard must catch the flat read, exit 4), runtime guard on every
   prefill forward and layer (every expert selected the same number of times +-1, or cv < 0.10 with sub-chance adjacent reuse, evaluated only at n_experts >= 16,
   top_k*4 <= n_experts and >= 2*n_experts selections; exit 3), offline guard in `tools/moe-trace/moe_trace_convert.py verify`. A forward with a missing layer or the
   wrong row count is exit 2, never a shorter trace. The `ggml.c:5373` anchor above has drifted; `ggml_argsort_top_k` is `ggml/src/ggml.c:5443` at ffc1bac82.

