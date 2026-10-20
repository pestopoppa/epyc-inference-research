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
