# VOID RUN — contains ZERO measurements. Do not read summary.csv or cells.jsonl as data.

Run aborted by mainA 2026-08-12T07:53Z. Every cell failed: the gemma4 MTP draft model
could not create a context on production v9, so llama-server exited ~1.7s after binding
and served zero requests.

The 8 rows in summary.csv and cells.jsonl carry success_count=0, total_count=0,
error_rate=1.0, aggregate_decode_tps=0.0. They are the driver's record of a failure,
NOT measurements. error_rate=1.0 here is computed from 0/0 — a value manufactured from
an empty denominator, not an observed error rate.

Root cause in logs/*.log (9 of 9):
  E llama_init_from_model: failed to initialize the context:
    Gemma4Assistant requires ctx_other to be set
  W srv load_model: [spec] failed to measure draft model memory:
    failed to create llama_context from model
