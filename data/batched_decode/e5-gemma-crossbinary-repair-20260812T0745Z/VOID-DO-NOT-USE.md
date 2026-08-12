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


## RETRACTION 2026-08-12 — the stated cause above is WRONG

This run is still VOID (zero measurements), but NOT for the reason given. The driver recorded
the real cause in `events.jsonl`: `affinity preflight exited 1` on all 8 cells — mainA's own
absolute-import sys.path bug in affinity_preflight.py, fixed in orchestrator `efbbbbe9`.

The `Gemma4Assistant requires ctx_other to be set` line quoted above is a BENIGN probe warning
— its own text says *(this warning is normal during memory fitting)* — and it appears in a
healthy MTP run too, immediately followed by `model loaded`.

**gemma4 MTP is verified working on production v9**: launched with the full 8-parameter recipe,
loaded and bound in 5.4s, served a correct completion, and speculative decoding engaged with
`draft acceptance = 1.00000 (4 accepted / 4 generated), mean len = 3.00`.

Diagnosed from the most alarming line in the server log instead of the cause the driver had
already recorded. The manifest's `void_detail.reason_code` is corrected alongside this note.
