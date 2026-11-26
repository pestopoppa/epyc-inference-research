# Proposal: hub page for np x ctx serving matrices (NOT applied)

Plane rule (dashboard/README.md): data contract with the producer, page + registry entry + health probe + freshness
envelope in the hub.

- **Data contract (producer side):** the capture rows `epyc.np_ctx_depth_matrix.v1` (self-hashed). The derived record comes from `np_ctx_depth_matrix.py export-catalog`; it is the epyc-web `data/catalog.json` matrix shape, so the hub page and the website read ONE format.
- **Page:** `/matrix`. For each (model, device) it shows a heat table of context depth × np, with a per-request / aggregate / prefill toggle, a draft-acceptance overlay and the per-cell status (SKIPPED_CAPACITY, STOPPED_STEP_CAP, REQUIRES_CTX_EXTENSION). There is an arm selector (INCUMBENT / C21 / C22) and a ratio view vs INCUMBENT.
- **Registry entry** (`dashboard/registry.json`): `{"id": "serving-matrix", "path": "/matrix", "source": "<capture dir>/capture.jsonl", "health": "/api/health?page=serving-matrix", "freshness": {"max_age_h": 168, "field": "max(row.utc)"}}`.
- **Health probe:** three-valued. A capture whose rows fail `verify_row` → RED. Every row hash-valid, but the newest row is older than max_age → STALE. Otherwise GREEN.
- **Vidya wiring (belief kernel):** add a source-table row in `scripts/vidya/adapters/README.md` and an adapter that projects `epyc.np_ctx_depth_matrix.v1` cells. The existing `np_ctx_kvu.py` reads the older `epyc.vidya.np_ctx_kvu_capture.v1` shape and does not read this one.
