# paddleocr vl first smoke — 20260717T194332Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/paddleocr-vl-first-smoke-20260717T194332Z` |
| measured (file mtimes, UTC) | 2026-07-17 19:43 |
| migrated | 2026-08-02 |
| carried | 6 files, 15,419 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8867** &nbsp;`roles.paddleocr_vl_16_gguf.performance.mi210_observation`
  > mi210_observation: "PaddleOCR-VL loaded through experimental-v7 llama-server with model+mmproj. First smoke at data/paddleocr_vl_first_smoke_20260717T194332Z/ passed digit OCR at 484.36 t/s and invoice markdown extraction at 489.82 t/s. Narrow receipt QA under a 96-token cap emitted broad OCR tex...
- **L8869** &nbsp;`roles.paddleocr_vl_16_gguf.performance.evidence`
  > - data/paddleocr_vl_first_smoke_20260717T194332Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/paddleocr_vl_first_smoke_20260717T194332Z/SHA256SUMS
```

