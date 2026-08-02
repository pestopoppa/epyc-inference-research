# paddleocr vl receipt extract — 20260717T194415Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/paddleocr-vl-receipt-extract-20260717T194415Z` |
| measured (file mtimes, UTC) | 2026-07-17 19:44 |
| migrated | 2026-08-02 |
| carried | 4 files, 7,471 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8867** &nbsp;`roles.paddleocr_vl_16_gguf.performance.mi210_observation`
  > mi210_observation: "PaddleOCR-VL loaded through experimental-v7 llama-server with model+mmproj. First smoke at data/paddleocr_vl_first_smoke_20260717T194332Z/ passed digit OCR at 484.36 t/s and invoice markdown extraction at 489.82 t/s. Narrow receipt QA under a 96-token cap emitted broad OCR tex...
- **L8870** &nbsp;`roles.paddleocr_vl_16_gguf.performance.evidence`
  > - data/paddleocr_vl_receipt_extract_20260717T194415Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/paddleocr_vl_receipt_extract_20260717T194415Z/SHA256SUMS
```


## WITHHELD FILES (2026-08-02)

`summary.json` and `response.json` are **deliberately not committed**. They contain the
OCR output of a real receipt — a third party's company name, business-registration
number, street address, telephone and GST/tax ID. That is another party's data captured
incidentally by an OCR benchmark; it does not belong in a repository regardless of this
repo's visibility.

Recorded hash-and-provenance-only, per MEASUREMENT.md §5 (evidence durability), which
permits this for artifacts that cannot be carried. Hashes are in the adjacent
`*.WITHHELD.sha256` files; the originals remain at
`/mnt/raid0/llm/tmp/paddleocr-vl-receipt-extract-20260717T194415Z/`.

The PII pre-commit hook caught these. It was correct, and this is not a false positive:
the 12-digit run it matched is the receipt's GST ID. The hook's `is_timestamp_or_log_line`
disambiguator WAS separately too narrow — it missed llama-server's single-letter severity
and non-epoch `t_last` counters, producing ~170 genuine false positives across this
migration — and that was fixed in `epyc-root scripts/hooks/pii_precommit.sh` rather than
bypassed. Both things were true at once; only one was a hook defect.
