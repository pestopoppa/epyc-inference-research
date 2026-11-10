# Hypothesis-portfolio re-seal review — 2026-11-10 (C6-C10-REPIN)

Scope: GPU discovery factory re-pinned to champion C10 `7f1bd044b` from row
`c10-c6-dispatch-reprofile-20261110-r2` (rc=0, verdict PASS: C10 0 new fails vs v10).
Trace: `/mnt/raid0/llm/autokernel/diagnostics/c10-rocprofv3-anchor-tg128-20261110/raw/c10_kernel_trace.csv`
(sha256 `2d950e3c…eac7f`, 60,054 rows; copied byte-identical from the row's `out/trace-c10/`).

## Portfolio changes (2 rows)

| Hypothesis | Change | Why |
|---|---|---|
| `akh-v2-q8-quantizer-new-mechanism` | current-frame signatures 2 -> 3: `quantize_q8_1_1d<false>` 15609@1024, `<false>` 1548@5120, `<true>` 1548@5120, workgroup 64 (was 256). total_calls 18705 unchanged | C10 splits the 1d quantizer by template; v10 already dispatched `quantize_q8_1_1d` at wg 64 |
| `akh-v2-fa-gqa7-pair-tail` | eligible -> false, blocking condition recorded; historical signature kept for provenance | v10 and C10 decode both dispatch `flash_attn_ext_vec<64,1,f16,f16>`; zero `flash_attn_tile` dispatches |

Fingerprints/mechanism blocks untouched. `validate --verify-evidence`: 21 hypotheses, 23 DNR, 3 eligible.

## New pins
- file `13c560267ae1b67519e06657cc64770221a19cfd4f87aeebf2b76913d2c64556`
- semantic `936ae176e3a43597046a472bced3c8efec9943472bd064f00ec456bb080f36f2`
- contract unchanged (`96f20773…`)

Consequence: native C6 S1+S2 capacity 20 -> 16 (GQA7's two candidates left the eligible set).
