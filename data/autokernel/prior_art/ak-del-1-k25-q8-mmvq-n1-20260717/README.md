# AK-DEL-1 K25 Q8 MMVQ profile input

This directory preserves one completed, real MI210 `rocprofv2` capture for the
AutoKernel AK-DEL-1 scope-reduction gate. It is an offline input, not a new run.

- Original capture: `/mnt/raid0/llm/tmp/k25-mmvq-verify-width-20260717-a/rocprof_n1/pmc_1/results_1786802.csv`
- Captured: 2026-07-17
- Experimental llama.cpp commit: `3dee86a5a`
- Workload: `test-backend-ops`, ROCm0, Q8_0 x F32 `MUL_MAT`, `m=16`, `n=1`, `k=256`
- Profile counter: `GRBM_GUI_ACTIVE`; AK-DEL-1 uses only the start/end dispatch timestamps
- Original raw SHA-256: `c34867879933ec0f1515b01565c585da436e0d8b27ed9de4df8a06ac9e4578a8`
- Normalized SHA-256: `3ad4800931477e44e1d60529e02771e6d03a5a0d89df4d4baa21779603d9ada8`

`rocprof_normalized.csv` is a deterministic derivative: it retains only
`Dispatch_ID`, `Kernel_Name`, `Start_Timestamp`, and `End_Timestamp`, then
subtracts the minimum raw start timestamp from every timestamp. Dispatch order,
gaps, names, and every duration are unchanged. This
keeps the replay input compact and avoids storing process identifiers or
machine-absolute timestamps; the original content hash remains above.

The capture is observation-grade historical evidence. Its role here is only to
exercise the four-way prior-art classification on real profiler findings; it
does not authorize a performance or production claim.

`scope_reduction_report.json` is the deterministic AK-DEL-1 result over that
normalized capture. The first
seed-catalogue pass exposed both a false generic-`mul` match and missing rows for
the three existing paths in this capture. After correcting those catalogue
defects and rebinding the scan to frozen v9 (`0db32c06e3e550065b78311a6031ef3dd2c4f27c`),
all three admitted families land in bucket (a). Existing/config/dispatch work
therefore dominates this bounded corpus, so the report recommends catalogue
expansion before building a novel-kernel proposal generator. This scope verdict
is corpus-bounded; it does not claim that every future profile will have the
same split.
