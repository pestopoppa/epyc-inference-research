# EXL3-2 dispatch timing, 2026-09-26

The remaining timing comparison is complete. The canonical region-lock runner
acquired q3, the harness ran on CPU95 with one OpenBLAS thread, and q3 was
released afterward. Its role lock is confirmed unheld and its attribution cleared;
subsequent status shows `autokernel-cpu` has reacquired q3. No inference ran.

These are **CANDIDATE observations**, instrument class `bench`, not protocol-eligible
performance or promotion claims. Lower batch latency is better. Each observation
averages four calls; each arm has five observations after a warmup. The payload is
eight synthetic native-layout 128x128 MUL1 matrices, one at each K1–K8, processing
one input row. The operator boundary includes validation, allocation, transforms,
decode and dot product. It is cache-resident and single-threaded; arms run in fixed
order, so these small differences do not establish a dispatch winner.

| ISA | Runtime K median (µs/batch) | Grouped K median (µs/batch) |
|---|---:|---:|
| Scalar | 3136.96 | 2949.34 |
| AVX512BW | 1268.65 | 1290.49 |
| VNNI | 1260.11 | 1264.28 |
| VBMI | 1265.40 | 1324.37 |

The largest absolute median difference is 5.98% in this sample. No dispatch default or many-row
optimization was selected from these observations. The native measurement rows
store arithmetic means and the complete raw vectors; the medians above are
rederived from those vectors.

The same final run passed **719,096 correctness checks**. Eight measurement rows
and one verifier row passed the frozen writer's reopen/hash validation and root's
strict adapter plus shared `claim_tuple.grade()` call. A lack of protocol citation
keeps these observations from becoming decision-gating performance evidence.

## Exact commands and evidence

The build command was:

```bash
EXL3_BUILD_DIR=/tmp/exl3-cpu-timing-build-hex \
EXL3_BLAS_LIBRARY=/mnt/raid0/llm/epyc-inference-research/.venv/lib/python3.13/site-packages/scipy.libs/libscipy_openblas-6cdc3b4a.so \
experiments/exl3_cpu/run_tests.sh --build-only
```

The exact region acquisition and writer-wrapped execution command is committed in
[`command.sh`](../../artifacts/exl3-cpu-dispatch-20260926/command.sh). It uses
`region-lock run --cpu-list 95 --timeout-s 5 --role exl3-cpu-probe`, then
`taskset -c 95 .../run_evidence.py --bench` with explicit writer SHA, binary, BLAS,
canonical bindings and live region-lock path. The wrapper verifies that its parent
owns the held q3 lock, captures that claim, and confirms it has not changed before
writing rows.

Evidence directory:
`artifacts/exl3-cpu-dispatch-20260926/run-final/`.
[Summary and all receipt digests](../../artifacts/exl3-cpu-dispatch-20260926/summary.json),
[projection results](../../artifacts/exl3-cpu-dispatch-20260926/projection-validation.json),
[acquisition log](../../artifacts/exl3-cpu-dispatch-20260926/region-final.stderr), and
[release status](../../artifacts/exl3-cpu-dispatch-20260926/released.txt) are committed.

Final verifier row:
`84bf196a5e32601df08511abc06d4838b9e51e331394e587c2c813cd0b66a32f`.
The frozen writer SHA-256 remains
`8caeb33dbb12986fadc385afe25d22bd791b036253c736f9527e67a55f85e268`.
The timed window contains 28 owned-process samples. Every timing row names the
same actual payload SHA-256:
`28217be82281df2a89760fdcad447b0a64aaca0fa0f0d9f9499b9c8ef7d6d355`.

## Narrow instrumentation correction

The original wrapper would have labeled synthetic timing with the preceding real
correctness fixtures' artifact identity. Timing now captures its actual matrices
and activation buffer in `workload.bin`, and each measurement binds that payload.
The paired comparator is the opposite grouping setting on the same ISA and
payload. Superseded captures (the inherited comparator label, then decimal clock values
rejected by the PII hook) were removed. This final capture uses hexadecimal clock
values so the unmodified hook accepts the raw sealed output.

`workload.bin` is a test-workload capture, not a new EXL3 artifact format. On this
little-endian x86 host it contains five u32 header values (magic `0x31425845`,
expert count 8, input 128, output 128, input stride 135), then for each expert four
u32 values (K, codebook enum, layout enum, trellis u16 count), its trellis words,
128 FP32 suh values, and 128 FP32 svh values; finally 135 FP32 activation values.
Routes are experts 0–7 in order, all reading input row 0; biases are empty.

The instrumentation change uses a separate worktree, preserving the earlier
normal and sanitizer verification read sets. Kernel/operator numerics did not
change. GitNexus impact remains unavailable due the pre-existing WAL mismatch;
this edit affects only the experimental test harness and receipt wrapper.
