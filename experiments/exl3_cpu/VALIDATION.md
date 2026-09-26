# EXL3-2 validation, 2026-09-26

Experimental CPU operators only; instrument class `bench`, category `CANDIDATE`.
No model inference, serving change, or performance/promotion claim.

The final normal prospective run used the frozen shared evidence writer
`8caeb33dbb12986fadc385afe25d22bd791b036253c736f9527e67a55f85e268`, reopened all
canonical real artifact payloads and fixture references, and passed **719,096
checks** on scalar, AVX512BW, VNNI, VBMI. Native row:
`f4eab798b161b6ac568a0b0bedbf888d022234f829b3f5a18a47851c5a997b6b` in
`/tmp/exl3-cpu-verifier-normal-v3/native/`; binary retained at
`/tmp/exl3-cpu-final-v2/test_cpu`. The exact compiler, binary, OpenBLAS library,
hardware, checker read set and during-process samples are in that run directory.
These are native experimental verifier records, not protocol-eligible performance
attestations. The final **ASan+UBSan** prospective rerun passed the same 719,096
checks: native row `b0b59f832297a197929290cd6410fb979250f8329e696f8b117cdd07d47531a2`
in `/tmp/exl3-cpu-verifier-asan-v3/native/`, binary
`/tmp/exl3-cpu-final-asan-v2/test_cpu`. Both binaries postdate their source files;
normal disassembly contains `vpmulld`, `vpdpbusd`, and `vpermb`.

The v1 prospective records predate the final MCG source-receipt revision and are
superseded by the later runs. The final v3 reruns also proved the bindings match
canonical research commit `bc9ac47b8e1cf79db3592553ef5e1ecc99cb0872` byte-for-byte. Earlier diagnostics and superseded records are not
used to close the final acceptance proposition.

| EXL3-2 clause | Implemented / checked scope |
|---|---|
| Scalar dense and indexed experts | Fused FP32 scalar path, per-expert scales/bias, mixed codebooks and K, duplicate routing, validated input rows |
| MUL1 AVX512BW/VNNI/VBMI | Vector window extraction, byte sums/BW integer products, VNNI dpbusd, VBMI activation-byte permutation; exact match to scalar Q8 proposition |
| Independent MCG vector path | Vector uint32 multiply/mask/xor, half conversion/add/round, FP32 vector dot; exact match to scalar fused proposition |
| Native and band-contiguous | Both layouts tested; lossless round trip; canonical fixture hash preserved through reconstruction; derived cache evidence requires canonical binding |
| ISA / K / rows / tails / padding / guards | Four executed ISA tiers, K1-K8, rows1-4, 1/17/128/131 input and 1/128/129/149 output extents, padded128/256, nontrivial strides, unchanged filled guards |
| Real fixtures cannot skip | Required real MUL1 K3/K4 and MCG K4, immutable provenance, SHA checks, independent raw/canonical golden, missing/corrupt/truncated failure |
| Grouped K versus runtime K | Both dispatch implementations and exact output comparison; timed comparison prepared but not run (region ownership below) |
| Canonical reconstruction + provider prefill | Stage-rounded normalized H128 ascending FP32 FMA; real reconstructed matrices bit-exact; seven-row existing OpenBLAS provider checked against deterministic output oracle |
| Avoid unprofiled many-row optimization | More-than-four-row fused requests rejected; prefill delegates to existing GEMM, no packed many-row kernel/cache |

The fused operator's fixed-input real-fixture relative-L2 observations against
materialized weight output were 0.000457453 (MUL1 K3), 0.000432122 (MUL1 K4), and
0.000424603 (MCG K4), within the predeclared 0.003 fixture envelope. The explicitly
approximate Q8 MUL1 path observed 0.00839703 and 0.0085257, within 0.025. Each is one
fixed operator input, not a model quality result or universal error bound.

Timing admission was attempted with `region-lock run --cpu-list 95 --timeout-s 1`
on 2026-09-26. It returned **75**, reporting q3 held by `autokernel-cpu`. The exact
refusal is retained in `/tmp/exl3-cpu-region-attempt.stderr`. No microbenchmark
executed. The specific event needed for timed comparison is release of that CPU
region by its owning session. Production/other-session processes were untouched.
