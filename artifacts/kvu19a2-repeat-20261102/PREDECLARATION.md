# KVU-19a-2 repeated exact-input diagnostic — declared before execution

Compare the exact retained baseline and candidate HIP shared libraries, not a
claim about an unverified source-to-binary build. Execute fresh-process pairs
for FA_CASE 78 (D=576/512) and 80 (D=512/512), five processes per arm and
case, alternating arm order by repeat. The model-free harness generates its
fixed-seed native Q, K/V and mask and saves the actual submitted host bytes.
Require the same input file set and SHA-256 per case across all ten processes;
different/missing bytes make that case INVALID. Record output bytes per process,
return status, source and executable hashes, native logs, same-process ggml
mapping samples and during-run KFD/VRAM samples; absent sampling is UNKNOWN.

For each case and arm, if all five original outputs are captured, input hashes
match and every run has own-PID KFD plus positive VRAM and expected binary and
mapped HIP, compare the full output SHA-256s. Identical = observed bit identity
across these five processes; two or more distinct digests = observed output
variation under those controls. An incomplete, unresidenced or identity-mismatched
run makes that arm UNKNOWN, never a no-variation verdict. Cross-arm output
differences are descriptive only. No tolerance, performance, source-causation,
quality, production, universal determinism or promotion claim follows.

Expected HIP SHA-256: baseline
`1970ea22b7045312b904604df4d50d06ace6f1c8fd8d493925807905462c07c1`;
candidate `ba4c98dacbb3c1b90f360131cd1e50450f022b95fc9c02e833e857127a5a6e8c`.
Harness source SHA-256 `b9f352a91ea975c682cbe78c2f4dbf9035c6a40e309b84035bfbdaf49b420091`;
harness executable SHA-256 `986c0137e499a9fe6a48910846c72973407aa5b56e30fac79978e6dfdf7de837`.
The archive digest is captured, but the previous compiler-visible source was
same-UID mutable, so its source/build relationship is **unproven**, even when
the binary content is pinned. Immutable build custody remains a separate gate.
Source-side belief observations cover only actual F32 output NaN element
counts, never inferred repeatability, GPU residency or a missing run.
