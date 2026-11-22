# AK-HACKSCAN-SOLX: covered vs missing exploit classes (2026-11-22)

Audit of `execution/reward_hack_scan.py` (RVP-C6-9) against the SOL-ExecBench exploit classes and
the amd/skills wave64 hazard sweep (intake-2732, intake-2745). Zero GPU, source-only. New
detectors live in `execution/reward_hack_scan_solx.py`; each has a mutant fixture in
`execution/test_reward_hack_scan_solx.py`. Classification: OBSERVATION.

Result: 9 classes audited, 0 fully covered before, 4 partial, 5 missing. All 9 now have a
detector. Champion HIP tree (`ggml/src/ggml-cuda`, `ggml/src/ggml-hip` at `74c18e2d8`, C18)
replayed as all-added diff: **0 hits** across all 11 detectors (`#if 0` blocks masked as dead code;
ggml's `cuda*`->`hip*` shim defines and explicit-width `__shfl_xor_sync(0xFFFFFFFF,..,32)` reductions
are deliberately not flagged).

| exploit class | source | pre-existing detector | status | added detector |
|---|---|---|---|---|
| precision downgrade with matching dtypes | SOL-ExecBench | - | MISSING | `precision_downgrade` |
| timer tamper | SOL-ExecBench | instrument_frame_edit (bench files only) | PARTIAL | `timer_tamper + timer_define_tamper` |
| cached output across timed iterations | SOL-ExecBench | pointer_memoization, content_specialization (keyed caches only) | PARTIAL | `cached_output_across_iterations` |
| work outside the timed region | SOL-ExecBench | async_escape, stream_creation | PARTIAL | `work_outside_timed_region + removed_sync` |
| 32-bit ballot/mask destinations | amd/skills wave64 | - | MISSING | `wave64_32bit_ballot` |
| 1<< on 64-bit masks | amd/skills wave64 | - | MISSING | `wave64_int_shift_mask` |
| logical 32-lane warps calling wave-wide collectives | amd/skills wave64 | - | MISSING | `logical_32lane_collective` |
| smid-indexed pools | amd/skills wave64 | - | MISSING | `smid_indexed_pool` |
| size-threshold selectors without an above-threshold case | SOL-ExecBench | structured_short_circuit (exact-size equality only) | PARTIAL | `size_threshold_selector` |

Limits: regex/line-window heuristics over added lines (not a compiler). `removed_sync` only sees a
sync call deleted from a file with no replacement sync added in the same diff. The detectors are
NOT wired into `loop/integrity.py`'s refusal list; wiring is a separate decision because it
widens what the integrity gate refuses.
