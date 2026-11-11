# race_eqv: GPU race / equivalence gate for seeded-defect kernel builds

Harness for the `race-eqv-gpu` rows (RVP-RACE-1, RVP-EQV-1), archived from the sealed row dir
`race-eqv-gpu-r8-20261111` so a tmp cleanup cannot lose it. Row launcher: `launch.sh`; config: `spec.json`.

## What the row does
`job.py` runs `test-backend-ops` on the MI210 for a BASELINE build, a BASELINE-REPEAT (determinism),
per-suite kernel-instance coverage, controls, then each seeded-defect mutant build. A mutant passes
only if it adds no new failing case key versus the BASELINE run of the same suite and seed
(`gpu_signals/newfail.py`, ANSI stripped before parsing). `gpuproof.py` samples KFD/VRAM residency
during the run (INVALID_NO_RESIDENCY if the GPU was not used). `check_complete.py` is the
filler's completion probe. Fixtures in `gpu_signals/fixtures/`. Paths to builds/outputs in `job.py`
and `launch.sh` are host-specific tmp paths; `D` (harness dir) is now derived from `__file__`.

## GDN+VT negate-transform exemption (@t4)
Value-transform (VT) runs apply identity, x3, x0p01, negate (`N_TRANSFORMS = 4`); the loop stops at the
first failing transform. For GATED_DELTA_NET+VT, negating the gate makes the recurrence diverge
(exp(+|g|) growth): ROCm overflows to NaN where the CPU reference is finite or also overflows. That is
outside the op's domain. Such failures at the 4th transform (after t1..t3 passed) do not count
against the baseline pass floor (`baseline_usable(..., tail_ok=True)`) and stay in `fail_counts`
tagged `@t4`. A mutant failing earlier (@t1..@t3) is a new key and is still caught.

## Fail-line parser change (r8)
`parse_log` now also parses the authoritative "Failing tests:" list that test-backend-ops prints at the
end (one indented case per line). Needed because VT failures print the case on one line and
`AK_VALUE_V1 ... completed=N | test failed` on another with backend debug output between, so the
`ERR =` and line-anchored `FAIL` forms missed them (r7: 59 GATED_DELTA_NET failures, 0 parsed).

## Tests (pure Python, from repo root)
    python3 -m pytest scripts/kernel_rnd/race_eqv -q
`test_rescore_r3.py` re-scores the r3 log through the gate and skips if that tmp log is gone.
