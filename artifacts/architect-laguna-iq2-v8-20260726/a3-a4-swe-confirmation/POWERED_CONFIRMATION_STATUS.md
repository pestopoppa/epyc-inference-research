# A3 vs A4 powered SWE confirmation status

Prepared 2026-07-26 after the Laguna GPU lane completed. This is a preparation
artifact, not a completed A3 or A4 measurement.

## Required order

1. Run the official SWE-bench gold evaluation against every ID in
   `powered_160_candidate_manifest.json` using the `gold` prediction source.
2. Retain the first 160 IDs in manifest order that pass official gold evaluation.
   Fewer than 160 passes is a hard stop: extend the deterministic candidate
   tranche and repeat validation before any model inference.
3. Materialize oracle prompts only from that accepted ID list using
   `build_powered_swebench_prompts.py`, then run A3 and A4 with the same v8 HIP /
   MI210 / 49,152-context / MTP-4 configuration.
4. Convert SEARCH/REPLACE output and evaluate both arms with the official
   FAIL_TO_PASS harness. Count empty patches as failures over the full accepted
   160-item denominator and use paired statistics only on that fixed slice.

## Current boundary

CPU-heavy gold validation and the later official model-patch scoring are deferred
until the AutoPilot E8 clean-CPU reseed window is released. No such Docker work
has run for this expanded candidate set. The two partial A3 raw responses under
`A3_27B_dense-port18091/` were intentionally aborted before a terminal result
when the 150-plus requirement was clarified; they are not an arm and must not be
scored or resumed.

The completed Laguna 40-item SWE result and 53-item LCB-hard result remain
separate, terminal candidate evidence.

Prompt-builder preparation is complete: eight focused tests pass, Ruff is clean,
and the parameterized implementation reproduces the historical 40-item question
file byte-for-byte at SHA-256
`f82a5191274048f2fdf432df7a0ebf4017ad982b954d6aa075326a1302df1c3c`.
No powered prompt file can be created until official gold acceptance exists.
