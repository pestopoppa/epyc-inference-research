# HIP thread-safety post-atomic differential

This evidence answers the external-audit request to rerun `test-thread-safety`
after candidate commit `6c44557bf`.

- Candidate: `67a433bf45a8a091d83b4ea0b32ff0735fd51800`
- Baseline: frozen v7 `6ad45fa3ff6718c07c000061dbc6e29c1771f6e3`
- Valid paired artifact:
  `run-20260725T135837Z-67a433bf4-paired/`
- Result: production and candidate both exited `139` in all three alternating
  repetitions under the same fixture and command.
- Classification: inherited HIP baseline failure, not a candidate regression.
  This is not a passing `test-thread-safety` claim.
- Postflight: no KFD clients remained.

The earlier `run-20260725T135517Z-67a433bf4/` attempt is invalid because the
deployed v7 HIP directory does not contain test binaries. Its production arms
therefore exited `127`; `INVALID.txt` records the reason. A detached v7 scratch
worktree was built for the valid paired run and removed after capture.
