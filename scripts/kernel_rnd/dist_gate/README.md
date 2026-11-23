# dist_gate - speculative-decoding distribution and identity gates

`spec_dist_gate.py` (stdlib only, no inference):

- **SPEC-PIT.1** `gate(arm, ref=None, select=..., match_keys=...)` -> PASS / FAIL / INCONCLUSIVE.
  Primary test: KS of randomized PIT (`F(x-) + U*m(x)`) over the arm's post-sampler target probs
  (`post_sampler_probs`: top_k -> top_p -> min_p -> temperature). FAIL on KS p <= 1e-5 or any token
  out of support. Secondary: rank (top-20 + other) chi-square vs the reference arm, SELECTION-MATCHED
  (`select` = the arm's selection rule applied to the reference, or `match_keys` = co-captured positions)
  and tested by request-level permutation (2000 perms); FAIL on p < 1e-3. INCONCLUSIVE when < 2000
  tokens, < 8 requests, or the reference arm fails its own KS. Thresholds: `THRESHOLDS`.
- **SPEC-ID-GATE-1** `margin_class`, `classify_divergence`, `paired_count_test`. Protocol:
  epyc-root `docs/guides/agent-workflows/spec-identity-gate-protocol.md`.

Tests: `python3 -m pytest scripts/kernel_rnd/dist_gate -q` (16 tests, ~5 s CPU). They cover exact samplers
passing under three sampler configs, a sharpened sampler failing, and the selection-bias case (an exact
arm that only emits at confident positions FAILS the unmatched chi-square and PASSES once matched).
Origin of the matching/permutation requirement: `data/pmin-dist-investigation-20261123/`.
