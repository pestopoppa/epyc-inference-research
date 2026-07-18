# GLM52 C-CRAB Accept-Control Machine Recommendations

Source packet: `glm52_ccrab_accept_control_n24_audit_packet_20260718.json`

This artifact is a machine-review recommendation only. It does not modify the packet's authoritative `signoff` fields, does not claim operator signoff, and is not a decision-grade signoff output.

## Counts

- Rows reviewed: 24
- `hard_accept_candidate`: 24
- `reject_or_ambiguous_candidate`: 0
- Candidate patches truncated: 0
- Candidate patches with redacted long digit runs: 5
- Tasks with redacted long digit runs: 0

## Report-Mode Helper Check

Helper was run in report mode without `--allow-unreviewed` and without output paths. It reported 24 unreviewed rows, 0 hard accepts, 0 rejected/ambiguous rows, and `decision_grade: false`.

## Format Concerns

Rows with `candidate_redacted_long_digit_runs: true`:

- `nearmiss-v1:c-crab:08cafcb6483d8389`
- `nearmiss-v1:c-crab:09584d0209952576`
- `nearmiss-v1:c-crab:10070430d41b73e9`
- `nearmiss-v1:c-crab:1600ca8239e2f6e0`
- `nearmiss-v1:c-crab:200003ca11cb7699`

No source rows were edited, and no official `*_signoff_*.json` outputs were written.
