"""EV-13 review-finding-F1 evaluation suite (clean-room).

Local code-review benchmark: micro-averaged Precision/Recall/F1 of model
review findings against a human-curated golden set. Clean-room per the
2026-06-03 Factory-methodology deep-dive; the unlicensed upstream harness is
NOT vendored. Absolute F1 is internal-only (not leaderboard-comparable).

Public surface:
  * scorer            — deterministic micro-averaged P/R/F1 + Mean-F1/StdDev.
  * harness           — /v1/chat/completions driver (mock/dry-run in tests).
  * assemble_golden_set — normalize Augment-v1 + PR sets to the case schema.
"""

__all__ = ["scorer", "harness", "assemble_golden_set"]
