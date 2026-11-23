# PREDECLARATION — SPEC-COUPLE-MTP-1 (sealed before any run, 2026-11-23)

Question: on the A3B MTP chain at served depth D=2, how much would routing MTP through the
existing rejection path (q = softmax(logit/T) over the MTP top-10) add over today's
sample-and-compare against the argmax draft?

- Data: capture cap-L01.jsonl sha256 00d810a82ba86172ef5860a3ed4055a3817cf66004b18ca88808beddddd74bb7
  (row ak-dice-capture-a3b-mtp-cpu-20261123-r2, prompts sha 438288dc..., T 0.1/0.3/1.0, captured D=4, truncated to D=2).
  Served T 0.2 is not in the capture; reported as interpolated between 0.1 and 0.3, not measured.
- Script: couple_mtp_replay.py sha256 0f722208c63a315cbd97f593e05ebfe3cbe8ce2fe22938ac095e532c436b9f45
- Arms: V0 = sum_k prod p(argmax chain_j); R = sum_k prod (1 - TV(q_j, p_j)). R is an optimistic
  upper bound (positions j>0 keep the argmax-prefix conditioning).
- Primary: tokens/pass ratio R/V0 per T on holdout prompts 32-63, 95% prompt-cluster bootstrap CI (2000, seed 1123).
- Futility (brief): CI-upper < 1.02 at every T -> stop.
- Box rule: holdout mean per-position gap 1 - TV - p(argmax q) < 1% at every T -> stop;
  >= 2% at any T with >= 20% token share -> file a build box. Between: report, no build box.
- Gate class: futility-only. Calibration half reported but not used for the verdict.
- Every captured round counted; no exclusion.
