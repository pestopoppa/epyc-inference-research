# Greedy-parked spec-decode decisions — served-T offline re-test — PREDECLARATION

Sealed 2026-11-23 BEFORE any number below is computed. Offline only: recorded data, no inference runs.
Served T: role defaults 0.1-0.3, :8083 at 1.0. "Greedy proxy" = DICE T=0.1 (no T=0 capture exists).

Inputs (read-only): DICE replay RESULT.json (research 03c100462, A3B MTP, holdout 32-63), capture
cap-L01.jsonl sha 00d810a8..., DTREE phase row r2 (A/A in band) `ak-dtree-phase-gpu-20261122-r2/out/`,
DTREE-STOCH row `ak-dtree-stoch-27b-gpu-20261121/out/RESULT.json` (measured T=1 tree vs chain),
c10 slope `c10-verify-slope-20261109/out/RESULT.json`, CAFE-1 `artifacts/compute/cafe1-replay-20261101/ANALYSIS.json`,
Q38FN depth `ak-q38fn-cpu-draft-depth-20261121-r2/out/RESULT.json`, hyp 18 seed text.

## Item 1 — DTREE tree-verify park (27B DFlash2, MI210)
Model: tok/s ratio S = tau_ratio / cost_ratio, cost_ratio = ms/round(tree)/ms/round(L).
- cost_ratio per W from phase r2 per-pair ms/round (8 pairs; T12, T16; no RS_REPLAY).
- tau_ratio at served T, two projections (both reported; the verdict uses the MORE favourable one, so a LOSS is robust):
  (P1) transfer: measured 27B greedy tau_ratio (phase r2 per-pair mean_len T/L) x DICE drift
       [V2/V0(T) / V2/V0(0.1)] at matching W (V2 = the implementable verifier per DICE note).
  (P2) absolute: DICE V2/V0(T) at matching W used directly as the 27B tau_ratio.
- CI: 2000 bootstrap draws, resampling phase pairs, DICE ratio drawn normal from its reported CI half-width.
- Verdict per (T in {0.1,0.3,1.0}, W in {12,16}): PROJECTED_WIN iff CI-lower(S) > 1.0084 (GPU floor);
  PROJECTED_LOSS iff CI-upper(S) < 1.0; else PROJECTED_PARITY.
- Reopen check: required tree round ms = 45.59 x tau_ratio (L ms/round of r2); compare with measured T12/T16 ms.
- Cross-check (not a verdict input): DTREE-STOCH measured T=1 tok/s ratios 0.963 (auto) / 0.933 (T12).
  If the projection says WIN while the measured T=1 row lost, the measurement wins and the verdict is LOSS.

## Item 2 — akm-c10 hybrid tree port
Break-even: S = tau_ratio / (1 + s*dM/t_base) >= 1.05, s = 2.39 ms/row [2.26, 2.51], t_base = c10 abs breadth M8
43.80 ms, dM = W - 8 for W in {12, 16} (rows beyond the 8-row chain). Required tau_ratio = 1.05 (1 + s dM / t_base).
Projected margin = DICE V2/V0(T, W) - required. Verdict: CLEARS iff DICE CI-lower > required at s CI-upper;
FAILS iff DICE CI-upper < required at s CI-lower; else MARGINAL. Caveat: DICE is A3B MTP D=4 proxy.

## Item 3 — CAFE-1 p-min
Rescore throughput alone with the box's own clauses: aggregate gain >= 1.0 + floor AND every class >= -(1.0 + class
floor) -> THROUGHPUT_PASS else THROUGHPUT_FAIL. CAFE-1 ran T=0, so served-T throughput is UNMEASURED; offline
served-T projection from DICE capture: p-min truncation of the MTP chain (stop drafting after depth j when
q_j(chain_j) < p-min), expected tokens/pass and verified rows per round at each T, priced with the CAFE-2 r2 verify
cost curve if its fit is on disk (else rows only). PROJECTED gain = tok/pass ratio / cost ratio vs p-min 0
(note: served :8070 already runs p-min 0.5 in the Q38FN row — reported as a fact if found, not assumed).
Report what SPEC-PIT.1 would need (distribution gate) and the minimal served-T row.

## Item 4 — Q38FN depth and hyp 18
(a) Per-depth conditional acceptance a_j(T) = E[p_j(chain_j) | prefix accepted] from the capture (all rounds, all
prompts), j = 1..4, at each T. Depth decision moves iff the projected Q38FN n6/n4 ratio at served T
(greedy measured +3.6% with depth-5..6 increment scaled by [a_4(T)/a_4(0.1)]^k, k = 1,2 for depths 5,6) crosses
1.0415 (it was below at greedy; a move UP needs a_4(T) > a_4(0.1)). Pre-stated expectation: no move.
(b) Hyp 18: from realized per-round acceptance in the capture, per request (prompt x T): request mean acceptance and
min rolling acceptance over 64-token windows (capture requests are 256 tokens, so the 512-token window cannot be
evaluated directly; 64 is a stricter proxy). Threshold moves iff ANY healthy request (mean acceptance >= 0.5) has a
rolling window < 0.2 at served T (false-trip risk), or the share of requests with mean < 0.5 grows by > 10 pp from
T 0.1 to 1.0.
