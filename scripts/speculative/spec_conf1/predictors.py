#!/usr/bin/env python3
"""SPEC-CONF.1 step-1 predictor logging and AUC (offline, no model, no timing claim).

Trace record (one JSON object per verify round per slot, JSONL):
  slot, round            ids
  n_drafted, n_accepted  ints, 0 <= n_accepted <= n_drafted
  top1_prob[i]           selector top-1 probability at draft position i (len n_drafted)
  cand_probs[i]          TARGET probabilities of the drafted candidates at position i
                         (chain: one element, the drafted token; tree: all siblings),
                         present for every VERIFIED position, i.e. i <= first rejection
                         (positions after the first rejection are never verified)

Predictors per draft position i (label = position accepted, i < n_accepted):
  position        prior: -i
  ema             CAFE-3 per-position acceptance EMA (state carried across rounds, by position)
  top1            top1_prob[i]
  cumprod         product of top1_prob[0..i]
  frontier_gap    1 - sum(cand_probs[i])   (intake-2759#record, Raptor eq. 4; lower = more likely
                  accepted; the AUC uses -gap). Defined only at verified positions. The round-level
                  field `frontier_gap_first_reject` is the gap AT the first rejected position
                  (None when the whole block was accepted).
"""
import json
import sys

PREDICTORS = ("position", "ema", "top1", "cumprod", "frontier_gap")


def frontier_gap(cands):
    """1 minus the summed target probability of the drafted candidates, clipped to [0, 1]."""
    return min(1.0, max(0.0, 1.0 - sum(cands)))


def round_fields(rec):
    """Return the logger output for one round: per-position predictor lists + round-level gap."""
    n, acc = rec["n_drafted"], rec["n_accepted"]
    if not 0 <= acc <= n:
        raise ValueError("n_accepted out of range")
    if len(rec["top1_prob"]) != n:
        raise ValueError("top1_prob length != n_drafted")
    verified = min(n, acc + 1)  # accepted positions plus the first rejected one
    cands = rec.get("cand_probs", [])
    if len(cands) < verified:
        raise ValueError("cand_probs must cover every verified position")
    gaps = [frontier_gap(cands[i]) if i < verified else None for i in range(n)]
    cum, run = [], 1.0
    for p in rec["top1_prob"]:
        run *= p
        cum.append(run)
    return {
        "top1": list(rec["top1_prob"]),
        "cumprod": cum,
        "frontier_gap": gaps,
        "frontier_gap_first_reject": gaps[acc] if acc < n else None,
    }


def auc(scores, labels):
    """Rank AUC with tie handling (Mann-Whitney); None if one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def build_table(records, ema_alpha=0.2):
    """AUC per predictor over all positions (frontier_gap: verified positions only)."""
    rows = {k: ([], []) for k in PREDICTORS}
    ema = {}
    for rec in records:
        f = round_fields(rec)
        for i in range(rec["n_drafted"]):
            y = 1 if i < rec["n_accepted"] else 0
            rows["position"][0].append(-i)
            rows["position"][1].append(y)
            rows["ema"][0].append(ema.get(i, 0.5))
            rows["ema"][1].append(y)
            rows["top1"][0].append(f["top1"][i])
            rows["top1"][1].append(y)
            rows["cumprod"][0].append(f["cumprod"][i])
            rows["cumprod"][1].append(y)
            if f["frontier_gap"][i] is not None:
                rows["frontier_gap"][0].append(-f["frontier_gap"][i])
                rows["frontier_gap"][1].append(y)
        for i in range(min(rec["n_drafted"], rec["n_accepted"] + 1)):  # EMA updates on verified positions
            y = 1.0 if i < rec["n_accepted"] else 0.0
            ema[i] = (1 - ema_alpha) * ema.get(i, 0.5) + ema_alpha * y
    return {k: {"auc": auc(*rows[k]), "n": len(rows[k][1])} for k in PREDICTORS}


def main(argv):
    recs = [json.loads(line) for line in open(argv[1]) if line.strip()]
    print(json.dumps(build_table(recs), indent=1))


if __name__ == "__main__":
    main(sys.argv)
