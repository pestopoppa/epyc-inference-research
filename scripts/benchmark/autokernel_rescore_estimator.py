#!/usr/bin/env python3
"""Re-score every historical AutoKernel two-arm screen under a like-for-like estimator.

WHY THIS EXISTS
---------------
`run_autokernel_gpu_discovery.py` computed the anchor centre as the **mean** of the
anchor samples and then reported the **median** of the per-sample candidate effects
against it:

    center  = sum(anchor_samples) / len(anchor_samples)     # mean
    effects = [(v - center) / center for v in candidate_samples]
    median_relative = median(effects)                        # median

Mixed estimators on the two arms. The anchor arm reliably carries a cold-start low
outlier, so `median(anchor) > mean(anchor)`, and comparing a candidate MEDIAN against
an anchor MEAN manufactures apparent improvement on every run.

This tool recomputes each screen three ways and prints the difference. It is a
measurement, not a fix: it makes the size of the historical bias auditable so the
corrected estimator can be adopted with the damage quantified.

    reported     median(effects vs mean(anchor))     -- what the loop recorded
    median/median  median(cand) vs median(anchor)    -- like-for-like, the correction
    mean/mean      mean(cand)   vs mean(anchor)      -- like-for-like cross-check

Usage:
    python3 scripts/benchmark/autokernel_rescore_estimator.py \
        --deployments-root /mnt/raid0/llm/autokernel/deployments \
        --out artifacts/autokernel-rescore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median, mean
import sys

NOMINATION_THRESHOLD = 0.03


def load_pair(result_path: Path) -> dict | None:
    """One screen: the result and its sibling baseline bank."""
    bank_path = result_path.parent / "baseline-bank.json"
    if not bank_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text())
        bank = json.loads(bank_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    cand = [float(v) for v in result.get("candidate_samples") or []]
    anchor = [float(v) for v in bank.get("anchor_samples") or []]
    if not cand or not anchor:
        return None

    contract = ((bank.get("frame") or {}).get("metric_contract") or {}).get("schema", "")

    # Replicate the producer's centre rule exactly (run_autokernel_gpu_discovery.py:3216):
    # the pair-max contract centres on the FIRST anchor run's own reported metric;
    # everything else centres on the MEAN of the anchor samples.
    anchor_runs = bank.get("anchor_runs") or []
    if contract == "epyc.autokernel.serialized_pair_max_metric.v1" and anchor_runs:
        produced_center = float(anchor_runs[0]["metric"])
    else:
        produced_center = mean(anchor)

    return {
        "path": str(result_path),
        "campaign": result.get("campaign_id", ""),
        "contract": contract or "(unset)",
        "reported": result.get("median_relative"),
        "reported_center": result.get("baseline_center"),
        "produced_center": produced_center,
        "anchor": anchor,
        "candidate": cand,
    }


def rescore(row: dict) -> dict:
    a, c = row["anchor"], row["candidate"]
    a_med, a_mean = median(a), mean(a)
    c_med, c_mean = median(c), mean(c)

    out = dict(row)
    out["anchor_median"] = a_med
    out["anchor_mean"] = a_mean
    out["candidate_median"] = c_med
    # The estimator the loop actually used, recomputed from raw samples through the
    # producer's own centre rule, so the stored value is verified rather than trusted.
    centre = row["produced_center"]
    out["recomputed_reported"] = median([(v - centre) / centre for v in c])
    out["median_over_median"] = (c_med / a_med) - 1.0
    out["mean_over_mean"] = (c_mean / a_mean) - 1.0
    # How much apparent improvement the mismatch alone contributes.
    out["anchor_median_vs_mean"] = (a_med / a_mean) - 1.0
    out["bias_pp"] = (out["recomputed_reported"] - out["median_over_median"]) * 100.0
    out["sign_flip"] = (out["recomputed_reported"] > 0) != (out["median_over_median"] > 0)
    out["nominated_reported"] = out["recomputed_reported"] > NOMINATION_THRESHOLD
    out["nominated_corrected"] = out["median_over_median"] > NOMINATION_THRESHOLD
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deployments-root", type=Path,
                    default=Path("/mnt/raid0/llm/autokernel/deployments"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.deployments_root.is_dir():
        print(f"REFUSED: no such deployments root: {args.deployments_root}", file=sys.stderr)
        return 2

    rows = []
    for result_path in sorted(args.deployments_root.rglob("result.json")):
        pair = load_pair(result_path)
        if pair is not None:
            rows.append(rescore(pair))

    if not rows:
        print("REFUSED: found no two-arm screens with both anchor and candidate samples",
              file=sys.stderr)
        return 3

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rescore.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{len(rows)} two-arm screens re-scored\n")
    hdr = (f"{'screen':<34} {'reported':>10} {'med/med':>10} {'mean/mean':>10} "
           f"{'bias pp':>9} {'flip':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tag = f"{r['campaign'][:22]}/{Path(r['path']).parent.name}"[:33]
        print(f"{tag:<34} {r['recomputed_reported']*100:>9.3f}% "
              f"{r['median_over_median']*100:>9.3f}% {r['mean_over_mean']*100:>9.3f}% "
              f"{r['bias_pp']:>+9.3f} {'YES' if r['sign_flip'] else '':>5}")

    n = len(rows)
    print("-" * len(hdr))
    print(f"{'MEAN across screens':<34} "
          f"{mean(r['recomputed_reported'] for r in rows)*100:>9.3f}% "
          f"{mean(r['median_over_median'] for r in rows)*100:>9.3f}% "
          f"{mean(r['mean_over_mean'] for r in rows)*100:>9.3f}% "
          f"{mean(r['bias_pp'] for r in rows):>+9.3f}")
    print(f"{'MEDIAN across screens':<34} "
          f"{median([r['recomputed_reported'] for r in rows])*100:>9.3f}% "
          f"{median([r['median_over_median'] for r in rows])*100:>9.3f}% "
          f"{median([r['mean_over_mean'] for r in rows])*100:>9.3f}%")

    flips = sum(1 for r in rows if r["sign_flip"])
    nom_rep = sum(1 for r in rows if r["nominated_reported"])
    nom_cor = sum(1 for r in rows if r["nominated_corrected"])
    drift = mean(r["anchor_median_vs_mean"] for r in rows) * 100.0

    print(f"\nsign flips under correction:            {flips} of {n}")
    print(f"crossed the {NOMINATION_THRESHOLD:.0%} nomination threshold: "
          f"{nom_rep} reported -> {nom_cor} corrected")
    print(f"mean anchor median-vs-mean gap:         {drift:+.3f}%  "
          f"(the mechanism: a cold-start low outlier in the anchor arm)")
    print(f"\nwrote {args.out / 'rescore.json'}")

    # Verification of the stored values: the recomputed reported figure must match
    # what the loop actually wrote, or this tool is reading the wrong samples.
    mismatched = [r for r in rows
                  if r["reported"] is not None
                  and abs(r["recomputed_reported"] - float(r["reported"])) > 1e-9]
    if mismatched:
        print(f"\nWARNING: {len(mismatched)} screen(s) whose stored median_relative does not "
              f"reproduce from their raw samples — inspect before trusting this table:")
        for r in mismatched[:5]:
            print(f"  {r['path']}\n    stored={r['reported']}  recomputed={r['recomputed_reported']}")
    else:
        print("\nall stored median_relative values reproduce exactly from raw samples "
              "(the reported column is verified, not trusted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
