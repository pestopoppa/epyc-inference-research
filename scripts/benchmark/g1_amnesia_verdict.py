#!/usr/bin/env python3
"""G1 — gate verdict computation from persisted summaries.

Gate (handoff multiscreen-attention-evaluation.md, 2026-08-22 spec, verbatim):
  >= 10 percentage points below the 4K control at any of 32K/64K/128K,
  with the paired 95% bootstrap CI over needle placements excluding zero,
  and surviving the f16-KV control.

Survival at f16: the drop at the f16-run length must also be >= 10pp below the
f16 4K control with CI excluding zero (i.e., not a KV-quant artifact).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GATE_LENGTHS = (32768, 65536, 131072)
ALL_TABLE_LENGTHS = ("4096", "32768", "65536", "131072", "262144")
# The 256K arm is a 5-10 trial DIAGNOSTIC (operator 2026-08-27), not a 50-trial cell: it can only be
# scored as "collapse-scale drop or not". The published effect is 57.8pp; with n=9 a 10pp gate cannot
# be resolved, so a drop of >= DIAG_DROP_PP vs the 4K control is a signal to ESCALATE to the full grid.
DIAG_LENGTH = "262144"
DIAG_DROP_PP = 30.0


def per_length_table(rep: dict) -> str:
    rows = []
    for L in ALL_TABLE_LENGTHS:
        if L in rep["per_length"]:
            p = rep["per_length"][L]
            acc = f"{p['accuracy_pct']:.1f}%" if p.get("accuracy_pct") is not None else "NA"
            rows.append(f"| {L} | {p['trials']} (+{p.get('unscored', 0)} unscored) | {p['hits']} | {acc} |")
    return "\n".join(rows)


def per_depth_table(rep: dict) -> str:
    rows = []
    for L in ALL_TABLE_LENGTHS:
        if L not in rep["per_depth"]:
            continue
        cells = []
        for d in ("0.1", "0.25", "0.5", "0.75", "0.9"):
            v = rep["per_depth"][L].get(d, {})
            acc = v.get("accuracy_pct")
            cells.append(f"{acc if acc is not None else 'NA'}%" if acc is not None else "NA")
        rows.append(f"| {L} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def gate_verdict(rep: dict) -> dict:
    """Returns per-length gate assessment using the bootstrap section."""
    out = {}
    for L in GATE_LENGTHS:
        key = str(L)
        if key not in rep.get("bootstrap", {}):
            out[key] = {"checked": False}
            continue
        b = rep["bootstrap"][key]
        drop = b.get("mean_drop_pp", 0.0)
        ci_low = b.get("ci95_low_pp", 0.0)
        open_signal = drop >= 10.0 and ci_low > 0.0 and b.get("n_placements", 0) > 0
        out[key] = {
            "checked": True,
            "mean_drop_pp": drop,
            "ci95_pp": [ci_low, b.get("ci95_high_pp")],
            "n_placements": b.get("n_placements"),
            "gate_signal": open_signal,
        }
    return out


def diagnostic_verdict(rep: dict) -> dict:
    """256K diagnostic: raw drop vs the 4K control, no CI claim (n is 5-10)."""
    pl = rep.get("per_length", {})
    if DIAG_LENGTH not in pl or not pl[DIAG_LENGTH].get("trials") or "4096" not in pl or pl["4096"].get("accuracy_pct") is None:
        return {"checked": False}
    drop = pl["4096"]["accuracy_pct"] - pl[DIAG_LENGTH]["accuracy_pct"]
    return {"checked": True, "n": pl[DIAG_LENGTH]["trials"], "hits": pl[DIAG_LENGTH]["hits"],
            "drop_pp_vs_4k": round(drop, 1), "escalate": drop >= DIAG_DROP_PP}


def shortest_drop_length(rep: dict) -> str:
    """Shortest length that shows a gate-signal drop (q8 grid), for choosing the f16 control length.
    Base lengths use the gate (>= 10pp and CI low > 0); the 256K diagnostic uses diagnostic_verdict().
    If nothing drops, 131072 (the run_g1.sh rule)."""
    gv = gate_verdict(rep)
    for L in GATE_LENGTHS:
        if gv[str(L)].get("gate_signal"):
            return str(L)
    if diagnostic_verdict(rep).get("escalate"):
        return DIAG_LENGTH
    return "131072"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q8", type=Path, required=True)
    ap.add_argument("--f16", type=Path, default=None)
    ap.add_argument("--shortest-drop", action="store_true",
                    help="print only the length the f16 control should use, then exit")
    args = ap.parse_args()

    if args.shortest_drop:
        print(shortest_drop_length(json.loads((args.q8 / "summary.json").read_text())))
        return 0
    if args.f16 is None:
        ap.error("--f16 is required unless --shortest-drop")

    q8 = json.loads((args.q8 / "summary.json").read_text())
    f16 = json.loads((args.f16 / "summary.json").read_text())

    print("=== per-length (q8_0/q8_0 full grid) ===")
    print("| length | trials | hits | accuracy |")
    print("|---|---|---|---|")
    print(per_length_table(q8))
    print()
    print("=== per-depth accuracy (q8_0/q8_0) ===")
    print("| length | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 |")
    print("|---|---|---|---|---|---|")
    print(per_depth_table(q8))
    print()
    print("=== gate assessment (q8 grid) ===")
    gv = gate_verdict(q8)
    any_signal = False
    for L in GATE_LENGTHS:
        v = gv[str(L)]
        if not v["checked"]:
            print(f"  {L}: not run")
            continue
        flag = "GATE-SIGNAL" if v["gate_signal"] else "clean"
        if v["gate_signal"]:
            any_signal = True
        print(
            f"  {L}: mean_drop={v['mean_drop_pp']}pp "
            f"CI95={v['ci95_pp']} n_placements={v['n_placements']} -> {flag}"
        )
    dv = diagnostic_verdict(q8)
    if dv["checked"]:
        print(f"  {DIAG_LENGTH} (DIAGNOSTIC, n={dv['n']}): {dv['hits']}/{dv['n']} hits, drop vs 4K = "
              f"{dv['drop_pp_vs_4k']}pp -> {'ESCALATE to the full 50-trial 256K grid' if dv['escalate'] else 'no collapse-scale drop'}")
    else:
        print(f"  {DIAG_LENGTH} (DIAGNOSTIC): not run")
    print()
    print("=== f16-KV control ===")
    ctrl4 = f16["per_length"].get("4096", {})
    print(f"  f16 4K control: {ctrl4.get('accuracy_pct', 'NA')}%")
    for L in ("32768", "65536", "131072", "262144"):
        if L in f16["per_length"]:
            p = f16["per_length"][L]
            print(f"  f16 {L}: {p['accuracy_pct']}% (drop vs f16-4K: "
                  f"{ctrl4.get('accuracy_pct', 0) - p['accuracy_pct']:.1f}pp)")
    print()
    b16 = f16.get("bootstrap", {})
    survived = {}
    for L in GATE_LENGTHS:
        key = str(L)
        if key not in b16:
            continue
        b = b16[key]
        drop = b.get("mean_drop_pp", 0.0)
        ok = drop >= 10.0 and b.get("ci95_low_pp", 0.0) > 0.0
        survived[key] = ok
        print(f"  f16 {L}: mean_drop={drop}pp CI95=[{b.get('ci95_low_pp')},{b.get('ci95_high_pp')}] "
              f"survives={'YES' if ok else 'no'}")

    # final gate: any q8 gate-signal that survives f16
    final = False
    for L in GATE_LENGTHS:
        if gv[str(L)].get("gate_signal") and survived.get(str(L)):
            final = True
    print()
    print(f"FINAL GATE: {'OPEN (ESCALATE — production defect)' if final else 'CLOSED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
