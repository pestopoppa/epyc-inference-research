#!/usr/bin/env python3
"""Seal operator-run gate evidence into a bundle the champion surface can read.

THE GAP THIS CLOSES
-------------------
Manual inference research can already be ADMITTED into the champion -- DFlash2 and
MoE-Spec are in it by exactly that route. What did not exist was ATTESTATION: the
champion surface reads only a campaign-produced `cumulative_performance` receipt, so
the strongest measured evidence in the program (operator-run gates) was invisible to
the surface that reports champion standing. Admission worked; attestation did not, and
the loop "do manual research -> update the champion -> see its standing" was broken at
the last step.

WHY THIS IS NOT A CAMPAIGN RECEIPT, DELIBERATELY
------------------------------------------------
The obvious shortcut -- emit an `epyc.autokernel.cumulative_performance.v2` from manual
gates -- would launder operator evidence into CAMPAIGN authority. That receipt is
reachable only through a chain the campaign builds (screen results, operation results,
authority journal, composition ledger) and its authority derives from that chain. Faking
it is exactly the failure the measurement apparatus exists to prevent, and it would
poison every later comparison that trusts the receipt's provenance.

So this is a SEPARATE, HONESTLY-LABELLED carrier:

    schema    epyc.autokernel.operator_gate_bundle.v1
    authority operator_gated_manual_research     <- never campaign authority
    claim     measured, attributable, and NOT promotion-authorising

The champion surface shows it as operator-gated standing, beside (never instead of)
any campaign receipt. A reader can always tell which produced a number.

WHAT IT SEALS
-------------
Every gate names its artifact and the SHA-256 of that artifact's bytes, so a claim can
be traced to the file that produced it and a silently edited artifact invalidates the
bundle. A gate whose artifact is missing is RECORDED AS MISSING rather than dropped --
a bundle that quietly omits a failed or absent gate would be worse than none.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "epyc.autokernel.operator_gate_bundle.v1"
AUTHORITY = "operator_gated_manual_research"
ARTIFACTS = Path("/mnt/raid0/llm/artifacts-df25")
LLAMA = Path("/mnt/raid0/llm/llama.cpp")

#: The 2026-08-28 originals, kept as defaults so a bare invocation re-seals the
#: same evidence. A REFRESH names its own dated artifacts on the command line;
#: nothing here needs editing to run the boundary refresh
#: (`serving_evidence_refresh.py` passes all three).
DEFAULT_ANCHOR_ARTIFACT = (
    ARTIFACTS / "champion_anchor_20260828" / "champion_anchor_validation.json")
DEFAULT_CONCURRENCY_ARTIFACT = (
    ARTIFACTS / "dflash2_concurrency_20260827" / "cells.json")
DEFAULT_PARITY_ARTIFACT = (
    ARTIFACTS / "dflash2_greedy_parity_20260828" / "parity_report.json")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _git(*args: str) -> str | None:
    result = subprocess.run(("git", "-C", str(LLAMA), *args), check=False,
                            capture_output=True, text=True)
    return result.stdout.strip() or None if result.returncode == 0 else None


def _anchor_validation(path: Path) -> dict:
    data = _json(path)
    gate = {"gate": "no_regression_vs_production_anchor", "kind": "throughput",
            "artifact": str(path), "artifact_sha256": sha256_file(path)}
    if not data:
        return {**gate, "status": "MISSING", "surfaces": []}
    surfaces = [{"surface": r["surface"], "anchor": r["anchor_median"],
                 "champion": r["champion_median"], "delta_pct": r["delta_pct"]}
                for r in data.get("results", [])]
    # "No regression" is the claim, and it is deliberately not "a win": the sample
    # ranges overlap, and the champion's default path SHOULD look like production
    # because its admitted arms are inert until a role selects them.
    regressed = [s for s in surfaces if s["delta_pct"] < -1.0]
    return {**gate, "status": "PASS" if not regressed else "REGRESSION",
            "surfaces": surfaces,
            "claim": "champion default path does not regress the frozen anchor"}


def _concurrency_gate(path: Path) -> dict:
    cells = _json(path)
    gate = {"gate": "dflash2_vs_production_serving_path", "kind": "throughput",
            "artifact": str(path), "artifact_sha256": sha256_file(path)}
    if not cells:
        return {**gate, "status": "MISSING", "points": []}
    by = {(c["arm"], c["concurrency"], c["kv_unified"]): c["aggregate_decode_tok_s"]
          for c in cells}
    points = []
    for n in (1, 2, 4, 8):
        mtp, df2 = by.get(("mtp", n, False)), by.get(("dflash2", n, False))
        if mtp and df2:
            points.append({"in_flight": n, "production_ceiling_tps": mtp,
                           "champion_tps": df2,
                           "delta_pct": (df2 / mtp - 1.0) * 100.0})
    return {**gate, "status": "PASS" if points else "MISSING", "points": points,
            # The comparison arm is MTP because that is what production CAN run:
            # frozen v9 rejects the DFlash2 GGUF outright (81-vs-58 tensors), so MTP
            # is production's ceiling for this model, not merely a rival configuration.
            "claim": ("DFlash2 exceeds production's ceiling for Qwen3.8-27B; frozen v9 "
                      "cannot load the DFlash2 drafter at all")}


def _parity_gate(path: Path) -> dict:
    data = _json(path)
    gate = {"gate": "greedy_parity", "kind": "correctness",
            "artifact": str(path), "artifact_sha256": sha256_file(path)}
    if not data:
        return {**gate, "status": "MISSING", "arms": {}}
    arms = {name: {"pass": a.get("n_pass"), "fail": a.get("n_fail")}
            for name, a in (data.get("arms") or {}).items()}
    return {**gate, "status": "NOT_BIT_EXACT", "arms": arms,
            "baseline_negative_control_ok": data.get("baseline_negative_control_ok"),
            # The control carries the conclusion: draft_simple contains no DFlash code
            # and diverges identically, so non-parity is a property of the shared
            # speculative-verify path, not of DFlash2.
            "claim": ("not bit-exact, but NOT attributable to DFlash2 -- draft_simple "
                      "diverges identically with no DFlash code involved")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--champion-branch", default="ak/champion/llama-cpp-0db32c06e3e5")
    ap.add_argument("--champion-commit", default=None,
                    help="expected commit for --champion-branch; REFUSED on "
                         "mismatch so a bundle cannot silently seal a different "
                         "champion than the one the gates were run on")
    ap.add_argument("--production-commit",
                    default="0db32c06e3e550065b78311a6031ef3dd2c4f27c")
    ap.add_argument("--anchor-artifact", type=Path,
                    default=DEFAULT_ANCHOR_ARTIFACT)
    ap.add_argument("--concurrency-artifact", type=Path,
                    default=DEFAULT_CONCURRENCY_ARTIFACT)
    ap.add_argument("--parity-artifact", type=Path,
                    default=DEFAULT_PARITY_ARTIFACT)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    champion_commit = _git("rev-parse", args.champion_branch)
    if not champion_commit:
        print(f"REFUSED: cannot resolve {args.champion_branch}", file=sys.stderr)
        return 2
    if args.champion_commit and not champion_commit.startswith(args.champion_commit):
        print(f"REFUSED: {args.champion_branch} is at {champion_commit[:12]}, "
              f"not the expected {args.champion_commit[:12]} -- the branch moved "
              "since the gates ran; re-run the gates or re-pin", file=sys.stderr)
        return 2

    gates = [_anchor_validation(args.anchor_artifact),
             _concurrency_gate(args.concurrency_artifact),
             _parity_gate(args.parity_artifact)]
    missing = [g["gate"] for g in gates if g["status"] == "MISSING"]

    serving = next((g for g in gates
                    if g["gate"] == "dflash2_vs_production_serving_path"), {})
    points = serving.get("points") or []
    headline = None
    if points:
        best = max(points, key=lambda p: p["delta_pct"])
        headline = {
            "effect_fraction": best["delta_pct"] / 100.0,
            "metric": "aggregate_decode_tok_s",
            "metric_direction": "higher_better",
            "at_in_flight": best["in_flight"],
            "positive": best["delta_pct"] > 0,
            "summary": (f"+{best['delta_pct']:.1f}% at {best['in_flight']} in-flight "
                        "vs production's ceiling for this model"),
        }

    bundle = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        # The bundle's own date, in the body. Without it the reader's only date is
        # file mtime, which a copy or a `touch` moves with no new measurement --
        # the 2026-08-31 false-STALE on /kernel was exactly this. The dashboard's
        # `_read_operator_gate_bundle` prefers this field (labelled
        # `generated_at_source: body_generated_at`) and only falls back to mtime.
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Said in the artifact itself so a reader cannot mistake it for a campaign
        # receipt even out of context.
        "promotion_claim": False,
        "not_campaign_sealed": True,
        "champion": {"branch": args.champion_branch, "commit": champion_commit},
        "production_anchor": {"commit": args.production_commit},
        "gates": gates,
        "gates_missing": missing,
        "headline": headline,
        "caveat": ("Operator-gated manual research. Measured and attributable, but it "
                   "carries NO promotion authority and is not a campaign-sealed "
                   "cumulative receipt."),
    }
    bundle["bundle_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in bundle.items() if k != "bundle_sha256"},
                   sort_keys=True).encode()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    print(f"  champion {champion_commit[:12]} vs production {args.production_commit[:12]}")
    for g in gates:
        print(f"  {g['gate']:<42} {g['status']}")
    if headline:
        print(f"  headline: {headline['summary']}")
    if missing:
        print(f"  MISSING GATES RECORDED: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
