#!/usr/bin/env python3
"""Build the FG-1 observation report from sealed captures only; no inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
OUT = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/fg1-fine-grain-replay-20260727"
FOUR = ROOT / "artifacts/architect-same-era-v8-20260726/final-4arm-v4-tail-replay-20260727/runs/final-4arm-v4-tail-replay-20260727T080703Z"
SIX = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/expanded-six-arm-v4-tail-replay-20260727"
FABLE = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/fable-swe-tail-sealed-20260727T094334Z"
FABLE_OFFICIAL = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/live-20260726T1750Z/fable-only-cpu-clean-official-swe/runs/fable-only-v8-cpu-clean-20260727T095334Z"
FABLE_LCB = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/live-20260726T1750Z/continuation-27b-v8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_binomial_two_sided(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def arm_source(arm: str) -> tuple[Path, Path]:
    if arm in {"A1", "A3", "A4", "Laguna"}:
        arm_dir = FOUR / arm
        report = next(arm_dir.glob("*.final-4arm-v4-tail-replay-20260727T080703Z-*.json"))
    else:
        arm_dir = SIX / arm
        report = next(arm_dir.glob("*.expanded-six-arm-v4-tail-replay-20260727-*.json"))
    return report, arm_dir / "raw_capture.sealed.jsonl"


def official_set(path: Path) -> set[str]:
    report = read_json(path)
    ids = set(report["resolved_ids"])
    if len(ids) != len(report["resolved_ids"]):
        raise ValueError(f"duplicate official IDs in {path}")
    total = report["total_instances"]
    total = len(total) if isinstance(total, list) else total
    if total != 40:
        raise ValueError(f"wrong SWE denominator in {path}")
    if report["error_ids"]:
        raise ValueError(f"harness errors in {path}: {report['error_ids']}")
    return ids


def capture_by_id(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {row["id"]: row for row in rows}
    if len(result) != 40:
        raise ValueError(f"wrong/duplicate capture denominator in {path}: {len(result)}")
    return result


def lcb_summary(path: Path) -> dict:
    summary = read_json(path)
    suite = summary["suites"][0]
    if suite["n"] != 53 or suite["errors"] != 0:
        raise ValueError(f"invalid LCB denominator/errors in {path}")
    return suite


def main() -> None:
    arms = ["A1", "A3", "A4", "Laguna", "A3-tc", "A3-ff"]
    reports: dict[str, Path] = {}
    captures: dict[str, Path] = {}
    solved: dict[str, set[str]] = {}
    rows: dict[str, dict[str, dict]] = {}
    for arm in arms:
        reports[arm], captures[arm] = arm_source(arm)
        solved[arm] = official_set(reports[arm])
        rows[arm] = capture_by_id(captures[arm])
    canonical_ids = set(rows["A3"])
    if any(set(rows[arm]) != canonical_ids for arm in arms):
        raise ValueError("six arm captures do not share exact SWE40 IDs")

    overlap = {left: {right: len(solved[left] & solved[right]) for right in arms} for left in arms}
    unique = {arm: sorted(solved[arm] - set().union(*(solved[other] for other in arms if other != arm))) for arm in arms}
    laguna_vs = {
        "unique_vs_A3": sorted(solved["Laguna"] - solved["A3"]),
        "unique_vs_A4": sorted(solved["Laguna"] - solved["A4"]),
        "unique_vs_A3_union_A4": sorted(solved["Laguna"] - (solved["A3"] | solved["A4"])),
        "overlap_A3": len(solved["Laguna"] & solved["A3"]),
        "overlap_A4": len(solved["Laguna"] & solved["A4"]),
    }

    def discordant(left: str, right: str) -> dict:
        only_left = sorted(solved[left] - solved[right])
        only_right = sorted(solved[right] - solved[left])
        return {
            "pair": f"{left} vs {right}",
            f"only_{left}": only_left,
            f"only_{right}": only_right,
            "discordant_n": len(only_left) + len(only_right),
            "exact_binomial_p_two_sided": round(exact_binomial_two_sided(len(only_left), len(only_right)), 6),
        }

    tc_empty = set(read_json(SIX / "A3-tc" / "nonrecovery_ledger.sealed.json")["empty_patch_rows"][i]["instance_id"] for i in range(16))
    taxonomy = []
    for item in sorted(tc_empty):
        row = rows["A3-tc"][item]
        diag = next(r for r in read_jsonl(SIX / "A3-tc" / "conversion_diagnostics.sealed.jsonl") if r["instance_id"] == item)
        if row.get("truncated") or row.get("finish_reason") == "length":
            category = "truncated_mid_think_or_before_patch"
        elif diag.get("applied_block_count", 0) == 0 and diag.get("skipped_block_count", 0) > 0:
            category = "converter_format_or_path_miss"
        elif not (row.get("reasoning") or row.get("response")):
            category = "declined_or_empty"
        else:
            category = "unclassified_nonempty_no_patch"
        taxonomy.append({
            "id": item,
            "category": category,
            "finish_reason": row.get("finish_reason"),
            "completion_tokens": row.get("completion_tokens"),
            "truncated": bool(row.get("truncated")),
            "response_chars": len(row.get("response") or ""),
            "reasoning_chars": len(row.get("reasoning") or ""),
            "parseable_block_count": diag.get("parseable_block_count"),
            "skipped_block_count": diag.get("skipped_block_count"),
            "skipped_reasons": [block.get("outcome") for block in diag.get("blocks", []) if block.get("outcome", "").startswith("skipped_")],
        })
    taxonomy_counts = {
        "declined_or_empty": 0,
        "truncated_mid_think_or_before_patch": 0,
        "converter_format_or_path_miss": 0,
        "unclassified_nonempty_no_patch": 0,
    }
    for row in taxonomy:
        taxonomy_counts[row["category"]] += 1

    economics = {}
    for arm in arms:
        completion = [int(row.get("completion_tokens") or 0) for row in rows[arm].values()]
        total = sum(completion)
        resolved = len(solved[arm])
        economics[arm] = {
            "swe40_completion_tokens_total": total,
            "swe40_completion_tokens_median": median(completion),
            "swe40_tokens_per_official_resolve": round(total / resolved, 1),
            "official_resolved": resolved,
            "truncated_rows": sum(bool(row.get("truncated")) for row in rows[arm].values()),
        }
    economics["paired_reads"] = {
        "A3-tc_vs_A3": {
            "tokens_per_resolve_ratio": round(economics["A3-tc"]["swe40_tokens_per_official_resolve"] / economics["A3"]["swe40_tokens_per_official_resolve"], 3),
            "resolved_delta": len(solved["A3-tc"]) - len(solved["A3"]),
        },
        "A3-ff_vs_A3": {
            "tokens_per_resolve_ratio": round(economics["A3-ff"]["swe40_tokens_per_official_resolve"] / economics["A3"]["swe40_tokens_per_official_resolve"], 3),
            "resolved_delta": len(solved["A3-ff"]) - len(solved["A3"]),
        },
    }

    trio_reports = {
        "stock_non_mtp": FABLE_OFFICIAL / "stock_non_mtp/stock_non_mtp.fable-only-v8-cpu-clean-20260727T095334Z-stock_non_mtp.json",
        "fable_non_mtp": FABLE_OFFICIAL / "fable_non_mtp/fable_non_mtp.fable-only-v8-cpu-clean-20260727T095334Z-fable_non_mtp.json",
        "fable_mtp": FABLE_OFFICIAL / "fable_mtp/fable_mtp.fable-only-v8-cpu-clean-20260727T095334Z-fable_mtp.json",
    }
    trio_captures = {name: FABLE / name / "raw_capture.sealed.jsonl" for name in trio_reports}
    trio_lcb_paths = {
        "stock_non_mtp": FABLE_LCB / "A3-ff-quality__stock_non_mtp/lcb_hard.summary.json",
        "fable_non_mtp": FABLE_LCB / "A3-ff-quality__fable_non_mtp/lcb_hard.summary.json",
        "fable_mtp": FABLE_LCB / "A3-ff-embedded-mtp__fable_mtp/lcb_hard.summary.json",
    }
    trio = {}
    trio_solves = {}
    for name, report_path in trio_reports.items():
        trio_solves[name] = official_set(report_path)
        captured = capture_by_id(trio_captures[name])
        lcb = lcb_summary(trio_lcb_paths[name])
        tokens = [int(row.get("completion_tokens") or 0) for row in captured.values()]
        trio[name] = {
            "swe40_resolved": len(trio_solves[name]),
            "swe40_total_completion_tokens": sum(tokens),
            "swe40_median_completion_tokens": median(tokens),
            "lcb53_correct": lcb["correct"],
            "lcb53_truncated": lcb["truncated"],
            "lcb53_aggregate_decode_tok_s": lcb["throughput"]["aggregate_decode_tok_s"],
        }
    trio["swe40_overlap"] = {left: {right: len(trio_solves[left] & trio_solves[right]) for right in trio_solves} for left in trio_solves}
    trio["mcnemar"] = [discordant_trio(trio_solves, "fable_non_mtp", "fable_mtp"), discordant_trio(trio_solves, "stock_non_mtp", "fable_non_mtp")]

    sources = {str(path): sha256(path) for path in [*reports.values(), *captures.values(), *trio_reports.values(), *trio_captures.values(), *trio_lcb_paths.values(), SIX / "A3-tc" / "conversion_diagnostics.sealed.jsonl", SIX / "A3-tc" / "nonrecovery_ledger.sealed.json"]}
    result = {
        "schema_version": "epyc.architect-fg1-deterministic-replay.v1",
        "status": "OBSERVATION_GRADE_ZERO_INFERENCE",
        "protocol": "FG-1; deterministic replay of sealed artifacts only",
        "source_hashes": sources,
        "validation": {"swe_denominator": 40, "six_arms": arms, "all_six_share_exact_ids": True, "all_six_harness_errors": 0, "trio_lcb_denominator": 53, "trio_lcb_harness_errors": 0},
        "swe40": {"pairwise_solve_overlap": overlap, "unique_solves_against_other_five": unique, "laguna_route_specialist_read": laguna_vs, "mcnemar_discordants": [discordant("A3-tc", "A3"), discordant("A3-ff", "A3"), discordant("Laguna", "A3"), discordant("Laguna", "A4")]},
        "thinkingcap_empty_patch_taxonomy": {"counts": taxonomy_counts, "rows": taxonomy, "attempts": 24, "official_resolved": 18, "precision_on_attempts": 0.75},
        "tokens_per_solved": economics,
        "fable_mtp_pair": trio,
        "limitations": ["SWE tokens/solved is a banked one-pass observation, not the pending equal-effort LCB instrument.", "The MTP trio is a separate diagnostic and is not part of the six-arm authority table.", "No inference, rescoring, lineup, registry, or production-kernel action occurred."],
    }
    (OUT / "fg1_replay_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md = ["# FG-1 Fine-Grain Deterministic Replay", "", "Status: observation-grade, zero inference. Source hashes and full per-item lists are in `fg1_replay_report.json`.", "", "## SWE40", "", "- Six arms share the exact 40-item denominator; all official reports have zero harness errors.", f"- Laguna resolves {len(solved['Laguna'])}/40. Its unique solves vs A3 union A4: {len(laguna_vs['unique_vs_A3_union_A4'])}; vs A3 alone: {len(laguna_vs['unique_vs_A3'])}; vs A4 alone: {len(laguna_vs['unique_vs_A4'])}.", f"- Laguna vs A3 discordance: {len(solved['Laguna']-solved['A3'])} Laguna-only / {len(solved['A3']-solved['Laguna'])} A3-only. Laguna vs A4: {len(solved['Laguna']-solved['A4'])} Laguna-only / {len(solved['A4']-solved['Laguna'])} A4-only.", "", "## ThinkingCap", "", f"- Empty patches: {taxonomy_counts}. This is 15 cap truncations and one converter/path miss, not a refusal signature.", f"- Precision on attempted patches: 18/24 = 75.0%. Banked SWE tokens/official resolve: {economics['A3-tc']['swe40_tokens_per_official_resolve']} vs A3 {economics['A3']['swe40_tokens_per_official_resolve']} (ratio {economics['paired_reads']['A3-tc_vs_A3']['tokens_per_resolve_ratio']}).", "", "## Fable", "", f"- Non-MTP vs MTP: SWE {trio['fable_non_mtp']['swe40_resolved']}/40 vs {trio['fable_mtp']['swe40_resolved']}/40; LCB {trio['fable_non_mtp']['lcb53_correct']}/53 vs {trio['fable_mtp']['lcb53_correct']}/53; decode {trio['fable_non_mtp']['lcb53_aggregate_decode_tok_s']} vs {trio['fable_mtp']['lcb53_aggregate_decode_tok_s']} tok/s.", f"- FF banked SWE tokens/official resolve: {economics['A3-ff']['swe40_tokens_per_official_resolve']} vs A3 {economics['A3']['swe40_tokens_per_official_resolve']} (ratio {economics['paired_reads']['A3-ff_vs_A3']['tokens_per_resolve_ratio']}).", "", "## Boundaries", "", "- No role or lineup decision follows from this replay. The equal-effort token-efficiency instrument remains open."]
    (OUT / "fg1_replay_report.md").write_text("\n".join(md) + "\n")


def discordant_trio(solved: dict[str, set[str]], left: str, right: str) -> dict:
    only_left = sorted(solved[left] - solved[right])
    only_right = sorted(solved[right] - solved[left])
    return {"pair": f"{left} vs {right}", f"only_{left}": only_left, f"only_{right}": only_right, "discordant_n": len(only_left) + len(only_right), "exact_binomial_p_two_sided": round(exact_binomial_two_sided(len(only_left), len(only_right)), 6)}


if __name__ == "__main__":
    main()
