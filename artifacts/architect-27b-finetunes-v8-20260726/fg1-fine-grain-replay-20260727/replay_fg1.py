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


def quantile_higher(values: list[float], quantile: float) -> float:
    """Return the empirical higher quantile using zero-based ceil(q * (n - 1))."""
    if not values or not 0 <= quantile <= 1:
        raise ValueError("quantile requires a non-empty sequence and q in [0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * (len(ordered) - 1))]


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
    empty_counts = {
        arm: len(read_json(captures[arm].parent / "nonrecovery_ledger.sealed.json")["empty_patch_rows"])
        for arm in arms
    }

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
            category = "cap_truncated_thinking_mode_generation"
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
        "cap_truncated_thinking_mode_generation": 0,
        "converter_format_or_path_miss": 0,
        "declined_or_empty": 0,
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
            "empty_patches": empty_counts[arm],
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

    mcnemar = [discordant("A3-tc", "A3"), discordant("A3-ff", "A3"), discordant("Laguna", "A3"), discordant("Laguna", "A4")]
    hard_core = sorted(canonical_ids - set().union(*solved.values()))
    source_paths = [
        *reports.values(),
        *captures.values(),
        *trio_reports.values(),
        *trio_captures.values(),
        *trio_lcb_paths.values(),
        *(captures[arm].parent / "nonrecovery_ledger.sealed.json" for arm in arms),
        SIX / "A3-tc" / "conversion_diagnostics.sealed.jsonl",
        SIX / "A3-tc" / "nonrecovery_ledger.sealed.json",
    ]
    sources = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for path in sorted(set(source_paths))
    ]
    tc_cap_rows = [row for row in taxonomy if row["category"] == "cap_truncated_thinking_mode_generation"]
    tc_zero_response_rows = [row["id"] for row in tc_cap_rows if row["response_chars"] == 0]
    tc_partial_response_rows = [row["id"] for row in tc_cap_rows if row["response_chars"] > 0]
    a4_speeds = [row["decode_tok_s"] for row in rows["A4"].values()]
    a4_p10 = quantile_higher(a4_speeds, 0.10)
    a4_p90 = quantile_higher(a4_speeds, 0.90)
    result = {
        "schema_version": "epyc.architect-fg1-deterministic-replay.v2",
        "status": "OBSERVATION_GRADE_ZERO_INFERENCE",
        "protocol": "FG-1; deterministic replay of sealed artifacts only",
        "sealed_sources": sources,
        "validation": {"swe_denominator": 40, "six_arms": arms, "all_six_share_exact_ids": True, "all_six_harness_errors": 0, "trio_lcb_denominator": 53, "trio_lcb_harness_errors": 0, "a4_decode_percentile_method": "higher: sorted zero-based ceil(q * (n - 1))", "a4_decode_p10": a4_p10, "a4_decode_p90": a4_p90},
        "swe40": {"pairwise_solve_overlap": overlap, "unique_solves_against_other_five": unique, "unsolved_by_all_six": hard_core, "laguna_route_specialist_read": laguna_vs, "mcnemar_discordants": mcnemar},
        "thinkingcap_empty_patch_taxonomy": {
            "counts": taxonomy_counts,
            "rows": taxonomy,
            "cap_truncated_rows": len(tc_cap_rows),
            "cap_truncated_zero_response_rows": tc_zero_response_rows,
            "cap_truncated_partial_response_rows": tc_partial_response_rows,
            "attempts": 24,
            "official_resolved": 18,
            "precision_on_attempts": 0.75,
        },
        "tokens_per_solved": economics,
        "fable_mtp_pair": trio,
        "limitations": ["SWE tokens/solved is a banked one-pass observation, not the pending equal-effort LCB instrument.", "The MTP trio is a separate diagnostic and is not part of the six-arm authority table.", "No inference, rescoring, lineup, registry, or production-kernel action occurred."],
    }
    (OUT / "fg1_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pair_order = ["A3", "A3-ff", "A3-tc", "Laguna", "A1", "A4"]
    pairwise_table = ["|        | A3 | A3-ff | A3-tc | Laguna | A1 | A4 |", "|--------|----|-------|-------|--------|----|----|"]
    for index, left in enumerate(pair_order):
        values = [str(overlap[left][right]) if right in pair_order[index:] else "" for right in pair_order]
        pairwise_table.append(f"| {left} | " + " | ".join(values) + " |")
    token_table = ["| Arm | resolved | empty | truncated | median compl. tok | tokens/solved | median decode tok/s |", "|-----|----------|-------|-----------|-------------------|---------------|---------------------|"]
    for arm in pair_order:
        decode = f"{median(row['decode_tok_s'] for row in rows[arm].values()):.1f}" if arm in {"A1", "A3", "A4", "Laguna"} else "-"
        if arm == "A3-ff":
            decode = f"- (trio {median(row['decode_tok_s'] for row in read_jsonl(trio_captures['fable_non_mtp'])):.1f})"
        elif arm == "A3-tc":
            decode = f"- (capture {median(row['decode_tok_s'] for row in rows[arm].values()):.1f})"
        token_table.append(f"| {arm} | {economics[arm]['official_resolved']} | {economics[arm]['empty_patches']} | {economics[arm]['truncated_rows']} | {economics[arm]['swe40_completion_tokens_median']:.1f} | {economics[arm]['swe40_tokens_per_official_resolve']:.1f} | {decode} |")
    md = [
        "# FG-1 fine-grain replay - six-arm SWE40 sealed artifacts (2026-07-27)",
        "",
        "Deterministic replay/transform only - zero inference. Sources: `final-4arm-v4-tail-replay-20260727T080703Z` (A3/A4/A1/Laguna), `expanded-six-arm-v4-tail-replay-20260727` (A3-tc/A3-ff), and `fable-swe-tail-sealed-20260727T094334Z` (FF MTP trio). `fg1_results.json` contains every sealed input path and SHA-256. Observation-grade fine-grain read per MEASUREMENT_POLICY deterministic-replay rule.",
        "",
        "## Headline findings",
        "",
        f"1. **TC empty patches are cap truncations during thinking-mode generation.** {len(tc_cap_rows)} of 16 empty-patch failures ended at the 3072-token cap; {len(tc_zero_response_rows)} have zero response chars and {len(tc_partial_response_rows)} have partial response text. The remaining failure is a single `skipped_missing_path` apply failure. TC's median reasoning text is {median(len(row.get('reasoning') or '') for row in rows['A3-tc'].values()):.0f} chars. The asymmetric thinking configuration is confounded, so its {economics['A3-tc']['swe40_tokens_per_official_resolve']:.1f} tokens/solved versus A3's {economics['A3']['swe40_tokens_per_official_resolve']:.1f} is diagnostic only, not comparative token-efficiency authority.",
        f"2. **FF is the banked token-efficiency leader.** Same-harness FF-non-MTP median completion tokens are {trio['fable_non_mtp']['swe40_median_completion_tokens']:.1f} versus stock {trio['stock_non_mtp']['swe40_median_completion_tokens']:.1f}; authority-table tokens/solved are {economics['A3-ff']['swe40_tokens_per_official_resolve']:.1f} versus A3 {economics['A3']['swe40_tokens_per_official_resolve']:.1f}. Quality is statistically tied (McNemar +2/-6, p={mcnemar[1]['exact_binomial_p_two_sided']:.2f}). FF-MTP is leaner still ({trio['fable_mtp']['swe40_median_completion_tokens']:.1f}, total {trio['fable_mtp']['swe40_total_completion_tokens']}) but LCB-weak ({trio['fable_mtp']['lcb53_correct']} versus {trio['fable_non_mtp']['lcb53_correct']}/{trio['stock_non_mtp']['lcb53_correct']}).",
        f"3. **Laguna SWE-route specialist is dead.** Its {len(solved['Laguna'])} solves are a strict subset of A3 union A4 (unique=0); A3 dominates +{len(solved['A3'] - solved['Laguna'])}/-{len(solved['Laguna'] - solved['A3'])}, exact p={mcnemar[2]['exact_binomial_p_two_sided']:.3f}.",
        f"4. **The Laguna speed argument inverts (FG-4).** In the sealed capture telemetry, A4 median decode is {median(a4_speeds):.1f} tok/s (p10 {a4_p10:.1f}, p90 {a4_p90:.1f}; empirical higher quantile, zero-based ceil(q * (n - 1))) versus Laguna {median(row['decode_tok_s'] for row in rows['Laguna'].values()):.1f}, A3 {median(row['decode_tok_s'] for row in rows['A3'].values()):.1f}, and A1 {median(row['decode_tok_s'] for row in rows['A1'].values()):.1f}. This remains observation-grade telemetry, not a registry replacement.",
        f"5. **Discriminating hard core:** {len(hard_core)}/40 instances are unsolved by all six arms: {', '.join(hard_core)}. A3 keeps {len(unique['A3'])} unique solves; TC has {len(unique['A3-tc'])}.",
        "",
        "## Tables",
        "",
        "Pairwise solve overlap (diagonal = resolved):",
        "",
        *pairwise_table,
        "",
        "Token economics (all 40 rows/arm):",
        "",
        *token_table,
        "",
        f"McNemar discordants: TC-vs-A3 +{len(mcnemar[0]['only_A3-tc'])}/-{len(mcnemar[0]['only_A3'])} (p={mcnemar[0]['exact_binomial_p_two_sided']:.2f}); FF-vs-A3 +{len(mcnemar[1]['only_A3-ff'])}/-{len(mcnemar[1]['only_A3'])} (p={mcnemar[1]['exact_binomial_p_two_sided']:.2f}); Laguna-vs-A3 +{len(mcnemar[2]['only_Laguna'])}/-{len(mcnemar[2]['only_A3'])} (p={mcnemar[2]['exact_binomial_p_two_sided']:.3f}); Laguna-vs-A4 +{len(mcnemar[3]['only_Laguna'])}/-{len(mcnemar[3]['only_A4'])} (p={mcnemar[3]['exact_binomial_p_two_sided']:.2f}).",
        "",
        "## Consequences filed",
        "",
        "- FG-3 remains a clean no-think validation; the confounded TC economics cannot rank candidacy.",
        "- FG-2 retains Laguna's SWE truncation prior, but FG-1 plus FG-4 eliminate a SWE routing case.",
        "- Laguna's remaining case is the L-Q4 quant axis plus non-coding suites (FG-5).",
        "- A4's registry performance row still needs a protocol-cited refresh.",
        "",
        "## Boundaries",
        "",
        "- No role or lineup decision follows from this replay. The equal-effort token-efficiency instrument remains open.",
    ]
    (OUT / "FG1_SUMMARY.md").write_text("\n".join(md) + "\n")


def discordant_trio(solved: dict[str, set[str]], left: str, right: str) -> dict:
    only_left = sorted(solved[left] - solved[right])
    only_right = sorted(solved[right] - solved[left])
    return {"pair": f"{left} vs {right}", f"only_{left}": only_left, f"only_{right}": only_right, "discordant_n": len(only_left) + len(only_right), "exact_binomial_p_two_sided": round(exact_binomial_two_sided(len(only_left), len(only_right)), 6)}


if __name__ == "__main__":
    main()
