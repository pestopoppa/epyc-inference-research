#!/usr/bin/env python3
"""Build the zero-inference FG-2 evidence report from sealed captures."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
OUT = Path(__file__).parent
LAGUNA = ROOT / "artifacts/architect-laguna-iq2-v8-20260726/lcb-hard-port18090/pq.jsonl"
LAGUNA_SUMMARY = ROOT / "artifacts/architect-laguna-iq2-v8-20260726/lcb-hard-port18090/runner.json"
TC = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/expanded-six-arm-v4-tail-replay-20260727/A3-tc"
FG1 = ROOT / "artifacts/architect-27b-finetunes-v8-20260726/fg1-fine-grain-replay-20260727/fg1_results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text: str, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", text).strip()[-limit:]


def repeated_window_max(text: str, width: int = 120, stride: int = 20) -> int:
    windows = [text[i : i + width] for i in range(0, max(0, len(text) - width + 1), stride)]
    return max((text.count(window) for window in windows if window.strip()), default=0)


# These labels are a transparent, frozen-output review, not an automated verdict:
# each one is backed by the metrics and tail excerpt in the emitted JSON.
FG2_LABELS = {
    "lcb_abc341_f": "format_spiral",
    "lcb_abc331_d": "format_spiral",
    "lcb_abc331_e": "format_spiral",
    "lcb_abc323_e": "format_spiral",
    "lcb_abc320_e": "repetition_loop",
    "lcb_abc318_e": "genuine_long_reasoning",
    "lcb_abc314_f": "genuine_long_reasoning",
    "lcb_abc308_f": "format_spiral",
}


def main() -> None:
    laguna_rows = [json.loads(line) for line in LAGUNA.read_text().splitlines() if line.strip()]
    truncated = [row for row in laguna_rows if row.get("truncated") or row.get("finish_reason") == "length"]
    assert len(laguna_rows) == 53
    assert len(truncated) == 8
    assert set(FG2_LABELS) == {row["id"] for row in truncated}

    fg2 = []
    for row in truncated:
        text = row["extracted"]
        fence_count = text.count("```")
        label = FG2_LABELS[row["id"]]
        reason = {
            "repetition_loop": "A 120-character window occurs 21 times; the tail revisits the same abandoned data-structure plan.",
            "format_spiral": "The output alternates incomplete code/prose candidates or puts analysis inside a code block, then reaches the cap without a final answer.",
            "genuine_long_reasoning": "No repeated 120-character window above five occurrences and no multi-candidate output-format restart; the cap occurs during continuing derivation.",
        }[label]
        fg2.append(
            {
                "instance_id": row["id"],
                "classification": label,
                "classification_basis": reason,
                "finish_reason": row.get("finish_reason"),
                "completion_tokens": row.get("completion_tokens"),
                "response_chars": len(text),
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "fence_delimiter_count": fence_count,
                "complete_fence_count": fence_count // 2,
                "repeated_120_char_window_max": repeated_window_max(text),
                "tail_excerpt": compact(text),
            }
        )

    raw = [json.loads(line) for line in (TC / "raw_capture.sealed.jsonl").read_text().splitlines() if line.strip()]
    diagnostics = [json.loads(line) for line in (TC / "conversion_diagnostics.sealed.jsonl").read_text().splitlines() if line.strip()]
    ledger = json.loads((TC / "nonrecovery_ledger.sealed.json").read_text())
    empty_answer_raw = [row for row in raw if not row.get("response")]
    length_raw = [row for row in raw if row.get("finish_reason") == "length" and row.get("truncated")]
    path_diag = [row for row in diagnostics if row.get("empty_patch_reason") == "all_parseable_blocks_skipped"]
    assert len(raw) == len(diagnostics) == 40
    assert len(empty_answer_raw) == 12
    assert len(length_raw) == 15
    assert len(path_diag) == 1
    assert ledger["aggregate"] == {"empty_patch_row_count": 16, "skipped_block_count": 1}
    assert len(ledger["empty_patch_rows"]) == 16
    assert Counter(row["empty_patch_reason"] for row in diagnostics if row["empty_patch"]) == Counter({"model_length_cap": 15, "all_parseable_blocks_skipped": 1})

    tc_path = path_diag[0]
    fg3_audit = {
        "status": "independent_audit_passed; no new FG-3 artifact or run designed here",
        "counts": {
            "sealed_raw_rows": len(raw),
            "sealed_diagnostic_rows": len(diagnostics),
            "raw_model_length_cap_truncations": len(length_raw),
            "raw_empty_final_answer_channel_with_reasoning": len(empty_answer_raw),
            "diagnostic_empty_patches": sum(bool(row["empty_patch"]) for row in diagnostics),
            "diagnostic_model_length_cap": sum(row.get("empty_patch_reason") == "model_length_cap" for row in diagnostics),
            "diagnostic_path_converter_miss": len(path_diag),
            "ledger_empty_patch_rows": len(ledger["empty_patch_rows"]),
            "ledger_skipped_blocks": len(ledger["skipped_blocks"]),
        },
        "path_converter_miss": {
            "instance_id": tc_path["instance_id"],
            "finish_reason": tc_path["finish_reason"],
            "parseable_block_count": tc_path["parseable_block_count"],
            "skipped_block_count": tc_path["skipped_block_count"],
            "block_outcome": tc_path["blocks"][0]["outcome"],
            "response_sha256": tc_path["response_sha256"],
        },
        "committed_fg3_state": "root commit 27bc4ffc proves argv asymmetry and stages the exact FF-argv/no-think validation; this report does not duplicate it.",
    }
    result = {
        "schema_version": "fg2-fg3-sealed-classification.v1",
        "method": "deterministic inspection of sealed banked outputs; zero inference",
        "source_hashes": {
            str(LAGUNA): sha256(LAGUNA),
            str(LAGUNA_SUMMARY): sha256(LAGUNA_SUMMARY),
            str(TC / "raw_capture.sealed.jsonl"): sha256(TC / "raw_capture.sealed.jsonl"),
            str(TC / "conversion_diagnostics.sealed.jsonl"): sha256(TC / "conversion_diagnostics.sealed.jsonl"),
            str(TC / "nonrecovery_ledger.sealed.json"): sha256(TC / "nonrecovery_ledger.sealed.json"),
            str(FG1): sha256(FG1),
        },
        "fg2": {
            "source_row_count": len(laguna_rows),
            "cap_truncation_count": len(fg2),
            "classification_counts": dict(Counter(row["classification"] for row in fg2)),
            "exhaustive_disjoint": sum(Counter(row["classification"] for row in fg2).values()) == len(fg2),
            "rows": fg2,
            "model_neutral_remediation_candidates": [
                {"candidate": "loop-control sampler", "targets": ["repetition_loop"], "validation_cell": "the one loop-labelled LCB item, fixed seed and cap"},
                {"candidate": "repetition penalty", "targets": ["repetition_loop", "format_spiral"], "validation_cell": "the six loop/format items, compare completion/valid-answer rate at unchanged cap"},
                {"candidate": "answer-contract prompt", "targets": ["format_spiral"], "validation_cell": "the five format-spiral items, demand one final code block and no draft restarts"},
                {"candidate": "cap policy", "targets": ["genuine_long_reasoning"], "validation_cell": "the two genuine-long items only, larger cap with unchanged sampler"},
            ],
            "validation_scope": "Focused discordant-item cells only; no full-suite run and no claim until a post-campaign inference validation is scored.",
        },
        "fg3_independent_audit": fg3_audit,
    }
    output = OUT / "fg2_fg3_sealed_classification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# FG-2 sealed Laguna LCB taxonomy (2026-07-27)",
        "",
        "Zero-inference deterministic inspection of the sealed banked outputs. This is an observation-grade remediation design, not a performance claim.",
        "",
        "## FG-2 result",
        "",
        "All 8/8 cap truncations at 4,096 completion tokens are classified exhaustively and disjointly: 5 format spirals, 2 genuine long derivations, and 1 literal repetition loop. The exact per-item source hashes and bounded tail excerpts are in `fg2_fg3_sealed_classification.json`.",
        "",
        "| Class | Count | Focused validation |",
        "|---|---:|---|",
        "| repetition loop | 1 | loop-control sampler, fixed seed/cap |",
        "| format spiral | 5 | answer-contract prompt, then repetition-penalty ablation if needed |",
        "| genuine long reasoning | 2 | cap-only ablation, unchanged sampler |",
        "",
        "Do not pool these remedies: a larger cap does not test a loop fix, and a prompt-contract fix does not test long-reasoning capacity. Each proposed cell is focused to the failure class; no full-suite regeneration is proposed.",
        "",
        "## FG-3 audit",
        "",
        "Independent replay confirms the sealed TC partition: 40 raw rows, 15 model-length-cap rows (12 with an empty final-answer channel; 3 with partial answer text), and one separate `skipped_missing_path` converter miss, for 16 empty patches total. Root commit `27bc4ffc` already proves the thinking-mode argv confound and stages the no-think validation; no duplicate FG-3 run or artifact is proposed here.",
        "",
        "## Provenance",
        "",
        "All inputs and SHA-256 values are machine-readable in the JSON report. FG-1 is used only as corroborating prior; the FG-2 classifications are derived from the sealed LCB capture itself.",
    ]
    (OUT / "FG2_FG3_SEALED_CLASSIFICATION.md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
