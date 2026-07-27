#!/usr/bin/env python3
"""Deterministically regenerate the sealed Laguna Q4-versus-IQ2 SWE40 read."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent
Q4_DIR = BASE / "official-v4" / "Laguna_Q4"
IQ2_DIR = Path(
    "/mnt/raid0/llm/epyc-inference-research/artifacts/architect-same-era-v8-20260726/"
    "final-4arm-v4-tail-replay-20260727/runs/"
    "final-4arm-v4-tail-replay-20260727T080703Z/Laguna"
)
Q4_REPORT = Q4_DIR / "Laguna_S_2_1_Q4_K_M_v8_cpu.laguna-q4-v4-tail-replay-20260727-laguna_q4.json"
IQ2_REPORT = IQ2_DIR / "Laguna_S_2_1_UD_IQ2_M_v8.final-4arm-v4-tail-replay-20260727T080703Z-laguna.json"
EXPECTED_REPORT_SHA256 = {
    "Q4": "0e6c0121202ec762ecce1888a799df73057b4441e94930b6d6350a97a316a13c",
    "IQ2": "a13192d3566191d25cbd78cec6c0c611ed1c61cd914d295ee5862b15cf876e20",
}


class ValidationError(ValueError):
    """A sealed input does not meet the paired-read contract."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a JSON object")
    return value


def validate_report(report: dict[str, Any], label: str) -> set[str]:
    required_counts = {
        "total_instances": 40,
        "submitted_instances": 40,
        "error_instances": 0,
    }
    for key, expected in required_counts.items():
        if report.get(key) != expected:
            raise ValidationError(f"{label}: {key} must be {expected}, got {report.get(key)!r}")
    submitted = report.get("submitted_ids")
    resolved = report.get("resolved_ids")
    if not isinstance(submitted, list) or len(submitted) != 40 or len(set(submitted)) != 40:
        raise ValidationError(f"{label}: submitted_ids must be 40 unique IDs")
    if not isinstance(resolved, list) or len(set(resolved)) != len(resolved):
        raise ValidationError(f"{label}: resolved_ids must be unique")
    resolved_set = set(resolved)
    if not resolved_set <= set(submitted):
        raise ValidationError(f"{label}: resolved_ids are outside the submitted denominator")
    if report.get("resolved_instances") != len(resolved_set):
        raise ValidationError(f"{label}: resolved_instances does not match resolved_ids")
    return resolved_set


def validate_pinned_report_hash(path: Path, label: str) -> str:
    observed = sha256(path)
    if observed != EXPECTED_REPORT_SHA256[label]:
        raise ValidationError(f"{label}: official report hash mismatch")
    return observed


def validate_sealed_sources(directory: Path, manifest: dict[str, Any], label: str) -> dict[str, str]:
    sealed = manifest.get("sealed")
    if not isinstance(sealed, dict):
        raise ValidationError(f"{label}: manifest has no sealed source map")
    expected_files = {
        "converter_source": "converter_v4.sealed.py",
        "dataset": "swebench_verified.sealed.json",
        "predictions": "predictions.sealed.json",
        "nonrecovery_ledger": "nonrecovery_ledger.sealed.json",
        "raw_capture": "raw_capture.sealed.jsonl",
    }
    actual: dict[str, str] = {}
    for key, filename in expected_files.items():
        expected = sealed.get(key, {}).get("sha256")
        if not isinstance(expected, str):
            raise ValidationError(f"{label}: missing sealed hash for {key}")
        observed = sha256(directory / filename)
        if observed != expected:
            raise ValidationError(f"{label}: {key} hash mismatch")
        actual[key] = observed
    return actual


def exact_two_sided_binomial(q4_only: int, iq2_only: int) -> float:
    discordant = q4_only + iq2_only
    if discordant == 0:
        return 1.0
    tail = min(q4_only, iq2_only)
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant))


def build_read(q4_dir: Path = Q4_DIR, iq2_dir: Path = IQ2_DIR) -> dict[str, Any]:
    q4_report_path = q4_dir / Q4_REPORT.name
    iq2_report_path = iq2_dir / IQ2_REPORT.name
    q4_manifest_path = q4_dir / "manifest.json"
    iq2_manifest_path = iq2_dir / "manifest.json"
    q4_report_hash = validate_pinned_report_hash(q4_report_path, "Q4")
    iq2_report_hash = validate_pinned_report_hash(iq2_report_path, "IQ2")
    q4_report = load_json(q4_report_path)
    iq2_report = load_json(iq2_report_path)
    q4_manifest = load_json(q4_manifest_path)
    iq2_manifest = load_json(iq2_manifest_path)
    q4_resolved = validate_report(q4_report, "Q4")
    iq2_resolved = validate_report(iq2_report, "IQ2")
    q4_sources = validate_sealed_sources(q4_dir, q4_manifest, "Q4")
    iq2_sources = validate_sealed_sources(iq2_dir, iq2_manifest, "IQ2")
    q4_only = sorted(q4_resolved - iq2_resolved)
    iq2_only = sorted(iq2_resolved - q4_resolved)
    overlap = len(q4_resolved & iq2_resolved)
    p_value = exact_two_sided_binomial(len(q4_only), len(iq2_only))

    return {
        "schema": "epyc.laguna_q4_vs_iq2_paired_quality_read.v2",
        "status": "OBSERVATION_GRADE_DETERMINISTIC_TAIL_REPLAY",
        "protocol": {
            "suite": "swebench_oracle",
            "fixed_denominator": 40,
            "scorer": "official SWE-bench FAIL_TO_PASS",
            "converter": "pinned v4 exact, trailing-whitespace, then unique-indent matching",
            "generation_reused": True,
            "additional_inference": False,
            "test_verdict_relaxed": False,
        },
        "q4": {
            "resolved": len(q4_resolved),
            "percent": 100 * len(q4_resolved) / 40,
            "harness_errors": q4_report["error_instances"],
            "empty_patches": q4_report["empty_patch_instances"],
            "official_report_sha256": q4_report_hash,
            "manifest_sha256": sha256(q4_manifest_path),
            "sealed_source_sha256": q4_sources,
        },
        "iq2": {
            "resolved": len(iq2_resolved),
            "percent": 100 * len(iq2_resolved) / 40,
            "harness_errors": iq2_report["error_instances"],
            "empty_patches": iq2_report["empty_patch_instances"],
            "official_report_sha256": iq2_report_hash,
            "manifest_sha256": sha256(iq2_manifest_path),
            "sealed_source_sha256": iq2_sources,
        },
        "paired_read": {
            "overlap": overlap,
            "q4_only": q4_only,
            "iq2_only": iq2_only,
            "discordant_n": len(q4_only) + len(iq2_only),
            "exact_binomial_two_sided_p": p_value,
            "interpretation": "This n=40 screen does not resolve a quant-axis quality difference.",
        },
        "disposition": {
            "additional_q4_quality_or_lcb_inference": "STOP_BY_CURRENT_SCREEN_ONLY",
            "cpu_performance_config_discovery": "DEFER_UNLESS_A_CONCRETE_CPU_ROLE_DECISION_REQUIRES_IT",
            "lineup_or_registry_action": "NONE",
            "reason": "The paired n=40 screen does not resolve a quant-axis difference; it is not an equivalence or absence-of-difference proof.",
        },
    }


def render_markdown(read: dict[str, Any]) -> str:
    paired = read["paired_read"]
    return "\n".join(
        [
            "# Laguna Q4 vs IQ2 paired quality read",
            "",
            "Status: observation-grade deterministic replay, zero additional inference.",
            "",
            "The official SWE40 result is Q4 "
            f"`{read['q4']['resolved']}/40` versus IQ2 `{read['iq2']['resolved']}/40`, with zero "
            "harness errors in both arms. "
            f"{paired['overlap']} solves overlap; Q4 has {len(paired['q4_only'])} unique solves "
            f"and IQ2 has {len(paired['iq2_only'])}. The exact paired two-sided result is "
            f"`p={paired['exact_binomial_two_sided_p']}`, so this n=40 screen does not resolve "
            "a quant-axis quality difference. It is not an equivalence or absence-of-difference proof.",
            "",
            "No additional Q4 quality or LCB inference is justified by this screen alone. CPU "
            "performance configuration discovery remains deferred unless a concrete CPU-role decision "
            "needs it; this replay authorizes no lineup or registry change.",
            "",
        ]
    )


def render_json(read: dict[str, Any]) -> str:
    return json.dumps(read, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--check", action="store_true", help="fail unless existing outputs match")
    args = parser.parse_args()
    read = build_read()
    outputs = {
        args.output_dir / "paired_q4_vs_iq2_quality_read.json": render_json(read),
        args.output_dir / "PAIRED_QUALITY_READ.md": render_markdown(read),
    }
    if args.check:
        mismatches = [str(path) for path, expected in outputs.items() if not path.exists() or path.read_text(encoding="utf-8") != expected]
        if mismatches:
            print("paired quality read is stale: " + ", ".join(mismatches), file=sys.stderr)
            return 1
        return 0
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
