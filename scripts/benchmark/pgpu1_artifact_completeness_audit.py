#!/usr/bin/env python3
"""Audit MI210 GPU artifacts against the draft P-GPU-1 field list.

This is intentionally artifact-only: it reads files, classifies missing fields,
and never launches inference, servers, benchmarks, or ROCm commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "epyc.pgpu1_artifact_completeness_audit.v1"
DEFAULT_MAX_BYTES = 2_000_000


@dataclass(frozen=True)
class FieldRule:
    key: str
    label: str
    required: bool
    patterns: tuple[str, ...]
    near_patterns: tuple[str, ...] = ()


FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule(
        key="summary_json",
        label="summary.json or explicit summary file",
        required=True,
        patterns=(r"__file__:summary\.json",),
    ),
    FieldRule(
        key="rocm_clocks_before_after",
        label="ROCm clocks before+after",
        required=True,
        patterns=(r"--showclocks?\b", r"\bsclk\b", r"\bmclk\b", r"\bclock\(s\)\b"),
        near_patterns=(r"rocm-smi", r"rocm system management interface"),
    ),
    FieldRule(
        key="rocm_power_before_after",
        label="ROCm power before+after",
        required=True,
        patterns=(r"--showpower\b", r"\bpower\b", r"\bwatt", r"\bavg.*power\b"),
        near_patterns=(r"rocm-smi", r"rocm system management interface"),
    ),
    FieldRule(
        key="rocm_temp_before_after",
        label="ROCm temperature before+after",
        required=True,
        patterns=(r"--showtemp\b", r"\btemperature\b", r"\btemp\b"),
        near_patterns=(r"rocm-smi", r"rocm system management interface"),
    ),
    FieldRule(
        key="vram_pid_util_samples",
        label="VRAM, utilization, and GPU PID samples",
        required=True,
        patterns=(
            r"--showpidgpus\b",
            r"--showmemuse\b",
            r"--showuse\b",
            r"\bgpu memory allocated\b",
            r"\bvram",
            r"\bgpu use",
        ),
    ),
    FieldRule(
        key="binary_model_identity",
        label="binary, git, model, and backend identity",
        required=True,
        patterns=(
            r"llama-server",
            r"llama\.cpp-experimental",
            r"\.gguf\b",
            r"rev-parse",
            r"ld_library_path",
            r"rocm0",
        ),
    ),
    FieldRule(
        key="production_named_kernel_identity",
        label="production-named kernel identity",
        required=True,
        patterns=(
            r"production_named_kernel['\"]?\s*:\s*true",
            r"production-consolidated-v[0-9]",
        ),
        near_patterns=(r"llama\.cpp-experimental", r"experimental-v7", r"v7-candidate"),
    ),
    FieldRule(
        key="warmup_discard_policy",
        label="explicit warm-up/discard policy",
        required=True,
        patterns=(
            r"\bwarm[-_ ]?up\b",
            r"\bdiscard(?:ed)?\b",
            r"\bgraph recapture\b",
            r"\bno warm[-_ ]?up\b",
        ),
        near_patterns=(r"\bfresh-server\b", r"\brep(?:s|licate)?\b"),
    ),
    FieldRule(
        key="rep_count",
        label="rep count sufficient for >=5% claim",
        required=True,
        patterns=(r'"rep"\s*:\s*5\b', r'"n"\s*:\s*5\b', r"\bn=5\b", r"\b5\s+fresh-server reps\b"),
    ),
    FieldRule(
        key="cpu_interference_policy",
        label="explicit CPU-stack interference policy",
        required=True,
        patterns=(
            r"\bcpu[-_ ]stack\b",
            r"\bquiesced\b",
            r"\bco[-_ ]?resident\b",
            r"\bhidden from rocm\b",
            r"\bproduction stack\b",
            r"\binterference policy\b",
        ),
        near_patterns=(r"process_blockers", r"cleanup_process_blockers"),
    ),
    FieldRule(
        key="result_grammar",
        label="median/MAD throughput, prompt/decode split, and draft counters",
        required=True,
        patterns=(
            r"\bmedian\b",
            r"\bmad\b",
            r"prompt_tps",
            r"decode_tps",
            r"avg_ts",
            r"stddev_ts",
            r"samples_ts",
            r"draft_n_accepted",
            r"accepted drafts",
        ),
    ),
    FieldRule(
        key="cleanup_proof",
        label="process cleanup proof",
        required=True,
        patterns=(
            r'"dead"\s*:\s*true',
            r"cleanup_process_blockers",
            r"independent_cleanup",
            r"no_exact_llama_processes",
            r"\bcleanup proof\b",
        ),
    ),
    FieldRule(
        key="post_cleanup_vram_sample",
        label="post-cleanup VRAM/KFD sample",
        required=True,
        patterns=(
            r"after_cleanup",
            r"post[-_ ]cleanup.*vram",
            r"post[-_ ]cleanup.*kfd",
            r"post_rocm_smi_no_kfd",
            r"no kfd pids",
            r"0% vram",
        ),
        near_patterns=(r"before_cleanup", r'"dead"\s*:\s*true'),
    ),
)


def _load_text(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        return ""
    if path.stat().st_size > max_bytes:
        return f"__file__:{path.name}\n__skipped_large_file__:{path.stat().st_size}\n"
    try:
        return path.read_text(errors="replace")
    except UnicodeDecodeError:
        return ""


def _candidate_files(path: Path, max_bytes: int) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.stat().st_size > max_bytes:
            continue
        files.append(file_path)
    return files


def _artifact_text(path: Path, max_bytes: int) -> tuple[str, list[str], list[str]]:
    files = _candidate_files(path, max_bytes)
    chunks: list[str] = []
    names: list[str] = []
    skipped: list[str] = []
    for file_path in files:
        names.append(str(file_path))
        rel_name = file_path.name
        text = _load_text(file_path, max_bytes)
        if "__skipped_large_file__" in text:
            skipped.append(str(file_path))
        chunks.append(f"\n__file__:{rel_name}\n__path__:{file_path}\n{text}\n")
    return "\n".join(chunks).lower(), names, skipped


def _match_patterns(text: str, patterns: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL):
            matches.append(pattern)
    return matches


def _summary_status(path: Path) -> str:
    summary_path = path if path.is_file() else path / "summary.json"
    if not summary_path.exists():
        return "missing_summary"
    try:
        data = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return "invalid_summary_json"
    status = data.get("status")
    if isinstance(status, str):
        return status
    return "summary_present"


def audit_artifact(path: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    path = path.expanduser()
    text, files, skipped = _artifact_text(path, max_bytes)
    field_results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    near_misses: list[str] = []
    present: list[str] = []

    for rule in FIELD_RULES:
        matches = _match_patterns(text, rule.patterns)
        near = _match_patterns(text, rule.near_patterns)
        state = "present" if matches else "missing"
        if matches:
            present.append(rule.key)
        elif rule.required:
            missing.append(rule.key)
        if near and not matches:
            near_misses.append(rule.key)
        field_results[rule.key] = {
            "label": rule.label,
            "required": rule.required,
            "state": state,
            "matched_patterns": matches,
            "near_miss_patterns": near,
        }

    status = "complete" if not missing else "incomplete"
    recommendation = "retro_cert_candidate" if status == "complete" else "rerun_required"
    return {
        "artifact": str(path),
        "summary_status": _summary_status(path),
        "status": status,
        "recommendation": recommendation,
        "present_required_fields": present,
        "missing_required_fields": missing,
        "near_miss_fields": near_misses,
        "files_scanned_n": len(files),
        "files_scanned": files,
        "files_skipped_large": skipped,
        "field_results": field_results,
    }


def audit_artifacts(paths: Iterable[Path], max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    artifacts = [audit_artifact(path, max_bytes=max_bytes) for path in paths]
    incomplete = [item for item in artifacts if item["status"] != "complete"]
    return {
        "schema": SCHEMA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "artifact-only; no inference, server, benchmark, build, or ROCm command executed",
        "policy": "draft P-GPU-1 mandatory-field audit",
        "status": "complete" if not incomplete else "incomplete",
        "recommendation": "retro_cert_candidates_present" if not incomplete else "rerun_required_for_incomplete_artifacts",
        "artifacts": artifacts,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P-GPU-1 Artifact Completeness Audit",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Created: `{report['created_at']}`",
        f"- Scope: {report['scope']}",
        f"- Overall status: `{report['status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "| Artifact | Status | Recommendation | Missing required fields | Near misses |",
        "|---|---|---|---|---|",
    ]
    for item in report["artifacts"]:
        missing = ", ".join(f"`{field}`" for field in item["missing_required_fields"]) or "-"
        near = ", ".join(f"`{field}`" for field in item["near_miss_fields"]) or "-"
        lines.append(
            f"| `{item['artifact']}` | `{item['status']}` | `{item['recommendation']}` | {missing} | {near} |"
        )
    lines.extend(
        [
            "",
            "## Field Semantics",
            "",
            "A near miss means related evidence exists but does not satisfy the explicit P-GPU-1 field.",
            "For example, `process_blockers: []` is not the same as an explicit CPU-stack interference policy.",
            "",
            "No inference, server, benchmark, build, or ROCm command is run by this audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path, help="Artifact directory or summary.json path")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_artifacts(args.artifacts, max_bytes=args.max_bytes)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded)
    else:
        print(encoded, end="")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
