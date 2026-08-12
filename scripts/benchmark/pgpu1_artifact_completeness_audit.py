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
class SubFieldRule:
    """One mandatory sub-field of a composite rule.

    Every sub-field of a composite rule must match independently — the rule is a
    CONJUNCTION, not an any-one-of. See ``binary_model_identity`` below.
    """

    key: str
    label: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class FieldRule:
    key: str
    label: str
    required: bool
    patterns: tuple[str, ...] = ()
    near_patterns: tuple[str, ...] = ()
    # When set, the rule is satisfied only if EVERY sub-rule matches.
    subfields: tuple[SubFieldRule, ...] = ()
    # "all"          -> scan every readable file in the artifact directory
    # "run_metadata" -> scan only recorded run metadata (JSON/YAML receipts and
    #                   recorded command transcripts). Harness/source files are
    #                   excluded so that code which merely MENTIONS a variable
    #                   cannot stand in for a recorded value.
    scope: str = "all"


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
    # P-GPU-1 field 3 (measurement/protocols/gpu-cross-device.md:36-38) is a
    # CONJUNCTION of mandatory sub-fields, and it is the field the ggml-linkage
    # audit exists to enforce. It used to OR six loose patterns, so an artifact
    # banking no LD_LIBRARY_PATH at all passed on a bare `llama-server` hit, and
    # the string `ld_library_path` inside a harness SOURCE file counted as a
    # recorded value (docs/reviews/gpu-linkage-retro-certification-20260812.md
    # §3.2a-bis; the "KEY too wide" member of the vacuous-verification family).
    # Two independent defences now apply: the scan is restricted to recorded run
    # metadata, and every sub-pattern demands a VALUE rather than a mention.
    FieldRule(
        key="binary_model_identity",
        label="binary path, git commit, LD_LIBRARY_PATH value, and backend list (all mandatory)",
        required=True,
        scope="run_metadata",
        subfields=(
            SubFieldRule(
                key="ld_library_path_value",
                label="recorded LD_LIBRARY_PATH value (not merely the variable name)",
                patterns=(
                    # JSON/YAML: "LD_LIBRARY_PATH": "/some/path..."
                    r'["\']ld_library_path["\']\s*[:=]\s*["\'][^"\'\n]*/[^"\'\n]*["\']',
                    # argv element or shell assignment: LD_LIBRARY_PATH=/some/path
                    r"\bld_library_path=[^\s\"',\]]*/",
                ),
            ),
            SubFieldRule(
                key="backend_device_list",
                label="enumerated backend/device list from the running binary or runtime",
                # Enumeration OUTPUT only. A hand-written device string such as
                # `"host_gpu": "AMD Instinct MI210 gfx90a"` is an assertion, not
                # an enumeration, and must not satisfy this sub-field.
                patterns=(
                    r"--list-devices",
                    r"available devices:",
                    r"\brocm[0-9]:\s",
                    r"\brocminfo\b",
                    r"\bhsa agents\b",
                    r"amdgcn-amd-amdhsa--gfx",
                    r'["\']backends["\']\s*:\s*[\[{"]',
                ),
            ),
            SubFieldRule(
                key="binary_path",
                label="absolute path of the binary that produced the measurement",
                patterns=(r"/[a-z0-9._/-]*bin/llama-(?:server|bench|cli)\b",),
            ),
            SubFieldRule(
                key="kernel_commit",
                label="recorded kernel commit/build id (a value, not a `rev-parse` mention)",
                patterns=(
                    r'["\'](?:commit|head|git_head|git_commit|revision|binary_version)["\']\s*[:=]\s*["\']?[0-9a-f]{7,40}\b',
                    r'["\']build_info["\']\s*[:=]\s*["\']?b[0-9]+-[0-9a-f]{7,40}\b',
                    r"\bb[0-9]{4,}-[0-9a-f]{7,40}\b",
                    r'rev-parse[\s\S]{0,600}?["\']stdout["\']\s*:\s*["\'][0-9a-f]{7,40}\b',
                ),
            ),
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


# --- run-metadata scoping -------------------------------------------------
#
# Recorded run metadata = structured receipts plus recorded command
# transcripts. Source files (a harness, a helper module) are NEVER run
# metadata: code that SETS an environment variable is not an artifact that
# RECORDS its value.
_METADATA_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml"})
_SOURCE_SUFFIXES = frozenset(
    {".py", ".pyi", ".c", ".cc", ".cpp", ".h", ".hpp", ".ipynb", ".md", ".rst", ".js", ".ts"}
)
_METADATA_NAME_RE = re.compile(
    r"^(?:commands|command|cmd|operator_run|run|env|environment|linkage[a-z0-9._-]*"
    r"|[a-z0-9._-]*receipt[a-z0-9._-]*|[a-z0-9._-]*argv[a-z0-9._-]*)"
    r"\.(?:sh|txt|log|env)$",
    re.IGNORECASE,
)


def _is_run_metadata(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in _SOURCE_SUFFIXES:
        return False
    if suffix in _METADATA_SUFFIXES:
        return True
    return bool(_METADATA_NAME_RE.match(path.name))


# --- kernel provenance ----------------------------------------------------
#
# measurement/protocols/gpu-cross-device.md:16-21,49-53 — a decision-grade claim
# MAY ONLY be produced on a production-named kernel. A measurement produced from
# a `llama.cpp-experimental` (or any other suffixed) tree is an OBSERVATION and
# can never be retro-certified, however complete its fields are.
#
# The reference must be to a BUILD/BINARY directory of the non-production tree.
# A bare mention of the tree (e.g. `git -C /mnt/raid0/llm/llama.cpp-experimental
# rev-parse HEAD`, which the v9 cert run banks as a guard-hygiene side probe)
# does not mean the measured binary came from there.
_NONPRODUCTION_KERNEL_RE = re.compile(
    r"/(llama\.cpp-([a-z0-9][a-z0-9._-]*))/(?=build|bin\b)", re.IGNORECASE
)
_PRODUCTION_KERNEL_SUFFIX_RE = re.compile(r"^production(?:[-_.]|$)", re.IGNORECASE)


def _kernel_provenance(text: str) -> list[dict[str, str]]:
    """Return one disqualification record per non-production kernel build tree."""
    seen: dict[str, dict[str, str]] = {}
    for match in _NONPRODUCTION_KERNEL_RE.finditer(text):
        tree, suffix = match.group(1), match.group(2)
        if _PRODUCTION_KERNEL_SUFFIX_RE.match(suffix):
            continue
        if tree in seen:
            continue
        seen[tree] = {
            "rule": "kernel_provenance",
            "kernel_tree": tree,
            "evidence": match.group(0),
            "reason": (
                f"measurement binaries resolve from non-production kernel tree '{tree}'; "
                "measurement/protocols/gpu-cross-device.md:16-21 makes such a run "
                "OBSERVATION-ONLY and :49-53 bars retro-certification regardless of "
                "field completeness"
            ),
        }
    return list(seen.values())


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
        if file_path.name != "summary.json" and file_path.stat().st_size > max_bytes:
            continue
        files.append(file_path)
    return files


def _artifact_text(path: Path, max_bytes: int) -> tuple[str, str, list[str], list[str], list[str]]:
    """Return (all-file text, run-metadata-only text, files, metadata files, skipped)."""
    files = _candidate_files(path, max_bytes)
    chunks: list[str] = []
    meta_chunks: list[str] = []
    names: list[str] = []
    meta_names: list[str] = []
    skipped: list[str] = []
    for file_path in files:
        names.append(str(file_path))
        rel_name = file_path.name
        text = _load_text(file_path, max_bytes)
        if "__skipped_large_file__" in text:
            skipped.append(str(file_path))
        chunk = f"\n__file__:{rel_name}\n__path__:{file_path}\n{text}\n"
        chunks.append(chunk)
        if _is_run_metadata(file_path):
            meta_names.append(str(file_path))
            meta_chunks.append(chunk)
    return (
        "\n".join(chunks).lower(),
        "\n".join(meta_chunks).lower(),
        names,
        meta_names,
        skipped,
    )


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
    text, metadata_text, files, metadata_files, skipped = _artifact_text(path, max_bytes)
    field_results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    near_misses: list[str] = []
    present: list[str] = []

    for rule in FIELD_RULES:
        scan_text = metadata_text if rule.scope == "run_metadata" else text
        matches = _match_patterns(scan_text, rule.patterns)
        near = _match_patterns(text, rule.near_patterns)
        subfield_results: dict[str, dict[str, Any]] = {}
        missing_subfields: list[str] = []
        if rule.subfields:
            for sub in rule.subfields:
                sub_matches = _match_patterns(scan_text, sub.patterns)
                subfield_results[sub.key] = {
                    "label": sub.label,
                    "state": "present" if sub_matches else "missing",
                    "matched_patterns": sub_matches,
                }
                if sub_matches:
                    matches = matches + sub_matches
                else:
                    missing_subfields.append(sub.key)
            satisfied = not missing_subfields
        else:
            satisfied = bool(matches)
        state = "present" if satisfied else "missing"
        if satisfied:
            present.append(rule.key)
        elif rule.required:
            missing.append(rule.key)
        if near and not satisfied:
            near_misses.append(rule.key)
        entry: dict[str, Any] = {
            "label": rule.label,
            "required": rule.required,
            "scope": rule.scope,
            "state": state,
            "matched_patterns": matches,
            "near_miss_patterns": near,
        }
        if rule.subfields:
            entry["subfields"] = subfield_results
            entry["missing_subfields"] = missing_subfields
        field_results[rule.key] = entry

    disqualifications = _kernel_provenance(text)
    status = "complete" if not missing else "incomplete"
    if disqualifications:
        # Never a silent downgrade: the reason travels with the verdict.
        recommendation = "retro_cert_disqualified"
    elif status == "complete":
        recommendation = "retro_cert_candidate"
    else:
        recommendation = "rerun_required"
    return {
        "artifact": str(path),
        "summary_status": _summary_status(path),
        "status": status,
        "recommendation": recommendation,
        "retro_cert_eligible": status == "complete" and not disqualifications,
        "disqualifications": disqualifications,
        "present_required_fields": present,
        "missing_required_fields": missing,
        "near_miss_fields": near_misses,
        "files_scanned_n": len(files),
        "files_scanned": files,
        "run_metadata_files": metadata_files,
        "files_skipped_large": skipped,
        "field_results": field_results,
    }


def audit_artifacts(paths: Iterable[Path], max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    artifacts = [audit_artifact(path, max_bytes=max_bytes) for path in paths]
    incomplete = [item for item in artifacts if item["status"] != "complete"]
    disqualified = [item for item in artifacts if item["disqualifications"]]
    return {
        "disqualified_artifacts": [item["artifact"] for item in disqualified],
        "disqualification_reasons": [
            {"artifact": item["artifact"], **record}
            for item in disqualified
            for record in item["disqualifications"]
        ],
        "schema": SCHEMA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "artifact-only; no inference, server, benchmark, build, or ROCm command executed",
        "policy": "draft P-GPU-1 mandatory-field audit",
        "status": "complete" if not incomplete else "incomplete",
        "recommendation": (
            "retro_cert_candidates_present"
            if not incomplete and not disqualified
            else "rerun_required_for_incomplete_artifacts"
        ),
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
    if report.get("disqualification_reasons"):
        lines.extend(["", "## Retro-certification disqualifications", ""])
        for record in report["disqualification_reasons"]:
            lines.append(f"- `{record['artifact']}` — {record['reason']} (evidence: `{record['evidence']}`)")
    lines.extend(
        [
            "",
            "## Field Semantics",
            "",
            "`binary_model_identity` is a CONJUNCTION: an artifact must independently record a",
            "LD_LIBRARY_PATH *value*, an enumerated backend/device list, the binary path, and the",
            "kernel commit. Those sub-fields are matched only against recorded run metadata",
            "(JSON/YAML receipts, recorded command transcripts) — a harness source file that merely",
            "mentions `LD_LIBRARY_PATH` is not evidence that the value was recorded.",
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
