#!/usr/bin/env python3
"""Fail-closed importer for the FG-4b A4 CPU tg512 re-anchor artifact.

This tool is deliberately read-only with respect to the artifact and model
registries.  It emits evidence and a JSON Patch *proposal* only after every
required attestation agrees with the fixed FG-4b launcher contract.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MODEL = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
EXPECTED_BINARY = "/mnt/raid0/llm/llama.cpp/build/bin/llama-bench"
EXPECTED_PROTOCOL = "bench_canonical.sh/canonical_recipe.py"
EXPECTED_METRIC = "llama-bench tg512 tokens_per_second"
REGISTRY_TARGET = "server_mode.frontdoor.canonical_benchmark_observations.fg4b_a4_cpu_reanchor_20260728"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESEARCH_REGISTRY = PROJECT_ROOT / "orchestration/model_registry.yaml"
DEFAULT_ORCHESTRATOR_REGISTRY = Path("/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml")

REQUIRED_FILES = (
    "COMPLETE",
    "region-status-before.json",
    "region-status-after.json",
    "provenance.txt",
    "binary.sha256",
    "model.sha256",
    "instrument.sha256",
    "binary-version.txt",
    "launcher.log",
    "bench.log",
)
SHA256_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
TG512_LINE = re.compile(
    r"^\|.*?\|\s*[^|]+\|\s*0\s*\|\s*96\s*\|\s*0\s*\|\s*none\s*"
    r"\|\s*tg512\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*±\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*$"
)


class EvidenceError(ValueError):
    """Raised when an artifact cannot prove the fixed FG-4b contract."""


@dataclass(frozen=True)
class Metric:
    mean_tokens_per_second: float
    spread_tokens_per_second: float


def _read_required(directory: Path, name: str) -> str:
    path = directory / name
    if not path.is_file():
        raise EvidenceError(f"missing required artifact file: {name}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"required artifact file is not UTF-8 text: {name}") from exc


def _require_complete(directory: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise EvidenceError("artifact is incomplete; missing: " + ", ".join(missing))
    if (directory / "COMPLETE").read_text(encoding="utf-8").strip():
        raise EvidenceError("COMPLETE must be an empty success sentinel")


def _parse_sha256_file(directory: Path, name: str, expected_path: str | None = None) -> dict[str, str]:
    lines = [line for line in _read_required(directory, name).splitlines() if line.strip()]
    if not lines:
        raise EvidenceError(f"empty digest file: {name}")
    records: dict[str, str] = {}
    for line in lines:
        match = SHA256_LINE.fullmatch(line)
        if not match:
            raise EvidenceError(f"invalid sha256 record in {name}: {line!r}")
        digest, recorded_path = match.groups()
        if recorded_path in records:
            raise EvidenceError(f"duplicate digest path in {name}: {recorded_path}")
        records[recorded_path] = digest
    if expected_path is not None and set(records) != {expected_path}:
        raise EvidenceError(f"{name} must attest only {expected_path}")
    return records


def _parse_provenance(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _read_required(directory, "provenance.txt").splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise EvidenceError(f"invalid provenance record: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise EvidenceError(f"invalid or duplicate provenance key: {key!r}")
        result[key] = value
    required = {
        "protocol_id": EXPECTED_PROTOCOL,
        "metric": EXPECTED_METRIC,
        "n_gen": "512",
        "reps": "2",
        "model": EXPECTED_MODEL,
        "binary": EXPECTED_BINARY,
        "binary_version_exit_code": "1",
        "exit_code": "0",
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise EvidenceError(f"provenance {key!r} must be {expected!r}")
    for key in ("started_at", "finished_at", "research_commit", "llama_commit", "llama_branch"):
        if not result.get(key):
            raise EvidenceError(f"missing provenance {key!r}")
    try:
        date.fromisoformat(result["finished_at"][:10])
    except ValueError as exc:
        raise EvidenceError("finished_at must begin with an ISO date") from exc
    return result


def _validate_region_status(directory: Path, name: str) -> None:
    try:
        payload = json.loads(_read_required(directory, name))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON in {name}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvidenceError(f"{name} must be a non-empty region-status list")
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("global_held"), bool):
            raise EvidenceError(f"{name} has no valid global_held records")
        if row["global_held"]:
            raise EvidenceError(f"{name} records a held CPU region")


def _option_value(tokens: list[str], option: str, expected: str) -> None:
    positions = [index for index, value in enumerate(tokens) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens) or tokens[positions[0] + 1] != expected:
        raise EvidenceError(f"canonical command requires {option} {expected}")


def _validate_bench_log(directory: Path) -> Metric:
    log = _read_required(directory, "bench.log")
    command_lines = [line[5:].strip() for line in log.splitlines() if line.startswith("Cmd: ")]
    env_lines = [line[5:].strip() for line in log.splitlines() if line.startswith("Env: ")]
    if len(command_lines) != 1 or len(env_lines) != 1:
        raise EvidenceError("bench.log must contain exactly one canonical Cmd and Env line")
    try:
        tokens = shlex.split(command_lines[0])
    except ValueError as exc:
        raise EvidenceError("canonical Cmd line is not shell-parseable") from exc
    prefix = ["taskset", "-c", "0-95", "numactl", "--interleave=all", EXPECTED_BINARY]
    if tokens[: len(prefix)] != prefix:
        raise EvidenceError("canonical command must start with taskset 0-95 then numactl --interleave=all and the canonical binary")
    for option, expected in (("-t", "96"), ("-p", "0"), ("-n", "512"), ("-r", "2"), ("-fa", "1"), ("-mmp", "0")):
        _option_value(tokens, option, expected)
    try:
        env = dict(item.split("=", 1) for item in shlex.split(env_lines[0]))
    except ValueError as exc:
        raise EvidenceError("canonical Env line is not parseable") from exc
    if env.get("GGML_IQK") != "1" or env.get("GGML_IQK_Q8_0") != "1":
        raise EvidenceError("canonical environment requires GGML_IQK=1 and GGML_IQK_Q8_0=1")

    rows = [TG512_LINE.fullmatch(line) for line in log.splitlines()]
    matched = [row for row in rows if row is not None]
    if len(matched) != 1:
        raise EvidenceError("bench.log must contain exactly one successful CPU tg512 row")
    return Metric(float(matched[0].group(1)), float(matched[0].group(2)))


def _validate_launcher_attestations(directory: Path) -> None:
    launcher_log = _read_required(directory, "launcher.log")
    if "FG-4b A4 CPU re-anchor watcher started:" not in launcher_log:
        raise EvidenceError("launcher.log lacks the FG-4b A4 start record")
    if f"Evidence directory: {directory}" not in launcher_log:
        raise EvidenceError("launcher.log does not attest this artifact directory")
    if "FG-4b completed successfully:" not in launcher_log:
        raise EvidenceError("launcher.log lacks the FG-4b completion record")
    version = _read_required(directory, "binary-version.txt")
    if f"usage: {EXPECTED_BINARY}" not in version:
        raise EvidenceError("binary-version.txt does not attest the canonical binary")


def _validate_registry(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"cannot parse registry: {path}") from exc
    try:
        frontdoor = payload["server_mode"]["frontdoor"]
        if frontdoor["model_path"] != EXPECTED_MODEL:
            raise EvidenceError(f"registry frontdoor model_path does not match FG-4b model: {path}")
    except (KeyError, TypeError) as exc:
        raise EvidenceError(f"registry lacks server_mode.frontdoor.model_path: {path}") from exc
    return payload


def import_evidence(directory: Path, research_registry: Path, orchestrator_registry: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completed artifact and return evidence plus a non-applying proposal."""
    directory = directory.resolve()
    _require_complete(directory)
    _validate_region_status(directory, "region-status-before.json")
    _validate_region_status(directory, "region-status-after.json")
    provenance = _parse_provenance(directory)
    binary_digest = _parse_sha256_file(directory, "binary.sha256", EXPECTED_BINARY)[EXPECTED_BINARY]
    model_digest = _parse_sha256_file(directory, "model.sha256", EXPECTED_MODEL)[EXPECTED_MODEL]
    instrument_digests = _parse_sha256_file(directory, "instrument.sha256")
    expected_instruments = {
        str(PROJECT_ROOT / "scripts/benchmark/bench_canonical.sh"),
        str(PROJECT_ROOT / "scripts/lib/canonical_recipe.py"),
    }
    if set(instrument_digests) != expected_instruments:
        raise EvidenceError("instrument.sha256 must attest exactly bench_canonical.sh and canonical_recipe.py")
    _validate_launcher_attestations(directory)
    metric = _validate_bench_log(directory)
    _validate_registry(research_registry)
    _validate_registry(orchestrator_registry)

    evidence: dict[str, Any] = {
        "schema": "epyc.fg4b_a4_cpu_evidence.v1",
        "metric": "tg512_tokens_per_second",
        "protocol_id": provenance["protocol_id"],
        "n": 512,
        "reps": 2,
        "date": provenance["finished_at"][:10],
        "artifact": str(directory),
        "model": {"path": EXPECTED_MODEL, "sha256": model_digest},
        "binary": {"path": EXPECTED_BINARY, "sha256": binary_digest, "version_attestation": "binary-version.txt"},
        "instrument_sha256": instrument_digests,
        "mean_tokens_per_second": metric.mean_tokens_per_second,
        "spread_tokens_per_second": metric.spread_tokens_per_second,
        "canonical_settings": {
            "taskset": "0-95",
            "numactl": "--interleave=all",
            "threads": 96,
            "n_prompt": 0,
            "n_gen": 512,
            "reps": 2,
            "flash_attention": 1,
            "mmap": 0,
            "ggml_iqk": 1,
            "ggml_iqk_q8_0": 1,
        },
    }
    proposal = {
        "schema": "epyc.registry_patch_proposal.v1",
        "mode": "proposal_only",
        "must_not_apply_automatically": True,
        "source_registry": str(research_registry.resolve()),
        "mirror_registry_checked": str(orchestrator_registry.resolve()),
        "intended_registry_field_targets": [REGISTRY_TARGET],
        "preconditions": {
            "source_target_model_path": EXPECTED_MODEL,
            "operation_is_observation_addition": True,
            "do_not_modify_live_throughput": True,
        },
        "json_patch": [{
            "op": "add",
            "path": "/server_mode/frontdoor/canonical_benchmark_observations/fg4b_a4_cpu_reanchor_20260728",
            "value": evidence,
        }],
    }
    return evidence, proposal


def _write_json(path: Path, payload: dict[str, Any], artifact: Path, registry_paths: set[Path]) -> None:
    resolved = path.resolve()
    if resolved == artifact or artifact in resolved.parents:
        raise EvidenceError("refusing to write output inside the input artifact directory")
    if resolved in registry_paths:
        raise EvidenceError("refusing to write a registry path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--evidence-out", required=True, type=Path)
    parser.add_argument("--proposal-out", required=True, type=Path)
    parser.add_argument("--research-registry", type=Path, default=DEFAULT_RESEARCH_REGISTRY)
    parser.add_argument("--orchestrator-registry", type=Path, default=DEFAULT_ORCHESTRATOR_REGISTRY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evidence, proposal = import_evidence(args.artifact_dir, args.research_registry, args.orchestrator_registry)
        registry_paths = {args.research_registry.resolve(), args.orchestrator_registry.resolve()}
        _write_json(args.evidence_out, evidence, args.artifact_dir.resolve(), registry_paths)
        _write_json(args.proposal_out, proposal, args.artifact_dir.resolve(), registry_paths)
    except EvidenceError as exc:
        print(f"FG-4b evidence import refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
