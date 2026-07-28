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
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MODEL = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
EXPECTED_BINARY = "/mnt/raid0/llm/llama.cpp/build/bin/llama-bench"
EXPECTED_PROTOCOL = "bench_canonical.sh/canonical_recipe.py"
EXPECTED_METRIC = "llama-bench tg512 tokens_per_second"
EXPECTED_BINARY_SHA256 = "68e1d37c200ffc9d9a0bcfa4bc6985475486600a20331e6524323d393ba5edd1"
EXPECTED_MODEL_SHA256 = "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131"
EXPECTED_BENCH_CANONICAL_SHA256 = "68e7c738fa0e7da407574f750fc2fadaa385aacd990af581ed19a225ef1b3655"
EXPECTED_CANONICAL_RECIPE_SHA256 = "c6d37ef99a8c291266c8ab8f7b7bb4789837b05e077f56d2e5be6eb6595574d3"
EXPECTED_LLAMA_BRANCH = "production-consolidated-v8"
EXPECTED_LLAMA_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
EXPECTED_OLD_BASELINE_TPS = 24.3
EXPECTED_OLD_BENCHMARK_DATE = "2026-05-04"
REGISTRY_TARGETS = (
    "roles.frontdoor.performance.baseline_tps",
    "roles.frontdoor.performance.benchmark_date",
)
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
    "binary-ldd.txt",
    "launcher.log",
    "bench.log",
)
SHA256_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
CPU_TG512_HEADER = ("model", "size", "params", "backend", "threads", "fa", "mmap", "test", "t/s")
TPS_VALUE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)\s*±\s*([0-9]+(?:\.[0-9]+)?)$")


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
    if result["llama_branch"] != EXPECTED_LLAMA_BRANCH:
        raise EvidenceError(f"provenance llama_branch must be {EXPECTED_LLAMA_BRANCH!r}")
    if result["llama_commit"] != EXPECTED_LLAMA_COMMIT:
        raise EvidenceError(f"provenance llama_commit must be frozen v8 {EXPECTED_LLAMA_COMMIT}")
    if not re.fullmatch(r"[0-9a-f]{40}", result["research_commit"]):
        raise EvidenceError("provenance research_commit must be a full 40-hex commit")
    resolved = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--verify", f"{result['research_commit']}^{{commit}}"],
        capture_output=True,
        check=False,
        text=True,
    )
    if resolved.returncode != 0:
        raise EvidenceError("provenance research_commit does not exist in the research repository")
    if "T" not in result["started_at"] or "T" not in result["finished_at"]:
        raise EvidenceError("started_at and finished_at must be full ISO datetimes")
    try:
        started_at = datetime.fromisoformat(result["started_at"])
        finished_at = datetime.fromisoformat(result["finished_at"])
    except ValueError as exc:
        raise EvidenceError("started_at and finished_at must be full ISO datetimes") from exc
    if started_at.tzinfo is None or finished_at.tzinfo is None:
        raise EvidenceError("started_at and finished_at must include timezones")
    if finished_at < started_at:
        raise EvidenceError("finished_at must not precede started_at")
    return result


def _validate_region_status(directory: Path, name: str) -> None:
    try:
        payload = json.loads(_read_required(directory, name))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON in {name}") from exc
    if not isinstance(payload, list):
        raise EvidenceError(f"{name} must be a region-status list")
    regions: list[str] = []
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("region"), str) or not isinstance(row.get("global_held"), bool):
            raise EvidenceError(f"{name} has no valid global_held records")
        regions.append(row["region"])
        if row["global_held"]:
            raise EvidenceError(f"{name} records a held CPU region")
    if sorted(regions) != ["q0", "q1", "q2", "q3"] or len(set(regions)) != 4:
        raise EvidenceError(f"{name} must contain q0, q1, q2, and q3 exactly once")


def _option_value(tokens: list[str], option: str, expected: str) -> None:
    positions = [index for index, value in enumerate(tokens) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens) or tokens[positions[0] + 1] != expected:
        raise EvidenceError(f"canonical command requires {option} {expected}")


def _table_cells(line: str) -> list[str] | None:
    if not line.startswith("|") or not line.endswith("|"):
        return None
    return [cell.strip() for cell in line[1:-1].split("|")]


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
    for option, expected in (("-m", EXPECTED_MODEL), ("-t", "96"), ("-p", "0"), ("-n", "512"), ("-r", "2"), ("-fa", "1"), ("-mmp", "0")):
        _option_value(tokens, option, expected)
    try:
        env = dict(item.split("=", 1) for item in shlex.split(env_lines[0]))
    except ValueError as exc:
        raise EvidenceError("canonical Env line is not parseable") from exc
    if env.get("GGML_IQK") != "1" or env.get("GGML_IQK_Q8_0") != "1":
        raise EvidenceError("canonical environment requires GGML_IQK=1 and GGML_IQK_Q8_0=1")

    table_rows = [cells for line in log.splitlines() if (cells := _table_cells(line)) is not None]
    headers = [row for row in table_rows if tuple(row) == CPU_TG512_HEADER]
    if len(headers) != 1:
        raise EvidenceError("bench.log must contain exactly the current CPU tg512 table header")
    header_index = table_rows.index(headers[0])
    candidates = table_rows[header_index + 1:]
    matched = [row for row in candidates if len(row) == len(CPU_TG512_HEADER) and row[3] == "CPU" and row[4] == "96" and row[5] == "1" and row[6] == "0" and row[7] == "tg512"]
    if len(matched) != 1:
        raise EvidenceError("bench.log must contain exactly one successful CPU tg512 row")
    value = TPS_VALUE.fullmatch(matched[0][8])
    if value is None:
        raise EvidenceError("CPU tg512 t/s field must contain mean ± spread")
    return Metric(float(value.group(1)), float(value.group(2)))


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
    ldd = _read_required(directory, "binary-ldd.txt")
    if not ldd.strip() or "libc.so" not in ldd or "ld-linux" not in ldd or "not found" in ldd.lower():
        raise EvidenceError("binary-ldd.txt lacks resolved libc and dynamic-loader evidence")


def _validate_registry(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"cannot parse registry: {path}") from exc
    try:
        role = payload["roles"]["frontdoor"]
        performance = role["performance"]
    except (KeyError, TypeError) as exc:
        raise EvidenceError(f"registry lacks roles.frontdoor.performance: {path}") from exc
    try:
        frontdoor = payload["server_mode"]["frontdoor"]
        if role["model"]["path"] != EXPECTED_MODEL or frontdoor["model_path"] != EXPECTED_MODEL:
            raise EvidenceError(f"registry frontdoor model_path does not match FG-4b model: {path}")
        if performance["baseline_tps"] != EXPECTED_OLD_BASELINE_TPS or str(performance["benchmark_date"]) != EXPECTED_OLD_BENCHMARK_DATE:
            raise EvidenceError(f"registry lacks the reviewed FG-4b target state: {path}")
    except (KeyError, TypeError) as exc:
        raise EvidenceError(f"registry lacks server_mode.frontdoor.model_path: {path}") from exc
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_reviewed_digests(directory: Path) -> tuple[str, str, dict[str, str]]:
    binary_digest = _parse_sha256_file(directory, "binary.sha256", EXPECTED_BINARY)[EXPECTED_BINARY]
    model_digest = _parse_sha256_file(directory, "model.sha256", EXPECTED_MODEL)[EXPECTED_MODEL]
    instrument_digests = _parse_sha256_file(directory, "instrument.sha256")
    expected_instruments = {
        str(PROJECT_ROOT / "scripts/benchmark/bench_canonical.sh"): EXPECTED_BENCH_CANONICAL_SHA256,
        str(PROJECT_ROOT / "scripts/lib/canonical_recipe.py"): EXPECTED_CANONICAL_RECIPE_SHA256,
    }
    if binary_digest != EXPECTED_BINARY_SHA256 or model_digest != EXPECTED_MODEL_SHA256 or instrument_digests != expected_instruments:
        raise EvidenceError("artifact digest records do not match reviewed FG-4b identities")
    current = {
        Path(EXPECTED_BINARY): EXPECTED_BINARY_SHA256,
        PROJECT_ROOT / "scripts/benchmark/bench_canonical.sh": EXPECTED_BENCH_CANONICAL_SHA256,
        PROJECT_ROOT / "scripts/lib/canonical_recipe.py": EXPECTED_CANONICAL_RECIPE_SHA256,
    }
    for path, expected in current.items():
        if not path.is_file() or _sha256(path) != expected:
            raise EvidenceError(f"current reviewed input hash does not match: {path}")
    return binary_digest, model_digest, instrument_digests


def import_evidence(directory: Path, research_registry: Path, orchestrator_registry: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one completed artifact and return evidence plus a non-applying proposal."""
    directory = directory.resolve()
    _require_complete(directory)
    _validate_region_status(directory, "region-status-before.json")
    _validate_region_status(directory, "region-status-after.json")
    provenance = _parse_provenance(directory)
    binary_digest, model_digest, instrument_digests = _validate_reviewed_digests(directory)
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
        "date": datetime.fromisoformat(provenance["finished_at"]).date().isoformat(),
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
        "intended_registry_field_targets": list(REGISTRY_TARGETS),
        "preconditions": {
            "source_target_model_path": EXPECTED_MODEL,
            "reviewed_old_baseline_tps": EXPECTED_OLD_BASELINE_TPS,
            "reviewed_old_benchmark_date": EXPECTED_OLD_BENCHMARK_DATE,
            "do_not_modify": ["roles.frontdoor.performance.optimized_tps", "server_mode.frontdoor.throughput"],
        },
        "mirror_regeneration": "After review applies this source patch, regenerate the orchestrator mirror with stack_change_pipeline; this importer never invokes it.",
        "json_patch": [
            {"op": "test", "path": "/roles/frontdoor/performance/baseline_tps", "value": EXPECTED_OLD_BASELINE_TPS},
            {"op": "replace", "path": "/roles/frontdoor/performance/baseline_tps", "value": metric.mean_tokens_per_second},
            {"op": "test", "path": "/roles/frontdoor/performance/benchmark_date", "value": EXPECTED_OLD_BENCHMARK_DATE},
            {"op": "replace", "path": "/roles/frontdoor/performance/benchmark_date", "value": evidence["date"]},
        ],
    }
    return evidence, proposal


def _validate_output_path(path: Path, artifact: Path, registry_paths: set[Path]) -> Path:
    resolved = path.resolve()
    if resolved == artifact or artifact in resolved.parents:
        raise EvidenceError("refusing to write output inside the input artifact directory")
    if resolved in registry_paths:
        raise EvidenceError("refusing to write a registry path")
    return resolved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        artifact = args.artifact_dir.resolve()
        evidence_out = _validate_output_path(args.evidence_out, artifact, registry_paths)
        proposal_out = _validate_output_path(args.proposal_out, artifact, registry_paths)
        if evidence_out == proposal_out:
            raise EvidenceError("evidence and proposal output paths must differ")
        _write_json(evidence_out, evidence)
        _write_json(proposal_out, proposal)
    except EvidenceError as exc:
        print(f"FG-4b evidence import refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
