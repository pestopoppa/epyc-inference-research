#!/usr/bin/env python3
from __future__ import annotations

"""Generate an offline AA-Omniscience benchmark prep manifest.

This script validates that the AA-Omniscience dataset adapter can be loaded,
then emits a per-role manifest and ready-to-run benchmark commands for later
live execution.

It does not start llama.cpp, the orchestrator, or any live inference stack.
"""

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BENCHMARK_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(BENCHMARK_DIR))

from dataset_adapters import ADAPTER_SUITES, get_adapter
from lib.executor import Executor
from lib.registry import load_registry
from suites import get_suites_for_role

SUITE_NAME = "omniscience"
RUN_BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark" / "run_benchmark.py"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "docs" / "data" / "aa_omniscience_measurement_manifest.json"
DEFAULT_COMMANDS_PATH = PROJECT_ROOT / "docs" / "data" / "aa_omniscience_measurement_commands.sh"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_adapter(sample_size: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if SUITE_NAME not in ADAPTER_SUITES:
        return {}, [f"{SUITE_NAME} is not registered in ADAPTER_SUITES"]

    adapter = get_adapter(SUITE_NAME)
    if adapter is None:
        return {}, [f"{SUITE_NAME} adapter could not be constructed"]

    try:
        total_available = adapter.total_available
    except Exception as exc:
        return {}, [f"{SUITE_NAME} dataset failed to load: {exc}"]

    if total_available <= 0:
        return {
            "suite": SUITE_NAME,
            "status": "empty",
            "total_available": 0,
            "sampled": 0,
            "sample_ids": [],
        }, [f"{SUITE_NAME} has no available questions"]

    actual_sample = min(sample_size, total_available)
    try:
        sample = adapter.sample(n=actual_sample, seed=42)
    except Exception as exc:
        return {}, [f"{SUITE_NAME} sampling failed: {exc}"]

    required_fields = {"id", "suite", "prompt", "expected", "scoring_method", "tier"}
    missing_fields: list[str] = []
    sample_ids: list[str] = []
    for idx, item in enumerate(sample):
        if missing := required_fields - set(item.keys()):
            missing_fields.append(f"sample[{idx}] missing {sorted(missing)}")
        if not item.get("prompt"):
            missing_fields.append(f"sample[{idx}] has empty prompt")
        if not item.get("suite") == SUITE_NAME:
            missing_fields.append(f"sample[{idx}] suite mismatch: {item.get('suite')!r}")
        sample_ids.append(str(item.get("id", "")))

    if missing_fields:
        errors.extend(missing_fields)

    return {
        "suite": SUITE_NAME,
        "status": "ok" if not errors else "warn",
        "total_available": total_available,
        "sampled": len(sample),
        "sample_ids": sample_ids,
        "sample_size_requested": sample_size,
    }, errors


def _build_role_entry(role: str, registry, executor: Executor) -> dict[str, Any] | None:
    suites = sorted(set(get_suites_for_role(role, registry)))
    if SUITE_NAME not in suites:
        return None

    architecture = registry.get_architecture(role)
    tier = registry.get_tier(role)
    model_path = registry.get_model_path(role)
    model_exists = bool(model_path and Path(model_path).exists())
    configs = executor.get_configs_for_architecture(architecture, role, registry)
    config_names = [cfg.name for cfg in configs]
    speed_only_configs = [cfg.name for cfg in configs if cfg.speed_test_only]
    live_command = shlex.join([
        sys.executable,
        str(RUN_BENCHMARK_SCRIPT),
        "--model",
        role,
        "--suite",
        SUITE_NAME,
        "--new-run",
    ])

    notes: list[str] = []
    if not model_exists:
        notes.append("model path missing")
    if not config_names:
        notes.append("no configs generated")

    return {
        "role": role,
        "tier": tier,
        "architecture": architecture,
        "model_path": model_path,
        "model_exists": model_exists,
        "suites": suites,
        "question_count": None,
        "config_names": config_names,
        "speed_only_configs": speed_only_configs,
        "command": live_command,
        "status": "ready" if model_exists and config_names else "blocked",
        "notes": notes,
    }


def build_manifest(sample_size: int = 8, role_filter: str | None = None) -> tuple[dict[str, Any], list[str]]:
    registry = load_registry()
    executor = Executor(registry, validate=False)

    dataset_validation, dataset_errors = _validate_adapter(sample_size)
    if not dataset_validation:
        dataset_validation = {"suite": SUITE_NAME, "status": "fail", "total_available": 0, "sampled": 0}

    roles: list[dict[str, Any]] = []
    validation_errors: list[str] = list(dataset_errors)

    for role in registry.get_all_roles(include_deprecated=False):
        if role_filter and role != role_filter:
            continue
        entry = _build_role_entry(role, registry, executor)
        if entry is None:
            continue
        roles.append(entry)
        if not entry["model_exists"]:
            validation_errors.append(f"{role}: model path missing")
        if not entry["config_names"]:
            validation_errors.append(f"{role}: no configs generated")

    roles.sort(key=lambda item: item["role"])
    for entry in roles:
        entry["question_count"] = dataset_validation.get("total_available", 0)

    commands = [
        {
            "role": entry["role"],
            "command": entry["command"],
            "status": entry["status"],
        }
        for entry in roles
    ]

    manifest = {
        "generated_at": _utc_now(),
        "suite": SUITE_NAME,
        "dataset_validation": dataset_validation,
        "roles": roles,
        "commands": commands,
        "summary": {
            "roles_total": len(roles),
            "roles_ready": sum(1 for entry in roles if entry["status"] == "ready"),
            "roles_blocked": sum(1 for entry in roles if entry["status"] != "ready"),
        },
    }
    return manifest, validation_errors


def _write_outputs(manifest: dict[str, Any], manifest_path: Path, commands_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    commands_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# Generated at {manifest['generated_at']}",
        f"# Suite: {manifest['suite']}",
        "",
    ]
    blocked: list[dict[str, Any]] = []
    for item in manifest.get("commands", []):
        lines.append(f"# {item['role']} [{item['status']}]")
        if item["status"] == "ready":
            lines.append(item["command"])
        else:
            blocked.append(item)
            lines.append(f"# blocked: {item['command']}")
        lines.append("")
    if blocked:
        lines.append("# Blocked roles are listed as comments above and in the JSON manifest.")
    commands_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_summary(manifest: dict[str, Any], errors: list[str]) -> None:
    validation = manifest.get("dataset_validation", {})
    print(f"Suite: {manifest['suite']}")
    print(f"Generated: {manifest['generated_at']}")
    print(f"Dataset status: {validation.get('status', 'unknown')}")
    print(f"Questions available: {validation.get('total_available', 0)}")
    print(f"Roles in manifest: {manifest.get('summary', {}).get('roles_total', 0)}")
    print(f"Ready roles: {manifest.get('summary', {}).get('roles_ready', 0)}")
    if errors:
        print("Validation issues:")
        for err in errors:
            print(f"  - {err}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an offline AA-Omniscience benchmark prep manifest"
    )
    parser.add_argument("--sample-size", type=int, default=8, help="Questions to sample for adapter validation")
    parser.add_argument("--role", help="Limit the manifest to a single role")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, help="Output JSON manifest path")
    parser.add_argument("--commands-path", type=Path, default=DEFAULT_COMMANDS_PATH, help="Output shell command list path")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; print the manifest summary")
    parser.add_argument("--validate", action="store_true", help="Exit non-zero if the manifest is not fully ready")
    args = parser.parse_args()

    manifest, errors = build_manifest(sample_size=args.sample_size, role_filter=args.role)
    _print_summary(manifest, errors)

    if not args.dry_run:
        _write_outputs(manifest, args.manifest_path, args.commands_path)
        print(f"\nWrote manifest: {args.manifest_path}")
        print(f"Wrote commands: {args.commands_path}")

    if args.validate and errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
