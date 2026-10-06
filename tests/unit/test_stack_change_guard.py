"""Tests for stack-prior drift validation."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import yaml

import pytest

import scripts.validate.stack_change_guard as stack_change_guard
from scripts.validate.stack_change_guard import (
    HARDCODED_SURFACE_RULES,
    RECOMPILE_PRIORS_COMMAND,
    REQUIRED_CONSUMER_SURFACE_IDS,
    RETIRED_LIVE_ROLE_FLOOR,
    ROLE_FACT_SURFACE_RULES,
    GuardResult,
    HardcodedSurfaceRule,
    RetiredRoleDerivationError,
    RoleFactSurfaceRule,
    derive_retired_live_roles,
    hardcoded_surface_rule_inventory,
    hardcoded_surface_warning_counts,
    main as stack_change_guard_main,
    scan_hardcoded_surfaces,
    scan_role_fact_surfaces,
    validate_surface_manifest,
    validate_stack_priors,
)
from src.registry.stack_priors import STACK_PRIORS_VERSION, stack_priors_contract


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def _synthetic_stack_manifest_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Make the two launch-helper controls independent of host RAM/kernel paths.

    stack_manifest reads /proc/meminfo once at import time. The captured GitHub
    runner has 15.61 GiB while the production capacity contract reserves 64 GiB,
    so these pure helper tests need a synthetic import-only value. The launch
    preview also resolves backend directories; empty synthetic backend directories
    keep that lookup independent of this runner's absent production kernel store.
    No binary is created or executed. The fixture primes the exact module under a
    narrowly patched Path.read_text, then removes its cached module/package
    attribute at teardown so synthetic inputs cannot leak to later tests. Every
    other path read delegates unchanged. The recipe separately records the
    runner's physical /proc/meminfo as context.
    """
    import importlib
    from src.registry import kernel_paths

    production_root = tmp_path / "synthetic-kernel-store" / "production"
    for backend in ("cpu", "gpu", "stt", "tts"):
        (production_root / backend).mkdir(parents=True)
    monkeypatch.setattr(kernel_paths, "PRODUCTION_ROOT", production_root)

    module_name = "scripts.server.stack_manifest"
    missing = object()
    cached_module = sys.modules.pop(module_name, missing)
    server_package = sys.modules.get("scripts.server")
    cached_attribute = (
        getattr(server_package, "stack_manifest", missing)
        if server_package is not None
        else missing
    )
    if server_package is not None and cached_attribute is not missing:
        delattr(server_package, "stack_manifest")

    original_read_text = Path.read_text

    def synthetic_meminfo(path: Path, *args, **kwargs) -> str:
        if path == Path("/proc/meminfo"):
            return "MemTotal: 1073741824 kB\n"
        return original_read_text(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_text", synthetic_meminfo)
            importlib.import_module(module_name)
        yield
    finally:
        sys.modules.pop(module_name, None)
        current_package = sys.modules.get("scripts.server")
        if current_package is not None and hasattr(current_package, "stack_manifest"):
            delattr(current_package, "stack_manifest")
        if server_package is not None and cached_attribute is not missing:
            setattr(server_package, "stack_manifest", cached_attribute)
        if cached_module is not missing:
            sys.modules[module_name] = cached_module


def _consumer_surface(surface_id: str = "unit_consumer") -> dict:
    return {
        "surface_id": surface_id,
        "classification": "typed_consumer",
        "owner": "stack-change-governance",
        "consumer_scope": "unit consumer surface",
        "source_of_truth": "generated stack priors",
        "promotion_blocker": True,
        "review_cadence": "every stack change",
        "validation_command": "uv run pytest -q tests/unit/test_stack_change_guard.py",
        "implementation_refs": ["src/unit.py"],
        "drift_response": "derive from generated truth",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _priors(registry: Path, descriptors: Path, *, memory_cost: float = 1.0) -> dict:
    stack_manifest = registry.parent / "stack_manifest.py"
    stack_manifest.write_text("ROLE_LAUNCH_META = {}\n", encoding="utf-8")
    stack_numa = registry.parent / "stack_numa.py"
    stack_numa.write_text("NUMA_CONFIG = {}\n", encoding="utf-8")
    orchestrator_stack = registry.parent / "orchestrator_stack.py"
    orchestrator_stack.write_text("# launcher\n", encoding="utf-8")
    stack_paths = registry.parent / "stack_paths.py"
    stack_paths.write_text("# paths\n", encoding="utf-8")
    stack_runtime = registry.parent / "stack_runtime.py"
    stack_runtime.write_text("# runtime\n", encoding="utf-8")
    return {
        "stack_priors_version": STACK_PRIORS_VERSION,
        "contract": stack_priors_contract(),
        "compiled_at": "2026-06-13T00:00:00Z",
        "status": "compiled",
        "coverage_scope": "test",
        "precedence_spec": "docs/reference/stack-truth-precedence.md",
        "source_artifacts": {
            "registry": {"path": str(registry), "sha256": _sha(registry)},
            "descriptors": {"path": str(descriptors), "sha256": _sha(descriptors)},
            "stack_manifest": {"path": str(stack_manifest), "sha256": _sha(stack_manifest)},
            "stack_numa": {"path": str(stack_numa), "sha256": _sha(stack_numa)},
            "orchestrator_stack": {
                "path": str(orchestrator_stack),
                "sha256": _sha(orchestrator_stack),
            },
            "stack_paths": {"path": str(stack_paths), "sha256": _sha(stack_paths)},
            "stack_runtime": {"path": str(stack_runtime), "sha256": _sha(stack_runtime)},
        },
        "roles": {
            "frontdoor": {
                "role": "frontdoor",
                "deployment_status": "live_stack",
                "status": "compiled",
                "model_id": "qwen",
                "display_name": "Qwen",
                "serving": {
                    "endpoint": "http://localhost:8070",
                    "server_role": "frontdoor",
                    "binding": "server_mode.direct",
                    "ports": [8070, 8080, 8180, 8280, 8380],
                    "slots": 1,
                    "tier": "hot",
                    "effective_context_tokens": 32768,
                    "binary": None,
                    "binary_dir": None,
                    "numa_policy": None,
                    "shared_mmap": False,
                    "launch": {
                        "entries": [
                            {
                                "port": 8070,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 0,
                            },
                            {
                                "port": 8080,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 1,
                            },
                            {
                                "port": 8180,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 2,
                            },
                            {
                                "port": 8280,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 3,
                            },
                            {
                                "port": 8380,
                                "primary_role": "frontdoor",
                                "mode": "default",
                                "alias": False,
                                "numa_instance": 4,
                            },
                        ],
                        "primary_roles": ["frontdoor"],
                        "modes": ["default"],
                        "requirements": {},
                        "runtime": {},
                    },
                },
                "priors": {
                    "throughput_tps": 24.3,
                    "quality_overall": 0.9,
                    "memory_cost": memory_cost,
                },
                "acceleration": {},
                "model": {},
                "evidence": {},
                "known_gaps": [],
            }
        },
        "known_global_gaps": {},
    }


def test_validate_stack_priors_accepts_fresh_live_artifact(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert result.ok
    assert result.warnings == []


def test_validate_stack_priors_rejects_stale_source_hash(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    registry.write_text("roles:\n  changed: true\n", encoding="utf-8")

    result = validate_stack_priors(priors)

    assert not result.ok
    assert any("hash mismatch" in error for error in result.errors)


def test_validate_stack_priors_rejects_stale_stack_manifest_hash(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    (tmp_path / "stack_manifest.py").write_text(
        "ROLE_LAUNCH_META = {'frontdoor': {'tier': 'warm'}}\n",
        encoding="utf-8",
    )

    result = validate_stack_priors(priors)

    assert not result.ok
    assert any("source_artifacts.stack_manifest hash mismatch" in error for error in result.errors)


def test_validate_stack_priors_rejects_retired_live_role(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["architect_coding"] = {
        "deployment_status": "live_stack",
        "model_id": "retired",
        "serving": {"endpoint": "http://localhost:8084", "tier": "hot"},
        "priors": {"memory_cost": 1.0},
        "known_gaps": [],
    }
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(priors)

    assert not result.ok
    assert "retired role 'architect_coding' appears as live_stack" in result.errors


def test_validate_stack_priors_rejects_missing_contract(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload.pop("contract")
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(priors)

    assert not result.ok
    assert "missing top-level stack-prior field: contract" in result.errors


def test_validate_stack_priors_rejects_hot_memory_penalty(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(
        tmp_path / "stack_priors.yaml",
        _priors(registry, descriptors, memory_cost=2.0),
    )

    result = validate_stack_priors(priors)

    assert not result.ok
    assert any("live HOT role 'frontdoor'" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_endpoint_drift(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["endpoint"] = "http://localhost:9999"
    payload["roles"]["frontdoor"]["serving"]["ports"] = [9999]
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert not result.ok
    assert any("serving.endpoint port 9999" in error for error in result.errors)
    assert any("does not include launch manifest port 8070" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_tier_drift(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "warm"}},
    )

    assert not result.ok
    assert any("serving.tier 'hot'" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_context_drift(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["effective_context_tokens"] = 8192
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "tier": "hot",
                "effective_context_tokens": 32768,
            }
        },
    )

    assert not result.ok
    assert any("serving.effective_context_tokens 8192" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_port_set_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["ports"] = [8070, 9999]
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "ports": [8070, 8080], "tier": "hot"}
        },
    )

    assert not result.ok
    assert any("missing launch manifest port(s) [8080]" in error for error in result.errors)
    assert any("include non-launch port(s) [9999]" in error for error in result.errors)


def test_validate_stack_priors_alias_uses_manifest_primary_without_registry_hint(
    tmp_path: Path,
) -> None:
    """A pure manifest alias inherits its primary's full serving fleet."""
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    alias = copy.deepcopy(payload["roles"]["frontdoor"])
    alias["role"] = "coder_escalation"
    alias["serving"]["server_role"] = "coder_escalation"
    alias["serving"]["launch"]["entries"] = [
        {
            "port": 8080,
            "primary_role": "frontdoor",
            "mode": "default",
            "alias": True,
            "numa_instance": 1,
        }
    ]
    payload["roles"]["coder_escalation"] = alias
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    host_target = {
        "port": 8070,
        "ports": [8070, 8080, 8180, 8280, 8380],
        "tier": "hot",
    }
    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": host_target,
            "coder_escalation": {
                "port": 8080,
                "ports": [8080],
                "tier": "hot",
                "launch_entries": [
                    {
                        "port": 8080,
                        "primary_role": "frontdoor",
                        "mode": "default",
                        "alias": True,
                        "numa_instance": 1,
                    }
                ],
            },
        },
    )

    assert result.ok, result.errors


def test_validate_stack_priors_rejects_launch_manifest_entry_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["launch"]["entries"] = [
        {"port": 8070, "primary_role": "frontdoor", "mode": "worker_pool", "alias": False}
    ]
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_entries": [
                    {
                        "port": 8070,
                        "primary_role": "frontdoor",
                        "mode": "default",
                        "alias": False,
                    }
                ],
            }
        },
    )

    assert not result.ok
    assert any("serving.launch.entries do not match" in error for error in result.errors)


def test_validate_stack_priors_rejects_launch_manifest_requirement_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["launch"]["requirements"] = {
        "model_path": "/stale/model.gguf"
    }
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_requirements": {
                    "model_path": "/current/model.gguf",
                    "mmproj_path": "/current/mmproj.gguf",
                },
            }
        },
    )

    assert not result.ok
    assert any("serving.launch.requirements do not match" in error for error in result.errors)
    assert any("mmproj_path" in error for error in result.errors)


def test_launch_manifest_targets_prefer_server_mode_launch_requirement_paths(
    tmp_path: Path,
) -> None:
    # `_launch_manifest_targets` derives its TARGETS and launch entries from the
    # LIVE launcher view; only the overrides come from the handed registry. The tmp
    # files made this test LOOK hermetic while it silently depended on one lineup:
    # `worker_general` was a worker_pool/explore launch target in 2026-06 and is an
    # ALIAS on another host role today, and an alias correctly compiles a DISABLED
    # spec (it inherits its host's draft rather than launching one). The role is
    # discovered from the same view the function uses, by SHAPE — its own
    # primary_role, not an alias — and every path below is a fixture literal that
    # appears nowhere in the lineup.
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    probe = _write_yaml(tmp_path / "probe.yaml", {"server_mode": {}, "roles": {}})
    probe_targets = stack_change_guard._launch_manifest_targets(
        registry_path=probe,
        descriptor_path=descriptors,
    )

    def _first_entry(target: dict) -> dict:
        entries = target.get("launch_entries") or [{}]
        return entries[0]

    host_role = next(
        (
            role
            for role, target in probe_targets.items()
            if _first_entry(target).get("mode") == "default"
            and not _first_entry(target).get("alias")
        ),
        None,
    )
    assert host_role, "the launch view declares at least one non-alias default-mode host role"

    registry = _write_yaml(
        tmp_path / "registry.yaml",
        {
            "server_mode": {
                # Keyed under a name that is NOT the launch role, so the
                # `model_role` binding step stays under test too.
                "fixture_server": {
                    "model_role": host_role,
                    "model_path": "/models/fixture-base.gguf",
                    "draft_model_path": "/models/fixture-draft.gguf",
                    "acceleration": {"spec_type": "draft-mtp", "draft_max": 4},
                }
            },
            "roles": {},
        },
    )

    targets = stack_change_guard._launch_manifest_targets(
        registry_path=registry,
        descriptor_path=descriptors,
    )

    requirements = targets[host_role]["launch_requirements"]
    assert requirements["model_path"] == "/models/fixture-base.gguf"
    assert requirements["draft_model_path"] == "/models/fixture-draft.gguf"
    assert (
        targets[host_role]["launch_runtime"]["flags"]["spec"]["draft_model_path"]
        == "/models/fixture-draft.gguf"
    )


def test_validate_stack_priors_rejects_launch_manifest_runtime_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["serving"]["launch"]["runtime"] = {
        "binary_family": "llama.cpp",
        "binary_path": "/stale/llama-server",
        "cache": {"kv_type_k": "q8_0", "kv_type_v": "q8_0", "slots": 1},
        "flags": {"spec": {"enabled": False}},
    }
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_runtime": {
                    "binary_family": "llama.cpp",
                    "binary_path": "/current/llama-server",
                    "cache": {"kv_type_k": "q8_0", "kv_type_v": "q8_0", "slots": 1},
                    "flags": {"spec": {"enabled": False}},
                },
            }
        },
    )

    assert not result.ok
    assert any("serving.launch.runtime does not match" in error for error in result.errors)
    assert any("/current/llama-server" in error for error in result.errors)


def test_validate_stack_priors_rejects_unclassified_launch_target(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "new_live_role": {
                "port": 9000,
                "ports": [9000],
                "tier": "hot",
                "launch_entries": [
                    {
                        "port": 9000,
                        "primary_role": "new_live_role",
                        "mode": "default",
                        "alias": False,
                    }
                ],
            },
        },
    )

    assert not result.ok
    assert (
        "launch manifest target 'new_live_role' has no generated stack-prior role record"
        in result.errors
    )


def test_validate_stack_priors_allows_partial_explicit_role_coverage(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["coverage_scope"] = "explicit_active_roles"
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "worker_general": {
                "port": 8072,
                "ports": [8072],
                "tier": "hot",
                "launch_entries": [
                    {
                        "port": 8072,
                        "primary_role": "worker_general",
                        "mode": "worker_pool",
                        "alias": False,
                    }
                ],
            },
        },
    )

    assert result.ok


def test_validate_stack_priors_allows_manifest_owned_embedding_targets(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "embedder_granite_97m_r2": {
                "port": 8096,
                "ports": [8096],
                "tier": "warm",
                "launch_entries": [
                    {
                        "port": 8096,
                        "primary_role": "embedder_granite_97m_r2",
                        "mode": "embedding",
                        "alias": False,
                    }
                ],
            },
        },
    )

    assert result.ok


def test_validate_stack_priors_allows_manifest_owned_eval_batch_frontdoor_targets(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "eval_batch_frontdoor": {
                "port": 18070,
                "ports": [18070],
                "tier": "warm",
                "launch_entries": [
                    {
                        "port": 18070,
                        "primary_role": "eval_batch_frontdoor",
                        "mode": "default",
                        "alias": False,
                    }
                ],
            },
        },
    )

    assert result.ok


def test_validate_stack_priors_allows_alias_targets_covered_by_live_primary(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    result = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "frontdoor_legacy_alias": {
                "port": 8070,
                "ports": [8070],
                "tier": "hot",
                "launch_entries": [
                    {
                        "port": 8070,
                        "primary_role": "frontdoor",
                        "mode": "default",
                        "alias": True,
                    }
                ],
            },
        },
    )

    assert result.ok


def test_validate_stack_priors_allows_only_explicit_worker_fast_auxiliary(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))

    allowed = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "worker_fast": {
                "port": 8102,
                "ports": [8102],
                "tier": "warm",
                "launch_entries": [
                    {
                        "port": 8102,
                        "primary_role": "worker_fast",
                        "mode": "worker_pool",
                        "alias": False,
                        "worker_type": "fast",
                    }
                ],
            },
        },
    )
    rejected = validate_stack_priors(
        priors,
        launch_manifest_targets={
            "frontdoor": {"port": 8070, "tier": "hot"},
            "worker_shadow": {
                "port": 8103,
                "ports": [8103],
                "tier": "warm",
                "launch_entries": [
                    {
                        "port": 8103,
                        "primary_role": "worker_shadow",
                        "mode": "worker_pool",
                        "alias": False,
                        "worker_type": "shadow",
                    }
                ],
            },
        },
    )

    assert allowed.ok
    assert not rejected.ok
    assert (
        "launch manifest target 'worker_shadow' has no generated stack-prior role record"
        in rejected.errors
    )


def test_validate_stack_priors_strict_fails_on_known_gaps(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["known_gaps"] = ["missing ctx"]
    payload["known_global_gaps"] = {"frontdoor": ["missing ctx"]}
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)

    launch_targets = {"frontdoor": {"port": 8070, "tier": "hot"}}
    loose = validate_stack_priors(priors, launch_manifest_targets=launch_targets)
    strict = validate_stack_priors(priors, strict=True, launch_manifest_targets=launch_targets)

    assert loose.ok
    assert loose.warnings
    assert not strict.ok
    assert any(error.startswith("strict:") for error in strict.errors)


def test_scan_hardcoded_surfaces_classifies_curated_surfaces(tmp_path: Path) -> None:
    active = tmp_path / "scripts" / "benchmark"
    tests = tmp_path / "tests" / "unit"
    docs = tmp_path / "docs"
    active.mkdir(parents=True)
    tests.mkdir(parents=True)
    docs.mkdir(parents=True)
    (active / "seeding_types.py").write_text(
        'DEFAULT_ROLES = ["architect_coding"]\n',
        encoding="utf-8",
    )
    (tests / "test_legacy.py").write_text(
        'def test_legacy():\n    assert "architect_coding"\n',
        encoding="utf-8",
    )
    (docs / "ARCHITECTURE.md").write_text(
        "Historical chain: architect_coding\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    categories = {finding.category for finding in findings}
    assert "production_blocker" in categories
    assert "legacy_test" in categories
    assert "historical_doc" in categories
    assert any(finding.path.as_posix() == "scripts/benchmark/seeding_types.py" for finding in findings)


def test_hardcoded_surface_rule_inventory_is_machine_readable() -> None:
    inventory = hardcoded_surface_rule_inventory()
    rule_ids = [rule["rule_id"] for rule in inventory["rules"]]

    assert inventory["version"] == 1
    assert inventory["rule_count"] == len(HARDCODED_SURFACE_RULES)
    assert inventory["consumer_surface_count"] == 0
    assert len(rule_ids) == len(set(rule_ids))
    assert "production_blocker" in inventory["categories"]
    assert any(
        rule["rule_id"] == "seeding_baseline_tps_table"
        and rule["category"] == "production_blocker"
        and "scripts/benchmark/seeding_rewards.py" in rule["path_globs"]
        for rule in inventory["rules"]
    )
    assert all("ownership" in rule for rule in inventory["rules"])


def test_hardcoded_surface_rule_inventory_includes_consumer_surfaces() -> None:
    inventory = hardcoded_surface_rule_inventory(
        ownership_manifest={
            "consumer_surfaces": [
                _consumer_surface("q_scorer_priors"),
                _consumer_surface("runtime_attestation"),
            ]
        }
    )

    assert inventory["consumer_surface_count"] == 2
    assert [surface["surface_id"] for surface in inventory["consumer_surfaces"]] == [
        "q_scorer_priors",
        "runtime_attestation",
    ]
    assert all(
        surface["classification"] == "typed_consumer"
        for surface in inventory["consumer_surfaces"]
    )


def test_validate_surface_manifest_accepts_complete_rule_ownership(tmp_path: Path) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit test surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "uv run python scripts/validate/stack_change_guard.py",
                    "drift_response": "remove stale hardcode",
                }
            ],
        },
    )

    assert validate_surface_manifest(manifest, rules=(rule,)) == []


def test_validate_surface_manifest_accepts_consumer_surface_ownership(tmp_path: Path) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit scanner surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "uv run python scripts/validate/stack_change_guard.py",
                    "drift_response": "remove stale hardcode",
                }
            ],
            "consumer_surfaces": [_consumer_surface()],
        },
    )

    assert validate_surface_manifest(
        manifest,
        rules=(rule,),
        required_consumer_surface_ids=frozenset({"unit_consumer"}),
    ) == []


def test_validate_surface_manifest_rejects_missing_consumer_surface(
    tmp_path: Path,
) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit scanner surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard",
                    "drift_response": "remove stale hardcode",
                }
            ],
        },
    )

    errors = validate_surface_manifest(
        manifest,
        rules=(rule,),
        required_consumer_surface_ids=frozenset({"unit_consumer"}),
    )

    assert any("'consumer_surfaces'" in error for error in errors)


def test_validate_surface_manifest_rejects_consumer_surface_metadata_drift(
    tmp_path: Path,
) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    bad_surface = {
        **_consumer_surface(),
        "classification": "unknown",
        "promotion_blocker": "yes",
        "implementation_refs": [],
    }
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit scanner surface",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard",
                    "drift_response": "remove stale hardcode",
                }
            ],
            "consumer_surfaces": [bad_surface, _consumer_surface()],
        },
    )

    errors = validate_surface_manifest(
        manifest,
        rules=(rule,),
        required_consumer_surface_ids=frozenset({"unit_consumer"}),
    )

    assert any("classification 'unknown'" in error for error in errors)
    assert any("missing boolean 'promotion_blocker'" in error for error in errors)
    assert any("implementation_refs" in error for error in errors)
    assert any("surface_id 'unit_consumer' is duplicated" in error for error in errors)


def test_validate_surface_manifest_rejects_missing_rule_ownership(tmp_path: Path) -> None:
    rules = (
        HardcodedSurfaceRule(
            rule_id="covered_rule",
            category="historical_doc",
            pattern=r"\bold\b",
            path_globs=("docs/**/*.md",),
            remediation="label historical",
        ),
        HardcodedSurfaceRule(
            rule_id="missing_rule",
            category="legacy_test",
            pattern=r"\bold\b",
            path_globs=("tests/**/*.py",),
            remediation="migrate fixture",
        ),
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "covered_rule",
                    "category": "historical_doc",
                    "owner": "documentation-governance",
                    "consumer_scope": "docs",
                    "promotion_blocker": False,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard --all-hardcoded-surfaces",
                    "drift_response": "label historical",
                }
            ],
        },
    )

    errors = validate_surface_manifest(manifest, rules=rules)

    assert any("missing_rule" in error for error in errors)


def test_validate_surface_manifest_rejects_category_policy_drift(tmp_path: Path) -> None:
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )
    manifest = _write_yaml(
        tmp_path / "surface_manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "legacy_test",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit test surface",
                    "promotion_blocker": False,
                    "review_cadence": "every stack change",
                    "evidence_command": "guard",
                    "drift_response": "remove stale hardcode",
                }
            ],
        },
    )

    errors = validate_surface_manifest(manifest, rules=(rule,))

    assert any("category 'legacy_test'" in error for error in errors)
    assert any("promotion_blocker=False" in error for error in errors)


def test_hardcoded_surface_warning_counts_bucket_unique_warnings() -> None:
    warnings = [
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.waived.production_blocker.retired_role_in_active_code: file:2",
        "hardcoded_surface.legacy_test.retired_role_in_tests: file:3",
        "non-surface warning",
    ]

    assert hardcoded_surface_warning_counts(warnings) == {
        "production_blocker": 1,
        "waived_production_blocker": 1,
        "legacy_test": 1,
    }


def test_stack_change_guard_can_print_surface_rule_inventory_json(capsys) -> None:
    rc = stack_change_guard_main(
        ["--list-hardcoded-surface-rules", "--surface-inventory-format", "json"]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rule_count"] == len(HARDCODED_SURFACE_RULES)
    assert payload["consumer_surface_count"] == len(REQUIRED_CONSUMER_SURFACE_IDS)
    assert set(surface["surface_id"] for surface in payload["consumer_surfaces"]) == set(
        REQUIRED_CONSUMER_SURFACE_IDS
    )
    assert any(
        rule["rule_id"] == "stale_autopilot_program_stack_guidance"
        and rule["ownership"]["owner"] == "autopilot-instrumentation"
        for rule in payload["rules"]
    )


def test_stack_change_guard_can_print_summary_only(monkeypatch, capsys) -> None:
    warnings = [
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.production_blocker.seeding_baseline_tps_table: file:1",
        "hardcoded_surface.waived.production_blocker.retired_role_in_active_code: file:2",
        "hardcoded_surface.historical_doc.retired_role_in_operator_docs: docs:1",
        "non-surface warning",
    ]

    def fake_validate_stack_priors(*args, **kwargs) -> GuardResult:
        return GuardResult(errors=[], warnings=warnings)

    monkeypatch.setattr(
        stack_change_guard,
        "validate_stack_priors",
        fake_validate_stack_priors,
    )

    rc = stack_change_guard_main(["--surface-summary-only"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "WARN: 4 unique stack-prior warning(s) (5 total)" in output
    assert (
        "surface_warnings: production_blocker=1, "
        "waived_production_blocker=1, historical_doc=1"
    ) in output
    assert "other_warnings: 1" in output
    assert "file:1" not in output


def test_scan_hardcoded_surfaces_flags_retired_launch_env_var(tmp_path: Path) -> None:
    server = tmp_path / "scripts" / "server"
    server.mkdir(parents=True)
    (server / "orchestrator_stack.py").write_text(
        'env["ORCHESTRATOR_LANGGRAPH_ARCHITECT_CODING"] = "1"\n',
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "retired_role_env_flag"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/server/orchestrator_stack.py"
        for finding in findings
    )

def test_scan_hardcoded_surfaces_flags_retired_source_access_role(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "source_registry.yaml").write_text(
        "role_access:\n  architect_coding:\n    enabled: true\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "retired_role_in_source_access"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "orchestration/source_registry.yaml"
        for finding in findings
    )

def test_scan_hardcoded_surfaces_flags_stale_autopilot_program_guidance(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
**Target ports**:
- coder: 8071
- architect_coding: 8084
- NOTE: Entire stack fits in HOT tier with mlock on 512GB RAM. WARM tier demotion is unnecessary.
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_autopilot_program_stack_guidance"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/program.md"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_stale_autopilot_program_model_claims(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
- **Q-scorer frontdoor throughput**: Currently uses 19.6 t/s but actual is 12.7 t/s.
try-cheap-first (Qwen3-Coder-30B-A3B, fastest)
  -> frontdoor (Qwen3.5-35B-A3B, quality gate)
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_autopilot_program_stack_guidance"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/program.md"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_autopilot_program_local_yaml_reader(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
```python
import yaml
data = yaml.safe_load(open("orchestration/derived/stack_priors.yaml")) or {}
```
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_autopilot_program_stack_guidance"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/program.md"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_stack_prior_autopilot_program_guidance(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "program.md").write_text(
        """
**Target endpoints**: derive live primary endpoints from `orchestration/derived/stack_priors.yaml`.
Tier demotion is not an open exploration surface by default.
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "stale_autopilot_program_stack_guidance"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_cli_degraded_status_targets(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "cli_orch.py").write_text(
        "FALLBACK_STATUS_TARGETS = [('frontdoor/coder_escalation', 8070)]\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_cli_degraded_status_targets"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/cli_orch.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_cli_status_excluded_roles(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "cli_orch.py").write_text(
        'FALLBACK_STATUS_EXCLUDED_ROLES = frozenset({"embedder"})\n',
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_cli_status_excluded_roles"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/cli_orch.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_autopilot_preflight_targets(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "preflight_audit.py").write_text(
        "FALLBACK_MODEL_SERVER_TARGETS = [('frontdoor', 'http://localhost:8070/health')]\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_autopilot_preflight_targets"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/preflight_audit.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_autopilot_preflight_excluded_roles(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "preflight_audit.py").write_text(
        'FALLBACK_MODEL_SERVER_EXCLUDED_ROLES = frozenset({"embedder"})\n',
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_autopilot_preflight_excluded_roles"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/preflight_audit.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_stale_corpus_quality_gate_models(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "scripts" / "benchmark"
    benchmark.mkdir(parents=True)
    (benchmark / "corpus_quality_gate.py").write_text(
        """
FALLBACK_MODELS = {
    "frontdoor": {"port": 8070, "name": "Qwen3.6-35B-A3B Q8_0"},
}
parser.add_argument("--models", default=["7b", "32b"])
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_corpus_quality_gate_models"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/benchmark/corpus_quality_gate.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_local_config_stack_prior_reader(
    tmp_path: Path,
) -> None:
    config = tmp_path / "src" / "config"
    config.mkdir(parents=True)
    (config / "models.py").write_text(
        """
import yaml

payload = yaml.safe_load(priors_path.read_text(encoding="utf-8")) or {}
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "local_config_stack_prior_yaml_reader"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/config/models.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_local_q_scorer_stack_prior_reader(
    tmp_path: Path,
) -> None:
    repl_memory = tmp_path / "orchestration" / "repl_memory"
    repl_memory.mkdir(parents=True)
    (repl_memory / "q_scorer.py").write_text(
        """
import yaml

data = yaml.safe_load(stack_priors_path.read_text(encoding="utf-8")) or {}
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "local_q_scorer_stack_prior_yaml_reader"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "orchestration/repl_memory/q_scorer.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_local_generated_docs_stack_prior_reader(
    tmp_path: Path,
) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    registry = tmp_path / "scripts" / "registry"
    autopilot.mkdir(parents=True)
    registry.mkdir(parents=True)
    (autopilot / "gen_system_card.py").write_text(
        'stack_priors = _load_yaml(root_path / "orchestration" / "derived" / "stack_priors.yaml")\n',
        encoding="utf-8",
    )
    (registry / "render_stack_summary.py").write_text(
        "stack_priors = load_yaml(stack_priors_path)\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    finding_paths = {
        finding.path.as_posix()
        for finding in findings
        if finding.rule_id == "local_generated_docs_stack_prior_yaml_reader"
        and finding.category == "production_blocker"
    }
    assert finding_paths == {
        "scripts/autopilot/gen_system_card.py",
        "scripts/registry/render_stack_summary.py",
    }


def test_scan_hardcoded_surfaces_flags_static_factual_risk_role_tiers(
    tmp_path: Path,
) -> None:
    classifier = tmp_path / "src" / "classifiers"
    classifier.mkdir(parents=True)
    (classifier / "factual_risk.py").write_text(
        "_ROLE_TO_TIER: dict[str, str] = {'frontdoor': 'tier_3'}\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_factual_risk_role_tiers"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/classifiers/factual_risk.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_openai_model_role_order(
    tmp_path: Path,
) -> None:
    route = tmp_path / "src" / "api" / "routes"
    route.mkdir(parents=True)
    (route / "openai_compat.py").write_text(
        "PREFERRED_ROLE_ORDER = ('frontdoor', 'coder_escalation')\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_openai_model_role_order"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/api/routes/openai_compat.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_chat_routing_prior_roles(
    tmp_path: Path,
) -> None:
    route = tmp_path / "src" / "api" / "routes"
    route.mkdir(parents=True)
    (route / "chat_routing.py").write_text(
        """
_HEURISTIC_PRIOR_ROLE_CANDIDATES = (
    "frontdoor",
    "worker_general",
)
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_chat_routing_heuristic_prior_roles"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/api/routes/chat_routing.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_inference_lock_role_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_lock.py").write_text(
        """
HEAVY_ROLES: frozenset[str] = frozenset({"frontdoor", "architect_general"})
LIGHT_ROLES = frozenset({"worker_general"})
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_inference_lock_role_policy"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/runtime/inference_lock.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_derived_inference_lock_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_lock.py").write_text(
        """
_LEGACY_HEAVY_ROLES = frozenset({"frontdoor", "architect_general"})
_LEGACY_LIGHT_ROLES = frozenset({"worker_general"})
_DERIVED_LOCK_ROLES = _lock_roles_from_stack_priors()
if _DERIVED_LOCK_ROLES is None:
    HEAVY_ROLES = _LEGACY_HEAVY_ROLES
    LIGHT_ROLES = _LEGACY_LIGHT_ROLES
else:
    HEAVY_ROLES, LIGHT_ROLES = _DERIVED_LOCK_ROLES
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "static_inference_lock_role_policy"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_inference_tap_stream_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_tap.py").write_text(
        'SAFE_NON_STREAM_ROLES: frozenset[str] = frozenset({"architect_general"})\n',
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_inference_tap_stream_policy"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "src/runtime/inference_tap.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_derived_inference_tap_stream_policy(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "src" / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inference_tap.py").write_text(
        """
_LEGACY_SAFE_NON_STREAM_ROLES: frozenset[str] = frozenset({"architect_general"})
_DERIVED_SAFE_NON_STREAM_ROLES = _safe_non_stream_roles_from_stack_priors()
SAFE_NON_STREAM_ROLES: frozenset[str] = (
    _LEGACY_SAFE_NON_STREAM_ROLES
    if _DERIVED_SAFE_NON_STREAM_ROLES is None
    else _DERIVED_SAFE_NON_STREAM_ROLES
)
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "static_inference_tap_stream_policy"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_static_autopilot_kv_port_map(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "kv_compress.py").write_text(
        "PRODUCTION_PORTS = {'frontdoor': 8070}\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "static_autopilot_kv_production_ports"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/autopilot/kv_compress.py"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_kv_degraded_fallback_port_map(tmp_path: Path) -> None:
    autopilot = tmp_path / "scripts" / "autopilot"
    autopilot.mkdir(parents=True)
    (autopilot / "kv_compress.py").write_text(
        "_FALLBACK_PRODUCTION_PORTS = {'frontdoor': 8070}\n"
        "PRODUCTION_PORTS = _FALLBACK_PRODUCTION_PORTS\n",
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "static_autopilot_kv_production_ports"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_flags_stale_launch_wrapper_inventory(tmp_path: Path) -> None:
    server = tmp_path / "scripts" / "server"
    server.mkdir(parents=True)
    (server / "launch_production.sh").write_text(
        """
echo "  - architect_coding (8084): Qwen3-Coder-480B-A35B"
echo "RAM breakdown (full mode):"
echo "  - Total: ~535GB"
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert any(
        finding.rule_id == "stale_launch_wrapper_static_inventory"
        and finding.category == "production_blocker"
        and finding.path.as_posix() == "scripts/server/launch_production.sh"
        for finding in findings
    )


def test_scan_hardcoded_surfaces_allows_manifest_derived_launch_wrapper(tmp_path: Path) -> None:
    server = tmp_path / "scripts" / "server"
    server.mkdir(parents=True)
    (server / "launch_production.sh").write_text(
        """
echo "Mode: FULL production stack (manifest-defined HOT tier)"
echo "Launch inventory comes from scripts/server/stack_manifest.py."
echo "Use --status after launch for live processes, ports, and residency."
""".lstrip(),
        encoding="utf-8",
    )

    findings = scan_hardcoded_surfaces(tmp_path)

    assert not any(
        finding.rule_id == "stale_launch_wrapper_static_inventory"
        for finding in findings
    )


def test_validate_stack_priors_can_include_surface_warnings(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    active = tmp_path / "scripts" / "benchmark"
    active.mkdir(parents=True)
    (active / "seeding_rewards.py").write_text(
        'DEFAULT_BASELINE_TPS = {"frontdoor": 10.3}\n',
        encoding="utf-8",
    )

    launch_targets = {"frontdoor": {"port": 8070, "tier": "hot"}}
    loose = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        launch_manifest_targets=launch_targets,
    )
    strict = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
        launch_manifest_targets=launch_targets,
    )

    assert loose.ok
    assert any("hardcoded_surface.production_blocker" in warning for warning in loose.warnings)
    assert not strict.ok
    assert any("hardcoded_surface.production_blocker" in error for error in strict.errors)


def test_validate_stack_priors_keeps_waived_surface_visible_in_strict_mode(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    active = tmp_path / "scripts" / "benchmark"
    active.mkdir(parents=True)
    (active / "seeding_rewards.py").write_text(
        'DEFAULT_BASELINE_TPS = {"frontdoor": 10.3}\n',
        encoding="utf-8",
    )
    exceptions = _write_yaml(
        tmp_path / "exceptions.yaml",
        {
            "exceptions": [
                {
                    "rule_id": "seeding_baseline_tps_table",
                    "category": "production_blocker",
                    "path_glob": "scripts/benchmark/seeding_rewards.py",
                    "classification": "degraded_fallback",
                    "owner": "stack-change-governance",
                    "rationale": "temporary fixture for strict-mode waiver behavior",
                    "expires": "2099-01-01",
                }
            ]
        },
    )

    result = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_exceptions_path=exceptions,
        allow_production_blocker_waivers=True,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert result.ok
    assert any("hardcoded_surface.waived.production_blocker" in warning for warning in result.warnings)


def test_validate_stack_priors_rejects_production_blocker_waiver_by_default(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    active = tmp_path / "scripts" / "benchmark"
    active.mkdir(parents=True)
    (active / "seeding_rewards.py").write_text(
        'DEFAULT_BASELINE_TPS = {"frontdoor": 10.3}\n',
        encoding="utf-8",
    )
    exceptions = _write_yaml(
        tmp_path / "exceptions.yaml",
        {
            "exceptions": [
                {
                    "rule_id": "seeding_baseline_tps_table",
                    "category": "production_blocker",
                    "path_glob": "scripts/benchmark/seeding_rewards.py",
                    "classification": "degraded_fallback",
                    "owner": "stack-change-governance",
                    "rationale": "temporary fixture for fail-closed waiver behavior",
                    "expires": "2099-01-01",
                }
            ]
        },
    )

    result = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_exceptions_path=exceptions,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert not result.ok
    assert any("--allow-production-blocker-waivers" in error for error in result.errors)
    assert any("hardcoded_surface.waived.production_blocker" in warning for warning in result.warnings)


def test_validate_stack_priors_rejects_invalid_surface_exception(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    exceptions = _write_yaml(
        tmp_path / "exceptions.yaml",
        {
            "exceptions": [
                {
                    "rule_id": "seeding_baseline_tps_table",
                    "category": "production_blocker",
                    "path_glob": "scripts/benchmark/seeding_rewards.py",
                    "classification": "degraded_fallback",
                    "rationale": "missing owner should fail",
                    "expires": "2099-01-01",
                }
            ]
        },
    )

    result = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_exceptions_path=exceptions,
    )

    assert not result.ok
    assert any("missing non-empty 'owner'" in error for error in result.errors)


def test_validate_stack_priors_rejects_unmatched_surface_exception(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    exceptions = _write_yaml(
        tmp_path / "exceptions.yaml",
        {
            "exceptions": [
                {
                    "rule_id": "seeding_baseline_tps_table",
                    "category": "production_blocker",
                    "path_glob": "scripts/benchmark/seeding_rewards.py",
                    "classification": "degraded_fallback",
                    "owner": "stack-change-governance",
                    "rationale": "stale waiver should fail once the finding is gone",
                    "expires": "2099-01-01",
                }
            ]
        },
    )

    result = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_exceptions_path=exceptions,
    )

    assert not result.ok
    assert any("no longer matches a hardcoded-surface finding" in error for error in result.errors)


def test_validate_stack_priors_rejects_stale_procedure_role_enum(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    procedure_dir = tmp_path / "orchestration" / "procedures"
    procedure_dir.mkdir(parents=True)
    _write_yaml(
        procedure_dir / "add_model_to_registry.yaml",
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "description": "Role assignment",
                    "validation": {"enum": ["frontdoor", "architect_coding"]},
                }
            ]
        },
    )

    result = validate_stack_priors(priors, scan_surfaces=True, repo_root=tmp_path)

    assert not result.ok
    assert any("procedure role enum drift" in error for error in result.errors)


def test_validate_stack_priors_rejects_stale_procedure_schema_role_enum(tmp_path: Path) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    orchestration_dir = tmp_path / "orchestration"
    orchestration_dir.mkdir()
    schema = {
        "properties": {
            "permissions": {
                "properties": {
                    "roles": {
                        "items": {
                            "enum": ["frontdoor", "architect_coding", "admin"],
                        }
                    }
                }
            }
        }
    }
    (orchestration_dir / "procedure.schema.json").write_text(
        json.dumps(schema),
        encoding="utf-8",
    )

    result = validate_stack_priors(priors, scan_surfaces=True, repo_root=tmp_path)

    assert not result.ok
    assert any("procedure schema permission enum drift" in error for error in result.errors)


def _legacy_test_finding_repo(tmp_path: Path) -> Path:
    """Create a fake repo_root containing one legacy_test hardcoded surface."""
    fixture = tmp_path / "tests" / "unit" / "legacy_role_fixture.py"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text('LEGACY_ROLE = "architect_' 'coding"\n', encoding="utf-8")
    return fixture


def _exceptions_file(tmp_path: Path, *, path_glob: str) -> Path:
    exc = tmp_path / "surface_exceptions.yaml"
    exc.write_text(
        "exceptions:\n"
        "  - rule_id: retired_role_in_tests\n"
        "    category: legacy_test\n"
        f"    path_glob: {path_glob}\n"
        "    classification: legacy_test\n"
        "    owner: test-infrastructure\n"
        "    rationale: intentional legacy-label regression coverage\n"
        '    expires: "2027-01-01"\n',
        encoding="utf-8",
    )
    return exc


def test_legacy_test_surface_is_classified_by_matching_exception(tmp_path: Path) -> None:
    _legacy_test_finding_repo(tmp_path)
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    exceptions = _exceptions_file(tmp_path, path_glob="tests/unit/legacy_role_fixture.py")

    result = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_categories=None,  # --all-hardcoded-surfaces
        surface_exceptions_path=exceptions,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert result.ok, result.errors
    assert any(
        w.startswith("hardcoded_surface.waived.legacy_test.retired_role_in_tests")
        and "owner=test-infrastructure" in w
        for w in result.warnings
    )
    assert hardcoded_surface_warning_counts(result.warnings) == {"waived_legacy_test": 1}


def test_production_blocker_scan_does_not_flag_classified_legacy_exception_stale(
    tmp_path: Path,
) -> None:
    # Regression: a legacy_test waiver must not be reported as a stale/unmatched
    # exception when the surface report is scoped to production_blocker only.
    _legacy_test_finding_repo(tmp_path)
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    exceptions = _exceptions_file(tmp_path, path_glob="tests/unit/legacy_role_fixture.py")

    result = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_categories=frozenset({"production_blocker"}),
        surface_exceptions_path=exceptions,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert result.ok, result.errors
    assert not any("no longer matches" in error for error in result.errors)
    # The legacy finding is out of the production_blocker report scope, so it
    # emits no warning under the default report.
    assert result.warnings == []


def test_genuinely_stale_legacy_exception_is_still_reported(tmp_path: Path) -> None:
    # Staleness is enforced against a full-category scan even when the report is
    # scoped to production_blocker, so a waiver matching nothing still errors.
    _legacy_test_finding_repo(tmp_path)
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    priors = _write_yaml(tmp_path / "stack_priors.yaml", _priors(registry, descriptors))
    exceptions = _exceptions_file(
        tmp_path, path_glob="tests/unit/does_not_exist_fixture.py"
    )

    result = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        surface_categories=frozenset({"production_blocker"}),
        surface_exceptions_path=exceptions,
        launch_manifest_targets={"frontdoor": {"port": 8070, "tier": "hot"}},
    )

    assert not result.ok
    assert any(
        "no longer matches a hardcoded-surface finding" in error
        for error in result.errors
    )


# ---------------------------------------------------------------------------
# Declared, expiring acceptance of known stack-prior gaps.
#
# The gate had exactly one severity, so "the operator knows about this and has
# accepted it" and "this is unsafe to launch" produced the same answer, and the
# only response was the total-bypass env flag. These tests pin the three states
# apart: declared+live -> visible warning, undeclared -> error, declared+expired
# -> error that names the expiry.
# ---------------------------------------------------------------------------


def _gap_priors(tmp_path: Path, gaps: dict[str, list[str]]) -> Path:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["known_gaps"] = list(gaps.get("frontdoor", []))
    payload["known_global_gaps"] = {
        role: list(role_gaps) for role, role_gaps in gaps.items() if role_gaps
    }
    return _write_yaml(tmp_path / "stack_priors.yaml", payload)


def _gap_declaration(**overrides) -> dict:
    declaration = {
        "role": "frontdoor",
        "gap": "Missing overall quality prior",
        "reason": "ratified SWE-bench evidence exists; canonical judge suite not run",
        "owner": "operator",
        "declared": "2026-08-01",
        "expires": "2099-01-01",
    }
    declaration.update(overrides)
    return declaration


def _gap_file(tmp_path: Path, *declarations: dict) -> Path:
    return _write_yaml(
        tmp_path / "accepted_gaps.yaml", {"accepted_gaps": list(declarations)}
    )


_GATE_TARGETS = {"frontdoor": {"port": 8070, "tier": "hot"}}


def test_declared_gap_stays_a_visible_warning_in_strict_mode(tmp_path: Path) -> None:
    priors = _gap_priors(tmp_path, {"frontdoor": ["Missing overall quality prior"]})
    gaps = _gap_file(tmp_path, _gap_declaration())

    result = validate_stack_priors(
        priors,
        strict=True,
        launch_manifest_targets=_GATE_TARGETS,
        accepted_gaps_path=gaps,
    )

    assert result.ok, result.errors
    # Visible, not silent: the operator must be able to read what is tolerated
    # and until when straight off the gate output.
    accepted = [w for w in result.warnings if w.startswith("accepted_gap.")]
    assert accepted
    assert all("expires=2099-01-01" in warning for warning in accepted)
    assert all("owner=operator" in warning for warning in accepted)


def test_undeclared_gap_still_blocks_strict_mode(tmp_path: Path) -> None:
    priors = _gap_priors(tmp_path, {"frontdoor": ["Missing overall quality prior"]})
    gaps = _gap_file(tmp_path)

    result = validate_stack_priors(
        priors,
        strict=True,
        launch_manifest_targets=_GATE_TARGETS,
        accepted_gaps_path=gaps,
    )

    assert not result.ok
    assert any(error.startswith("strict:") for error in result.errors)


def test_expired_declaration_blocks_and_reports_its_expiry(tmp_path: Path) -> None:
    priors = _gap_priors(tmp_path, {"frontdoor": ["Missing overall quality prior"]})
    gaps = _gap_file(tmp_path, _gap_declaration(expires="2020-01-01"))

    result = validate_stack_priors(
        priors,
        strict=True,
        launch_manifest_targets=_GATE_TARGETS,
        accepted_gaps_path=gaps,
    )

    assert not result.ok
    assert any("expired on 2020-01-01" in error for error in result.errors)
    # The declaration is dropped, so the gap it covered goes back to blocking —
    # an expired waiver must not keep working.
    assert any(error.startswith("strict:") for error in result.errors)


def test_declaration_requires_an_expiry(tmp_path: Path) -> None:
    priors = _gap_priors(tmp_path, {"frontdoor": ["Missing overall quality prior"]})
    declaration = _gap_declaration()
    del declaration["expires"]
    gaps = _gap_file(tmp_path, declaration)

    result = validate_stack_priors(
        priors, launch_manifest_targets=_GATE_TARGETS, accepted_gaps_path=gaps
    )

    assert not result.ok
    assert any("missing non-empty 'expires'" in error for error in result.errors)


def test_stale_declaration_for_an_absent_gap_is_reported(tmp_path: Path) -> None:
    priors = _gap_priors(tmp_path, {})
    gaps = _gap_file(tmp_path, _gap_declaration())

    result = validate_stack_priors(
        priors, launch_manifest_targets=_GATE_TARGETS, accepted_gaps_path=gaps
    )

    assert not result.ok
    assert any(
        "no longer matches a stack-prior gap" in error for error in result.errors
    )


def test_declaration_matches_role_and_gap_exactly_never_by_wildcard(
    tmp_path: Path,
) -> None:
    # A declaration filed for a missing quality prior must not also swallow a
    # missing live server binding on the same role, nor the same gap on a
    # different role. Both are the failure the declaration file exists to avoid.
    priors = _gap_priors(
        tmp_path,
        {
            "frontdoor": [
                "Missing overall quality prior",
                "Missing live server binding",
            ]
        },
    )
    same_role_other_gap = _gap_file(tmp_path, _gap_declaration())

    result = validate_stack_priors(
        priors,
        strict=True,
        launch_manifest_targets=_GATE_TARGETS,
        accepted_gaps_path=same_role_other_gap,
    )

    assert not result.ok
    assert any(error.startswith("strict:") for error in result.errors)

    other_role = _write_yaml(
        tmp_path / "other_role_gaps.yaml",
        {
            "accepted_gaps": [
                _gap_declaration(role="architect_general"),
                _gap_declaration(gap="Missing live server binding"),
            ]
        },
    )
    result = validate_stack_priors(
        priors,
        strict=True,
        launch_manifest_targets=_GATE_TARGETS,
        accepted_gaps_path=other_role,
    )

    assert not result.ok
    # The architect_general declaration matches nothing here.
    assert any(
        "no longer matches a stack-prior gap" in error and "architect_general" in error
        for error in result.errors
    )


def test_production_accepted_gaps_file_carries_no_malformed_or_expired_waiver() -> None:
    """Every waiver in the production file must be well-formed and still live.

    Deliberately does NOT restate the file's contents. Waivers open and close as
    gaps do — the three 2026-08-01 "Missing overall quality prior" entries were
    removed on 2026-08-03 once the gap stopped firing — so a hardcoded roster
    fails on every legitimate edit instead of on a real defect. The invariants
    that actually matter are: it parses, nothing is expired, and each entry names
    an accountable owner and a reason. That a waiver must still match a live gap
    is covered by the stale-waiver test above.
    """
    declarations, errors = stack_change_guard.load_accepted_gaps()

    assert not errors, errors
    today = datetime.date.today().isoformat()
    for declaration in declarations:
        label = f"{declaration.role}/{declaration.gap}"
        assert declaration.role and declaration.gap, f"waiver missing role or gap: {declaration}"
        assert declaration.owner, f"{label}: waiver needs an accountable owner"
        assert declaration.reason, f"{label}: waiver needs a reason"
        assert declaration.expires, f"{label}: a waiver without an expiry is a deleted check"
        assert declaration.expires > today, (
            f"{label}: waiver expired on {declaration.expires}; renew with fresh "
            "evidence or close the gap"
        )


# ---------------------------------------------------------------------------
# Retired-role set: derived, not restated by hand.
# ---------------------------------------------------------------------------


def _retired_role_sources(tmp_path: Path, master_roles: dict, lean_roles: dict):
    master = _write_yaml(tmp_path / "master.yaml", {"roles": master_roles})
    lean = _write_yaml(tmp_path / "lean.yaml", {"roles": lean_roles})
    return master, lean


def test_derive_retired_live_roles_reads_master_retirement_markers(
    tmp_path: Path,
) -> None:
    master, lean = _retired_role_sources(
        tmp_path,
        {
            "frontdoor": {"tier": "A"},
            "old_by_deprecated": {"deprecated": True},
            "old_by_retired_date": {"retired": "2026-07-31"},
            "old_by_reason_only": {"deprecated_reason": "GGUF deleted"},
        },
        {"frontdoor": {}},
    )

    derived = derive_retired_live_roles(
        master_registry_path=master, lean_registry_path=lean, floor=frozenset()
    )

    # Nobody had to enumerate these three names anywhere in the guard.
    assert derived == {
        "old_by_deprecated",
        "old_by_retired_date",
        "old_by_reason_only",
    }


def test_derive_retired_live_roles_never_flags_a_live_role(tmp_path: Path) -> None:
    # Master says deprecated, the compiled lean registry still serves it: that is
    # a registry contradiction, not a retired role. Flagging it would fail every
    # stack change on a name the fleet is actually running.
    master, lean = _retired_role_sources(
        tmp_path,
        {"frontdoor": {"deprecated": True}, "gone": {"deprecated": True}},
        {"frontdoor": {}},
    )

    derived = derive_retired_live_roles(
        master_registry_path=master, lean_registry_path=lean, floor=frozenset()
    )

    assert derived == {"gone"}


def test_derive_retired_live_roles_keeps_the_documented_floor(tmp_path: Path) -> None:
    master, lean = _retired_role_sources(tmp_path, {"frontdoor": {}}, {"frontdoor": {}})

    derived = derive_retired_live_roles(
        master_registry_path=master, lean_registry_path=lean
    )

    assert RETIRED_LIVE_ROLE_FLOOR <= derived


def test_derive_retired_live_roles_raises_instead_of_degrading_to_the_floor(
    tmp_path: Path,
) -> None:
    _, lean = _retired_role_sources(tmp_path, {"frontdoor": {}}, {"frontdoor": {}})

    with pytest.raises(RetiredRoleDerivationError):
        derive_retired_live_roles(
            master_registry_path=tmp_path / "absent.yaml", lean_registry_path=lean
        )

    shapeless = _write_yaml(tmp_path / "shapeless.yaml", {"roles": []})
    with pytest.raises(RetiredRoleDerivationError):
        derive_retired_live_roles(
            master_registry_path=shapeless, lean_registry_path=lean
        )


def test_retired_role_derivation_failure_is_reported_not_swallowed(monkeypatch) -> None:
    def _boom(**_kwargs):
        raise RetiredRoleDerivationError("master registry is missing: /nope.yaml")

    monkeypatch.setattr(stack_change_guard, "_RETIRED_LIVE_ROLES_CACHE", None)
    monkeypatch.setattr(stack_change_guard, "derive_retired_live_roles", _boom)

    roles, errors = stack_change_guard._retired_live_roles_or_error()

    assert roles == RETIRED_LIVE_ROLE_FLOOR
    assert any("retired-role derivation failed" in error for error in errors)


def test_production_retired_role_set_is_derived_and_supersets_the_floor() -> None:
    derived = stack_change_guard.retired_live_roles(refresh=True)

    assert RETIRED_LIVE_ROLE_FLOOR <= derived
    # A derived set that is only the floor means derivation silently did nothing.
    assert len(derived) > len(RETIRED_LIVE_ROLE_FLOOR)


# ---------------------------------------------------------------------------
# Value staleness: compare against the compiled artifact, do not grep for names.
#
# This is the model_quality_signatures.yaml failure: every row named a CURRENT
# role, so the name-matching rule passed clean while every VALUE described the
# fleet retired 2026-05-08.
# ---------------------------------------------------------------------------


_ROLE_FACT_RULE = RoleFactSurfaceRule(
    rule_id="unit_role_fact_rule",
    category="production_blocker",
    path_globs=("stack_templates/*.yaml",),
    roles_key="roles",
    fields=("model", "quant", "tier"),
    remediation="restate from the compiled stack priors",
)


def _compiled_priors_doc() -> dict:
    return {
        "roles": {
            "frontdoor": {
                "display_name": "Qwen3.6-35B-A3B-MTP-Q8_0",
                "model_id": "qwen3.6-35b-a3b-mtp-q8_0",
                "model": {"quant": "Q8_0"},
                "serving": {
                    "tier": "hot",
                    "launch": {
                        "requirements": {
                            "model_path": "/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
                        }
                    },
                },
            }
        }
    }


def _template(tmp_path: Path, roles: dict) -> Path:
    templates = tmp_path / "stack_templates"
    templates.mkdir(parents=True, exist_ok=True)
    return _write_yaml(templates / "default.yaml", {"name": "default", "roles": roles})


def test_role_fact_scan_reports_value_drift_on_a_current_role_name(
    tmp_path: Path,
) -> None:
    # TEETH: the surface names only a live role, so every name-matching rule in
    # the guard passes it clean. Its VALUES describe a different model.
    _template(
        tmp_path,
        {"frontdoor": {"model": "Qwen3.5-122B-A10B", "quant": "Q4_K_M", "tier": "HOT"}},
    )

    findings = scan_role_fact_surfaces(
        _compiled_priors_doc(), tmp_path, rules=(_ROLE_FACT_RULE,)
    )

    assert len(findings) == 1
    assert findings[0].rule_id == "unit_role_fact_rule"
    assert findings[0].category == "production_blocker"
    assert findings[0].path.as_posix() == "stack_templates/default.yaml"
    assert "Qwen3.5-122B-A10B" in findings[0].snippet
    assert "Qwen3.6-35B-A3B-MTP-Q8_0" in findings[0].snippet
    assert "quant 'Q4_K_M' != compiled 'Q8_0'" in findings[0].snippet
    # A drifted row must point at itself, not at line 1 of the file.
    assert findings[0].line > 0


def test_role_fact_scan_catches_a_distinguishing_build_token(tmp_path: Path) -> None:
    # The non-MTP GGUF is a DIFFERENT FILE with the same family name; launching it
    # silently disables draft-mtp speculative decoding.
    _template(
        tmp_path,
        {"frontdoor": {"model": "Qwen3.6-35B-A3B-Q8_0", "quant": "Q8_0", "tier": "HOT"}},
    )

    findings = scan_role_fact_surfaces(
        _compiled_priors_doc(), tmp_path, rules=(_ROLE_FACT_RULE,)
    )

    assert len(findings) == 1
    assert "Qwen3.6-35B-A3B-Q8_0" in findings[0].snippet


def test_role_fact_scan_tolerates_an_informal_but_agreeing_name(tmp_path: Path) -> None:
    # Config surfaces write short names for the model the compiled artifact spells
    # out. Treating that as drift would bury the real signal.
    _template(
        tmp_path,
        {"frontdoor": {"model": "Qwen3.6-35B-A3B-MTP", "quant": "Q8_0", "tier": "hot"}},
    )

    assert not scan_role_fact_surfaces(
        _compiled_priors_doc(), tmp_path, rules=(_ROLE_FACT_RULE,)
    )


def test_role_fact_scan_skips_alias_rows_and_unknown_roles(tmp_path: Path) -> None:
    # `tier: ALIAS` is a structural marker meaning "launches no server", not a
    # serving tier; comparing it against the compiled `hot` is a category error.
    # A role the compiled artifact does not declare has nothing to compare to.
    _template(
        tmp_path,
        {
            "frontdoor": {"model": "Qwen3.6-35B-A3B-MTP-Q8_0", "quant": "Q8_0", "tier": "HOT"},
            "worker_summarize": {"tier": "ALIAS", "alias_to": "frontdoor"},
            "embedder": {"model": "bge-large-en-v1.5-f16", "quant": "f16", "tier": "HOT"},
        },
    )

    assert not scan_role_fact_surfaces(
        _compiled_priors_doc(), tmp_path, rules=(_ROLE_FACT_RULE,)
    )


def test_role_fact_scan_honours_the_inline_allow_marker(tmp_path: Path) -> None:
    templates = tmp_path / "stack_templates"
    templates.mkdir(parents=True)
    (templates / "default.yaml").write_text(
        "name: default\n"
        "roles:\n"
        f"  frontdoor:  # {stack_change_guard.SURFACE_SCAN_ALLOW_MARKER}\n"
        "    model: Qwen3.5-122B-A10B\n"
        "    quant: Q4_K_M\n"
        "    tier: HOT\n",
        encoding="utf-8",
    )

    assert not scan_role_fact_surfaces(
        _compiled_priors_doc(), tmp_path, rules=(_ROLE_FACT_RULE,)
    )


def test_validate_stack_priors_blocks_strict_mode_on_role_fact_drift(
    tmp_path: Path,
) -> None:
    registry = _write_yaml(tmp_path / "registry.yaml", {"roles": {}})
    descriptors = _write_yaml(tmp_path / "descriptors.yaml", {"models": []})
    payload = _priors(registry, descriptors)
    payload["roles"]["frontdoor"]["display_name"] = "Qwen3.6-35B-A3B-MTP-Q8_0"
    payload["roles"]["frontdoor"]["model"] = {"quant": "Q8_0"}
    priors = _write_yaml(tmp_path / "stack_priors.yaml", payload)
    _template(
        tmp_path,
        {"frontdoor": {"model": "Qwen2.5-VL-7B", "quant": "Q4_K_M", "tier": "HOT"}},
    )

    loose = validate_stack_priors(
        priors,
        scan_surfaces=True,
        repo_root=tmp_path,
        launch_manifest_targets=_GATE_TARGETS,
    )
    strict = validate_stack_priors(
        priors,
        strict=True,
        scan_surfaces=True,
        repo_root=tmp_path,
        launch_manifest_targets=_GATE_TARGETS,
    )

    assert loose.ok
    assert any("stale_role_fact_table" in warning for warning in loose.warnings)
    assert not strict.ok
    assert any("stale_role_fact_table" in error for error in strict.errors)


def test_surface_manifest_bijection_covers_role_fact_rules(tmp_path: Path) -> None:
    manifest = _write_yaml(
        tmp_path / "manifest.yaml",
        {
            "version": 1,
            "surfaces": [
                {
                    "rule_id": "unit_rule",
                    "category": "production_blocker",
                    "owner": "stack-change-governance",
                    "consumer_scope": "unit",
                    "promotion_blocker": True,
                    "review_cadence": "every stack change",
                    "evidence_command": "uv run python scripts/validate/stack_change_guard.py",
                    "drift_response": "derive from generated truth",
                }
            ],
        },
    )
    rule = HardcodedSurfaceRule(
        rule_id="unit_rule",
        category="production_blocker",
        pattern=r"\bunit\b",
        path_globs=("src/**/*.py",),
        remediation="derive from generated truth",
    )

    errors = validate_surface_manifest(
        manifest, rules=(rule,), role_fact_rules=(_ROLE_FACT_RULE,)
    )

    assert any("unit_role_fact_rule" in error for error in errors)


def test_production_surface_manifest_owns_every_role_fact_rule() -> None:
    assert ROLE_FACT_SURFACE_RULES
    assert not validate_surface_manifest()
    manifest, manifest_errors = stack_change_guard.load_surface_manifest()
    assert not manifest_errors, manifest_errors
    inventory = hardcoded_surface_rule_inventory(ownership_manifest=manifest)
    assert inventory["role_fact_rule_count"] == len(ROLE_FACT_SURFACE_RULES)
    assert all(
        rule["ownership"].get("owner") for rule in inventory["role_fact_rules"]
    )
    assert all(
        rule["compared_against"] == "orchestration/derived/stack_priors.yaml"
        for rule in inventory["role_fact_rules"]
    )


# ---------------------------------------------------------------------------
# Derived retired-role surface patterns (2026-08-02)
#
# These rules each carried the literal `\barchitect_coding\b` while this module
# already derived the authoritative retired-role set a few hundred lines above.
# The two had diverged: the derivation returns dozens of names, the scanner looked
# for one. Nothing was duplicated, a name was simply MISSING from a hand-written
# pattern — so no review of what the rule DID match could reveal what it did not.
# ---------------------------------------------------------------------------


def test_retired_role_rules_derive_their_pattern_from_the_producer() -> None:
    derived_rules, errors = stack_change_guard._derive_retired_role_patterns(
        HARDCODED_SURFACE_RULES
    )
    assert not errors, errors

    marked = [rule for rule in HARDCODED_SURFACE_RULES if rule.derive_retired_roles]
    assert marked, "no rule is marked derive_retired_roles"

    retired = stack_change_guard.retired_live_roles()
    by_id = {rule.rule_id: rule for rule in derived_rules}
    for rule in marked:
        pattern = re.compile(by_id[rule.rule_id].pattern)
        # every DERIVED retired role is now scanned for...
        for role in retired:
            assert pattern.search(role), f"{rule.rule_id} does not match {role}"
        # ...and the FLOOR is still covered even if the derivation shrinks.
        for role in RETIRED_LIVE_ROLE_FLOOR:
            assert pattern.search(role), f"{rule.rule_id} lost floor role {role}"
        # a live role must NOT match, or every stack change fails on a name the
        # fleet is actually serving.
        assert not pattern.search("architect_general")


def test_retired_role_derivation_keeps_the_floor_when_derivation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken source degrades to the floor and REPORTS, never silently passes."""

    def _boom() -> frozenset[str]:
        raise RetiredRoleDerivationError("master registry unreadable")

    monkeypatch.setattr(stack_change_guard, "retired_live_roles", _boom)
    derived_rules, errors = stack_change_guard._derive_retired_role_patterns(
        HARDCODED_SURFACE_RULES
    )
    assert errors and "retired-role derivation failed" in errors[0]
    marked = next(r for r in derived_rules if r.derive_retired_roles)
    pattern = re.compile(marked.pattern)
    for role in RETIRED_LIVE_ROLE_FLOOR:
        assert pattern.search(role)


def test_derived_retired_role_scan_gains_coverage_and_loses_none(tmp_path: Path) -> None:
    """Teeth: a retired role the hand-written pattern never named is now found."""
    (tmp_path / "src").mkdir()
    retired = sorted(stack_change_guard.retired_live_roles() - RETIRED_LIVE_ROLE_FLOOR)
    if not retired:
        pytest.skip("no retired role beyond the floor to demonstrate with")
    probe = retired[0]
    (tmp_path / "src" / "live_module.py").write_text(
        f'ROLE_MAP = {{"{probe}": "x", "{sorted(RETIRED_LIVE_ROLE_FLOOR)[0]}": "y"}}\n',
        encoding="utf-8",
    )

    rule = next(
        r for r in HARDCODED_SURFACE_RULES if r.rule_id == "retired_role_in_active_code"
    )
    hand_written = dataclasses.replace(rule, derive_retired_roles=False, exclude_globs=())
    derived = dataclasses.replace(rule, exclude_globs=())

    hand_findings = stack_change_guard.scan_hardcoded_surfaces(
        tmp_path, rules=(hand_written,), categories=None
    )
    derived_findings = stack_change_guard.scan_hardcoded_surfaces(
        tmp_path, rules=(derived,), categories=None
    )

    assert len(hand_findings) == 1  # only the floor role is named by hand
    assert len(derived_findings) == 1
    hand_snippets = {f.snippet for f in hand_findings}
    assert probe in derived_findings[0].snippet
    # coverage gained, none lost
    assert hand_snippets <= {f.snippet for f in derived_findings}


# ---------------------------------------------------------------------------
# Fail-open regression suite (W6, 2026-08-02)
#
# Each of these constructs the condition the guard exists to detect and asserts
# it now reports COULD-NOT-CHECK. Before the repair every one of them returned a
# clean result: the guard reported success on exactly the condition it detects.
# The screen each test encodes: "can I make this check pass by DELETING the
# thing it inspects?"
# ---------------------------------------------------------------------------


def _real_priors() -> dict:
    return yaml.safe_load(
        stack_change_guard.DEFAULT_PRIORS.read_text(encoding="utf-8")
    )


def _could_not_check(errors: list[str]) -> list[str]:
    return [e for e in errors if stack_change_guard.COULD_NOT_CHECK in e]


def _block_import(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"simulated: {module_name} unimportable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_launch_view_import_failure_is_could_not_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocking the launch-manifest import must not take the gate clean."""
    priors = _real_priors()
    _block_import(monkeypatch, "scripts.server.stack_manifest")

    targets, view_errors = stack_change_guard._launch_manifest_targets_or_error()
    errors = stack_change_guard.validate_launch_manifest_serving_alignment(priors)

    assert targets == {}
    assert _could_not_check(view_errors)
    assert _could_not_check(errors)
    assert any("did not import" in e for e in errors)


def test_launch_requirements_unavailable_is_could_not_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the launch-path constants must be reported, not absorbed."""

    def _raise(_server: dict) -> dict:
        raise stack_change_guard.LaunchViewUnavailableError("simulated")

    monkeypatch.setattr(stack_change_guard, "_launch_requirements_for_server", _raise)
    errors = stack_change_guard.validate_launch_manifest_serving_alignment(_real_priors())

    assert any("launch requirements unavailable" in e for e in _could_not_check(errors))


def test_launch_context_unavailable_is_could_not_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0-of-N context assertions skipped, with the target count unchanged."""

    def _raise(_server: dict) -> int:
        raise stack_change_guard.LaunchViewUnavailableError("simulated")

    monkeypatch.setattr(stack_change_guard, "_effective_context_for_server", _raise)
    errors = stack_change_guard.validate_launch_manifest_serving_alignment(_real_priors())

    assert any("launch context unavailable" in e for e in _could_not_check(errors))


def test_declared_fact_without_launch_counterpart_is_could_not_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checklist is derived from the producer, so ANY cause of a dropped
    launch-view value is caught — not only an import error."""
    monkeypatch.setattr(
        stack_change_guard, "_effective_context_for_server", lambda _server: None
    )
    errors = stack_change_guard.validate_launch_manifest_serving_alignment(_real_priors())

    skipped = [
        e
        for e in _could_not_check(errors)
        if "declares serving.effective_context_tokens" in e
    ]
    assert skipped, "a declared context with no launch counterpart must be reported"


def test_empty_launch_view_is_could_not_check() -> None:
    """0 launch targets is an instrument failure, not 'no alignment errors'."""
    errors = stack_change_guard.validate_launch_manifest_serving_alignment(
        _real_priors(), launch_manifest_targets={}
    )

    assert any("0 launch targets" in e for e in _could_not_check(errors))


def test_unreadable_registry_is_could_not_check(tmp_path: Path) -> None:
    """A registry that will not parse silently halved the launch view."""
    bad = tmp_path / "registry.yaml"
    bad.write_text("this: is: not: valid: yaml: [\n", encoding="utf-8")

    errors = stack_change_guard.validate_launch_manifest_serving_alignment(
        _real_priors(), registry_path=bad
    )

    assert any("launch-view registry is unreadable" in e for e in _could_not_check(errors))


def test_deleted_registry_is_could_not_check(tmp_path: Path) -> None:
    """The delete screen, applied directly."""
    errors = stack_change_guard.validate_launch_manifest_serving_alignment(
        _real_priors(), registry_path=tmp_path / "not-here.yaml"
    )

    assert any("launch-view registry is missing" in e for e in _could_not_check(errors))


def test_missing_procedure_artifact_is_could_not_check(tmp_path: Path) -> None:
    """Deleting the procedure used to make its enum check pass."""
    errors = stack_change_guard.validate_procedure_role_enums(
        _real_priors(), procedure_path=tmp_path / "not-here.yaml"
    )

    assert any("procedure artifact is missing" in e for e in _could_not_check(errors))


def test_missing_procedure_schema_is_could_not_check(tmp_path: Path) -> None:
    errors = stack_change_guard.validate_procedure_role_enums(
        _real_priors(), schema_path=tmp_path / "not-here.json"
    )

    assert any(
        "procedure schema artifact is missing" in e for e in _could_not_check(errors)
    )


def test_healthy_artifacts_emit_no_could_not_check() -> None:
    """The other half of a guard with teeth: it must stay quiet when it CAN check."""
    priors = _real_priors()

    assert _could_not_check(
        stack_change_guard.validate_launch_manifest_serving_alignment(priors)
    ) == []
    assert _could_not_check(stack_change_guard.validate_procedure_role_enums(priors)) == []


# ---------------------------------------------------------------------------
# REQUIRED_SOURCE_ARTIFACTS staleness — commit-time gate (P1-10)
#
# Every one of these runs against a THROWAWAY git repo under tmp_path. The
# checker reads a real git index, so a fake would only prove the fake works.
# ---------------------------------------------------------------------------

_STALENESS_LABELS = (
    "registry",
    "descriptors",
    "stack_manifest",
    "stack_numa",
    "orchestrator_stack",
    "stack_paths",
    "stack_runtime",
)


def _git_available() -> bool:
    try:
        return (
            subprocess.run(["git", "--version"], capture_output=True, timeout=10).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


requires_git = pytest.mark.skipif(not _git_available(), reason="git not available")


def _git(repo: Path, *args: str) -> str:
    """Run git in an ISOLATED sandbox: no global config, no inherited hooks."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "staleness-test",
        "GIT_AUTHOR_EMAIL": "staleness@test.invalid",
        "GIT_COMMITTER_NAME": "staleness-test",
        "GIT_COMMITTER_EMAIL": "staleness@test.invalid",
    }
    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout


def _write_staleness_priors(priors_path: Path, artifacts: dict[str, Path]) -> None:
    priors_path.write_text(
        yaml.safe_dump(
            {
                "stack_priors_version": STACK_PRIORS_VERSION,
                "source_artifacts": {
                    label: {"path": str(path), "sha256": _sha(path)}
                    for label, path in artifacts.items()
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def staleness_repo(tmp_path: Path):
    """A committed sandbox whose priors pin every REQUIRED_SOURCE_ARTIFACTS file."""
    repo = (tmp_path / "sandbox").resolve()
    (repo / "orchestration" / "derived").mkdir(parents=True)
    (repo / "scripts" / "server").mkdir(parents=True)
    artifacts = {
        "registry": repo / "orchestration" / "model_registry.yaml",
        "descriptors": repo / "orchestration" / "model_descriptors.yaml",
        "stack_manifest": repo / "scripts" / "server" / "stack_manifest.py",
        "stack_numa": repo / "scripts" / "server" / "stack_numa.py",
        "orchestrator_stack": repo / "scripts" / "server" / "orchestrator_stack.py",
        "stack_paths": repo / "scripts" / "server" / "stack_paths.py",
        "stack_runtime": repo / "scripts" / "server" / "stack_runtime.py",
    }
    for label, path in artifacts.items():
        path.write_text(f"# {label} v1\n", encoding="utf-8")
    (repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    priors_path = repo / "orchestration" / "derived" / "stack_priors.yaml"
    _write_staleness_priors(priors_path, artifacts)

    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "-m", "sandbox baseline")

    return types.SimpleNamespace(repo=repo, priors=priors_path, artifacts=artifacts)


def _staleness(sandbox, **kwargs) -> GuardResult:
    return stack_change_guard.check_source_artifact_staleness(
        sandbox.priors, repo_root=sandbox.repo, **kwargs
    )


def _repin(sandbox) -> None:
    """Stand-in for `stack_change_pipeline.py update`: re-pin from current bytes."""
    _write_staleness_priors(sandbox.priors, sandbox.artifacts)


@requires_git
def test_staleness_passes_when_the_commit_ships_nothing_stale(staleness_repo) -> None:
    (staleness_repo.repo / "README.md").write_text("edited\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", "README.md")

    result = _staleness(staleness_repo)

    assert result.ok, result.errors
    assert result.warnings == []


@requires_git
def test_staleness_fails_then_passes_again_after_repinning(staleness_repo) -> None:
    """MUTATION: stage a pinned artifact without regenerating priors."""
    numa = staleness_repo.artifacts["stack_numa"]
    original = numa.read_text(encoding="utf-8")

    numa.write_text("# stack_numa v2 — NUMA_CONFIG changed\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", str(numa))
    stale = _staleness(staleness_repo)

    assert not stale.ok
    assert any("source_artifacts.stack_numa is STALE" in e for e in stale.errors)
    assert any(
        "uv run python scripts/registry/stack_change_pipeline.py update" in e
        for e in stale.errors
    )

    # Remedy 1: regenerate the priors (what the guard tells you to do).
    _repin(staleness_repo)
    _git(staleness_repo.repo, "add", "-A")
    assert _staleness(staleness_repo).ok

    # Remedy 2: restore the artifact instead.
    numa.write_text(original, encoding="utf-8")
    _write_staleness_priors(staleness_repo.priors, staleness_repo.artifacts)
    _git(staleness_repo.repo, "add", "-A")
    restored = _staleness(staleness_repo)
    assert restored.ok, restored.errors


@requires_git
def test_staleness_reads_the_index_not_the_working_tree(staleness_repo) -> None:
    """The scope distinction is load-bearing: an UNSTAGED edit ships nothing."""
    numa = staleness_repo.artifacts["stack_numa"]
    numa.write_text("# stack_numa v2 — not staged\n", encoding="utf-8")

    staged = _staleness(staleness_repo)
    worktree = _staleness(staleness_repo, scope="worktree")

    assert staged.ok, staged.errors
    assert staged.warnings == []
    assert not worktree.ok
    assert any("source_artifacts.stack_numa is STALE" in e for e in worktree.errors)


@requires_git
def test_staleness_inherited_by_an_untouched_commit_warns_and_strict_promotes(
    staleness_repo,
) -> None:
    numa = staleness_repo.artifacts["stack_numa"]
    numa.write_text("# stack_numa v2 — landed earlier\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", str(numa))
    _git(staleness_repo.repo, "commit", "-q", "--no-verify", "-m", "inherited drift")

    (staleness_repo.repo / "README.md").write_text("unrelated change\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", "README.md")

    lenient = _staleness(staleness_repo)
    strict = _staleness(staleness_repo, strict=True)

    assert lenient.ok, lenient.errors
    assert any("pre-existing staleness" in w for w in lenient.warnings)
    assert not strict.ok
    assert any("source_artifacts.stack_numa is STALE" in e for e in strict.errors)


@requires_git
def test_rewriting_the_priors_blames_every_pin_in_it(staleness_repo) -> None:
    """Hand-editing the derived artifact makes all of its pins this commit's claim."""
    numa = staleness_repo.artifacts["stack_numa"]
    numa.write_text("# stack_numa v2 — landed earlier\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", str(numa))
    _git(staleness_repo.repo, "commit", "-q", "--no-verify", "-m", "inherited drift")

    priors = yaml.safe_load(staleness_repo.priors.read_text(encoding="utf-8"))
    priors["status"] = "hand edited"
    staleness_repo.priors.write_text(yaml.safe_dump(priors, sort_keys=False), encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("source_artifacts.stack_numa is STALE" in e for e in result.errors)


# --- Killer question: can this check be PASSED by deleting what it inspects? ---


@requires_git
def test_deleting_a_pinned_artifact_is_a_loud_failure(staleness_repo) -> None:
    numa = staleness_repo.artifacts["stack_numa"]
    _git(staleness_repo.repo, "rm", "-q", "--cached", str(numa))
    numa.unlink()

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any(
        "source_artifacts.stack_numa" in e and "DELETION" in e for e in result.errors
    )


@requires_git
def test_a_pin_pointing_at_a_nonexistent_file_is_a_loud_failure(staleness_repo) -> None:
    priors = yaml.safe_load(staleness_repo.priors.read_text(encoding="utf-8"))
    priors["source_artifacts"]["stack_numa"]["path"] = str(
        staleness_repo.repo / "scripts" / "server" / "vanished.py"
    )
    staleness_repo.priors.write_text(yaml.safe_dump(priors, sort_keys=False), encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any(
        "source_artifacts.stack_numa" in e and "unreadable on disk" in e
        for e in result.errors
    )


@requires_git
def test_deleting_the_priors_artifact_is_a_loud_failure(staleness_repo) -> None:
    _git(staleness_repo.repo, "rm", "-q", "--cached", str(staleness_repo.priors))
    staleness_repo.priors.unlink()

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("stack priors artifact" in e and "DELETION" in e for e in result.errors)


@requires_git
def test_an_empty_source_artifacts_section_is_a_loud_failure(staleness_repo) -> None:
    """Emptying the list must not turn 'nothing to compare' into a pass."""
    staleness_repo.priors.write_text(
        yaml.safe_dump({"source_artifacts": {}}, sort_keys=False), encoding="utf-8"
    )
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("no non-empty source_artifacts section" in e for e in result.errors)
    # ...and the FLOOR still names every artifact that vanished with it.
    for label in _STALENESS_LABELS:
        assert any(f"source_artifacts.{label} is absent" in e for e in result.errors), label


@requires_git
def test_an_empty_required_floor_is_itself_a_loud_failure(
    staleness_repo, monkeypatch
) -> None:
    """The floor is the backstop; a floor of zero names would be vacuous."""
    monkeypatch.setattr(stack_change_guard, "REQUIRED_SOURCE_ARTIFACTS", ())
    staleness_repo.priors.write_text(
        yaml.safe_dump({"source_artifacts": {}}, sort_keys=False), encoding="utf-8"
    )
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("REQUIRED_SOURCE_ARTIFACTS is EMPTY" in e for e in result.errors)


@requires_git
def test_dropping_one_required_pin_is_a_loud_failure(staleness_repo) -> None:
    priors = yaml.safe_load(staleness_repo.priors.read_text(encoding="utf-8"))
    priors["source_artifacts"].pop("stack_runtime")
    staleness_repo.priors.write_text(yaml.safe_dump(priors, sort_keys=False), encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("source_artifacts.stack_runtime is absent" in e for e in result.errors)


@requires_git
@pytest.mark.parametrize("bad_path", ["", None, 42, []])
def test_an_unresolvable_pin_path_is_a_loud_failure(staleness_repo, bad_path) -> None:
    priors = yaml.safe_load(staleness_repo.priors.read_text(encoding="utf-8"))
    priors["source_artifacts"]["stack_numa"]["path"] = bad_path
    staleness_repo.priors.write_text(yaml.safe_dump(priors, sort_keys=False), encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any(
        "source_artifacts.stack_numa.path does not resolve" in e for e in result.errors
    )


@requires_git
@pytest.mark.parametrize("bad_sha", ["", None, "deadbeef", "Z" * 64])
def test_a_malformed_pin_digest_is_a_loud_failure(staleness_repo, bad_sha) -> None:
    priors = yaml.safe_load(staleness_repo.priors.read_text(encoding="utf-8"))
    priors["source_artifacts"]["stack_numa"]["sha256"] = bad_sha
    staleness_repo.priors.write_text(yaml.safe_dump(priors, sort_keys=False), encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any(
        "source_artifacts.stack_numa.sha256 is not a sha256 digest" in e
        for e in result.errors
    )


@requires_git
def test_unparseable_priors_fail_closed(staleness_repo) -> None:
    staleness_repo.priors.write_text("source_artifacts: [\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("did not parse" in e for e in result.errors)


@requires_git
def test_priors_that_parse_to_a_non_mapping_fail_closed(staleness_repo) -> None:
    staleness_repo.priors.write_text("- not a mapping\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", str(staleness_repo.priors))

    result = _staleness(staleness_repo)

    assert not result.ok
    assert any("did not parse to a mapping" in e for e in result.errors)


def test_an_unreadable_git_index_fails_closed(tmp_path: Path) -> None:
    """No index to read is a REFUSAL, never a quiet pass."""
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    priors = not_a_repo / "stack_priors.yaml"
    priors.write_text("source_artifacts: {}\n", encoding="utf-8")

    result = stack_change_guard.check_source_artifact_staleness(
        priors, repo_root=not_a_repo
    )

    assert not result.ok
    assert any("cannot read the git index" in e for e in result.errors)


def test_an_unknown_staleness_scope_fails_closed(tmp_path: Path) -> None:
    result = stack_change_guard.check_source_artifact_staleness(
        tmp_path / "stack_priors.yaml", scope="whatever", repo_root=tmp_path
    )

    assert not result.ok
    assert any("unknown staleness scope" in e for e in result.errors)


@requires_git
def test_staleness_check_is_reachable_from_the_guard_cli(staleness_repo, capsys) -> None:
    """The consumer: `python scripts/validate/stack_change_guard.py ...`."""
    numa = staleness_repo.artifacts["stack_numa"]
    numa.write_text("# stack_numa v2\n", encoding="utf-8")
    _git(staleness_repo.repo, "add", str(numa))

    argv = [
        "--check-source-artifact-staleness",
        "--priors",
        str(staleness_repo.priors),
        "--repo-root",
        str(staleness_repo.repo),
    ]
    rc = stack_change_guard_main(argv)
    stale_out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL: 1 source-artifact staleness error(s)" in stale_out
    assert "source_artifacts.stack_numa is STALE" in stale_out
    assert RECOMPILE_PRIORS_COMMAND in stale_out

    _repin(staleness_repo)
    _git(staleness_repo.repo, "add", "-A")
    rc_clean = stack_change_guard_main(argv)
    clean_out = capsys.readouterr().out

    assert rc_clean == 0
    assert "OK: source artifacts match the pins" in clean_out


def test_launch_view_rejects_invalid_explicit_numa_mode_as_could_not_check(
    _synthetic_stack_manifest_inputs: None,
) -> None:
    """NIB2-69: an explicit launch mode is never silently normalised to a default."""
    targets, view_errors = stack_change_guard._launch_manifest_targets_or_error(
        launch_numa_mode="halves"
    )

    assert targets == {}
    assert _could_not_check(view_errors)
    assert "invalid launch NUMA mode" in view_errors[0]


def test_explicit_launch_numa_mode_bypasses_realized_fleet_probe(
    monkeypatch: pytest.MonkeyPatch,
    _synthetic_stack_manifest_inputs: None,
) -> None:
    def _probe_must_not_run() -> str:
        raise AssertionError("explicit launch mode must not consult the realized fleet")

    monkeypatch.setattr(stack_change_guard, "_realized_launch_numa_mode", _probe_must_not_run)
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")

    both, both_errors = stack_change_guard._launch_manifest_targets_or_error(
        launch_numa_mode="both"
    )
    full, _ = stack_change_guard._launch_manifest_targets_or_error(launch_numa_mode="full")

    assert not both_errors
    assert set(full["frontdoor"]["ports"]) < set(both["frontdoor"]["ports"])


def test_standalone_guard_uses_declared_mode_despite_ambient_and_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topology = tmp_path / "orchestration" / "stack_topology.yaml"
    topology.parent.mkdir(parents=True)
    _write_yaml(topology, {"numa_mode": "both"})
    scanner_root = tmp_path / "scanner-root"
    scanner_topology = scanner_root / "orchestration" / "stack_topology.yaml"
    scanner_topology.parent.mkdir(parents=True)
    _write_yaml(scanner_topology, {"numa_mode": "quarter"})
    monkeypatch.setattr(stack_change_guard, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")

    def fail_if_fleet_probed() -> str:
        raise AssertionError("standalone guard must not probe fleet")

    monkeypatch.setattr(
        stack_change_guard, "_realized_launch_numa_mode", fail_if_fleet_probed
    )
    calls: dict[str, object] = {}

    def fake_validate(_priors, **kwargs):
        calls.update(kwargs)
        return GuardResult(errors=[], warnings=[])

    monkeypatch.setattr(stack_change_guard, "validate_stack_priors", fake_validate)

    result = stack_change_guard_main(
        ["--skip-hardcoded-surface-scan", "--repo-root", str(scanner_root)]
    )

    assert result == 0
    assert calls["launch_numa_mode"] == "both"


def test_standalone_guard_explicit_mode_overrides_declared_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    topology = tmp_path / "orchestration" / "stack_topology.yaml"
    topology.parent.mkdir(parents=True)
    topology.write_text("numa_mode: halves\n", encoding="utf-8")
    monkeypatch.setattr(stack_change_guard, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    calls: dict[str, object] = {}

    def fake_validate(_priors, **kwargs):
        calls.update(kwargs)
        return GuardResult(errors=[], warnings=[])

    monkeypatch.setattr(stack_change_guard, "validate_stack_priors", fake_validate)

    result = stack_change_guard_main(
        ["--skip-hardcoded-surface-scan", "--numa-mode", "quarter"]
    )

    assert result == 0
    assert calls["launch_numa_mode"] == "quarter"


@pytest.mark.parametrize(
    "topology_state",
    [
        "missing_file",
        "malformed_yaml",
        "not_mapping",
        "missing_mode",
        "invalid_mode",
        "unreadable",
    ],
)
def test_standalone_guard_fails_closed_on_unresolvable_declared_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    topology_state: str,
) -> None:
    topology = tmp_path / "orchestration" / "stack_topology.yaml"
    topology.parent.mkdir(parents=True)
    contents = {
        "malformed_yaml": "numa_mode: [\n",
        "not_mapping": "- full\n",
        "missing_mode": "{}\n",
        "invalid_mode": "numa_mode: halves\n",
        "unreadable": "numa_mode: both\n",
    }
    if topology_state != "missing_file":
        topology.write_text(contents[topology_state], encoding="utf-8")
    monkeypatch.setattr(stack_change_guard, "REPO_ROOT", tmp_path)
    validated = False

    def fake_validate(_priors, **_kwargs):
        nonlocal validated
        validated = True
        return GuardResult(errors=[], warnings=[])

    monkeypatch.setattr(stack_change_guard, "validate_stack_priors", fake_validate)
    if topology_state == "unreadable":
        real_read_text = Path.read_text

        def unreadable_read_text(path, *args, **kwargs):
            if path == topology:
                raise PermissionError("fixture denied read")
            return real_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable_read_text)

    result = stack_change_guard_main(["--skip-hardcoded-surface-scan"])

    output = capsys.readouterr().out
    assert result == 1
    assert output.startswith(f"{stack_change_guard.COULD_NOT_CHECK}:")
    assert not validated


def test_standalone_guard_rejects_invalid_explicit_mode_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(stack_change_guard, "REPO_ROOT", tmp_path)
    validated = False

    def fake_validate(_priors, **_kwargs):
        nonlocal validated
        validated = True
        return GuardResult(errors=[], warnings=[])

    monkeypatch.setattr(stack_change_guard, "validate_stack_priors", fake_validate)

    result = stack_change_guard_main(
        ["--skip-hardcoded-surface-scan", "--numa-mode", "halves"]
    )

    assert result == 1
    assert capsys.readouterr().out.startswith(f"{stack_change_guard.COULD_NOT_CHECK}:")
    assert not validated


@pytest.mark.parametrize("special_args", ["list", "staleness"])
def test_special_cli_commands_bypass_standalone_numa_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    special_args: str,
) -> None:
    monkeypatch.setattr(stack_change_guard, "REPO_ROOT", tmp_path)
    if special_args == "list":
        monkeypatch.setattr(
            stack_change_guard, "load_surface_manifest", lambda _path: ({}, [])
        )
        monkeypatch.setattr(
            stack_change_guard, "validate_surface_manifest", lambda _path: []
        )
        monkeypatch.setattr(
            stack_change_guard,
            "hardcoded_surface_rule_inventory",
            lambda **_kwargs: {},
        )
        args = ["--list-hardcoded-surface-rules", "--numa-mode", "invalid"]
    else:
        monkeypatch.setattr(
            stack_change_guard,
            "check_source_artifact_staleness",
            lambda *_args, **_kwargs: GuardResult(errors=[], warnings=[]),
        )
        args = ["--check-source-artifact-staleness", "--numa-mode", "invalid"]

    assert stack_change_guard_main(args) == 0


# ── Portable source pins (2026-09-24) ────────────────────────────────────────
# The compiled artifacts used to record the ABSOLUTE path of whichever checkout
# compiled them. Every worktree shares the committed artifacts, so a gate run in
# a worktree hash-checked the PRIMARY clone's files (and failed on another
# session's uncommitted edits there) and saw its descriptors as stale for path
# spelling alone. These tests pin the fix: in-repo pins are repo-relative, and a
# relative pin resolves against the checkout running the guard.


def test_in_repo_source_pins_are_repo_relative_and_external_ones_absolute(tmp_path: Path) -> None:
    from src.registry.model_descriptors import (
        portable_source_path,
        resolve_portable_source_path,
    )

    in_repo = stack_change_guard.REPO_ROOT / "orchestration" / "launch_manifest.yaml"
    assert portable_source_path(in_repo) == "orchestration/launch_manifest.yaml"
    assert resolve_portable_source_path("orchestration/launch_manifest.yaml") == in_repo.resolve()

    external = tmp_path / "model_registry.yaml"
    external.write_text("roles: {}\n", encoding="utf-8")
    assert portable_source_path(external) == str(external)
    assert resolve_portable_source_path(str(external)) == external


def test_relative_source_pin_resolves_against_guard_checkout_not_priors_dir(tmp_path: Path) -> None:
    priors_path = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    resolved = stack_change_guard._source_path(
        priors_path, {"path": "scripts/server/stack_numa.py"}
    )
    assert resolved == (stack_change_guard.REPO_ROOT / "scripts/server/stack_numa.py").resolve()
    assert resolved is not None and resolved.is_file()


def test_shipped_priors_and_descriptors_carry_no_checkout_absolute_pins() -> None:
    root = stack_change_guard.REPO_ROOT
    priors = yaml.safe_load(
        (root / "orchestration" / "derived" / "stack_priors.yaml").read_text(encoding="utf-8")
    )
    descriptors = yaml.safe_load(
        (root / "orchestration" / "model_descriptors.yaml").read_text(encoding="utf-8")
    )
    pins = dict(priors["source_artifacts"])
    pins.update({f"descriptors.{k}": v for k, v in descriptors["source_registries"].items()})
    for label, pin in pins.items():
        path = Path(pin["path"])
        if path.is_absolute():
            # Only files OUTSIDE this repository may be pinned absolutely.
            assert not str(path.resolve()).startswith(str(root.resolve()) + os.sep), label
            assert "epyc-orchestrator" not in path.parts, label
        else:
            assert (root / path).is_file(), label
    assert not Path(priors["precedence_spec"]).is_absolute()
