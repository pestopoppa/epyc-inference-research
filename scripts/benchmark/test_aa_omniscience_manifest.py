#!/usr/bin/env python3
"""Tests for the AA-Omniscience manifest generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import aa_omniscience_manifest as manifest


class _FakeConfig:
    def __init__(self, name: str, speed_test_only: bool = False):
        self.name = name
        self.speed_test_only = speed_test_only


class _FakeExecutor:
    def __init__(self, registry, validate: bool = False):
        self.registry = registry

    def get_configs_for_architecture(self, architecture: str, role: str, registry=None):
        return [_FakeConfig("baseline"), _FakeConfig("lookup_n0", speed_test_only=True)]


class _FakeRegistry:
    def __init__(self, roles: dict[str, dict[str, object]]):
        self._roles = roles

    def get_all_roles(self, include_deprecated: bool = False):
        return list(self._roles)

    def get_architecture(self, role: str) -> str:
        return self._roles[role].get("architecture", "dense")  # type: ignore[return-value]

    def get_tier(self, role: str):
        return self._roles[role].get("tier", "A")

    def get_model_path(self, role: str):
        return self._roles[role].get("model_path")


class _FakeAdapter:
    def __init__(self, total: int = 12):
        self._total = total
        self._sample = [
            {
                "id": f"q{i}",
                "suite": "omniscience",
                "prompt": f"prompt {i}",
                "expected": f"answer {i}",
                "scoring_method": "f1",
                "tier": 1,
            }
            for i in range(total)
        ]

    @property
    def total_available(self):
        return self._total

    def sample(self, n: int = 10, seed: int = 42):
        return self._sample[:n]


def test_build_manifest_includes_ready_role(monkeypatch):
    registry = _FakeRegistry(
        {
            "role_ready": {
                "architecture": "dense",
                "tier": "A",
                "model_path": str(Path("/tmp") / "model.gguf"),
            },
            "role_other": {
                "architecture": "dense",
                "tier": "B",
                "model_path": str(Path("/tmp") / "other.gguf"),
            },
        }
    )
    monkeypatch.setattr(manifest, "load_registry", lambda: registry)
    monkeypatch.setattr(manifest, "Executor", _FakeExecutor)
    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _FakeAdapter())
    monkeypatch.setattr(
        manifest,
        "get_suites_for_role",
        lambda role, registry=None: ["general", "omniscience"] if role == "role_ready" else ["general"],
    )
    monkeypatch.setattr(manifest.Path, "exists", lambda self: True)

    built, errors = manifest.build_manifest(sample_size=3)

    assert built["suite"] == "omniscience"
    assert built["dataset_validation"]["status"] == "ok"
    assert built["summary"]["roles_total"] == 1
    assert built["summary"]["roles_ready"] == 1
    assert not errors
    assert built["roles"][0]["role"] == "role_ready"
    assert "--suite omniscience" in built["roles"][0]["command"]
    assert "--skip-preflight" not in built["roles"][0]["command"]
    assert built["commands"][0]["role"] == "role_ready"


def test_build_manifest_flags_missing_model(monkeypatch):
    registry = _FakeRegistry(
        {
            "role_missing": {
                "architecture": "dense",
                "tier": "A",
                "model_path": None,
            }
        }
    )
    monkeypatch.setattr(manifest, "load_registry", lambda: registry)
    monkeypatch.setattr(manifest, "Executor", _FakeExecutor)
    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _FakeAdapter())
    monkeypatch.setattr(
        manifest,
        "get_suites_for_role",
        lambda role, registry=None: ["omniscience"],
    )

    built, errors = manifest.build_manifest(sample_size=2)

    assert built["summary"]["roles_total"] == 1
    assert built["summary"]["roles_blocked"] == 1
    assert built["roles"][0]["status"] == "blocked"
    assert any("model path missing" in err for err in errors)


def test_validate_adapter_rejects_bad_rows(monkeypatch):
    class _BrokenAdapter:
        @property
        def total_available(self):
            return 1

        def sample(self, n: int = 10, seed: int = 42):
            return [{"id": "q1", "suite": "omniscience"}]

    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _BrokenAdapter())

    dataset_validation, errors = manifest._validate_adapter(sample_size=1)
    assert dataset_validation["status"] == "warn"
    assert errors


def test_write_outputs_comments_blocked_commands(tmp_path):
    built = {
        "generated_at": "2026-06-19T00:00:00+00:00",
        "suite": "omniscience",
        "commands": [
            {"role": "ready_role", "status": "ready", "command": "run ready"},
            {"role": "blocked_role", "status": "blocked", "command": "run blocked"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    commands_path = tmp_path / "commands.sh"

    manifest._write_outputs(built, manifest_path, commands_path)

    commands = commands_path.read_text()
    assert "\nrun ready\n" in commands
    assert "\nrun blocked\n" not in commands
    assert "# blocked: run blocked" in commands


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
