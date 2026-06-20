#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import clean_window_manifest as manifest


class _FakeRegistry:
    data = {
        "server_mode": {
            "frontdoor": {"model_role": "frontdoor", "port": 8070, "url": "http://localhost:8070"},
            "worker": {"model_role": "worker_general", "port": 8072, "url": "http://localhost:8072"},
        }
    }

    def __init__(self):
        self.roles = {
            "frontdoor": {
                "tier": "A",
                "architecture": "moe_hybrid",
                "model_path": "/tmp/shared.gguf",
                "max_context": 32768,
            },
            "coder_escalation": {
                "tier": "A",
                "architecture": "moe_hybrid",
                "model_path": "/tmp/shared.gguf",
                "max_context": 32768,
            },
            "worker_general": {
                "tier": "B",
                "architecture": "gemma4",
                "model_path": "/tmp/worker.gguf",
                "max_context": 16384,
            },
            "ingest_long_context": {
                "tier": "B",
                "architecture": "ssm",
                "model_path": "/tmp/ingest.gguf",
                "max_context": 32768,
            },
        }

    def get_role_config(self, role):
        return self.roles.get(role)

    def get_model_path(self, role):
        item = self.roles.get(role)
        return item["model_path"] if item else None

    def get_tier(self, role):
        return self.roles[role]["tier"]

    def get_architecture(self, role):
        return self.roles[role]["architecture"]

    def get_max_context(self, role):
        return self.roles[role]["max_context"]


class _FakeAdapter:
    @property
    def total_available(self):
        return 10


def test_build_manifest_groups_by_model_path(monkeypatch):
    registry = _FakeRegistry()
    monkeypatch.setattr(manifest, "load_registry", lambda: registry)
    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _FakeAdapter())
    monkeypatch.setattr(
        manifest,
        "get_suites_for_role",
        lambda role, registry=None: ["omniscience", "tulving_episodic", "math", "gpqa"],
    )
    monkeypatch.setattr(manifest.Path, "exists", lambda self: str(self).startswith("/tmp/"))

    built = manifest.build_manifest(
        aa_roles=["frontdoor", "coder_escalation"],
        k_mem_roles=[],
        k_rope_roles=[],
        g5_roles=[],
        live_registry_path=None,
    )

    shared = [group for group in built["groups"] if group["model_path"] == "/tmp/shared.gguf"]
    assert len(shared) == 1
    assert shared[0]["roles"] == ["coder_escalation", "frontdoor"]
    assert shared[0]["summary"]["ready"] == 2


def test_rope_cells_block_when_context_exceeds_registered_max(monkeypatch, tmp_path):
    registry = _FakeRegistry()
    monkeypatch.setattr(manifest, "load_registry", lambda: registry)
    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _FakeAdapter())
    monkeypatch.setattr(manifest, "get_suites_for_role", lambda role, registry=None: ["omniscience"])
    monkeypatch.setattr(manifest.Path, "exists", lambda self: str(self).startswith("/tmp/"))

    built = manifest.build_manifest(
        aa_roles=[],
        k_mem_roles=[],
        k_rope_roles=["worker_general"],
        g5_roles=[],
        output_root=tmp_path,
        live_registry_path=None,
    )

    contexts = {
        entry["context_length"]: entry
        for entry in built["entries"]
        if entry["kind"] == "rope_position_probe"
    }
    assert contexts[4096]["status"] == "ready"
    assert " --api chat " in contexts[4096]["command"]
    assert contexts[32768]["status"] == "blocked"
    assert "exceeds registered max_context" in contexts[32768]["notes"][0]


def test_g5_is_blocked_when_runner_is_missing(monkeypatch):
    registry = _FakeRegistry()
    monkeypatch.setattr(manifest, "load_registry", lambda: registry)
    monkeypatch.setattr(manifest.Path, "exists", lambda self: str(self).startswith("/tmp/"))

    built = manifest.build_manifest(
        aa_roles=[],
        k_mem_roles=[],
        k_rope_roles=[],
        g5_roles=["frontdoor"],
        live_registry_path=None,
    )

    assert built["entries"][0]["package"] == "G5"
    assert built["entries"][0]["status"] == "blocked"
    assert built["entries"][0]["command"] is None


def test_xmas_entry_records_constrained_policy_command(monkeypatch, tmp_path):
    monkeypatch.setattr(
        manifest,
        "XMAS_LIVE_AB_SCRIPT",
        tmp_path / "epyc-orchestrator" / "scripts" / "benchmark" / "xmas_live_ab.py",
    )
    monkeypatch.setattr(
        manifest,
        "XMAS_HELDOUT_PROMPTS",
        tmp_path
        / "epyc-orchestrator"
        / "benchmarks"
        / "results"
        / "runs"
        / "xmas_live_ab"
        / "20260618-heldout-resilient"
        / "prompts.jsonl",
    )
    monkeypatch.setattr(manifest.Path, "exists", lambda self: True)

    entry = manifest._xmas_live_ab_entry()

    assert entry["package"] == "X-MAS"
    assert entry["status"] == "ready"
    assert "--host-quiet-confirmed" in entry["command"]
    assert "--reps 2" in entry["command"]
    assert "$(date -u +%Y%m%dT%H%M%SZ)-constrained-policy" in entry["command"]
    assert any("incumbent_constrained_v1" in note for note in entry["notes"])


def test_xmas_entry_blocks_when_prompt_manifest_missing(monkeypatch, tmp_path):
    runner = tmp_path / "xmas_live_ab.py"
    runner.touch()
    missing_prompts = tmp_path / "missing-prompts.jsonl"
    monkeypatch.setattr(manifest, "XMAS_LIVE_AB_SCRIPT", runner)
    monkeypatch.setattr(manifest, "XMAS_HELDOUT_PROMPTS", missing_prompts)

    entry = manifest._xmas_live_ab_entry()

    assert entry["status"] == "blocked"
    assert entry["command"] is None
    assert "X-MAS held-out prompt manifest missing" in entry["notes"]


def test_e1_entry_records_decision_grade_np_sweep_command(monkeypatch, tmp_path):
    runner = tmp_path / "server_np_sweep.py"
    runner.touch()
    monkeypatch.setattr(manifest, "SERVER_NP_SWEEP_SCRIPT", runner)

    entry = manifest._e1_batched_decode_entry()

    assert entry["package"] == "E1"
    assert entry["status"] == "ready"
    assert "scripts/benchmark/server_np_sweep.py" in entry["command"]
    assert "--np-levels 1,2,4,8,16" in entry["command"]
    assert "e1-pbench3-$(date -u +%Y%m%dT%H%M%SZ)" in entry["command"]
    assert any("refuses decision-grade execution" in note for note in entry["notes"])


def test_e2_entry_records_plan_generation_command(monkeypatch, tmp_path):
    runner = tmp_path / "e2_eval_driver_ab.py"
    runner.touch()
    monkeypatch.setattr(manifest, "E2_EVAL_DRIVER_AB_SCRIPT", runner)

    entry = manifest._e2_eval_driver_ab_entry()

    assert entry["package"] == "E2"
    assert entry["status"] == "ready"
    assert "scripts/benchmark/e2_eval_driver_ab.py" in entry["command"]
    assert "--batch-np 8" in entry["command"]
    assert "--current-concurrency 3" in entry["command"]
    assert any("generated commands.sh" in note for note in entry["notes"])


def test_queue2_entries_are_included_in_manifest(monkeypatch):
    registry = _FakeRegistry()
    monkeypatch.setattr(manifest, "load_registry", lambda: registry)
    monkeypatch.setattr(manifest.Path, "exists", lambda self: str(self).startswith("/tmp/"))
    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _FakeAdapter())
    monkeypatch.setattr(manifest, "get_suites_for_role", lambda role, registry=None: ["omniscience"])

    built = manifest.build_manifest(
        aa_roles=[],
        k_mem_roles=[],
        k_rope_roles=[],
        g5_roles=[],
        live_registry_path=None,
    )

    packages = {(entry["package"], entry["kind"]) for entry in built["entries"]}
    assert ("E1", "batched_decode_np_sweep") in packages
    assert ("E2", "eval_driver_ab_plan") in packages
    assert "Queue-2 E1/E2" in built["purpose"]


def test_suite_entry_blocks_when_live_registry_differs(monkeypatch):
    registry = _FakeRegistry()
    live_registry = {
        "roles": {
            "worker_general": {
                "tier": "C",
                "model": {
                    "path": "/tmp/live-worker.gguf",
                    "architecture": "gemma4",
                    "max_context": 16384,
                },
            }
        },
        "server_mode": {
            "worker": {
                "model_role": "worker_general",
                "port": 8072,
            }
        },
    }
    monkeypatch.setattr(manifest.Path, "exists", lambda self: str(self).startswith("/tmp/"))
    monkeypatch.setattr(manifest, "get_adapter", lambda suite: _FakeAdapter())
    monkeypatch.setattr(manifest, "get_suites_for_role", lambda role, registry=None: ["omniscience"])

    entry = manifest._suite_entry(
        registry,
        package="G11",
        role="worker_general",
        suite="omniscience",
        live_registry=live_registry,
    )

    assert entry["status"] == "blocked"
    assert entry["model"]["benchmark_registry_mismatch"] is True
    assert "run_benchmark.py would not measure the live role" in entry["notes"][0]
    assert "--server-mode" in entry["command"]
    assert "--skip-speed-tests" in entry["command"]


def test_rope_entry_uses_live_context_override(monkeypatch, tmp_path):
    registry = _FakeRegistry()
    monkeypatch.setattr(manifest.Path, "exists", lambda self: str(self).startswith("/tmp/"))

    entries = manifest._rope_entries(
        registry,
        "frontdoor",
        {"frontdoor": 8070},
        {"frontdoor": 16384},
        tmp_path,
        live_registry=None,
    )

    contexts = {entry["context_length"]: entry for entry in entries}
    assert contexts[8192]["status"] == "ready"
    assert contexts[16384]["status"] == "blocked"
    assert "needs chat-template headroom below live server context 16384" in contexts[16384]["notes"][0]


def test_write_outputs_comments_blocked_commands(tmp_path):
    built = {
        "generated_at": "2026-06-19T00:00:00+00:00",
        "groups": [
            {
                "model_path": "/tmp/model.gguf",
                "roles": ["frontdoor"],
                "entries": [
                    {
                        "package": "G11",
                        "kind": "run_benchmark_suite",
                        "role": "frontdoor",
                        "suite": "omniscience",
                        "status": "ready",
                        "command": "run ready",
                        "notes": [],
                    },
                    {
                        "package": "K-ROPE-1",
                        "kind": "rope_position_probe",
                        "role": "frontdoor",
                        "context_length": 32768,
                        "status": "blocked",
                        "command": "run blocked",
                        "notes": ["port unavailable"],
                    },
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    commands_path = tmp_path / "commands.sh"

    manifest._write_outputs(built, manifest_path, commands_path)

    commands = commands_path.read_text()
    assert "\nrun ready\n" in commands
    assert "\nrun blocked\n" not in commands
    assert "# blocked: run blocked" in commands
    assert "# note: port unavailable" in commands


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
