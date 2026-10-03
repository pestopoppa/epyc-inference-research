"""Tests for the canonical stack-change pipeline command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.registry import stack_change_pipeline as pipeline
from scripts.registry.render_stack_summary import render_current_stack_summary
from scripts.registry.stack_change_pipeline import (
    BENCHMARK_PREFLIGHT_TARGETS,
    PipelineReport,
    PipelineStep,
    PROMOTION_GATE_TARGETS,
    SIMULATED_FIXTURE_TARGET,
    SURFACE_INVENTORY_COMMAND,
    StackChangePipelineConfig,
    _print_report,
    run_stack_change_pipeline,
)

PROMOTION_GATE_COMMAND = "promotion_gate: run uv run pytest -q " + " ".join(
    PROMOTION_GATE_TARGETS
)
SURFACE_INVENTORY_LINE = f"surface_inventory: run {SURFACE_INVENTORY_COMMAND}"


@pytest.fixture(autouse=True)
def _clean_runtime_attestation(monkeypatch):
    monkeypatch.setattr(pipeline, "_runtime_attestation_warnings", lambda: [])
    # Unit tests never read the live /proc: pin the declared-env attestation to a clean result.
    from scripts.server.env_attestation import EnvAttestation

    monkeypatch.setattr(
        pipeline,
        "_declared_env_attestation_result",
        lambda: EnvAttestation(compared=["fixture:0 pid 0 (fixture, 0 declared keys)"]),
    )


@pytest.fixture(autouse=True)
def _pin_realized_compile_mode(monkeypatch):
    """ESC-8 Fix 6: the update/check compile now resolves the NUMA mode from the
    realized fleet (a bare TCP probe) or refuses when nothing is listening. Pin
    it deterministically to the launcher ``full`` default so these pipeline
    tests stay hermetic (no live sockets) and keep their pre-Fix-6 expectations.
    The guard's launch-view probe is pinned to no-signal for the same reason
    (env fallback governs, matching the fixtures)."""
    from scripts.validate import stack_change_guard
    from src.registry import stack_priors

    monkeypatch.setattr(stack_priors, "_realized_compile_numa_mode", lambda **_kw: "full")
    monkeypatch.setattr(stack_change_guard, "_realized_launch_numa_mode", lambda: None)


def test_promotion_gate_includes_benchmark_preflight_regressions() -> None:
    assert set(BENCHMARK_PREFLIGHT_TARGETS).issubset(PROMOTION_GATE_TARGETS)


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _registry(path: Path, *, throughput: float = 24.3) -> Path:
    return _write_yaml(
        path,
        {
            "process_layout": {
                "hot_resident": ["frontdoor"],
                "warm_mmap": [],
            },
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "frontdoor",
                    "memory_gb": 37,
                    "throughput": throughput,
                    "benchmark_score": "170/183 (92.9%)",
                    "benchmark_date": "2026-05-04",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                }
            },
            "roles": {
                "frontdoor": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q8_0",
                        "quant": "Q8_0",
                        "architecture": "qwen35moe",
                        "size_gb": 37,
                        "ctx_max": 131072,
                    },
                    "performance": {
                        "quality_pct": 93,
                        "baseline_tps": throughput,
                        "benchmark_date": "2026-05-04",
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                }
            },
        },
    )


def _procedure(path: Path) -> Path:
    return _write_yaml(
        path,
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "description": "Role assignment",
                    "validation": {"enum": ["frontdoor"]},
                }
            ]
        },
    )


def _schema(path: Path) -> Path:
    return _write_json(
        path,
        {
            "properties": {
                "permissions": {
                    "properties": {
                        "roles": {
                            "items": {
                                "enum": ["frontdoor", "admin"],
                            }
                        }
                    }
                }
            }
        },
    )


def _config(tmp_path: Path, *, mode: str) -> StackChangePipelineConfig:
    repo_root = tmp_path
    registry = _registry(tmp_path / "orchestration" / "model_registry.yaml")
    descriptors = tmp_path / "orchestration" / "model_descriptors.yaml"
    priors = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    operator_summary = tmp_path / "docs" / "generated" / "current_stack_summary.md"
    procedure = _procedure(tmp_path / "orchestration" / "procedures" / "add_model.yaml")
    schema = _schema(tmp_path / "orchestration" / "procedure.schema.json")
    return StackChangePipelineConfig(
        mode=mode,  # type: ignore[arg-type]
        repo_root=repo_root,
        lean_registry=registry,
        research_registry=None,
        descriptors=descriptors,
        stack_priors=priors,
        operator_summary=operator_summary,
        procedure=procedure,
        schema=schema,
        surface_exceptions=tmp_path / "missing_exceptions.yaml",
        roles={"frontdoor"},
        allow_known_gaps=True,
        # NIB2-69: check refuses to run without a resolvable NUMA mode. These
        # fixtures carry no stack_topology.yaml, so they declare the lineup
        # explicitly — the same "full" the realized-compile pin above supplies.
        numa_mode="full",
    )


def test_update_merges_shared_alias_mismatch_into_runtime_descriptor(tmp_path: Path) -> None:
    """An alias that declares a DIFFERENT model merges into its host's descriptor.

    Re-fixtured for the 2026-09-22 lineup cutover (860b0b2d). The old fixture made
    worker_general the PRIMARY of a gemma4 ``server_mode.worker`` row on :8072.
    The guard's launch view is built from the REAL launch manifest, where
    worker_general is now an alias on frontdoor's :8070, so it resolved the role to
    a frontdoor host the tmp registry did not declare and correctly reported
    COULD-NOT-CHECK on the requirement comparison. The fixture now mirrors the
    current shape: frontdoor hosts the worker lane via ``shared_with``;
    worker_general declares the host's model, while worker_math declares a
    different one (the conflict under test). Expected ids come from the compiler's
    own identity helper, not literals.
    """
    from src.registry.model_descriptors import _model_id_from_configs

    config = _config(tmp_path, mode="update")
    registry = yaml.safe_load(config.lean_registry.read_text(encoding="utf-8"))
    host_cfg = registry["server_mode"]["frontdoor"]
    host_cfg["shared_with"] = ["worker_general", "worker_math"]
    registry["process_layout"]["hot_resident"] = ["frontdoor", "worker_general", "worker_math"]
    host_role_cfg = registry["roles"]["frontdoor"]
    # The base fixture spells the server GGUF `Qwen_Qwen3.6-...` but the role model
    # `Qwen3.6-...`; a direct binding never compares the two, but a shared_with
    # alias does (Role-server conflict). Spell them consistently, as the real
    # master does (server `Qwen3.6-35B-A3B-MTP-Q8_0.gguf` / role
    # `Qwen3.6-35B-A3B-MTP-Q8_0`), so the only model conflict is worker_math's.
    host_gguf = f"{host_role_cfg['model']['name']}.gguf"
    host_cfg["model"] = host_gguf
    host_cfg["model_path"] = f"/models/{host_gguf}"
    registry["roles"]["worker_general"] = {
        # Same model as the host: an alias, not a conflict.
        "model": dict(host_role_cfg["model"]),
        "performance": {"quality_pct": 90, "baseline_tps": 44.7},
        "acceleration": {"type": "none", "lookup": False},
        "memory": {"pinned": True, "residency": "hot"},
    }
    registry["roles"]["worker_math"] = {
        "model": {
            "name": "Qwen2.5-Math-7B-Instruct",
            "quant": "Q4_K_M",
            "architecture": "dense",
            "size_gb": 4.4,
            "ctx_max": 32768,
        },
        "performance": {"quality_pct": 88, "baseline_tps": 12.4},
        "acceleration": {"type": "none", "lookup": False},
        "memory": {"pinned": True, "residency": "hot"},
    }
    _write_yaml(config.lean_registry, registry)
    roles = {"frontdoor", "worker_general", "worker_math"}
    conflict_config = StackChangePipelineConfig(**{**config.__dict__, "roles": roles})
    _write_yaml(
        conflict_config.procedure,
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "validation": {"enum": sorted(roles)},
                }
            ]
        },
    )
    _write_json(
        conflict_config.schema,
        {
            "properties": {
                "permissions": {
                    "properties": {
                        "roles": {
                            "items": {
                                "enum": [*sorted(roles), "admin"],
                            }
                        }
                    }
                }
            }
        },
    )

    report = run_stack_change_pipeline(conflict_config)

    assert report.ok, [(step.name, step.errors) for step in report.steps if step.errors]
    host_model_id = _model_id_from_configs(host_role_cfg)
    math_model_id = _model_id_from_configs(registry["roles"]["worker_math"])
    assert host_model_id and math_model_id and host_model_id != math_model_id
    descriptors = yaml.safe_load(conflict_config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [host_model_id]
    model = descriptors["models"][0]
    assert model["role_bindings"]["roles"] == sorted(roles)
    assert model["role_bindings"]["alias_overrides"] == [
        {
            "role": "worker_math",
            "served_by": "frontdoor",
            "ignored_model_id": math_model_id,
            "reason": "server_mode.shared_with runtime takes precedence",
        }
    ]
    assert not any("ignored non-live role model metadata" in gap for gap in model["known_gaps"])
    assert not any(
        gap.startswith("Role-server conflict:")
        for gap in model["known_gaps"]
    )
    assert conflict_config.stack_priors.exists()


def test_check_reports_shared_alias_mismatch_without_conflict_error(
    tmp_path: Path,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    config = StackChangePipelineConfig(
        **{**update_config.__dict__, "mode": "check"}
    )
    _write_yaml(
        config.lean_registry,
        {
            "server_mode": {
                "worker": {
                    "url": "http://localhost:8072",
                    "port": 8072,
                    "tier": "hot",
                    "slots": 1,
                    "model": "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "model_role": "worker_general",
                    "shared_with": ["worker_math"],
                    "memory_gb": 16,
                    "throughput": 60.7,
                    "benchmark_score": "90%",
                    "runtime_requirements": {
                        "binary_dir": "/mnt/raid0/llm/ik_llama.cpp/build/bin",
                        "ld_library_path": [
                            "/mnt/raid0/llm/ik_llama.cpp/build/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/ggml/src",
                            "/mnt/raid0/llm/ik_llama.cpp/build/examples/mtmd",
                        ],
                    },
                    "numa_instances": 4,
                    "numa_ports": [8082, 8182, 8282, 8382],
                }
            },
            "roles": {
                "worker_general": {
                    "model": {
                        "name": "gemma-4-26B-A4B-it-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "gemma4",
                        "size_gb": 16,
                        "ctx_max": 16384,
                    },
                    "performance": {"quality_pct": 90, "baseline_tps": 44.7},
                    "acceleration": {"type": "speculative_decoding", "spec_type": "mtp"},
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "worker_math": {
                    "model": {
                        "name": "Qwen2.5-Math-7B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "dense",
                        "size_gb": 4.4,
                        "ctx_max": 32768,
                    },
                    "performance": {"quality_pct": 88, "baseline_tps": 12.4},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                }
            },
        },
    )
    conflict_config = StackChangePipelineConfig(
        **{**config.__dict__, "roles": {"worker_general", "worker_math"}}
    )

    report = run_stack_change_pipeline(conflict_config)

    descriptor_step = next(step for step in report.steps if step.name == "descriptors")
    assert not report.ok
    assert any("descriptor artifact is stale" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert not any("descriptor generated role/server conflict" in error for error in descriptor_step.errors)


def test_update_then_check_succeeds_with_known_gaps_allowed(tmp_path: Path) -> None:
    update_report = run_stack_change_pipeline(_config(tmp_path, mode="update"))
    check_report = run_stack_change_pipeline(_config(tmp_path, mode="check"))

    assert update_report.ok
    assert check_report.ok
    assert {step.name for step in check_report.steps} == {
        # 2026-07-31: master -> lean is now the pipeline's FIRST step. Before this
        # it was absent entirely and the pipeline regenerated descriptors and
        # priors from a lean registry nothing kept fresh, reporting green over a
        # stale input. This set is the step inventory of record — an addition here
        # must be deliberate.
        # 2026-09-15 NIB2-69: numa_mode leads — the ONE lineup the compile and the
        # guard's launch view both evaluate, printed with its provenance.
        "numa_mode",
        # 2026-09-23 SSU-F4: the shared_with recompute-and-diff runs second, before
        # the lean compile, so all four restating surfaces (port_map,
        # role_launch_meta, numa_config, roles.<alias>.model) fail in ONE run
        # instead of one class per run. It SKIPS when this config declares no
        # master registry / launch manifest / topology.
        "shared_with_derivations",
        "lean_registry",
        "descriptors",
        "stack_priors",
        "procedure_enums",
        "operator_summary",
        "guard",
        "guard_all_surfaces",
        "guard_strict",
        "reasoning_effort_certifications",
        # 2026-10-03 STACKCHG-DFLASH2: the import-time capacity gate re-run in a fresh
        # process on the lean THIS run wrote (update was green over an over-capacity lean).
        "serving_shape_capacity",
        "stack_manifest_registry",
        "q_scorer_priors",
        "runtime_attestation",
        # 2026-09-26: declared-vs-live launch env (/proc/<pid>/environ). Added because
        # every declared GGML knob was dropped at launch for eight weeks (b060dd56) and
        # runtime_attestation never compared the env blocks.
        "declared_env_attestation",
        "simulated_fixtures",
        "promotion_gate",
    }
    manifest_step = next(
        step for step in check_report.steps if step.name == "stack_manifest_registry"
    )
    assert manifest_step.status == "ok"
    q_scorer_step = next(step for step in check_report.steps if step.name == "q_scorer_priors")
    assert q_scorer_step.status == "ok"
    runtime_step = next(step for step in check_report.steps if step.name == "runtime_attestation")
    assert runtime_step.status == "ok"
    operator_step = next(step for step in check_report.steps if step.name == "operator_summary")
    assert operator_step.status == "ok"
    check_config = _config(tmp_path, mode="check")
    assert check_config.operator_summary.read_text(
        encoding="utf-8"
    ) == render_current_stack_summary(
        stack_priors_path=check_config.stack_priors,
        registry_path=check_config.lean_registry,
        descriptor_path=check_config.descriptors,
    )
    promotion_step = next(step for step in check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "reference"
    assert any(PROMOTION_GATE_COMMAND.removeprefix("promotion_gate: run ") in detail for detail in promotion_step.details)
    assert check_report.acceptance_lines() == [
        "acceptance: no-inference checks passed",
        PROMOTION_GATE_COMMAND,
        SURFACE_INVENTORY_LINE,
    ]


def test_check_rejects_production_blocker_surface_waiver_by_default(
    tmp_path: Path,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    surface_path = tmp_path / "scripts" / "benchmark" / "seeding_rewards.py"
    surface_path.parent.mkdir(parents=True)
    surface_path.write_text(
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
                    "rationale": "temporary fixture for fail-closed pipeline behavior",
                    "expires": "2099-01-01",
                }
            ]
        },
    )
    check_config = StackChangePipelineConfig(
        **{
            **update_config.__dict__,
            "mode": "check",
            "surface_exceptions": exceptions,
        }
    )

    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("--allow-production-blocker-waivers" in error for error in report.errors)
    assert any(
        "hardcoded_surface.waived.production_blocker" in warning
        for warning in report.warnings
    )


def test_acceptance_lines_summarize_unique_hardcoded_surface_warnings() -> None:
    duplicate_warning = (
        "hardcoded_surface.production_blocker.retired_role_in_active_code: "
        "src/example.py:1: retired_role"
    )
    report = PipelineReport(
        steps=[
            PipelineStep(
                name="guard",
                status="warnings",
                warnings=[
                    duplicate_warning,
                    duplicate_warning,
                    "hardcoded_surface.waived.production_blocker.retired_role_in_active_code: src/example.py:2",
                    "hardcoded_surface.legacy_test.retired_role_in_tests: tests/example.py:3",
                    "hardcoded_surface.historical_doc.retired_role_in_operator_docs: docs/example.md:4",
                    "role 'example' has 1 known gap(s)",
                ],
            )
        ]
    )

    assert report.hardcoded_surface_warning_counts() == {
        "production_blocker": 1,
        "waived_production_blocker": 1,
        "legacy_test": 1,
        "historical_doc": 1,
    }
    assert report.acceptance_lines() == [
        "acceptance: no-inference checks passed",
        "warnings: 5 unique (6 total)",
        "surface_warnings: production_blocker=1, waived_production_blocker=1, legacy_test=1, historical_doc=1",
        PROMOTION_GATE_COMMAND,
        SURFACE_INVENTORY_LINE,
    ]


def test_print_report_includes_promotion_gate_for_passing_check(
    tmp_path: Path,
    capsys,
) -> None:
    assert run_stack_change_pipeline(_config(tmp_path, mode="update")).ok
    report = run_stack_change_pipeline(_config(tmp_path, mode="check"))

    _print_report(report)

    output = capsys.readouterr().out
    assert "summary: ok" in output
    assert "acceptance: no-inference checks passed" in output
    assert PROMOTION_GATE_COMMAND in output
    assert SURFACE_INVENTORY_LINE in output
    assert SIMULATED_FIXTURE_TARGET in output


def test_print_report_blocks_promotion_for_failed_check(
    tmp_path: Path,
    capsys,
) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    _registry(config.lean_registry, throughput=42.0)
    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    _print_report(report)

    output = capsys.readouterr().out
    assert not report.ok
    assert "summary: failed" in output
    assert "acceptance: blocked" in output
    assert "promotion_gate: fix " in output


def test_run_promotion_gate_executes_combined_no_inference_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        if cmd[:4] != ["uv", "run", "pytest", "-q"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(cmd, 0, stdout="53 passed\n", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    config = StackChangePipelineConfig(
        **{
            **update_config.__dict__,
            "run_promotion_gate": True,
            "mode": "check",
        }
    )
    report = run_stack_change_pipeline(config)
    promotion_step = next(step for step in report.steps if step.name == "promotion_gate")

    assert report.ok
    assert captured["cmd"] == ["uv", "run", "pytest", "-q", *PROMOTION_GATE_TARGETS]
    assert captured["cwd"] == tmp_path
    assert promotion_step.status == "ok"
    assert any("53 passed" in detail for detail in promotion_step.details)


def test_q_scorer_prior_source_errors_block_promotion_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    monkeypatch.setattr(
        pipeline,
        "validate_live_q_scorer_prior_sources",
        lambda _path: [
            "live q_scorer role 'frontdoor' uses throughput source "
            "degraded_fallback; expected stack_priors"
        ],
    )

    config = StackChangePipelineConfig(
        **{
            **update_config.__dict__,
            "run_promotion_gate": True,
            "mode": "check",
        }
    )
    report = run_stack_change_pipeline(config)

    q_scorer_step = next(step for step in report.steps if step.name == "q_scorer_priors")
    promotion_step = next(step for step in report.steps if step.name == "promotion_gate")
    assert not report.ok
    assert q_scorer_step.status == "failed"
    assert q_scorer_step.errors == [
        "live q_scorer role 'frontdoor' uses throughput source "
        "degraded_fallback; expected stack_priors"
    ]
    assert promotion_step.status == "skipped"


def test_runtime_attestation_warnings_block_promotion_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    monkeypatch.setattr(
        pipeline,
        "_runtime_attestation_warnings",
        lambda: ["frontdoor pid 123 expected current.gguf; live cmdline has stale.gguf"],
    )

    def fake_run(cmd, **_kwargs):
        if cmd[:4] == ["uv", "run", "pytest", "-q"]:
            raise AssertionError("promotion gate should be skipped on runtime drift")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    config = StackChangePipelineConfig(
        **{
            **update_config.__dict__,
            "run_promotion_gate": True,
            "mode": "check",
        }
    )
    report = run_stack_change_pipeline(config)

    runtime_step = next(step for step in report.steps if step.name == "runtime_attestation")
    promotion_step = next(step for step in report.steps if step.name == "promotion_gate")
    assert not report.ok
    assert runtime_step.status == "failed"
    assert runtime_step.errors == [
        "live process drift: frontdoor pid 123 expected current.gguf; live cmdline has stale.gguf"
    ]
    assert promotion_step.status == "skipped"


def test_stack_manifest_registry_warnings_block_promotion_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    monkeypatch.setattr(
        pipeline,
        "_stack_manifest_registry_warnings",
        lambda _config: ["role 'frontdoor': PORT_MAP says port 8071"],
    )

    def fake_run(cmd, **_kwargs):
        if cmd[:4] == ["uv", "run", "pytest", "-q"]:
            raise AssertionError("promotion gate should be skipped on manifest drift")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    config = StackChangePipelineConfig(
        **{
            **update_config.__dict__,
            "run_promotion_gate": True,
            "mode": "check",
        }
    )
    report = run_stack_change_pipeline(config)

    manifest_step = next(
        step for step in report.steps if step.name == "stack_manifest_registry"
    )
    q_scorer_step = next(step for step in report.steps if step.name == "q_scorer_priors")
    promotion_step = next(step for step in report.steps if step.name == "promotion_gate")
    assert not report.ok
    assert manifest_step.status == "failed"
    assert manifest_step.errors == [
        "stack manifest registry drift: role 'frontdoor': PORT_MAP says port 8071"
    ]
    assert q_scorer_step.status == "skipped"
    assert promotion_step.status == "skipped"


def test_check_reports_stale_generated_artifact_without_writing(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _registry(config.lean_registry, throughput=42.0)

    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    descriptor_step = next(step for step in report.steps if step.name == "descriptors")
    assert any("artifact is stale" in error for error in report.errors)
    assert any("descriptor changed qwen3.6-35b-a3b-q8_0" in detail for detail in descriptor_step.details)
    assert any(
        "operator decision required: review descriptor drift details" in detail
        for detail in descriptor_step.details
    )
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before


def test_check_reports_stale_operator_summary_without_writing(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    summary_before = config.operator_summary.read_text(encoding="utf-8")
    config.operator_summary.write_text("stale\n", encoding="utf-8")

    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    summary_step = next(step for step in report.steps if step.name == "operator_summary")
    assert summary_step.status == "stale"
    assert any("operator stack summary is stale" in error for error in summary_step.errors)
    assert config.operator_summary.read_text(encoding="utf-8") == "stale\n"
    config.operator_summary.write_text(summary_before, encoding="utf-8")


def test_update_refuses_to_remove_existing_descriptor_model_ids(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    _write_yaml(
        config.descriptors,
        {
            "models": [
                {
                    "model_id": "qwen3.6-35b-a3b-q8_0",
                },
                {
                    "model_id": "benchmark-only-reap",
                },
            ]
        },
    )
    descriptor_before = config.descriptors.read_text(encoding="utf-8")

    report = run_stack_change_pipeline(config)

    assert not report.ok
    assert any("benchmark-only-reap" in error for error in report.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptor_before
    assert not config.stack_priors.exists()
    assert not config.operator_summary.exists()
    assert {step.name: step.status for step in report.steps}["stack_priors"] == "skipped"
    assert {step.name: step.status for step in report.steps}["operator_summary"] == "skipped"


def test_check_reports_descriptor_model_removal_blocker(tmp_path: Path) -> None:
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    loaded = yaml.safe_load(update_config.descriptors.read_text(encoding="utf-8"))
    loaded["models"].append({"model_id": "benchmark-only-reap"})
    _write_yaml(update_config.descriptors, loaded)

    config = StackChangePipelineConfig(
        **{**update_config.__dict__, "mode": "check"}
    )

    report = run_stack_change_pipeline(config)

    assert not report.ok
    descriptor_step = next(step for step in report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any(
        "descriptor generated removes model_id(s): benchmark-only-reap" in detail
        for detail in descriptor_step.details
    )
    assert any("benchmark-only-reap" in error for error in descriptor_step.errors)
    assert any(
        "operator decision required: descriptor generation removes model_id(s)" in detail
        for detail in descriptor_step.details
    )
    assert not any(
        error.endswith("stack_change_pipeline.py update") for error in descriptor_step.errors
    )


def test_check_fails_on_stale_procedure_enums(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(config).ok
    retired_role = "architect" + "_coding"
    _procedure(config.procedure).write_text(
        f"inputs:\n- name: role\n  validation:\n    enum: [{retired_role}]\n",
        encoding="utf-8",
    )

    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check"}
    )
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("procedure role enums are stale" in error for error in report.errors)


# ---------------------------------------------------------------------------
# NIB2-69 (2026-09-15): `check` evaluates ONE explicit NUMA lineup and any
# launch-manifest/port error fails it.
#
# Root cause of the 39 errors on origin/main: the compile resolved the declared
# topology mode (`both`) while the guard's launch view fell through realized
# fleet -> ambient env -> "full" in a clean shell, filtering the half instances
# out of the view. 13 half-port mismatches x 3 guard steps = 39.
# ---------------------------------------------------------------------------


def _topology(path: Path, mode: str) -> Path:
    return _write_yaml(path, {"schema_version": "stack_topology.v1", "numa_mode": mode})


def test_resolution_uses_declared_topology_and_ignores_ambient_env(tmp_path: Path) -> None:
    config = StackChangePipelineConfig(
        mode="check",
        repo_root=tmp_path,
        stack_topology=_topology(tmp_path / "stack_topology.yaml", "both"),
    )

    resolution = pipeline.resolve_pipeline_numa_mode(
        config, environ={"ORCHESTRATOR_STACK_NUMA_MODE": "full"}
    )

    assert resolution.mode == "both"
    assert resolution.source == "declared:stack_topology.yaml"
    assert resolution.errors == ()
    assert any("IGNORED" in warning and "'full'" in warning for warning in resolution.warnings)


def test_resolution_explicit_mode_wins_and_says_it_is_not_production(tmp_path: Path) -> None:
    config = StackChangePipelineConfig(
        mode="check",
        repo_root=tmp_path,
        stack_topology=_topology(tmp_path / "stack_topology.yaml", "both"),
        numa_mode="quarter",
    )

    resolution = pipeline.resolve_pipeline_numa_mode(config, environ={})

    assert resolution.mode == "quarter"
    assert resolution.source == "explicit:--numa-mode"
    assert any("does NOT evaluate production" in warning for warning in resolution.warnings)


def test_resolution_rejects_invalid_explicit_mode(tmp_path: Path) -> None:
    config = StackChangePipelineConfig(mode="check", repo_root=tmp_path, numa_mode="halves")

    resolution = pipeline.resolve_pipeline_numa_mode(config, environ={})

    assert resolution.mode is None
    assert resolution.errors and "invalid --numa-mode" in resolution.errors[0]


def test_resolution_without_declaration_fails_check_but_not_update(tmp_path: Path) -> None:
    check = pipeline.resolve_pipeline_numa_mode(
        StackChangePipelineConfig(mode="check", repo_root=tmp_path), environ={}
    )
    update = pipeline.resolve_pipeline_numa_mode(
        StackChangePipelineConfig(mode="update", repo_root=tmp_path), environ={}
    )

    assert check.mode is None and check.errors
    assert "refuses to inherit a lineup" in check.errors[0]
    assert update.mode is None and update.errors == ()
    assert any("realized-fleet probe" in warning for warning in update.warnings)


def test_check_without_resolvable_numa_mode_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ORCHESTRATOR_STACK_NUMA_MODE", raising=False)
    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    check_config = StackChangePipelineConfig(
        **{**update_config.__dict__, "mode": "check", "numa_mode": None}
    )

    report = run_stack_change_pipeline(check_config)

    numa_step = report.steps[0]
    assert numa_step.name == "numa_mode"
    assert numa_step.status == "failed"
    assert not report.ok


def test_check_records_evaluated_numa_mode_and_threads_it_into_every_guard_step(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.validate import stack_change_guard

    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    seen_modes: list[object] = []
    real_view = stack_change_guard._launch_manifest_targets_or_error

    def recording_view(**kwargs):
        seen_modes.append(kwargs.get("launch_numa_mode"))
        return real_view(**kwargs)

    monkeypatch.setattr(stack_change_guard, "_launch_manifest_targets_or_error", recording_view)
    # An ambient export that disagrees must not leak into the guard's view.
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "both")
    check_config = StackChangePipelineConfig(**{**update_config.__dict__, "mode": "check"})

    report = run_stack_change_pipeline(check_config)

    numa_step = report.steps[0]
    assert numa_step.name == "numa_mode"
    assert numa_step.details == ["evaluated numa_mode: full (source: explicit:--numa-mode)"]
    assert any("IGNORED" in warning for warning in numa_step.warnings)
    assert seen_modes == ["full", "full", "full"]
    assert report.ok


def test_check_fails_on_synthetic_launch_manifest_port_error_even_with_known_gaps(
    tmp_path: Path, monkeypatch
) -> None:
    """One manifest/port mismatch must fail `check`, and --allow-known-gaps must
    neither pass it nor relabel it as a known gap."""
    from scripts.validate import stack_change_guard

    update_config = _config(tmp_path, mode="update")
    assert run_stack_change_pipeline(update_config).ok
    assert update_config.allow_known_gaps is True
    real_view = stack_change_guard._launch_manifest_targets_or_error

    def drifted_view(**kwargs):
        targets, errors = real_view(**kwargs)
        frontdoor = targets["frontdoor"]
        frontdoor["ports"] = [*frontdoor["ports"], 8999]
        return targets, errors

    monkeypatch.setattr(stack_change_guard, "_launch_manifest_targets_or_error", drifted_view)
    check_config = StackChangePipelineConfig(**{**update_config.__dict__, "mode": "check"})

    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    for name in ("guard", "guard_all_surfaces", "guard_strict"):
        step = next(step for step in report.steps if step.name == name)
        assert step.status == "failed", name
        assert any("missing launch manifest port(s) [8999]" in e for e in step.errors), name
    strict_step = next(step for step in report.steps if step.name == "guard_strict")
    assert not any("8999" in warning for warning in strict_step.warnings)
    assert report.acceptance_lines()[0] == "acceptance: blocked"


def test_production_topology_declares_both_and_launch_alignment_is_clean_in_it() -> None:
    """The committed priors align with the launch manifest in the DECLARED mode.

    The `full` half of this test pins the root cause: evaluating the launch view
    for a lineup the priors were not compiled for manufactures half-port errors.
    """
    from scripts.validate import stack_change_guard

    mode, source = pipeline.resolve_declared_numa_mode(pipeline.DEFAULT_STACK_TOPOLOGY)
    assert (mode, source) == ("both", "declared:stack_topology.yaml")
    priors = yaml.safe_load(stack_change_guard.DEFAULT_PRIORS.read_text(encoding="utf-8"))

    production = stack_change_guard.validate_launch_manifest_serving_alignment(
        priors, launch_numa_mode=mode
    )
    wrong_lineup = stack_change_guard.validate_launch_manifest_serving_alignment(
        priors, launch_numa_mode="full"
    )

    assert production == []
    assert any("include non-launch port(s)" in error for error in wrong_lineup)


def test_lean_check_judges_committed_content_not_the_gitignored_cache_key(tmp_path: Path) -> None:
    """NIB2-69: `.lean_cache_key` is gitignored and per-clone. A fresh worktree
    (no key) whose committed lean equals the master projection is fresh; a lean
    that differs from the projection is stale even when the key matches."""
    from src.registry.registry_compiler import cache_key, compile_lean

    master = _registry(tmp_path / "master" / "model_registry.yaml")
    lean = tmp_path / "orchestration" / "model_registry.yaml"
    roles = {"frontdoor"}
    _write_yaml(lean, compile_lean(master, roles))
    config = StackChangePipelineConfig(
        mode="check", repo_root=tmp_path, lean_registry=lean, research_registry=master, roles=roles
    )

    no_key = pipeline._lean_registry_step(config, check=True)
    assert no_key.status == "ok", no_key.errors
    assert any("local cache key (gitignored): <none> !=" in d for d in no_key.details)

    # DRAFT-SEL-1: the drafter selection is a compile input, so the key is taken over
    # THIS config's topology (as the step does), not the repo default.
    (lean.parent / ".lean_cache_key").write_text(
        cache_key(master, roles, pipeline._topology_path(config))
    )
    _registry(lean, throughput=99.0)  # hand-edited lean, key still matches master
    edited = pipeline._lean_registry_step(config, check=True)
    assert edited.status == "stale"
    assert "local cache key: matches" in edited.errors[0]


def test_declared_env_attestation_step_maps_verdicts(monkeypatch) -> None:
    from scripts.server.env_attestation import EnvAttestation

    monkeypatch.setattr(pipeline, "_declared_env_attestation_result",
                        lambda: EnvAttestation(compared=["x"], errors=["x: declared K='1' but live has it MISSING"]))
    failed = pipeline._declared_env_attestation_step()
    assert failed.status == "failed" and not failed.ok and "MISSING" in failed.errors[0]

    monkeypatch.setattr(pipeline, "_declared_env_attestation_result", lambda: EnvAttestation())
    skipped = pipeline._declared_env_attestation_step()
    assert skipped.status == "skipped" and "COULD-NOT-CHECK" in skipped.warnings[0]

    def boom():
        raise RuntimeError("proc unreadable")
    monkeypatch.setattr(pipeline, "_declared_env_attestation_result", boom)
    assert pipeline._declared_env_attestation_step().status == "failed"


# --- STACKCHG-DFLASH2-20261003: capacity gate re-run on the lean this run wrote ---------

def _default_lean_config(tmp_path: Path) -> StackChangePipelineConfig:
    (tmp_path / "orchestration").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "server").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "server" / "stack_manifest.py").write_text("")
    return StackChangePipelineConfig(
        mode="check", repo_root=tmp_path,
        lean_registry=tmp_path / "orchestration" / "model_registry.yaml",
    )


def test_serving_shape_capacity_fails_when_fresh_import_refuses(monkeypatch, tmp_path) -> None:
    refusal = "ValueError: device ROCm0 (GPU) OVERSUBSCRIBED by 3.16 GiB"
    monkeypatch.setattr(
        pipeline, "_fresh_capacity_import",
        lambda root: subprocess.CompletedProcess([], 1, "", f"Traceback\n{refusal}\n"),
    )
    step = pipeline._serving_shape_capacity_step(_default_lean_config(tmp_path))
    assert step.status == "failed"
    assert any("OVERSUBSCRIBED by 3.16 GiB" in e for e in step.errors)


def test_serving_shape_capacity_passes_on_clean_import(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline, "_fresh_capacity_import",
                        lambda root: subprocess.CompletedProcess([], 0, "", ""))
    assert pipeline._serving_shape_capacity_step(_default_lean_config(tmp_path)).status == "ok"


def test_serving_shape_capacity_is_not_a_pass_for_a_non_default_lean(tmp_path) -> None:
    cfg = StackChangePipelineConfig(mode="check", repo_root=tmp_path,
                                    lean_registry=tmp_path / "elsewhere.yaml")
    step = pipeline._serving_shape_capacity_step(cfg)
    assert step.status == "skipped"
    assert step.ok is True and step.errors == []
    assert any("COULD-NOT-CHECK" in d for d in step.details)


def test_lean_bootstrap_roles_from_sources_match_role_launch_meta() -> None:
    """The bootstrap path (stack_manifest refused against a stale lean) must derive the
    SAME active set the compiler gets from ROLE_LAUNCH_META on a consistent tree."""
    from scripts.server.stack_manifest import ROLE_LAUNCH_META
    from src.registry.registry_compiler import active_roles_from_launch_meta

    root = pipeline.REPO_ROOT
    cfg = StackChangePipelineConfig(
        mode="update", repo_root=root,
        research_registry=root / "orchestration" / "model_registry_full.yaml",
    )
    assert active_roles_from_launch_meta(pipeline._launch_meta_from_sources(cfg)) == \
        active_roles_from_launch_meta(ROLE_LAUNCH_META)


# --- STACKCHG-KVPOOL-20261003: `update` must not compile priors from the lean it replaced ---

_SM = "scripts.server.stack_manifest"


@pytest.fixture
def pin_stack_manifest():
    """Swap BOTH bindings eviction touches (sys.modules and the package attribute) and
    put the real ones back at teardown, whatever eviction left behind."""
    import scripts.server as server_pkg

    had_mod, mod = _SM in sys.modules, sys.modules.get(_SM)
    had_attr, attr = "stack_manifest" in vars(server_pkg), vars(server_pkg).get("stack_manifest")

    def pin(value) -> None:
        sys.modules[_SM] = value
        server_pkg.stack_manifest = value

    yield pin
    if had_mod:
        sys.modules[_SM] = mod
    else:
        sys.modules.pop(_SM, None)
    if had_attr:
        server_pkg.stack_manifest = attr
    elif "stack_manifest" in vars(server_pkg):
        delattr(server_pkg, "stack_manifest")


def test_evict_lean_derived_modules_only_for_the_lean_stack_manifest_reads(
    pin_stack_manifest, tmp_path
) -> None:
    sentinel = object()
    pin_stack_manifest(sentinel)
    elsewhere = StackChangePipelineConfig(mode="update", repo_root=tmp_path,
                                          lean_registry=tmp_path / "elsewhere.yaml")
    assert pipeline._evict_lean_derived_modules(elsewhere) == []
    assert sys.modules[_SM] is sentinel

    default = StackChangePipelineConfig(
        mode="update", repo_root=tmp_path,
        lean_registry=tmp_path / "orchestration" / "model_registry.yaml")
    assert _SM in pipeline._evict_lean_derived_modules(default)
    assert _SM not in sys.modules


def test_lean_update_evicts_stack_manifest_only_when_the_lean_changed(
    pin_stack_manifest, tmp_path
) -> None:
    """Measured on the KVPOOL candidate: with stack_manifest imported before the lean was
    rewritten, ONE `update` wrote `effective_context_tokens: 196608` for a 393216 lean and
    the next `check` failed (evidence/pipeline-stale-priors-repro.txt)."""
    master = _registry(tmp_path / "master" / "model_registry.yaml")
    lean = tmp_path / "orchestration" / "model_registry.yaml"
    config = StackChangePipelineConfig(mode="update", repo_root=tmp_path, lean_registry=lean,
                                       research_registry=master, roles={"frontdoor"})
    sentinel = object()

    pin_stack_manifest(sentinel)
    first = pipeline._lean_registry_step(config, check=False)
    assert first.status == "ok", first.errors
    assert _SM not in sys.modules
    assert any("evicted scripts.server.stack_manifest" in d for d in first.details)

    pin_stack_manifest(sentinel)
    again = pipeline._lean_registry_step(config, check=False)  # same master: lean unchanged
    assert again.status == "ok", again.errors
    assert sys.modules[_SM] is sentinel
    assert not any("evicted" in d for d in again.details)

    _registry(master, throughput=99.0)  # master edit -> the lean changes -> evict again
    third = pipeline._lean_registry_step(config, check=False)
    assert _SM not in sys.modules and any("evicted" in d for d in third.details)
