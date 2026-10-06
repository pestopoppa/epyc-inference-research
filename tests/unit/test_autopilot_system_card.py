"""Tests for AutoPilot W8 controller guidance split."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
AUTOPILOT_DIR = ROOT / "scripts" / "autopilot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AUTOPILOT_DIR))

gen_system_card = importlib.import_module("gen_system_card")
autopilot = importlib.import_module("autopilot")
render_stack_summary = importlib.import_module("scripts.registry.render_stack_summary")

LEGACY_ARCHITECT_ROLE = "architect" "_coding"


def _load_server_mode(registry_path: Path) -> dict:
    import yaml

    return yaml.safe_load(registry_path.read_text(encoding="utf-8"))["server_mode"]


def _live_host_of(role: str) -> str:
    """The LIVE registry server_mode row that physically serves ``role``.

    Follows the repository's own ``orchestration/model_registry.yaml`` (host row
    itself, ``model_role`` or ``shared_with``), never an alias row.
    """
    server_mode = _load_server_mode(ROOT / "orchestration" / "model_registry.yaml")
    for host, row in server_mode.items():
        if row.get("alias_of"):
            continue
        if host == role or row.get("model_role") == role or role in (row.get("shared_with") or []):
            return host
    raise AssertionError(f"live registry declares no host row for {role!r}")


def _compiled_fallback_port(tmp_path: Path, role: str) -> int:
    """Port the compiled fallback must publish for ``role`` in the fixture tree.

    The fixture registry deliberately binds nothing to ``worker_general`` (its
    ``worker`` row is a bare legacy row with no ``model_role``/``shared_with``),
    so the compiler resolves the role through the LIVE stack manifest's alias for
    it and projects the FIXTURE row of that host. Until the operator-signed
    2026-09-22 cutover (orchestrator 860b0b2d) the live host was the ``worker``
    row (fixture :8072); it is now frontdoor's fleet (fixture :8070). Derived
    from the live registry so the next lineup change moves the expectation too.
    """
    host = _live_host_of(role)
    fixture = _load_server_mode(tmp_path / "orchestration" / "model_registry.yaml")
    assert host in fixture, f"fixture registry lacks the live host row {host!r} of {role!r}"
    return int(fixture[host]["port"])


def _write_minimal_root(tmp_path: Path) -> None:
    orchestration = tmp_path / "orchestration"
    orchestration.mkdir()
    (orchestration / "derived").mkdir()
    registry_yaml = """
server_mode:
  frontdoor:
    port: 8070
    model: frontdoor.gguf
    tier: hot
    acceleration: {type: none, lookup: false}
    throughput: 24.3
    description: Root model
  worker:
    port: 8072
    model: worker.gguf
    tier: hot
    acceleration: {type: speculative_decoding, draft_max: 2, lookup: false}
    throughput: 60.7
    description: Worker alias without a matching roles entry
  __LEGACY_ARCHITECT_ROLE__:
    port: 8099
    model: removed-role.gguf
    tier: hot
    throughput: 4.2
    description: Retired architect role must stay out of live summaries
roles:
  frontdoor:
    backend: {type: local}
  worker_general:
    backend: {type: local}
  __LEGACY_ARCHITECT_ROLE__:  # legacy fixture: stack priors must exclude this from live roles
    backend: {type: local}
    model: {name: removed-role.gguf}
""".lstrip()
    (orchestration / "model_registry.yaml").write_text(
        registry_yaml.replace("__LEGACY_ARCHITECT_ROLE__", LEGACY_ARCHITECT_ROLE)
    )
    (orchestration / "derived" / "stack_priors.yaml").write_text(
        """
roles:
  frontdoor:
    role: frontdoor
    deployment_status: live_stack
    status: compiled
    display_name: frontdoor-prior.gguf
    serving:
      ports: [8070, 8080]
      endpoint: http://localhost:8070
      tier: hot
      binding: server_mode.direct
    priors:
      throughput_tps: 24.3
    acceleration:
      spec_type: none
  worker_general:
    role: worker_general
    deployment_status: live_stack
    status: compiled
    display_name: worker-prior.gguf
    serving:
      ports: [8072]
      endpoint: http://localhost:8072
      tier: hot
      binding: stack_manifest.role
    priors:
      throughput_tps: 60.7
    acceleration:
      spec_type: draft
      draft_max: 2
  worker_vision:
    role: worker_vision
    deployment_status: live_stack
    status: compiled
    display_name: vision-prior.gguf
    serving:
      ports: [8086]
      endpoint: http://localhost:8086
      tier: hot
      binding: stack_manifest.role
      launch:
        requirements:
          model_path: /models/vision-prior.gguf
          mmproj_path: /models/mmproj-model-f16.gguf
    priors:
      throughput_tps: 20.0
    acceleration:
      spec_type: baseline
  candidate_only:
    role: candidate_only
    deployment_status: benchmark_or_candidate
    status: compiled
    display_name: candidate.gguf
    serving:
      ports: [9999]
    priors:
      throughput_tps: 1.0
    acceleration:
      spec_type: none
""".lstrip()
    )
    (orchestration / "autopilot_baseline.yaml").write_text(
        """
baselines_by_tier:
  1: 0.5
per_suite_quality_by_tier:
  1:
    math: 3.0
""".lstrip()
    )


def _write_minimal_descriptors(tmp_path: Path) -> None:
    (tmp_path / "orchestration" / "model_descriptors.yaml").write_text(
        """
models:
  - model_id: qwen36-frontdoor
    display_name: frontdoor-compiled.gguf
    family: qwen
    arch: moe
    params_b: 35
    active_b: 3
    quant: Q8_0
    mem_gb: 37
    ctx_max: 131072
    modalities: [text]
    role_bindings:
      roles: [frontdoor]
      server_roles: [frontdoor]
    quality:
      suite_vector: {overall: 0.93}
      measured: []
    speed:
      solo_96t_tps: 24.3
      measured: []
    acceleration: {spec_type: none}
    serving:
      ports: [8070]
      binary: llama.cpp
    known_gaps: []
  - model_id: gemma-worker
    display_name: worker-compiled.gguf
    family: gemma
    arch: dense
    params_b: 26
    active_b: 4
    quant: Q4_K_M
    mem_gb: 16
    ctx_max: 16384
    modalities: [text]
    role_bindings:
      roles: [worker_general]
      server_roles: [worker]
    quality:
      suite_vector: {overall: 0.84}
      measured: []
    speed:
      quarter_48t_tps: 60.7
      measured: []
    acceleration: {spec_type: mtp, draft_max: 2}
    serving:
      ports: [8072]
      binary: ik-pr1744
    known_gaps: []
""".lstrip(),
        encoding="utf-8",
    )


def test_system_card_uses_stack_priors_not_registry_or_removed_role(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "paused": True,
            "trial_counter": 12,
            "baseline_state": {
                "baselines_by_tier": {"1": 1.9},
                "per_suite_quality_by_tier": {"1": {"math": 3.0}},
            },
        },
    )

    assert "Source: orchestration/derived/stack_priors.yaml" in card
    assert "| frontdoor | 8070, 8080 | frontdoor-prior.gguf |" in card
    assert "| worker_general | 8072 | worker-prior.gguf |" in card
    assert "| worker_vision | 8086 | vision-prior.gguf |" in card
    assert "mmproj=mmproj-model-f16.gguf" in card
    assert "worker.gguf" not in card
    assert "candidate.gguf" not in card
    assert f"| {LEGACY_ARCHITECT_ROLE} |" not in card
    assert (
        f"{LEGACY_ARCHITECT_ROLE} is historical only; use architect_general "
        "as the live architect server role and port."
    ) in card


@pytest.mark.usefixtures("_declared_backend_metadata")
def test_system_card_compiles_fallback_rows_when_stack_priors_missing(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    _write_minimal_descriptors(tmp_path)
    (tmp_path / "orchestration" / "derived" / "stack_priors.yaml").unlink()

    card = gen_system_card.generate_system_card(tmp_path, state_override={})

    assert (
        "Source: orchestration/model_registry.yaml + "
        "orchestration/model_descriptors.yaml (compiled fallback)"
    ) in card
    assert (
        "| frontdoor | 8070 | "
        "frontdoor-compiled.gguf |"
    ) in card
    wg_port = _compiled_fallback_port(tmp_path, "worker_general")
    assert (
        f"| worker_general | {wg_port} | "
        "worker-compiled.gguf |"
    ) in card
    assert "worker.gguf" not in card
    assert f"| {LEGACY_ARCHITECT_ROLE} |" not in card


@pytest.mark.usefixtures("_declared_backend_metadata")
def test_renderer_compiles_fallback_rows_when_stack_priors_missing(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    _write_minimal_descriptors(tmp_path)
    stack_priors_path = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    stack_priors_path.unlink()

    summary = render_stack_summary.render_current_stack_summary(
        stack_priors_path=stack_priors_path,
        registry_path=tmp_path / "orchestration" / "model_registry.yaml",
        descriptor_path=tmp_path / "orchestration" / "model_descriptors.yaml",
    )

    assert (
        "Source: `orchestration/model_registry.yaml + "
        "orchestration/model_descriptors.yaml (compiled fallback)`"
    ) in summary
    assert (
        "| frontdoor | 8070 | "
        "frontdoor-compiled.gguf |"
    ) in summary
    wg_port = _compiled_fallback_port(tmp_path, "worker_general")
    assert (
        f"| worker_general | {wg_port} | "
        "worker-compiled.gguf |"
    ) in summary
    assert "worker.gguf" not in summary


def test_renderer_embeds_source_fingerprints_for_freshness(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    stack_priors_path = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    registry_path = tmp_path / "orchestration" / "model_registry.yaml"
    descriptor_path = tmp_path / "orchestration" / "model_descriptors.yaml"
    descriptor_path.write_text("models: []\n", encoding="utf-8")

    summary = render_stack_summary.render_current_stack_summary(
        stack_priors_path=stack_priors_path,
        registry_path=registry_path,
        descriptor_path=descriptor_path,
    )
    stack_hash = render_stack_summary.file_sha256(stack_priors_path)
    registry_hash = render_stack_summary.file_sha256(registry_path)
    descriptor_hash = render_stack_summary.file_sha256(descriptor_path)

    assert "Source fingerprints:" in summary
    assert f"- orchestration/derived/stack_priors.yaml: `{stack_hash}`" in summary
    assert f"- orchestration/model_registry.yaml: `{registry_hash}`" in summary
    assert f"- orchestration/model_descriptors.yaml: `{descriptor_hash}`" in summary

    descriptor_path.write_text("models: []\n# whitespace-only table no-op\n", encoding="utf-8")
    refreshed = render_stack_summary.render_current_stack_summary(
        stack_priors_path=stack_priors_path,
        registry_path=registry_path,
        descriptor_path=descriptor_path,
    )

    assert refreshed != summary
    assert render_stack_summary.file_sha256(descriptor_path) in refreshed
    assert descriptor_hash not in refreshed


def test_renderer_uses_endpoint_port_when_stack_prior_ports_missing(
    tmp_path: Path,
) -> None:
    stack_priors = {
        "roles": {
            "frontdoor": {
                "deployment_status": "live_stack",
                "status": "compiled",
                "display_name": "frontdoor-prior.gguf",
                "serving": {
                    "endpoint": "http://localhost:8070",
                    "tier": "hot",
                    "binding": "server_mode.direct",
                },
                "priors": {"throughput_tps": 24.3},
                "acceleration": {"spec_type": "none"},
            }
        }
    }
    rows = render_stack_summary.stack_prior_role_rows(stack_priors)

    assert rows == [
        "| frontdoor | 8070 | frontdoor-prior.gguf | hot | none | none | "
        "24.3 | live_stack; binding=server_mode.direct; status=compiled |"
    ]


def test_renderer_distinguishes_embedded_nextn_from_separate_draft(
    tmp_path: Path,
) -> None:
    qwen_model = tmp_path / "Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
    gemma_model = tmp_path / "gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf"
    gemma_draft = tmp_path / "gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf"
    stack_priors = {
        "roles": {
            "frontdoor": {
                "deployment_status": "live_stack",
                "status": "compiled",
                "display_name": qwen_model.name,
                "serving": {
                    "ports": [8070],
                    "tier": "hot",
                    "binding": "server_mode.direct",
                    "launch": {
                        "requirements": {
                            "model_path": str(qwen_model),
                            "draft_model_path": str(qwen_model),
                        }
                    },
                },
                "priors": {"throughput_tps": 24.3},
                "acceleration": {"spec_type": "draft-mtp", "draft_max": 4},
            },
            "worker_general": {
                "deployment_status": "live_stack",
                "status": "compiled",
                "display_name": gemma_model.name,
                "serving": {
                    "ports": [8072],
                    "tier": "hot",
                    "binding": "server_mode.model_role",
                    "launch": {
                        "requirements": {
                            "model_path": str(gemma_model),
                            "draft_model_path": str(gemma_draft),
                        }
                    },
                },
                "priors": {"throughput_tps": 60.7},
                "acceleration": {"spec_type": "draft-mtp", "draft_max": 2},
            },
        }
    }

    rendered = "\n".join(render_stack_summary.stack_prior_role_rows(stack_priors))

    assert f"embedded_nextn={qwen_model.name}" in rendered
    assert f"draft={qwen_model.name}" not in rendered
    assert f"draft={gemma_draft.name}" in rendered


def test_renderer_degraded_registry_fallback_canonicalizes_live_aliases(
    tmp_path: Path,
) -> None:
    _write_minimal_root(tmp_path)
    stack_priors_path = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    stack_priors_path.unlink()

    summary = render_stack_summary.render_current_stack_summary(
        stack_priors_path=stack_priors_path,
        registry_path=tmp_path / "orchestration" / "model_registry.yaml",
        descriptor_path=tmp_path / "orchestration" / "missing_model_descriptors.yaml",
    )

    assert "Source: `orchestration/model_registry.yaml (degraded fallback)`" in summary
    assert "| frontdoor | 8070 | frontdoor.gguf |" in summary
    assert "| worker_general | 8072 | worker.gguf |" in summary
    assert "| worker | 8072 |" not in summary
    assert f"| {LEGACY_ARCHITECT_ROLE} |" not in summary


def test_system_card_prefers_state_baseline_over_yaml(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "baseline_state": {
                "baselines_by_tier": {"1": 2.25},
                "per_suite_quality_by_tier": {"1": {"math": 3.0, "coder": 1.5}},
            }
        },
    )

    assert "Source: orchestration/autopilot_state.json:baseline_state" in card
    assert "T1: quality baseline 2.25" in card
    assert "quality baseline 0.5" not in card


def test_system_card_uses_baseline_ledger_fold_when_state_cache_absent(
    tmp_path: Path,
) -> None:
    _write_minimal_root(tmp_path)
    journal_path = tmp_path / "orchestration" / "autopilot_journal.jsonl"
    journal_path.write_text(
        json.dumps(
            {
                "type": "baseline_promotion",
                "source_trial_id": 7,
                "tier": 1,
                "previous_quality": 0.5,
                "new_quality": 2.75,
                "baseline_state": {
                    "baselines_by_tier": {"1": 2.75},
                    "per_suite_quality_by_tier": {"1": {"coder": 2.5}},
                },
            }
        )
        + "\n"
    )

    card = gen_system_card.generate_system_card(tmp_path, state_override={})

    assert "Source: orchestration/autopilot_journal.jsonl:baseline_promotion fold" in card
    assert "T1: quality baseline 2.75" in card
    assert "Active T1 suites: coder" in card
    assert "quality baseline 0.5" not in card


def test_system_card_hides_stale_pause_reason_when_unpaused(tmp_path: Path) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "paused": False,
            "pause_reason": "old contested-window reason",
            "trial_counter": 99,
        },
    )

    assert "- paused: false" in card
    assert "old contested-window reason" not in card


def test_system_card_surfaces_instrument_era_and_frontier_rerun(
    tmp_path: Path,
) -> None:
    _write_minimal_root(tmp_path)
    card = gen_system_card.generate_system_card(
        tmp_path,
        state_override={
            "paused": True,
            "trial_counter": 1001,
            "pareto_epoch_ts": 1782511631.0,
            "pareto_exclude_before_ts": 1782511631.0,
            "active_instrument_eras": {
                "autopilot_speed": "E5-autopilot-speed",
                "cpu_bench": "E5-cpu-kernel",
            },
            "frontier_rerun_required": {
                "required": True,
                "reason": "v6 kernel era opened; rebuild current-stack frontier",
            },
        },
    )

    assert "- pareto_epoch_ts: 1782511631.0" in card
    assert "- pareto_exclude_before_ts: 1782511631.0" in card
    assert (
        "- active_instrument_eras: "
        "autopilot_speed=E5-autopilot-speed, cpu_bench=E5-cpu-kernel"
    ) in card
    assert (
        "- frontier_rerun_required: true "
        "(v6 kernel era opened; rebuild current-stack frontier)"
    ) in card


def test_controller_template_uses_constitution_and_system_card() -> None:
    assert "{program}" not in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert "{constitution}" in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert "{system_card}" in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert "{planner_evidence}" in autopilot.CONTROLLER_PROMPT_TEMPLATE
    assert (
        autopilot.CONTROLLER_PROMPT_TEMPLATE.index("### Pareto Frontier Geometry")
        < autopilot.CONTROLLER_PROMPT_TEMPLATE.index(
            "### Evidence Power and Sequential Candidate Status"
        )
        < autopilot.CONTROLLER_PROMPT_TEMPLATE.index("### Journal Trustworthiness")
    )


def test_render_system_card_fails_closed_when_generator_unavailable(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("broken stack-prior render")

    monkeypatch.setattr(gen_system_card, "generate_system_card", _boom)

    card = autopilot._render_system_card({})

    assert "SYSTEM CARD GENERATION FAILED" in card
    assert "broken stack-prior render" in card
    assert "Live role, port, tier, throughput" in card
    assert "Do not use checked-in `system_card.md`" in card
    assert "| Role | Port | Model |" not in card
    assert "frontdoor-prior.gguf" not in card


@pytest.fixture
def _declared_backend_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Owned path metadata for pure compilation; no binary or store evidence."""
    import subprocess
    from src.registry import kernel_paths

    directories = {}
    for backend in ("cpu", "gpu"):
        directory = tmp_path / "declared-backends" / backend
        directory.mkdir(mode=0o700, parents=True)
        directories[backend] = directory

    def declared_backend_dir(backend):
        if backend not in directories:
            raise AssertionError(f"unexpected metadata backend: {backend!r}")
        return directories[backend]

    original_popen = subprocess.Popen
    provenance_children = []
    expected_git = ["git", "-C", str(Path(__file__).resolve().parents[2]),
                    "rev-parse", "--short", "HEAD"]
    expected_options = {"text": True, "stderr": subprocess.DEVNULL,
                        "stdout": subprocess.PIPE}

    def refuse_child(*args, **kwargs):
        # Exact check_output contract: no executable/env/shell or creation flags.
        if len(args) != 1 or args[0] != expected_git or kwargs != expected_options:
            raise AssertionError("pure metadata fixture permits only readonly Git provenance")
        child = original_popen(*args, **kwargs)
        provenance_children.append(child)
        return child

    monkeypatch.setattr(kernel_paths, "backend_dir", declared_backend_dir)
    # Preserve real CPU [] and GPU vendor-path library policy.
    monkeypatch.setattr(subprocess, "Popen", refuse_child)
    yield directories
    assert provenance_children, "missing actual Git provenance read"
    assert all(child.poll() == 0 for child in provenance_children)
    for directory in directories.values():
        assert not list(directory.iterdir()), "metadata fixture acquired a binary or output"
