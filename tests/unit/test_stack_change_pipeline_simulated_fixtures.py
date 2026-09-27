"""Scenario-level simulated stack-change fixtures.

These tests use only temporary registries/artifacts. They exercise the
stack-change pipeline against realistic data-only edits without touching live
generated files or running inference.
"""

from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "autopilot"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "benchmark"))

from orchestration.repl_memory.q_scorer import (  # noqa: E402
    PRIOR_SOURCE_STACK_PRIORS,
    stack_prior_q_scorer_priors_by_role,
    validate_live_q_scorer_prior_sources,
)
from seeding_rewards import (  # noqa: E402
    ALLOW_DEGRADED_CONFIG_KEY,
    MODEL_DESCRIPTORS_CONFIG_KEY,
    PRIOR_SOURCE_MODEL_DESCRIPTORS,
    STACK_PRIORS_CONFIG_KEY,
    RoleResult,
    compute_comparative_rewards,
    descriptor_throughput_by_role,
    throughput_prior_provenance,
)
from scripts.registry.stack_change_pipeline import (  # noqa: E402
    SIMULATED_FIXTURE_TARGET,
    StackChangePipelineConfig,
    run_stack_change_pipeline,
)
from scripts.registry import stack_change_pipeline as pipeline  # noqa: E402

gen_system_card = importlib.import_module("gen_system_card")


@pytest.fixture(autouse=True)
def _clean_runtime_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_runtime_attestation_warnings", lambda: [])
    # Unit tests never read the live /proc: pin the declared-env attestation to a clean result.
    from scripts.server.env_attestation import EnvAttestation

    monkeypatch.setattr(
        pipeline,
        "_declared_env_attestation_result",
        lambda: EnvAttestation(compared=["fixture:0 pid 0 (fixture, 0 declared keys)"]),
    )


@pytest.fixture(autouse=True)
def _pin_realized_compile_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """ESC-8 Fix 6: the update/check compile now resolves the NUMA mode from the
    realized fleet (a bare TCP probe) or refuses when nothing is listening. Pin
    it deterministically to the launcher ``full`` default so these simulated
    fixtures stay hermetic (no live sockets) and keep prior expectations.
    The guard's launch-view probe is pinned to no-signal for the same reason
    (env fallback governs, matching the fixtures)."""
    from scripts.validate import stack_change_guard
    from src.registry import stack_priors

    # The promotion gate inherits the caller's environment. These simulated
    # fixtures define a full-topology launch view, so prevent a live quarter
    # selector from changing their data-only compile fallback.
    monkeypatch.setenv("ORCHESTRATOR_STACK_NUMA_MODE", "full")
    monkeypatch.setattr(stack_priors, "_realized_compile_numa_mode", lambda **_kw: "full")
    monkeypatch.setattr(stack_change_guard, "_realized_launch_numa_mode", lambda: None)


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _write_role_enum_files(config: StackChangePipelineConfig, roles: set[str]) -> None:
    ordered = sorted(roles)
    permission_roles = [*ordered, "admin"]
    _write_yaml(
        config.procedure,
        {
            "inputs": [
                {
                    "name": "role",
                    "type": "string",
                    "validation": {"enum": ordered},
                }
            ]
        },
    )
    config.schema.parent.mkdir(parents=True, exist_ok=True)
    config.schema.write_text(
        """{
  "properties": {
    "permissions": {
      "properties": {
        "roles": {
          "items": {
            "enum": %s
          }
        }
      }
    }
  }
}
"""
        % json.dumps(permission_roles),
        encoding="utf-8",
    )


# 2026-08-01 W1 cutover: frontdoor's :8070 alias is now `worker_summarize`, its
# ONLY alias. `coder_escalation` — which these two builders used to name as the
# second role on :8070 — LEFT that process and is now an alias on
# architect_general's :8083 (registry `server_mode.architect_general.shared_with`,
# launch manifest PORT_MAP coder_escalation -> 8083). Naming it here made the
# pipeline's stack_manifest_registry step report real drift against the launcher.
# The scenario is unchanged: one primary plus one role sharing its process.
def _base_frontdoor_registry(path: Path, *, throughput: float = 24.3) -> Path:
    return _write_yaml(
        path,
        {
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
                },
                "worker_summarize": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf",
                    "model_role": "worker_summarize",
                    "memory_gb": 37,
                    "throughput": throughput,
                    "benchmark_score": "170/183 (92.9%)",
                    "benchmark_date": "2026-05-04",
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                },
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
                },
                "worker_summarize": {
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
                },
            },
        },
    )


def _swapped_frontdoor_registry(path: Path, *, throughput: float = 18.5) -> Path:
    return _write_yaml(
        path,
        {
            "server_mode": {
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_role": "frontdoor",
                    "memory_gb": 20,
                    "throughput": throughput,
                    "benchmark_score": "160/183 (87.4%)",
                    "benchmark_date": "2026-06-13",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                },
                "worker_summarize": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "slots": 2,
                    "model": "Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_path": "/models/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "model_role": "worker_summarize",
                    "memory_gb": 20,
                    "throughput": throughput,
                    "benchmark_score": "160/183 (87.4%)",
                    "benchmark_date": "2026-06-13",
                    "kv_quant": {"k": "q8_0", "v": "q8_0"},
                    "numa_instances": 1,
                    "numa_ports": [8070],
                },
            },
            "roles": {
                "frontdoor": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "qwen35moe",
                        "size_gb": 20,
                        "ctx_max": 131072,
                    },
                    "performance": {
                        "quality_pct": 87.4,
                        "baseline_tps": throughput,
                        "benchmark_date": "2026-06-13",
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "worker_summarize": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q4_K_M",
                        "quant": "Q4_K_M",
                        "architecture": "qwen35moe",
                        "size_gb": 20,
                        "ctx_max": 131072,
                    },
                    "performance": {
                        "quality_pct": 87.4,
                        "baseline_tps": throughput,
                        "benchmark_date": "2026-06-13",
                    },
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                },
            },
        },
    )


# 2026-09-22 lineup cutover (epyc-orchestrator 860b0b2d): the dedicated worker
# pool on :8072 (quarters 8082/8182/8282/8382) is RETIRED. worker_general and
# its worker_math / toolrunner aliases now ride frontdoor's :8070 process — the
# computed launch manifest (orchestration/launch_manifest.yaml, read by
# scripts/server/stack_manifest.py at import, NOT by this tmp repo) declares
# PORT_MAP worker_general/worker_math/toolrunner -> 8070 with frontdoor as the
# primary of that server, and role_launch_meta no longer has a worker_general
# entry. Both the compiler and the guard resolve these roles to that host.
#
# These builders used to declare `server_mode.worker` as its OWN process on
# :8072. That world no longer exists on the launch surface, so the guard
# resolved worker_general to its manifest host (frontdoor), found no frontdoor
# row in this simulated registry, and reported COULD-NOT-CHECK for the
# requirement comparison (and, where the fixture DID declare a frontdoor row with
# a different model, a model_path mismatch). The simulated world now matches the
# launch surface: ONE :8070 process whose `server_mode.frontdoor` row is the
# host and whose shared_with names the worker lane. The model data is still the
# gemma4 fixture data — only the topology had to move — so the swap assertions
# keep biting on the Q4_K_M -> Q8_0 change.
#
# frontdoor is an ACTIVE role in these scenarios on purpose: WP-13 validates a
# shared process's runtime once, on the host's own row
# (stack_change_guard.py `if target_launch_runtime and not host_role`). With the
# host outside the evaluated role set, a runtime drift on :8070 would be
# observable nowhere and the requirement-drift scenario would pass vacuously.
WORKER_LANE_ROLES = {"frontdoor", "worker_general", "worker_math", "toolrunner"}


def _worker_lane_registry(
    path: Path,
    *,
    model: str,
    quant: str,
    size_gb: float,
    memory_gb: float,
    throughput: float,
    baseline_tps: float,
    benchmark_score: str,
    model_path: str | None = None,
    binary_dir: str = "/mnt/raid0/llm/llama.cpp/build/bin",
    ld_library_path: list[str] | None = None,
) -> Path:
    if ld_library_path is None:
        ld_library_path = [
            "/mnt/raid0/llm/llama.cpp/build/src",
            "/mnt/raid0/llm/llama.cpp/build/ggml/src",
            "/mnt/raid0/llm/llama.cpp/build/examples/mtmd",
        ]
    host: dict[str, Any] = {
        "url": "http://localhost:8070",
        "port": 8070,
        "tier": "hot",
        "slots": 1,
        "model": f"{model}.gguf",
    }
    if model_path is not None:
        host["model_path"] = model_path
    host.update(
        {
            "model_role": "frontdoor",
            "shared_with": ["worker_general", "worker_math", "toolrunner"],
            "memory_gb": memory_gb,
            "throughput": throughput,
            "benchmark_score": benchmark_score,
            # 2026-06-26 v6 cutover: canonical llama.cpp (v6); ik_llama.cpp deprecated.
            "runtime_requirements": {
                "binary_dir": binary_dir,
                "ld_library_path": ld_library_path,
            },
        }
    )

    def lane_role() -> dict[str, Any]:
        # frontdoor and worker_general describe the SAME resident model: one
        # process, one GGUF. Built fresh per role so the YAML carries no anchors.
        return {
            "model": {
                "name": model,
                "quant": quant,
                "architecture": "gemma4",
                "size_gb": size_gb,
                "ctx_max": 16384,
            },
            "performance": {"quality_pct": 90, "baseline_tps": baseline_tps},
            # 2026-06-26 v6 cutover: MTP spec token is now 'draft-mtp'.
            "acceleration": {"type": "speculative_decoding", "spec_type": "draft-mtp"},
            "memory": {"pinned": True, "residency": "hot"},
        }

    return _write_yaml(
        path,
        {
            "server_mode": {"frontdoor": host},
            "roles": {
                "frontdoor": lane_role(),
                "worker_general": lane_role(),
                # worker_math / toolrunner keep their stale standalone model
                # metadata on purpose: the descriptor compile must record them
                # as alias_overrides of the one resident model, not as
                # role-server conflicts.
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
                },
                "toolrunner": {
                    "model": {
                        "name": "Qwen3-Coder-30B-A3B-Instruct",
                        "quant": "Q4_K_M",
                        "architecture": "qwen3coder",
                        "size_gb": 16,
                        "ctx_max": 32768,
                    },
                    "performance": {"quality_pct": 84, "baseline_tps": 39.1},
                    "acceleration": {"type": "none", "lookup": False},
                    "memory": {"pinned": True, "residency": "hot"},
                },
            },
        },
    )


def _swapped_worker_registry(path: Path, *, throughput: float = 66.2) -> Path:
    return _worker_lane_registry(
        path,
        model="gemma-4-26B-A4B-it-Q8_0",
        quant="Q8_0",
        size_gb=30,
        memory_gb=30,
        throughput=throughput,
        baseline_tps=throughput,
        benchmark_score="92%",
        model_path="/models/gemma-4-26B-A4B-it-Q8_0.gguf",
    )


def _worker_alias_registry(
    path: Path,
    *,
    binary_dir: str = "/mnt/raid0/llm/llama.cpp/build/bin",
    ld_library_path: list[str] | None = None,
) -> Path:
    return _worker_lane_registry(
        path,
        model="gemma-4-26B-A4B-it-Q4_K_M",
        quant="Q4_K_M",
        size_gb=16,
        memory_gb=16,
        throughput=60.7,
        baseline_tps=44.7,
        benchmark_score="90%",
        binary_dir=binary_dir,
        ld_library_path=ld_library_path,
    )


# 2026-07-31 vision unification + 2026-08-01 W1 cutover: worker_vision and
# vision_escalation are ONE :8086 GPU process serving ONE VL model. Port 8087 is
# RETIRED. The registry encodes this as
# `server_mode.worker_vision.shared_with: [vision_escalation]` with
# `roles.vision_escalation.alias_of: worker_vision`, and the alias declares no
# model of its own ("There is NO second VL model resident").
#
# This builder used to declare TWO servers (8086 + 8087) with TWO different
# models. That shape is no longer expressible: the launch-manifest guard resolves
# vision_escalation to its host and requires the alias's compiled
# serving.launch.requirements to EQUAL worker_vision's, so the two-model form
# failed with a model_path mismatch before any of the swap assertions ran.
#
# The swap this fixture simulates is now the real one — the 7B incumbent replaced
# by the 30B-A3B — so both "old model name is gone" assertions still bite.
def _vision_registry(
    path: Path,
    *,
    model: str = "Qwen2.5-VL-7B-Instruct-Q4_K_M",
    memory_gb: int = 7,
    throughput: float = 20.0,
    quality_pct: float = 81.0,
    quant: str = "Q4_K_M",
    architecture: str = "qwen2vl",
    benchmark_date: str = "2026-05-04",
) -> Path:
    gguf = f"{model}.gguf"
    return _write_yaml(
        path,
        {
            "server_mode": {
                "worker_vision": {
                    "url": "http://localhost:8086",
                    "port": 8086,
                    "tier": "hot",
                    "slots": 2,
                    "model": gguf,
                    "model_path": f"/models/{gguf}",
                    "model_role": "worker_vision",
                    "shared_with": ["vision_escalation"],
                    "memory_gb": memory_gb,
                    "throughput": throughput,
                    "benchmark_score": f"{quality_pct:g}%",
                    "benchmark_date": benchmark_date,
                    "mmproj": f"/models/{model}-mmproj.gguf",
                },
            },
            "roles": {
                "worker_vision": {
                    "model": {
                        "name": model,
                        "quant": quant,
                        "architecture": architecture,
                        "size_gb": memory_gb,
                        "ctx_max": 8192,
                    },
                    "performance": {
                        "quality_pct": quality_pct,
                        "baseline_tps": throughput,
                        "benchmark_date": benchmark_date,
                    },
                    "memory": {"pinned": True, "residency": "hot"},
                },
                "vision_escalation": {
                    "alias_of": "worker_vision",
                    "model": {
                        "name": model,
                        "quant": quant,
                        "architecture": architecture,
                        "size_gb": memory_gb,
                        "ctx_max": 8192,
                        "shared_gguf_with": "worker_vision",
                    },
                    "performance": {
                        "inherits_from": "worker_vision",
                        "quality_pct": quality_pct,
                        "baseline_tps": throughput,
                        "benchmark_date": benchmark_date,
                    },
                    "memory": {"pinned": True, "residency": "hot"},
                },
            },
        },
    )


def _swapped_vision_registry(path: Path) -> Path:
    return _vision_registry(
        path,
        model="Qwen3-VL-30B-A3B-Instruct-Q8_0",
        memory_gb=38,
        throughput=5.9,
        quality_pct=92.0,
        quant="Q8_0",
        architecture="qwen3vlmoe",
        benchmark_date="2026-06-13",
    )


# 2026-09-22 lineup cutover (epyc-orchestrator 860b0b2d): ingest_long_context's
# three dedicated CPU instances on :8085/:8185/:8285 are RETIRED. The role is an
# ALIAS on architect_critic's :8083 process — launch manifest PORT_MAP
# ingest_long_context -> 8083, its role_launch_meta entry removed, and the real
# master's `server_mode.architect_critic.shared_with` names it. The launch view
# is the REAL repo's (scripts/server/stack_manifest.py loads
# orchestration/launch_manifest.yaml at import), so this builder's old
# standalone :8085 row made the guard resolve the role to its manifest host,
# find no architect_critic row here, and report COULD-NOT-CHECK for the
# model_path comparison. The simulated registry now declares the host row and the
# alias rides it; the model data is still the Qwen3-Next fixture data, so the
# Q4_K_M -> Q8_0 swap assertions are unchanged.
#
# architect_critic is an ACTIVE role in the scenario: an alias launches nothing,
# so the process-level consumers (dashboard port labels, the host's runtime row)
# only exist through the host's record. See INGEST_PROCESS_ROLES.
# 2026-09-27 ARCHITECT SWAP (operator-decided): the :8083 process that hosts
# ingest_long_context is architect_critic now (it was architect_general).
INGEST_PROCESS_ROLES = {"architect_critic", "ingest_long_context"}


def _ingest_registry(
    path: Path,
    *,
    model: str = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M",
    memory_gb: int = 46,
    throughput: float = 20.8,
    quality_pct: float = 92.59,
    quant: str = "Q4_K_M",
    ctx_max: int = 262144,
    benchmark_date: str = "2026-05-04",
) -> Path:
    gguf = f"{model}.gguf"

    def process_role() -> dict[str, Any]:
        # architect_critic and ingest_long_context describe the SAME resident
        # model: one process, one GGUF. Built fresh per role (no YAML anchors).
        return {
            "model": {
                "name": model,
                "quant": quant,
                "architecture": "qwen3next",
                "size_gb": memory_gb,
                "ctx_max": ctx_max,
            },
            "performance": {
                "quality_pct": quality_pct,
                "baseline_tps": throughput,
                "long_context_quality": f"{quality_pct:g}%",
                "benchmark_date": benchmark_date,
            },
            "acceleration": {"type": "none", "lookup": False},
            "memory": {"pinned": True, "residency": "hot"},
        }

    ingest_role = process_role()
    ingest_role["alias_of"] = "architect_critic"
    ingest_role["model"]["shared_gguf_with"] = "architect_critic"
    return _write_yaml(
        path,
        {
            "server_mode": {
                "architect_critic": {
                    "url": "http://localhost:8083",
                    "port": 8083,
                    "tier": "hot",
                    "slots": 1,
                    "model": gguf,
                    "model_path": f"/models/{gguf}",
                    "model_role": "architect_critic",
                    "shared_with": ["ingest_long_context"],
                    "memory_gb": memory_gb,
                    "throughput": throughput,
                    "benchmark_score": f"{quality_pct:g}%",
                    "benchmark_date": benchmark_date,
                }
            },
            "roles": {
                "architect_critic": process_role(),
                "ingest_long_context": ingest_role,
            },
        },
    )


def _swapped_ingest_registry(path: Path) -> Path:
    return _ingest_registry(
        path,
        model="Qwen3-Next-80B-A3B-Instruct-Q8_0",
        memory_gb=82,
        throughput=14.2,
        quality_pct=96.3,
        quant="Q8_0",
        ctx_max=262144,
        benchmark_date="2026-06-13",
    )


def _config(tmp_path: Path, *, mode: str, roles: set[str]) -> StackChangePipelineConfig:
    registry = tmp_path / "orchestration" / "model_registry.yaml"
    descriptors = tmp_path / "orchestration" / "model_descriptors.yaml"
    priors = tmp_path / "orchestration" / "derived" / "stack_priors.yaml"
    operator_summary = tmp_path / "docs" / "generated" / "current_stack_summary.md"
    procedure = tmp_path / "orchestration" / "procedures" / "add_model.yaml"
    schema = tmp_path / "orchestration" / "procedure.schema.json"
    _write_role_enum_files(
        StackChangePipelineConfig(
            mode=mode,  # type: ignore[arg-type]
            repo_root=tmp_path,
            lean_registry=registry,
            research_registry=None,
            descriptors=descriptors,
            stack_priors=priors,
            operator_summary=operator_summary,
            procedure=procedure,
            schema=schema,
            surface_exceptions=tmp_path / "missing_exceptions.yaml",
            roles=roles,
            allow_known_gaps=True,
            numa_mode="full",
        ),
        roles,
    )
    return StackChangePipelineConfig(
        mode=mode,  # type: ignore[arg-type]
        repo_root=tmp_path,
        lean_registry=registry,
        research_registry=None,
        descriptors=descriptors,
        stack_priors=priors,
        operator_summary=operator_summary,
        procedure=procedure,
        schema=schema,
        surface_exceptions=tmp_path / "missing_exceptions.yaml",
        roles=roles,
        allow_known_gaps=True,
        # NIB2-69: no stack_topology.yaml in the fixture repo, so the evaluated
        # lineup is explicit (matches the realized-compile "full" pin above).
        numa_mode="full",
    )


def _assert_text_stack_primary_port_consumers(
    stack_priors_path: Path,
    *,
    expected_roles: set[str],
    expected_port: int,
) -> None:
    from scripts.autopilot.preflight_audit import _model_server_target_groups
    from scripts.benchmark import corpus_quality_gate
    from scripts.graph_router.train_graph_router import load_model_fleet
    from src.api.routes.openai_compat import _ordered_live_role_ids
    from src.cli_orch import _stack_status_targets

    artifact = yaml.safe_load(stack_priors_path.read_text(encoding="utf-8"))
    records = artifact["roles"]
    grouped_role_name = "/".join(sorted(expected_roles))

    assert (grouped_role_name, expected_port) in _stack_status_targets(stack_priors_path)

    live_models = corpus_quality_gate._load_live_models(stack_priors_path)
    assert {role: live_models[role]["port"] for role in expected_roles} == {
        role: expected_port for role in expected_roles
    }

    fleet = {record["role_id"]: record for record in load_model_fleet(stack_priors_path)}
    assert {role: fleet[role]["port"] for role in expected_roles} == {
        role: expected_port for role in expected_roles
    }

    _, names_by_health_url = _model_server_target_groups(records, "http://localhost:8000")
    assert sorted(names_by_health_url[f"http://localhost:{expected_port}/health"]) == sorted(
        expected_roles
    )

    sentinel_records = {
        **records,
        "_sentinel_before": {"serving": {"ports": [expected_port - 1]}},
        "_sentinel_after": {"serving": {"ports": [expected_port + 1]}},
    }
    ordered_roles = _ordered_live_role_ids(sentinel_records)
    # `_ordered_live_role_ids` pins frontdoor FIRST regardless of port (it is the
    # default /v1/models entry); every other role is ordered by primary port. Since
    # the 2026-09-22 cutover the worker lane shares frontdoor's :8070, so the
    # port-order bracket applies to the non-frontdoor roles only.
    port_ordered_roles = expected_roles - {"frontdoor"}
    if "frontdoor" in expected_roles:
        assert ordered_roles[0] == "frontdoor"
    assert port_ordered_roles
    assert ordered_roles.index("_sentinel_before") < min(
        ordered_roles.index(role) for role in port_ordered_roles
    )
    assert max(ordered_roles.index(role) for role in port_ordered_roles) < ordered_roles.index(
        "_sentinel_after"
    )


def _primary_port_for_roles(priors: dict[str, Any], roles: set[str]) -> int:
    ports = {
        port
        for role in roles
        for port in ((priors["roles"][role].get("serving") or {}).get("ports") or [])
        if isinstance(port, int)
    }
    assert ports
    return min(ports)


def _assert_worker_pool_stack_prior_consumer(
    records: dict[str, dict[str, Any]],
    *,
    expected_port: int,
    expected_model_path_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.registry import stack_priors
    from src.services.worker_pool import WorkerPoolManager, WorkerTier

    monkeypatch.setattr(stack_priors, "live_stack_role_records", lambda _path=None: records)
    config = WorkerPoolManager().config

    assert list(config.workers) == ["worker_general"]
    worker = config.workers["worker_general"]
    assert worker.port == expected_port
    assert expected_model_path_fragment in worker.model_path
    assert worker.tier is WorkerTier.HOT
    assert worker.managed_process is False


def _assert_factual_risk_stack_prior_consumer(
    stack_priors_path: Path,
    *,
    expected_tiers: dict[str, str],
) -> None:
    from src.classifiers.factual_risk import _role_adjustment, _role_tier_for_role

    adjustment_by_tier = {"tier_1": 0.11, "tier_2": 0.22, "tier_3": 0.33}
    config = {"role_adjustments": adjustment_by_tier}

    for role, expected_tier in expected_tiers.items():
        assert _role_tier_for_role(role, stack_priors_path=stack_priors_path) == expected_tier
        assert _role_adjustment(
            role,
            config=config,
            stack_priors_path=stack_priors_path,
        ) == pytest.approx(adjustment_by_tier[expected_tier])


def _assert_seeding_descriptor_fallback_consumer(
    descriptors_path: Path,
    *,
    expected_roles: set[str],
    expected_tps: float,
) -> None:
    throughput = descriptor_throughput_by_role(descriptors_path)
    assert {role: throughput[role] for role in expected_roles} == {
        role: expected_tps for role in expected_roles
    }

    missing_stack_priors = descriptors_path.with_name("missing_stack_priors.yaml")
    cost_config = {
        STACK_PRIORS_CONFIG_KEY: missing_stack_priors,
        MODEL_DESCRIPTORS_CONFIG_KEY: descriptors_path,
        ALLOW_DEGRADED_CONFIG_KEY: True,
    }
    provenance = throughput_prior_provenance(cost_config)
    assert provenance["source"] == PRIOR_SOURCE_MODEL_DESCRIPTORS
    assert set(expected_roles) <= set(provenance["roles"])
    assert provenance["model_descriptors_path"] == str(descriptors_path)

    rewards = compute_comparative_rewards(
        {
            "frontdoor:direct": RoleResult(
                role="frontdoor",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=1.0,
            ),
            "worker_general:direct": RoleResult(
                role="worker_general",
                mode="direct",
                answer="ok",
                passed=True,
                elapsed_seconds=2.0,
                generation_ms=2000,
                tokens_generated=100,
            ),
        },
        cost_config=cost_config,
    )
    expected_elapsed = 100 / expected_tps
    expected_reward = 0.5 - 0.15 * max(0.0, (2.0 / expected_elapsed) - 1.0)
    assert math.isclose(rewards["worker_general:direct"], expected_reward)


def test_pipeline_report_names_simulated_fixture_target(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "worker_summarize"})
    _base_frontdoor_registry(config.lean_registry)

    report = run_stack_change_pipeline(config)

    step = next(step for step in report.steps if step.name == "simulated_fixtures")
    assert step.name == "simulated_fixtures"
    assert step.status == "reference"
    assert SIMULATED_FIXTURE_TARGET in step.details[0]


def test_simulated_check_runs_promotion_gate_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "worker_summarize"})
    _base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check", "run_promotion_gate": True}
    )

    report = run_stack_change_pipeline(check_config)

    assert report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    step = next(step for step in report.steps if step.name == "promotion_gate")
    assert step.status == "ok"
    assert any("promotion gate ok" in detail for detail in step.details)


def test_simulated_check_fails_when_promotion_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "worker_summarize"})
    _base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **_)
        return subprocess.CompletedProcess(
            args=command,
            returncode=7,
            stdout="partial output\n",
            stderr="promotion gate failed\n",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    check_config = StackChangePipelineConfig(
        **{**config.__dict__, "mode": "check", "run_promotion_gate": True}
    )

    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    step = next(step for step in report.steps if step.name == "promotion_gate")
    assert step.status == "failed"
    assert step.errors == ["promotion gate exited 7"]
    assert any("partial output" in detail for detail in step.details)
    assert any("promotion gate failed" in detail for detail in step.details)


def test_simulated_update_does_not_write_real_operator_summary(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="update", roles={"frontdoor", "worker_summarize"})
    _base_frontdoor_registry(config.lean_registry)
    real_summary = StackChangePipelineConfig(mode="check").operator_summary
    before = real_summary.read_text(encoding="utf-8")

    assert run_stack_change_pipeline(config).ok

    assert config.operator_summary.exists()
    assert config.operator_summary != real_summary
    assert real_summary.read_text(encoding="utf-8") == before


def test_simulated_frontdoor_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = {"frontdoor", "worker_summarize"}
    config = _config(tmp_path, mode="update", roles=roles)
    _base_frontdoor_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_frontdoor_registry(config.lean_registry)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    descriptor_step = next(step for step in check_report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptors_before
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    approved_config = StackChangePipelineConfig(
        **{**config.__dict__, "allow_descriptor_model_removal": True}
    )
    update_report = run_stack_change_pipeline(approved_config)

    assert update_report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "qwen3.6-35b-a3b-q4_k_m"
    ]
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert set(priors["roles"]) == roles
    for role in roles:
        assert priors["roles"][role]["model_id"] == "qwen3.6-35b-a3b-q4_k_m"
        assert priors["roles"][role]["priors"]["throughput_tps"] == 18.5
        assert priors["roles"][role]["priors"]["quality_overall"] == pytest.approx(0.874)

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    for role in roles:
        assert f"| {role}" in operator_summary
        assert priors["roles"][role]["display_name"] in operator_summary
    assert "Qwen_Qwen3.6-35B-A3B-Q8_0" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    for role in roles:
        assert f"| {role} |" in system_card
        assert priors["roles"][role]["display_name"] in system_card
    assert "Qwen_Qwen3.6-35B-A3B-Q8_0" not in system_card

    from src.api.routes.dashboard_topology import _stack_prior_port_hints
    from src.api.routes.health import _stack_prior_backend_urls

    expected_port = _primary_port_for_roles(priors, roles)
    assert _stack_prior_backend_urls(config.stack_priors) == {
        # Grouped name is the sorted role set on one port; W1 moved
        # coder_escalation off :8070, so the pair is now frontdoor+worker_summarize.
        "frontdoor/worker_summarize": f"http://localhost:{expected_port}"
    }
    port_hints = _stack_prior_port_hints(config.stack_priors)
    assert port_hints[expected_port].split(".", 1)[0] in roles

    from src.api.routes import chat_routing, openai_compat

    monkeypatch.setattr(openai_compat, "live_stack_role_records", lambda: priors["roles"])
    openai_roles = openai_compat.available_roles()
    assert openai_roles[:3] == ["orchestrator", "architect", "worker"]
    assert set(roles) <= set(openai_roles)
    retired_architect_role = "architect" + "_coding"
    assert retired_architect_role not in openai_roles

    monkeypatch.setattr(chat_routing, "live_stack_role_records", lambda: priors["roles"])
    assert set(chat_routing._live_heuristic_prior_roles()) == roles

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["frontdoor"] == 18.5
    assert q_priors.baseline_tps_by_role["worker_summarize"] == 18.5
    assert q_priors.baseline_tps_source_by_role["frontdoor"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["frontdoor"] == pytest.approx(0.874)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_worker_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = WORKER_LANE_ROLES
    config = _config(tmp_path, mode="update", roles=roles)
    _worker_alias_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_worker_registry(config.lean_registry)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    descriptor_step = next(step for step in check_report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptors_before
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    approved_config = StackChangePipelineConfig(
        **{**config.__dict__, "allow_descriptor_model_removal": True}
    )
    update_report = run_stack_change_pipeline(approved_config)

    assert update_report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "gemma4-26b-a4b-q8_0"
    ]
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert set(priors["roles"]) == roles
    for role in roles:
        assert priors["roles"][role]["model_id"] == "gemma4-26b-a4b-q8_0"
        assert priors["roles"][role]["priors"]["throughput_tps"] == 66.2
        assert priors["roles"][role]["priors"]["quality_overall"] == pytest.approx(0.9)

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    for role in roles:
        assert f"| {role}" in operator_summary
        assert priors["roles"][role]["display_name"] in operator_summary
    assert "gemma-4-26B-A4B-it-Q4_K_M" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    for role in roles:
        assert f"| {role} |" in system_card
        assert priors["roles"][role]["display_name"] in system_card
    assert "gemma-4-26B-A4B-it-Q4_K_M" not in system_card

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["worker_general"] == 66.2
    assert q_priors.baseline_tps_source_by_role["worker_general"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["worker_general"] == pytest.approx(0.9)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []
    _assert_seeding_descriptor_fallback_consumer(
        config.descriptors,
        expected_roles=roles,
        expected_tps=66.2,
    )
    expected_port = _primary_port_for_roles(priors, roles)
    _assert_text_stack_primary_port_consumers(
        config.stack_priors,
        expected_roles=roles,
        expected_port=expected_port,
    )
    _assert_worker_pool_stack_prior_consumer(
        priors["roles"],
        expected_port=expected_port,
        expected_model_path_fragment="gemma-4-26B-A4B-it-Q8_0.gguf",
        monkeypatch=monkeypatch,
    )
    _assert_factual_risk_stack_prior_consumer(
        config.stack_priors,
        expected_tiers={"worker_general": "tier_2"},
    )

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_vision_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = {"worker_vision", "vision_escalation"}
    config = _config(tmp_path, mode="update", roles=roles)
    _vision_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_vision_registry(config.lean_registry)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    descriptor_step = next(step for step in check_report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptors_before
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    approved_config = StackChangePipelineConfig(
        **{**config.__dict__, "allow_descriptor_model_removal": True}
    )
    update_report = run_stack_change_pipeline(approved_config)

    assert update_report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    # 2026-07-31 vision unification: ONE :8086 process, so ONE VL descriptor.
    # Was {"qwen2.5-vl-7b-q8_0", "qwen3-vl-30b-a3b-q8_0"} when vision_escalation
    # had its own :8087 server and its own model.
    assert {model["model_id"] for model in descriptors["models"]} == {
        "qwen3-vl-30b-a3b-q8_0",
    }
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert set(priors["roles"]) == roles
    # Both roles ride one process, so the swap must land identically on both —
    # worker_vision was 14.5 t/s @ 0.84 on its own 7B before the unification.
    for role in ("worker_vision", "vision_escalation"):
        assert priors["roles"][role]["model_id"] == "qwen3-vl-30b-a3b-q8_0"
        assert priors["roles"][role]["priors"]["throughput_tps"] == 5.9
        assert priors["roles"][role]["priors"]["quality_overall"] == pytest.approx(0.92)
    assert priors["roles"]["vision_escalation"]["serving"]["binding"] == "server_mode.shared_with"

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    for role in roles:
        assert f"| {role}" in operator_summary
        assert priors["roles"][role]["display_name"] in operator_summary
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M" not in operator_summary
    assert "Qwen3-VL-30B-A3B-Instruct-Q4_K_M" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    for role in roles:
        assert f"| {role} |" in system_card
        assert priors["roles"][role]["display_name"] in system_card
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M" not in system_card
    assert "Qwen3-VL-30B-A3B-Instruct-Q4_K_M" not in system_card

    from src.api.routes.chat_pipeline.vision_stage import _vl_port_for_role
    from src.api.routes.chat_vision import _vl_url_for_port, _vl_url_for_role
    from src.api.routes.vision_serving import stack_prior_vl_ports

    # 2026-07-31 vision unification: :8087 is RETIRED. vision_escalation is a
    # routing label on worker_vision's :8086 process, so every VL consumer must
    # resolve BOTH roles to 8086 (was worker_vision 8086 / vision_escalation 8087).
    assert stack_prior_vl_ports(config.stack_priors) == {
        "worker_vision": 8086,
        "vision_escalation": 8086,
    }
    assert _vl_port_for_role("worker_vision", config.stack_priors) == 8086
    assert _vl_port_for_role("vision_escalation", config.stack_priors) == 8086
    assert _vl_url_for_role("worker_vision", config.stack_priors) == "http://localhost:8086"
    assert _vl_url_for_role("vision_escalation", config.stack_priors) == "http://localhost:8086"
    assert _vl_url_for_port(8086, config.stack_priors) == "http://localhost:8086"
    # The retired port must be REFUSED, not silently fallen back to localhost:8087.
    with pytest.raises(ValueError, match="No generated VL URL for port 8087"):
        _vl_url_for_port(8087, config.stack_priors)

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    # worker_vision was 14.5 pre-unification (its own 7B); it now reads the
    # single :8086 model's prior, same as its alias.
    assert q_priors.baseline_tps_by_role["worker_vision"] == 5.9
    assert q_priors.baseline_tps_by_role["vision_escalation"] == 5.9
    assert q_priors.baseline_tps_source_by_role["worker_vision"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["vision_escalation"] == pytest.approx(0.92)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []
    # Tier is derived from model memory (src/classifiers/factual_risk.py), and
    # both roles now name the same 38 GB model — worker_vision was tier_3 only
    # while it carried its own 13 GB 7B.
    _assert_factual_risk_stack_prior_consumer(
        config.stack_priors,
        expected_tiers={
            "worker_vision": "tier_2",
            "vision_escalation": "tier_2",
        },
    )

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    assert calls[0]["text"] is True
    assert calls[0]["capture_output"] is True
    assert calls[0]["check"] is False
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_ingest_swap_updates_generated_consumers_with_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roles = INGEST_PROCESS_ROLES
    config = _config(tmp_path, mode="update", roles=roles)
    _ingest_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    descriptors_before = config.descriptors.read_text(encoding="utf-8")
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _swapped_ingest_registry(config.lean_registry)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    descriptor_step = next(step for step in check_report.steps if step.name == "descriptors")
    assert any("descriptor artifact is stale:" in error for error in descriptor_step.errors)
    assert any("descriptor update would remove existing model_id" in error for error in descriptor_step.errors)
    assert config.descriptors.read_text(encoding="utf-8") == descriptors_before
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    approved_config = StackChangePipelineConfig(
        **{**config.__dict__, "allow_descriptor_model_removal": True}
    )
    update_report = run_stack_change_pipeline(approved_config)

    assert update_report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "qwen3-next-80b-a3b-q8_0"
    ]
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    role = priors["roles"]["ingest_long_context"]
    assert set(priors["roles"]) == roles
    assert role["model_id"] == "qwen3-next-80b-a3b-q8_0"
    assert role["priors"]["throughput_tps"] == 14.2
    assert role["priors"]["quality_overall"] == pytest.approx(0.963)
    assert role["model"]["ctx_max"] == 262144
    # This simulated registry declares no `serving_shape` of its own, so the
    # context comes from the AMBIENT launcher table (derived from the real
    # master's serving_shape.n_ctx). Since the 2026-09-22 cutover (860b0b2d) the
    # role is an alias and has NO launch context of its own — it inherits its
    # host's, architect_general's (196608 at the time of writing; it was the
    # role's own 262144 before; the host is architect_critic since the 2026-09-27
    # ARCHITECT SWAP). The fixture exercises the model/descriptor swap,
    # not the serving shape, so it tracks the ambient HOST value rather than
    # pinning a literal that has moved twice.
    from scripts.server.stack_manifest import LAUNCH_CONTEXT_TOKENS

    assert "ingest_long_context" not in LAUNCH_CONTEXT_TOKENS
    host_context = LAUNCH_CONTEXT_TOKENS["architect_critic"]
    assert role["serving"]["effective_context_tokens"] == host_context
    assert role["serving"]["launch"]["runtime"]["cache"]["context_tokens"] == host_context

    operator_summary = config.operator_summary.read_text(encoding="utf-8")
    assert "Source: `orchestration/derived/stack_priors.yaml`" in operator_summary
    assert "| ingest_long_context" in operator_summary
    assert role["display_name"] in operator_summary
    assert "Qwen3-Next-80B-A3B-Instruct-Q4_K_M" not in operator_summary

    system_card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert "Source: orchestration/derived/stack_priors.yaml" in system_card
    assert "| ingest_long_context |" in system_card
    assert role["display_name"] in system_card
    assert "Qwen3-Next-80B-A3B-Instruct-Q4_K_M" not in system_card

    from src.api.routes.dashboard_topology import _stack_prior_port_hints
    from src.api.routes.health import _stack_prior_backend_urls

    expected_port = _primary_port_for_roles(priors, roles)
    assert _stack_prior_backend_urls(config.stack_priors) == {
        # One :8083 process, grouped under the sorted role set on that port.
        "architect_critic/ingest_long_context": f"http://localhost:{expected_port}"
    }
    port_hints = _stack_prior_port_hints(config.stack_priors)
    # The dashboard labels a port by the process that LAUNCHES it; an alias
    # (launch.primary_roles excludes it) never claims a port label.
    assert port_hints[expected_port].split(".", 1)[0] == "architect_critic"
    assert "ingest_long_context" not in {label.split(".", 1)[0] for label in port_hints.values()}
    # The alias rides :8083; none of its retired dedicated ports may resurface.
    assert expected_port == 8083
    assert not ({8085, 8185, 8285, 8385, 8485} & set(port_hints))

    q_priors = stack_prior_q_scorer_priors_by_role(config.stack_priors)
    assert q_priors.baseline_tps_by_role["ingest_long_context"] == 14.2
    assert q_priors.baseline_tps_source_by_role["ingest_long_context"] == PRIOR_SOURCE_STACK_PRIORS
    assert q_priors.baseline_quality_by_role["ingest_long_context"] == pytest.approx(0.963)
    assert validate_live_q_scorer_prior_sources(config.stack_priors) == []
    _assert_factual_risk_stack_prior_consumer(
        config.stack_priors,
        expected_tiers={"ingest_long_context": "tier_1"},
    )

    calls: list[dict[str, Any]] = []
    original_run = pipeline.subprocess.run

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command != pipeline._promotion_gate_command():
            return original_run(command, **kwargs)
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="promotion gate ok\n",
            stderr="",
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    swap_check_config = StackChangePipelineConfig(
        **{**approved_config.__dict__, "mode": "check", "run_promotion_gate": True}
    )
    swap_check_report = run_stack_change_pipeline(swap_check_config)

    assert swap_check_report.ok
    assert len(calls) == 1
    assert calls[0]["command"] == pipeline._promotion_gate_command()
    assert calls[0]["cwd"] == tmp_path
    promotion_step = next(step for step in swap_check_report.steps if step.name == "promotion_gate")
    assert promotion_step.status == "ok"
    assert any("promotion gate ok" in detail for detail in promotion_step.details)
    assert any(step.name == "operator_summary" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "q_scorer_priors" and step.status == "ok" for step in swap_check_report.steps)
    assert any(step.name == "runtime_attestation" and step.status == "ok" for step in swap_check_report.steps)


def test_simulated_shared_runtime_aliases_compile_as_one_runtime_descriptor(
    tmp_path: Path,
) -> None:
    roles = WORKER_LANE_ROLES
    config = _config(tmp_path, mode="update", roles=roles)
    _worker_alias_registry(config.lean_registry)

    report = run_stack_change_pipeline(config)

    assert report.ok
    descriptors = yaml.safe_load(config.descriptors.read_text(encoding="utf-8"))
    assert [model["model_id"] for model in descriptors["models"]] == [
        "gemma4-26b-a4b-q4_k_m"
    ]
    model = descriptors["models"][0]
    assert model["role_bindings"]["roles"] == [
        "frontdoor",
        "toolrunner",
        "worker_general",
        "worker_math",
    ]
    assert not any(gap.startswith("Role-server conflict:") for gap in model["known_gaps"])
    assert not any("ignored non-live role model metadata" in gap for gap in model["known_gaps"])
    alias_overrides = model["role_bindings"]["alias_overrides"]
    ignored_models = {override["ignored_model_id"] for override in alias_overrides}
    assert ignored_models == {"qwen2.5-math-7b-q4_k_m", "qwen3-coder-30b-a3b-q4_k_m"}

    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    # One :8070 process: every alias compiles the HOST's runtime and launch
    # requirements (worker_general included, since the 2026-09-22 cutover).
    primary_runtime = priors["roles"]["frontdoor"]["serving"]["launch"]["runtime"]
    primary_requirements = priors["roles"]["frontdoor"]["serving"]["launch"]["requirements"]
    assert primary_requirements["model_path"].endswith("gemma-4-26B-A4B-it-Q4_K_M.gguf")
    # The host launches the process and owns its NEXTN/MTP draft; aliases ride
    # it and compile a DISABLED spec block by contract (stack_priors.py
    # `_spec_type_has_mtp(spec_type_prior) and role == primary_role`), so their
    # records match the launch manifest, which nulls the draft for aliases. The
    # old :8072 worker_pool host hid this: worker_pool mode stamps spec onto every
    # role on the pool. Everything else in the runtime must be the host's.
    primary_spec = primary_runtime["flags"]["spec"]
    assert primary_spec["enabled"] is True
    assert primary_spec["type"] == "draft-mtp"

    def without_spec(runtime: dict[str, Any]) -> dict[str, Any]:
        stripped = json.loads(json.dumps(runtime))
        stripped["flags"].pop("spec")
        return stripped

    for alias in ("worker_general", "worker_math", "toolrunner"):
        alias_runtime = priors["roles"][alias]["serving"]["launch"]["runtime"]
        assert priors["roles"][alias]["serving"]["binding"] == "server_mode.shared_with"
        assert without_spec(alias_runtime) == without_spec(primary_runtime)
        assert alias_runtime["flags"]["spec"]["enabled"] is False
        assert alias_runtime["flags"]["spec"]["draft_model_path"] is None
        assert priors["roles"][alias]["serving"]["launch"]["requirements"] == primary_requirements


def test_simulated_retired_role_enum_is_removed_by_update(tmp_path: Path) -> None:
    roles = {"frontdoor", "worker_summarize"}
    config = _config(tmp_path, mode="update", roles=roles)
    _base_frontdoor_registry(config.lean_registry)
    retired_role = "architect" + "_coding"
    assert run_stack_change_pipeline(config).ok
    _write_role_enum_files(config, roles | {retired_role})

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    assert any("procedure role enums are stale" in error for error in check_report.errors)

    update_report = run_stack_change_pipeline(config)

    assert update_report.ok
    procedure = yaml.safe_load(config.procedure.read_text(encoding="utf-8"))
    role_enum = procedure["inputs"][0]["validation"]["enum"]
    assert retired_role not in role_enum
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    assert retired_role not in priors["roles"]
    card = gen_system_card.generate_system_card(config.repo_root, state_override={})
    assert f"| {retired_role} |" not in card
    assert f"{retired_role} is not an active server role" in card


def test_simulated_runtime_requirement_drift_fails_until_regenerated(tmp_path: Path) -> None:
    roles = WORKER_LANE_ROLES
    config = _config(tmp_path, mode="update", roles=roles)
    _worker_alias_registry(config.lean_registry)
    assert run_stack_change_pipeline(config).ok
    priors_before = config.stack_priors.read_text(encoding="utf-8")
    _worker_alias_registry(
        config.lean_registry,
        binary_dir="/tmp/simulated-ik-build/bin",
        ld_library_path=["/tmp/simulated-ik-build/lib"],
    )

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    check_report = run_stack_change_pipeline(check_config)

    assert not check_report.ok
    assert any("stack-prior artifact is stale" in error for error in check_report.errors)
    assert any("serving.launch.runtime does not match" in error for error in check_report.errors)
    assert any("/tmp/simulated-ik-build/bin" in error for error in check_report.errors)
    assert config.stack_priors.read_text(encoding="utf-8") == priors_before

    update_report = run_stack_change_pipeline(config)

    assert update_report.ok
    priors = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    runtime = priors["roles"]["worker_general"]["serving"]["launch"]["runtime"]
    assert runtime["binary_dir"] == "/tmp/simulated-ik-build/bin"
    assert runtime["binary_path"] == "/tmp/simulated-ik-build/bin/llama-server"
    assert runtime["ld_library_path"] == ["/tmp/simulated-ik-build/lib"]
    assert runtime["env_policy"] == "binary_override_strip_ggml"


def test_simulated_context_kv_and_acceleration_drift_are_rejected(
    tmp_path: Path,
) -> None:
    roles = {
        "frontdoor",
        "worker_general",
        "worker_vision",
        "vision_escalation",
        "architect_general",
    }
    config = _config(tmp_path, mode="update", roles=roles)
    _write_yaml(
        config.lean_registry,
        {
            "server_mode": {
                # 2026-09-22 lineup cutover (860b0b2d): the :8072 worker pool is
                # RETIRED and worker_general is an alias on frontdoor's :8070
                # process (launch manifest PORT_MAP worker_general -> 8070). Its
                # own `server_mode.worker` row serving a DIFFERENT model (gemma4)
                # is no longer expressible: the guard resolves worker_general to
                # its manifest host and requires the alias's compiled launch
                # requirements to EQUAL frontdoor's, so that form failed with a
                # model_path mismatch before any drift detector ran.
                "frontdoor": {
                    "url": "http://localhost:8070",
                    "port": 8070,
                    "tier": "hot",
                    "model_role": "frontdoor",
                    # GGUF name canonicalises to the same model_id as the role
                    # metadata (`Qwen_` publisher prefix dropped), so the shared
                    # alias is not reported as a role/server model conflict.
                    "model": "Qwen3.6-35B-A3B-Q8_0.gguf",
                    "shared_with": ["worker_general"],
                    "throughput": 24.3,
                    "memory_gb": 37,
                },
                # 2026-07-31 vision unification: ONE :8086 process, two role
                # names. vision_escalation's own :8087 server_mode entry (its own
                # 30B model) is gone — the port is RETIRED and the launch guard
                # now requires the alias's compiled launch requirements to equal
                # its host's.
                "worker_vision": {
                    "url": "http://localhost:8086",
                    "port": 8086,
                    "tier": "hot",
                    "model_role": "worker_vision",
                    "model": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                    "shared_with": ["vision_escalation"],
                    "throughput": 20.0,
                    "memory_gb": 7,
                },
                # 2026-09-27 ARCHITECT SWAP: architect_general launches on :8074
                # (the full-CPU instance); the real launch manifest is ambient here.
                "architect_general": {
                    "url": "http://localhost:8074",
                    "port": 8074,
                    "tier": "hot",
                    "model_role": "architect_general",
                    "model": "Qwen3.5-122B-A3B-Instruct-Q4_K_M.gguf",
                    "throughput": 12.19,
                    "memory_gb": 133,
                },
            },
            "roles": {
                "frontdoor": {
                    "model": {"name": "Qwen3.6-35B-A3B-Q8_0", "ctx_max": 131072},
                    "performance": {"quality_pct": 93, "baseline_tps": 24.3},
                    # The :8070 process owns the MTP self-draft (v6 token
                    # 'draft-mtp'); its aliases inherit it and compile none.
                    "acceleration": {"type": "speculative_decoding", "spec_type": "draft-mtp"},
                    "memory": {"residency": "hot"},
                },
                "worker_general": {
                    "model": {
                        "name": "Qwen3.6-35B-A3B-Q8_0",
                        "ctx_max": 131072,
                        "shared_gguf_with": "frontdoor",
                    },
                    "performance": {"quality_pct": 93, "baseline_tps": 24.3},
                    "memory": {"residency": "hot"},
                },
                "worker_vision": {
                    "model": {"name": "Qwen2.5-VL-7B-Instruct-Q4_K_M", "ctx_max": 8192},
                    "performance": {"quality_pct": 81, "baseline_tps": 20.0},
                    # The VL model is a MoE with a forced-expert override, so the
                    # :8086 runtime carries an override_kv the guard must police.
                    "acceleration": {
                        "type": "moe_expert_reduction",
                        "override_key": "qwen3vlmoe.expert_used_count",
                        "experts": 4,
                    },
                    "memory": {"residency": "hot"},
                },
                "vision_escalation": {
                    "alias_of": "worker_vision",
                    "model": {
                        "name": "Qwen2.5-VL-7B-Instruct-Q4_K_M",
                        "ctx_max": 8192,
                        "shared_gguf_with": "worker_vision",
                    },
                    "performance": {
                        "inherits_from": "worker_vision",
                        "quality_pct": 81,
                        "baseline_tps": 20.0,
                    },
                    "memory": {"residency": "hot"},
                },
                "architect_general": {
                    "model": {"name": "Qwen3.5-122B-A3B-Instruct-Q4_K_M", "ctx_max": 16384},
                    "performance": {"quality_pct": 94, "baseline_tps": 12.19},
                    "acceleration": {
                        "type": "moe_expert_reduction",
                        "override_key": "qwen35moe.expert_used_count",
                        "experts": 8,
                        "draft_max": 4,
                    },
                    "memory": {"residency": "hot"},
                },
            },
        },
    )
    assert run_stack_change_pipeline(config).ok
    payload = yaml.safe_load(config.stack_priors.read_text(encoding="utf-8"))
    payload["roles"]["frontdoor"]["serving"]["launch"]["runtime"]["cache"]["kv_type_k"] = "f16"
    # Acceleration drift. This corruption used to be written onto worker_general's
    # own :8072 runtime; since 860b0b2d worker_general is an alias on :8070 and
    # WP-13 validates the shared process's RUNTIME once, on the host's row, so
    # the spec flag goes on frontdoor — same detector, the row that owns it.
    assert payload["roles"]["frontdoor"]["serving"]["launch"]["runtime"]["flags"]["spec"][
        "enabled"
    ] is True
    payload["roles"]["frontdoor"]["serving"]["launch"]["runtime"]["flags"]["spec"][
        "enabled"
    ] = False
    # Context drift stays on the ALIAS: an alias's declared context IS judged
    # against its host's launch context.
    payload["roles"]["worker_general"]["serving"]["effective_context_tokens"] = 8192
    payload["roles"]["architect_general"]["serving"]["launch"]["runtime"]["flags"][
        "override_kv"
    ] = []
    payload["roles"]["worker_vision"]["serving"]["launch"]["requirements"][
        "mmproj_path"
    ] = "/tmp/stale-mmproj.gguf"
    # Spurious-override detector. This corruption used to be written onto
    # vision_escalation, but the 2026-07-31 unification made that role an alias:
    # WP-13 deliberately validates an alias's RUNTIME once, on its host's row
    # (stack_change_guard.py: `if target_launch_runtime and not host_role`), so a
    # runtime corruption written to the alias is no longer observable anywhere.
    # The row that owns the :8086 runtime is worker_vision, so the fabricated
    # override goes there — same detector, correct row.
    payload["roles"]["worker_vision"]["serving"]["launch"]["runtime"]["flags"][
        "override_kv"
    ] = ["stale.vision_override=int:1"]
    _write_yaml(config.stack_priors, payload)

    check_config = StackChangePipelineConfig(**{**config.__dict__, "mode": "check"})
    report = run_stack_change_pipeline(check_config)

    assert not report.ok
    assert any("serving.effective_context_tokens 8192" in error for error in report.errors)
    assert any("serving.launch.runtime does not match" in error for error in report.errors)
    assert any('"kv_type_k": "f16"' in error for error in report.errors)
    assert any('"enabled": false' in error for error in report.errors)
    assert any("qwen35moe.expert_used_count=int:8" in error for error in report.errors)
    assert any("stale.vision_override=int:1" in error for error in report.errors)
    assert any("mmproj_path" in error and "stale-mmproj" in error for error in report.errors)
