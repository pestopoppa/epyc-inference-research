from __future__ import annotations

from pathlib import Path

import yaml

from src.registry.registry_compiler import (
    DEFAULT_MASTER_PATH,
    _format_header_banner,
    active_roles_from_launch_meta,
    cache_key,
    compile_lean,
    load_or_compile,
)


def test_generated_banner_labels_canonical_and_worktree_master_inputs(tmp_path: Path) -> None:
    canonical = _format_header_banner(DEFAULT_MASTER_PATH, set(), "abc123")
    assert f"# canonical master registry at {DEFAULT_MASTER_PATH.resolve()}\n" in canonical
    assert "# edit registry data, change the master source; this generated view is disposable.\n" in canonical
    assert "# noncanonical master input" not in canonical

    worktree_master = tmp_path / "orchestration" / "model_registry.yaml"
    worktree_banner = _format_header_banner(worktree_master, set(), "def456")
    assert f"# noncanonical master input at {worktree_master.resolve()}\n" in worktree_banner
    assert f"# canonical source of truth: {DEFAULT_MASTER_PATH.resolve()}\n" in worktree_banner


def test_active_roles_from_launch_meta_includes_shared_aliases() -> None:
    launch_meta = {
        "frontdoor": {
            "tier": "hot",
            "shared_with_first_n": ["coder_escalation", "worker_summarize"],
        },
        "worker_general": {
            "tier": "hot",
            "shared_with_first_n": ["worker_math", "toolrunner"],
        },
        "embedder": {"tier": "hot", "mode": "embedding"},
    }

    assert active_roles_from_launch_meta(launch_meta) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
        "worker_general",
        "worker_math",
        "toolrunner",
        "embedder",
    }


def test_active_roles_from_launch_meta_excludes_launcher_only_servers() -> None:
    launch_meta = {
        "frontdoor": {"tier": "hot"},
        "eval_batch_frontdoor": {
            "tier": "warm",
            "mode": "eval_batch_frontdoor",
            "launcher_only": True,
            "shared_with_first_n": ["should_not_leak"],
        },
    }

    assert active_roles_from_launch_meta(launch_meta) == {"frontdoor"}


def test_compile_lean_keeps_alias_records_when_active_roles_include_aliases(
    tmp_path: Path,
) -> None:
    master = {
        "runtime_defaults": {"temperature": 0.2},
        "server_mode": {
            "frontdoor": {
                "port": 8070,
                "shared_with": ["coder_escalation", "worker_summarize"],
            },
            "coder_escalation": {"port": 8070},
            "worker_summarize": {"port": 8070},
            "cold_candidate": {"port": 9000},
        },
        "roles": {
            "frontdoor": {"model": {"path": "frontdoor.gguf"}},
            "coder_escalation": {"model": {"path": "frontdoor.gguf"}},
            "worker_summarize": {"model": {"path": "frontdoor.gguf"}},
            "cold_candidate": {"model": {"path": "cold.gguf"}},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    active = active_roles_from_launch_meta(
        {
            "frontdoor": {
                "shared_with_first_n": ["coder_escalation", "worker_summarize"]
            }
        }
    )
    lean = compile_lean(master_path, active)

    assert set(lean["server_mode"]) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
    }
    assert set(lean["roles"]) == {
        "frontdoor",
        "coder_escalation",
        "worker_summarize",
    }


def test_compile_lean_keeps_server_mode_backing_process_for_model_role(
    tmp_path: Path,
) -> None:
    master = {
        "server_mode": {
            "worker": {
                "port": 8082,
                "model_role": "worker_general",
                "shared_with": ["worker_math", "toolrunner"],
            },
            "cold_candidate": {"port": 9000, "model_role": "cold_candidate"},
        },
        "roles": {
            "worker_general": {"model": {"path": "worker.gguf"}},
            "worker_math": {"model": {"path": "math.gguf"}},
            "toolrunner": {"model": {"path": "toolrunner.gguf"}},
            "cold_candidate": {"model": {"path": "cold.gguf"}},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    active = active_roles_from_launch_meta(
        {
            "worker_general": {
                "shared_with_first_n": ["worker_math", "toolrunner"]
            }
        }
    )
    lean = compile_lean(master_path, active)

    assert set(lean["server_mode"]) == {"worker"}
    assert set(lean["roles"]) == {
        "worker_general",
        "worker_math",
        "toolrunner",
    }


def test_compile_lean_drops_retired_runtime_timeout_aliases(tmp_path: Path) -> None:
    master = {
        "runtime_defaults": {
            "timeouts": {
                "default": 600,
                "roles": {
                    "worker_coder": 30,
                    "worker_code": 60,
                    "worker_general": 60,
                },
            }
        },
        "roles": {
            "worker_general": {"model": {"path": "worker.gguf"}},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    lean = compile_lean(master_path, {"worker_general"})

    role_timeouts = lean["runtime_defaults"]["timeouts"]["roles"]
    assert role_timeouts == {
        "worker_coder": 30,
        "worker_general": 60,
    }


def test_compile_lean_preserves_small_registry_metadata_sections(tmp_path: Path) -> None:
    master = {
        "runtime_quirks": {"qwen3_vl_30b": {"quirks": [{"issue": "needs mmproj"}]}},
        "deprecated_models": [{"name": "old-model", "reason": "superseded"}],
        "optimized_params": {"frontdoor": {"ubatch": 512}},
        "observations": {"kernel": ["v6"]},
        "kernel_audits": {"v6": {"status": "current"}},
        "roles": {
            "worker_general": {"model": {"path": "worker.gguf"}},
            "cold_candidate": {"model": {"path": "cold.gguf"}},
        },
        "server_mode": {
            "worker": {"model_role": "worker_general"},
            "cold_candidate": {"model_role": "cold_candidate"},
        },
    }
    master_path = tmp_path / "model_registry.yaml"
    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")

    lean = compile_lean(master_path, {"worker_general"})

    assert lean["runtime_quirks"] == master["runtime_quirks"]
    assert lean["deprecated_models"] == master["deprecated_models"]
    assert lean["optimized_params"] == master["optimized_params"]
    assert lean["observations"] == master["observations"]
    assert lean["kernel_audits"] == master["kernel_audits"]
    assert "cold_candidate" not in lean["roles"]
    assert "cold_candidate" not in lean["server_mode"]


def test_load_or_compile_does_not_rewrite_equivalent_output_without_cache_key(
    tmp_path: Path,
) -> None:
    master = {
        "runtime_defaults": {"temperature": 0.2},
        "roles": {
            "worker_general": {"model": {"path": "worker.gguf"}},
            "cold_candidate": {"model": {"path": "cold.gguf"}},
        },
        "server_mode": {
            "worker": {"model_role": "worker_general"},
            "cold_candidate": {"model_role": "cold_candidate"},
        },
    }
    master_path = tmp_path / "master.yaml"
    output_path = tmp_path / "model_registry.yaml"
    cache_key_path = tmp_path / ".lean_cache_key"
    active_roles = {"worker_general"}

    master_path.write_text(yaml.safe_dump(master), encoding="utf-8")
    compiled = compile_lean(master_path, active_roles)
    output_text = "# old generated banner\n" + yaml.safe_dump(
        compiled,
        sort_keys=False,
        default_flow_style=False,
    )
    output_path.write_text(output_text, encoding="utf-8")

    result = load_or_compile(
        master_path=master_path,
        active_roles=active_roles,
        output_path=output_path,
        cache_key_path=cache_key_path,
    )

    assert result == compiled
    assert output_path.read_text(encoding="utf-8") == output_text
    assert cache_key_path.read_text(encoding="utf-8") == cache_key(
        master_path,
        active_roles,
    )
