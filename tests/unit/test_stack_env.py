"""Tests for orchestrator stack launch-env composition."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.server.stack_env import (
    _CANONICAL_OMP_ENV,
    _LLVM20_LIBDIR,
    _ROLE_ENV_BLOCKS,
    _role_env_overrides,
    build_launch_env,
)


def test_canonical_omp_env_applied_to_every_role() -> None:
    env = build_launch_env("frontdoor", base_env={})
    for key, value in _CANONICAL_OMP_ENV.items():
        assert env[key] == value


def test_llvm20_libdir_prepended_when_missing() -> None:
    env = build_launch_env("worker", base_env={"LD_LIBRARY_PATH": "/opt/foo/lib"})
    assert env["LD_LIBRARY_PATH"] == f"{_LLVM20_LIBDIR}:/opt/foo/lib"


def test_llvm20_libdir_set_when_ld_library_path_unset() -> None:
    env = build_launch_env("worker", base_env={})
    assert env["LD_LIBRARY_PATH"] == _LLVM20_LIBDIR


def test_llvm20_libdir_idempotent_when_already_present() -> None:
    initial = f"{_LLVM20_LIBDIR}:/opt/foo/lib"
    env = build_launch_env("worker", base_env={"LD_LIBRARY_PATH": initial})
    assert env["LD_LIBRARY_PATH"] == initial


def test_worker_role_gets_v6_iqk_without_vestigial_ccd_stack() -> None:
    env = build_launch_env("worker", base_env={})
    assert env["GGML_IQK"] == "1"
    assert "GGML_CCD_POOLS" not in env
    assert "GGML_CCD_WORK_DIST" not in env
    assert "GGML_BARRIER_LOCAL_BETWEEN_OPS" not in env


def test_frontdoor_role_gets_no_ggml_env() -> None:
    env = build_launch_env("frontdoor", base_env={})
    for key in env:
        if key == "GGML_IQK":
            continue
        assert not key.startswith("GGML_"), f"frontdoor must not set {key}"


def test_explicit_iqk_override_survives_canonical_default() -> None:
    env = build_launch_env("worker_general", base_env={"GGML_IQK": "0"})
    assert env["GGML_IQK"] == "0"


def test_arch_alias_fallthrough_worker_summarize_inherits_frontdoor() -> None:
    direct = _role_env_overrides("frontdoor")
    aliased = _role_env_overrides("worker_summarize")
    assert direct == aliased == {}


def test_arch_alias_fallthrough_toolrunner_inherits_worker() -> None:
    aliased = _role_env_overrides("toolrunner")
    expected = _ROLE_ENV_BLOCKS["worker_general"]
    assert aliased == expected


def test_worker_explore_alias_inherits_worker() -> None:
    aliased = _role_env_overrides("worker_explore")
    expected = _ROLE_ENV_BLOCKS["worker_general"]
    assert aliased == expected


def test_worker_general_role_gets_v6_iqk_directly() -> None:
    env = build_launch_env("worker_general", base_env={})
    assert env["GGML_IQK"] == "1"
    assert "GGML_CCD_POOLS" not in env
    assert "GGML_CCD_WORK_DIST" not in env
    assert "GGML_BARRIER_LOCAL_BETWEEN_OPS" not in env


def test_unknown_role_returns_empty_overrides() -> None:
    assert _role_env_overrides("nonexistent_role_xyz") == {}


def test_unknown_role_still_gets_canonical_omp_and_libdir() -> None:
    env = build_launch_env("nonexistent_role_xyz", base_env={})
    for key, value in _CANONICAL_OMP_ENV.items():
        assert env[key] == value
    assert env["LD_LIBRARY_PATH"] == _LLVM20_LIBDIR


def test_base_env_preserved() -> None:
    env = build_launch_env("worker", base_env={"USER": "tester", "HOME": "/home/tester"})
    assert env["USER"] == "tester"
    assert env["HOME"] == "/home/tester"


def test_role_overrides_returns_independent_dict_copies() -> None:
    first = _role_env_overrides("worker")
    first["MUTATED_KEY"] = "MUTATED"
    second = _role_env_overrides("worker")
    assert "MUTATED_KEY" not in second


def test_dead_numa_repack_interleave_knob_is_not_emitted() -> None:
    """GGML_NUMA_REPACK_INTERLEAVE is a dead knob and must not come back.

    Removed 2026-09-26: the env var is not compiled into production v10
    (ffc1bac82eec) or the AutoKernel champion. `git grep` finds no NUMA_REPACK in
    either tree, `strings libggml-cpu.so` finds 0 hits in both builds, and the
    feature (a7b0e9644 / kill-switch b1dec7ae9) survives only on the v5 lineage.
    Emitting it would claim a tuning the binary never applies. Re-porting the
    interleave is tracked as AutoKernel hypothesis AK-H-NRI-1.
    """
    registry = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml")
        .read_text()
    )["server_mode"]
    for role in registry:
        env = build_launch_env(role, base_env={})
        assert "GGML_NUMA_REPACK_INTERLEAVE" not in env, role

    # ...and the ROCm guard still holds for the GPU architect (the MI210 27B,
    # architect_critic since the 2026-09-27 ARCHITECT SWAP).
    assert registry["architect_critic"]["device"] == "ROCm0"
    gpu_env = build_launch_env("architect_critic", base_env={})
    assert "GGML_NUMA_REPACK_INTERLEAVE" not in gpu_env
