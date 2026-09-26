"""Every launch branch strips ambient GGML_* the same way and keeps the role's declared knobs.

Since b060dd56 (2026-07-31) every role's compiled prior carries a binary_dir, so the
binary-override strip fired on every launch and removed the role's own stack_env block
along with ambient GGML_* keys. The only knob it ever dropped was architect_critic's dead
GGML_NUMA_REPACK_INTERLEAVE (DAR-LAT-3h). The embedding and eval-batch branches never
stripped at all, so ambient GGML_* reached those processes.

The neutrality tests pin the landing as a QUIET correction: with today's blocks, no launch
branch's env changes.
"""

from __future__ import annotations

import inspect

import pytest

from scripts.server import orchestrator_stack as launcher
from scripts.server.stack_env import _ROLE_ENV_BLOCKS, _role_env_overrides, build_launch_env
from scripts.server.stack_numa import NUMA_CONFIG
from src.roles import Role


def test_strip_removes_ambient_and_keeps_declared() -> None:
    env = {"GGML_AMBIENT_LEAK": "1", "GGML_DECLARED": "0", "GGML_IQK": "1"}
    launcher._apply_runtime_requirements_env(
        env, binary_override="/k/llama-server", ld_paths=None, preserve={"GGML_DECLARED"}
    )
    assert env == {"GGML_DECLARED": "0", "GGML_IQK": "1", "KMP_BLOCKTIME": "10"}


def test_strip_without_preserve_is_unchanged_behaviour() -> None:
    env = {"GGML_AMBIENT_LEAK": "1", "GGML_IQK": "1"}
    launcher._apply_runtime_requirements_env(env, binary_override="/k/llama-server", ld_paths=None)
    assert env == {"GGML_IQK": "1", "KMP_BLOCKTIME": "10"}


def test_strip_no_longer_keys_on_binary_override() -> None:
    # binary_dir presence was the misread signal (b060dd56 made it universal). Ambient
    # GGML_* is stripped with or without an override; KMP_BLOCKTIME stays override-only.
    env = {"GGML_AMBIENT_LEAK": "1", "GGML_IQK": "1"}
    launcher._apply_runtime_requirements_env(env, binary_override=None, ld_paths=None)
    assert env == {"GGML_IQK": "1"}


def test_shared_strip_helper() -> None:
    env = {"GGML_AMBIENT_LEAK": "1", "GGML_IQK": "1", "GGML_KEEP": "2", "OMP_PLACES": "cores"}
    assert launcher._strip_ambient_ggml(env, {"GGML_KEEP"}, label="t") == ["GGML_AMBIENT_LEAK"]
    assert env == {"GGML_IQK": "1", "GGML_KEEP": "2", "OMP_PLACES": "cores"}


def _launch_role_names() -> set[str]:
    names = set(NUMA_CONFIG) | {"worker", "worker_general", "frontdoor", "embedder",
                                "formalizer", "general_gemma_3_27b_it_qat", "dense_q8", "dense_q4"}
    names |= {r.value for r in Role}
    names |= {k for k in _ROLE_ENV_BLOCKS if not k.endswith(tuple("0123456789"))}
    return names


@pytest.mark.parametrize("role", sorted(_launch_role_names()))
def test_neutral_today_every_role_resolves_no_ggml_beyond_iqk(role: str) -> None:
    """The QUIET-correction proof: `preserve` is empty for every role a launch branch can
    name, so the strip removes exactly what it removed before this change."""
    declared = {k for k in _role_env_overrides(role) if k.startswith("GGML_")}
    assert declared <= {"GGML_IQK"}, f"{role} now declares {declared}; this landing is no longer neutral"


@pytest.mark.parametrize("role", sorted(_launch_role_names()))
def test_neutral_today_launch_env_is_identical_with_and_without_preserve(role: str) -> None:
    ambient = {"GGML_AMBIENT_LEAK": "1", "PATH": "/usr/bin"}
    before = build_launch_env(role, dict(ambient))
    after = dict(before)
    launcher._apply_runtime_requirements_env(before, binary_override="/k/llama-server", ld_paths=None)
    launcher._apply_runtime_requirements_env(
        after, binary_override="/k/llama-server", ld_paths=None, preserve=_role_env_overrides(role)
    )
    assert before == after


def test_embedding_env_with_the_live_ambient_is_unchanged_by_the_strip() -> None:
    # The live embedders' environ carries GGML_IQK=1 as its only GGML_* key (2026-09-26), so
    # the ambient the stack launches from has none. Under that ambient the new strip is a no-op.
    env = build_launch_env("embedder", {"PATH": "/usr/bin"})
    snapshot = dict(env)
    assert launcher._strip_ambient_ggml(env, _role_env_overrides("embedder"), label="embedding") == []
    assert env == snapshot


def test_every_launch_branch_strips() -> None:
    """Structural: every build_launch_env call in start_server is followed by a strip."""
    src = inspect.getsource(launcher.start_server)
    n_env = src.count("build_launch_env(")
    n_strip = src.count("_apply_runtime_requirements_env(") + src.count("_strip_ambient_ggml(")
    n_preserve = src.count("preserve=_role_env_overrides(") + src.count("_strip_ambient_ggml(")
    assert n_env == n_strip == n_preserve, (n_env, n_strip, n_preserve)
