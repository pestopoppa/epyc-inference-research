"""STACKCHG-8083BATCH-20261004 part B: `-b 512 -ub 512` on :8083.

A declared `serving_shape.batch` / `.ubatch` must travel registry -> compiled priors ->
launcher argv -> runtime attestation; the compiler must refuse an inert ubatch (K4) and a
`vram_non_kv_gib` derived at a different ubatch (the KQ-mask term scales with -ub).
Offline: synthetic priors, no server, no inference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.server import orchestrator_stack as oss
from scripts.server import stack_commands
from scripts.server.stack_manifest import DEFAULT_UBATCH_TOKENS
from src.registry.stack_priors import _launch_runtime_record

REPO = Path(__file__).resolve().parents[2]


def _record(server_cfg: dict, mode: str = "default") -> dict:
    return _launch_runtime_record(
        role="architect_critic",
        descriptor={},
        server_cfg=server_cfg,
        role_cfg={},
        launch_cfg={
            "effective_context_tokens": 8192,
            "launch": {"primary_roles": ["architect_critic"], "modes": [mode],
                       "requirements": {}, "runtime": {}},
        },
    )


# -- compiler ------------------------------------------------------------------------


def test_declared_batch_and_ubatch_compile() -> None:
    cache = _record({"serving_shape": {"batch": 512, "ubatch": 512}})["cache"]
    assert cache["ubatch"] == 512 and cache["batch"] == 512


def test_undeclared_keeps_the_default_and_no_batch_key() -> None:
    cache = _record({"serving_shape": {"n_ctx": 65536}})["cache"]
    assert cache["ubatch"] == DEFAULT_UBATCH_TOKENS
    assert "batch" not in cache


def test_compiler_refuses_an_inert_ubatch_k4() -> None:
    with pytest.raises(ValueError, match="K4"):
        _record({"serving_shape": {"batch": 512, "ubatch": 1024}})
    with pytest.raises(ValueError, match="K4"):   # no -b -> llama.cpp's 2048
        _record({"serving_shape": {"ubatch": 4096}})


def test_compiler_refuses_vram_figure_derived_at_another_ubatch() -> None:
    with pytest.raises(ValueError, match="vram_non_kv_ubatch"):
        _record({"serving_shape": {"batch": 512, "ubatch": 512, "vram_non_kv_ubatch": 2048}})
    # undeclared ubatch resolves to the default, which must equal the named one
    with pytest.raises(ValueError, match="vram_non_kv_ubatch"):
        _record({"serving_shape": {"vram_non_kv_ubatch": 512}})
    ok = _record({"serving_shape": {"batch": 512, "ubatch": 512, "vram_non_kv_ubatch": 512}})
    assert ok["cache"]["ubatch"] == 512


# -- launcher ------------------------------------------------------------------------


def _write_prior(tmp_path: Path, cache: dict) -> Path:
    path = tmp_path / "stack_priors.yaml"
    path.write_text(yaml.safe_dump({"roles": {"architect_critic": {
        "deployment_status": "live_stack",
        "serving": {"launch": {"requirements": {}, "runtime": {
            "binary_path": "/prior/llama-server",
            "cache": {"context_tokens": 393216, "slots": 4, **cache},
            "flags": {"flash_attn": True, "jinja": True, "spec": {"enabled": False}},
        }}},
    }}}, sort_keys=False), encoding="utf-8")
    return path


def _role() -> SimpleNamespace:
    return SimpleNamespace(
        name="architect_critic",
        model=SimpleNamespace(full_path="/m/q38.gguf"),
        acceleration=SimpleNamespace(type="none", experts=None, draft_role=None),
    )


def _flag(cmd: list[str], flag: str) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_launcher_emits_declared_batch_before_ubatch(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"batch": 512, "ubatch": 512}))
    cmd = oss._build_role_command(_role(), port=9083, prepare_runtime_dirs=False)
    assert _flag(cmd, "-b") == "512" and _flag(cmd, "-ub") == "512"
    assert cmd.index("-b") + 2 == cmd.index("-ub")
    assert "[WARN]" not in capsys.readouterr().out


def test_launcher_emits_no_batch_when_undeclared(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"ubatch": 2048}))
    cmd = oss._build_role_command(_role(), port=9083, prepare_runtime_dirs=False)
    assert "-b" not in cmd and _flag(cmd, "-ub") == "2048"


def test_k4_guard_compares_against_the_declared_batch(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(oss, "STACK_PRIORS_PATH", _write_prior(tmp_path, {"batch": 512, "ubatch": 1024}))
    oss._build_role_command(_role(), port=9083, prepare_runtime_dirs=False)
    out = capsys.readouterr().out
    assert "[WARN]" in out and "-b 512" in out and "no -b is emitted" not in out


# -- runtime attestation -------------------------------------------------------------


def _info():
    return stack_commands.ProcessInfo(role="architect_critic", pid=321, port=8083,
                                      started_at="now", model_path="/models/m.gguf",
                                      log_file="a.log")


_BASE = ["/opt/llama/bin/llama-server", "-m", "/models/m.gguf", "--device", "ROCm0"]


def _contract(cache: dict) -> dict:
    return {"requirements": {"model_path": "/models/m.gguf"},
            "runtime": {"binary_path": "/opt/llama/bin/llama-server", "cache": cache,
                        "flags": {"device": "ROCm0"}},
            "ports": [8083]}


def test_attestation_clean_when_live_matches() -> None:
    assert stack_commands._runtime_attestation_warnings(
        "architect_critic", _info(), _BASE + ["-b", "512", "-ub", "512"],
        _contract({"batch": 512, "ubatch": 512})) == []


def test_attestation_reports_the_pre_relaunch_argv() -> None:
    """Today's :8083 argv (no -b, -ub 2048) against the new declaration."""
    assert stack_commands._runtime_attestation_warnings(
        "architect_critic", _info(), _BASE + ["-ub", "2048"],
        _contract({"batch": 512, "ubatch": 512})) == [
        "architect_critic pid 321 runtime ubatch expected 512; live cmdline has 2048",
        "architect_critic pid 321 runtime batch expected 512; live cmdline has no -b",
    ]


def test_the_new_field_is_mapped_for_attestation() -> None:
    assert "runtime.cache.batch" in stack_commands._RUNTIME_FIELD_CHECKS


# -- this tree: master declaration -> compiled priors (recompute, not a literal) -------


def test_this_trees_8083_priors_carry_the_master_batch_shape() -> None:
    lean = yaml.safe_load((REPO / "orchestration/model_registry.yaml").read_text())
    shape = lean["server_mode"]["architect_critic"]["serving_shape"]
    priors = yaml.safe_load((REPO / "orchestration/derived/stack_priors.yaml").read_text())
    cache = priors["roles"]["architect_critic"]["serving"]["launch"]["runtime"]["cache"]
    assert cache["ubatch"] == shape.get("ubatch", DEFAULT_UBATCH_TOKENS)
    assert cache.get("batch") == shape.get("batch")
    if "vram_non_kv_ubatch" in shape:
        assert shape["vram_non_kv_ubatch"] == cache["ubatch"]
