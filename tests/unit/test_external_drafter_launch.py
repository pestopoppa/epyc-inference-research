"""DRAFT-SEL-1: an EXTERNAL drafter (DFlash2) survives priors compilation and reaches argv.

Before the fix, `_launch_runtime_record` only enabled spec for a spec_type containing
`draft-mtp`; `draft-dflash` fell through to the DISABLED spec, so selecting DFlash2 in
the registry would have launched :8083 with NO speculation. And the launcher had no
`-ngld` emission, which the DFlash2 recipe (ngld 99) requires.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "server"))

from src.registry.stack_priors import _launch_runtime_record  # noqa: E402

orchestrator_stack = importlib.import_module("orchestrator_stack")

MODEL = "/models/Qwen3.8-27B-Q8_0.gguf"
DFLASH = "/models/Qwen3.8-27B-DFlash2-Q8_0.gguf"


def _runtime(accel: dict, *, role: str = "architect_critic", primary: str = "architect_critic"):
    return _launch_runtime_record(
        role=role,
        descriptor={},
        server_cfg={"acceleration": accel, "draft_model": accel.get("draft_model")},
        role_cfg=None,
        launch_cfg={
            "launch": {
                "primary_roles": [primary],
                "modes": ["default"],
                "requirements": {"model_path": MODEL,
                                 "draft_model_path": accel.get("draft_model")},
                "runtime": {},
            }
        },
    )


def test_draft_dflash_spec_is_enabled_with_drafter_path_and_ngld() -> None:
    spec = _runtime({"type": "speculative_decoding", "spec_type": "draft-dflash",
                     "draft_model": DFLASH, "draft_max": 8,
                     "n_gpu_layers_draft": 99})["flags"]["spec"]
    assert spec["enabled"] is True
    assert spec["type"] == "draft-dflash"
    assert spec["draft_model_path"] == DFLASH
    assert spec["draft_max"] == 8
    assert spec["n_gpu_layers_draft"] == 99


def test_draft_mtp_spec_is_unchanged_and_carries_no_ngld() -> None:
    spec = _runtime({"type": "speculative_decoding", "spec_type": "draft-mtp",
                     "draft_model": MODEL, "draft_max": 4})["flags"]["spec"]
    assert spec["enabled"] is True and spec["type"] == "draft-mtp"
    assert spec["n_gpu_layers_draft"] is None


def test_alias_of_a_dflash_host_does_not_launch_its_own_drafter() -> None:
    spec = _runtime({"type": "none", "spec_type": "draft-dflash", "draft_model": DFLASH,
                     "draft_max": 8, "n_gpu_layers_draft": 99},
                    role="coder_escalation")["flags"]["spec"]
    assert spec["enabled"] is False


def test_argv_for_dflash2_emits_md_ngld_spec_type_and_depth() -> None:
    cmd: list[str] = []
    orchestrator_stack._append_spec_decode_args(
        cmd, model_path=MODEL, draft_model_path=DFLASH, spec_type="draft-dflash",
        draft_max="8", n_gpu_layers_draft="99",
    )
    assert cmd == ["-md", DFLASH, "-ngld", "99",
                   "--spec-type", "draft-dflash", "--spec-draft-n-max", "8"]


def test_argv_for_same_file_mtp_omits_md_and_ngld() -> None:
    cmd: list[str] = []
    orchestrator_stack._append_spec_decode_args(
        cmd, model_path=MODEL, draft_model_path=MODEL, spec_type="draft-mtp",
        draft_max="4", n_gpu_layers_draft="99",
    )
    assert "-md" not in cmd and "-ngld" not in cmd
    assert cmd == ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4"]


@pytest.mark.parametrize("ngld,expected", [(99, "99"), ("all", "all"), (None, None), (-1, None)])
def test_runtime_spec_args_project_ngld(ngld, expected) -> None:
    cmd: list[str] = []
    runtime = {"flags": {"spec": {"enabled": True, "type": "draft-dflash",
                                  "draft_model_path": DFLASH, "draft_max": 8,
                                  "n_gpu_layers_draft": ngld}}}
    orchestrator_stack._append_runtime_spec_args(cmd, runtime, MODEL)
    if expected is None:
        assert "-ngld" not in cmd
    else:
        assert cmd[cmd.index("-ngld") + 1] == expected


# --- STACKCHG-DFLASH2-20261003: fail closed at compile, and attest -ngld live ---

@pytest.mark.parametrize("draft_model", [MODEL, None])
def test_external_drafter_without_a_distinct_drafter_refuses_at_compile(draft_model) -> None:
    """draft-dflash whose drafter resolves to the target itself (or nothing) would launch
    `--spec-type draft-dflash` with no -md and die at load; refuse at compile instead."""
    with pytest.raises(ValueError, match="external drafter"):
        _runtime({"type": "speculative_decoding", "spec_type": "draft-dflash",
                  "draft_model": draft_model, "draft_max": 8})


def test_runtime_attestation_maps_the_ngld_field() -> None:
    """Every speculating role's priors now carry spec.n_gpu_layers_draft; the derived
    attestation checklist reports an unmapped declared field, so it must be mapped."""
    stack_commands = importlib.import_module("stack_commands")
    mode, flags, _ = stack_commands._RUNTIME_FIELD_CHECKS["runtime.flags.spec.n_gpu_layers_draft"]
    assert mode == "int_or_token_flag"
    assert "-ngld" in flags
