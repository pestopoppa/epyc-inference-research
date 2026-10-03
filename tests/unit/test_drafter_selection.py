"""DRAFT-SEL-1: master lists every acceptable drafter, topology selects one per server.

Origin: production :8083 ran Qwen3.8-27B with `--spec-type draft-mtp` although the
operator had ruled DFlash2 its spec-decode path. The drafter was hand-copied per role
and re-inherited across two model swaps. These tests pin the structure that replaces
the copy: one list per model (master), one selection per server (topology), one
writer (the compiler), and a validator over the compiled artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.registry.drafter_selection import (
    DrafterSelectionError,
    NO_SPECULATION,
    check_lean,
    load_drafter_selection,
    resolve,
)
from src.registry.registry_compiler import cache_key, compile_lean

MODEL = "/models/Q38-27B.gguf"
DFLASH = "/models/Q38-27B-DFlash2.gguf"


def _master(*, drafters=None, host_accel=None, alias_accel=None, host_extra=None):
    host = {
        "port": 8083,
        "model_role": "q38_local",
        "model": MODEL,
        "shared_with": ["coder_escalation"],
        "acceleration": dict(host_accel or {"type": "speculative_decoding", "lookup": False}),
        **(host_extra or {}),
    }
    q38 = {"model": {"path": MODEL}}
    if drafters is not None:
        q38["drafters"] = drafters
    return {
        "runtime_defaults": {},
        "server_mode": {
            "architect_critic": host,
            "coder_escalation": {
                "port": 8083,
                "model_role": "q38_local",
                "acceleration": dict(alias_accel or {"type": "none",
                                                     "inherits_spec_from": "architect_critic"}),
            },
        },
        "roles": {
            "architect_critic": {"acceleration": {"type": "speculative_decoding", "p_split": 0}},
            "coder_escalation": {"acceleration": {"type": "none"}},
            "q38_local": q38,
        },
    }


TWO = {
    "mtp": {"spec_type": "draft-mtp", "draft_model": MODEL, "draft_max": 4},
    "dflash2": {"spec_type": "draft-dflash", "draft_model": DFLASH, "draft_max": 8,
                "ngld": 99, "recipe": "artifacts/serving-recipes/x.json"},
}
ACTIVE = {"architect_critic", "coder_escalation"}


def _write(tmp_path: Path, master: dict, selection: dict | None) -> tuple[Path, Path]:
    m = tmp_path / "master.yaml"
    m.write_text(yaml.safe_dump(master))
    t = tmp_path / "stack_topology.yaml"
    t.write_text(yaml.safe_dump({"schema_version": "stack_topology.v1",
                                 **({"drafter_selection": selection} if selection else {})}))
    return m, t


# --- resolution ---------------------------------------------------------------------

def test_topology_selection_is_projected_into_host_roles_and_aliases(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": "dflash2"})
    lean = compile_lean(m, ACTIVE, t)
    host = lean["server_mode"]["architect_critic"]
    assert host["acceleration"]["spec_type"] == "draft-dflash"
    assert host["acceleration"]["draft_model"] == DFLASH
    assert host["acceleration"]["draft_max"] == 8
    assert host["acceleration"]["k"] == 8  # RegistryLoader reads k
    assert host["acceleration"]["n_gpu_layers_draft"] == 99
    assert host["draft_model"] == DFLASH  # becomes requirements.draft_model_path in priors
    assert host["acceleration"]["drafter_selection"]["drafter"] == "dflash2"
    assert lean["roles"]["architect_critic"]["acceleration"]["spec_type"] == "draft-dflash"
    assert lean["roles"]["architect_critic"]["acceleration"]["p_split"] == 0  # non-drafter kept
    alias = lean["server_mode"]["coder_escalation"]["acceleration"]
    assert alias["type"] == "none" and alias["inherits_spec_from"] == "architect_critic"
    assert alias["spec_type"] == "draft-dflash"


def test_switching_the_selection_changes_only_the_projection(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": "mtp"})
    host = compile_lean(m, ACTIVE, t)["server_mode"]["architect_critic"]
    assert host["acceleration"]["spec_type"] == "draft-mtp"
    assert host["acceleration"]["draft_max"] == 4
    assert "n_gpu_layers_draft" not in host["acceleration"]


def test_sole_drafter_needs_no_selection(tmp_path):
    m, t = _write(tmp_path, _master(drafters={"mtp": TWO["mtp"]}), None)
    acc = compile_lean(m, ACTIVE, t)["server_mode"]["architect_critic"]["acceleration"]
    assert acc["spec_type"] == "draft-mtp"
    assert acc["drafter_selection"]["source"] == "sole_drafter"


def test_explicit_none_disables_speculation(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": NO_SPECULATION})
    acc = compile_lean(m, ACTIVE, t)["server_mode"]["architect_critic"]["acceleration"]
    assert acc["type"] == "none" and "spec_type" not in acc


# --- the failure modes ----------------------------------------------------------------

def test_selection_of_unlisted_drafter_fails(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": "eagle3"})
    with pytest.raises(DrafterSelectionError, match="not an acceptable drafter"):
        compile_lean(m, ACTIVE, t)


def test_selection_for_model_without_drafters_fails(tmp_path):
    m, t = _write(tmp_path, _master(drafters=None), {"architect_critic": "dflash2"})
    with pytest.raises(DrafterSelectionError, match="lists no `drafters`"):
        compile_lean(m, ACTIVE, t)


def test_multiple_drafters_without_selection_fails_no_silent_default(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), None)
    with pytest.raises(DrafterSelectionError, match="selects none"):
        compile_lean(m, ACTIVE, t)


@pytest.mark.parametrize("where", ["host", "alias", "host_top_level"])
def test_hand_carried_drafter_field_fails_when_model_declares_drafters(tmp_path, where):
    kwargs = {"drafters": TWO}
    if where == "host":
        kwargs["host_accel"] = {"type": "speculative_decoding", "spec_type": "draft-mtp"}
    elif where == "alias":
        kwargs["alias_accel"] = {"type": "none", "spec_type": "draft-mtp", "draft_max": 4}
    else:
        kwargs["host_extra"] = {"draft_model": MODEL}
    m, t = _write(tmp_path, _master(**kwargs), {"architect_critic": "dflash2"})
    with pytest.raises(DrafterSelectionError, match="hand-carried|hand-carries"):
        compile_lean(m, ACTIVE, t)


def test_selection_keyed_by_alias_fails(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO),
                  {"architect_critic": "dflash2", "coder_escalation": "mtp"})
    with pytest.raises(DrafterSelectionError, match="ALIAS"):
        compile_lean(m, ACTIVE, t)


def test_recipe_missing_required_key_or_unknown_spec_type_fails(tmp_path):
    bad = {"mtp": TWO["mtp"], "dflash2": {"spec_type": "dflash", "draft_max": 8}}
    m, t = _write(tmp_path, _master(drafters=bad), {"architect_critic": "dflash2"})
    with pytest.raises(DrafterSelectionError) as exc:
        compile_lean(m, ACTIVE, t)
    assert "draft_model" in str(exc.value) and "'dflash'" in str(exc.value)


def test_legacy_hand_carried_row_without_drafters_passes_with_warning():
    master = _master(drafters=None,
                     host_accel={"type": "speculative_decoding", "spec_type": "draft-mtp"})
    res = resolve(master, ACTIVE, {})
    assert res.errors == [] and any("LEGACY" in w for w in res.warnings)
    assert res.selected == {}


def test_inactive_selection_is_a_warning_not_an_error():
    res = resolve(_master(drafters=TWO), ACTIVE,
                  {"architect_critic": "dflash2", "retired_role": "mtp"})
    assert res.errors == [] and any("inert" in w for w in res.warnings)


# --- cache key and validator --------------------------------------------------------

def test_cache_key_changes_when_only_the_selection_changes(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": "dflash2"})
    k1 = cache_key(m, ACTIVE, t)
    t.write_text(yaml.safe_dump({"drafter_selection": {"architect_critic": "mtp"}}))
    assert cache_key(m, ACTIVE, t) != k1


def test_check_lean_passes_fresh_projection_and_catches_hand_edit(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": "dflash2"})
    lean = compile_lean(m, ACTIVE, t)
    selection = load_drafter_selection(t)
    assert check_lean(lean, selection) == []
    lean["server_mode"]["architect_critic"]["acceleration"]["spec_type"] = "draft-mtp"
    assert any("differs" in e for e in check_lean(lean, selection))


def test_check_lean_catches_stale_lean_after_topology_change(tmp_path):
    m, t = _write(tmp_path, _master(drafters=TWO), {"architect_critic": "dflash2"})
    lean = compile_lean(m, ACTIVE, t)
    assert any("stale" in e for e in check_lean(lean, {"architect_critic": "mtp"}))


def test_live_27b_is_selected_to_dflash2():
    """Pins the operator's 2026-10-01 rule against the checked-in inputs."""
    selection = load_drafter_selection()
    assert selection.get("architect_critic") == "dflash2"


def test_validate_all_actually_calls_the_drafter_check(monkeypatch, tmp_path):
    from src.registry import registry_validator

    monkeypatch.setattr(registry_validator, "_check_drafter_selection",
                        lambda registry, topology_path=None: ["drafter-sentinel"])
    path = tmp_path / "lean.yaml"
    path.write_text(yaml.safe_dump({"server_mode": {}, "roles": {}}))
    assert "drafter-sentinel" in registry_validator.validate_all(path)


# --- STACKCHG-DFLASH2-20261003: VRAM figure is bound to the drafter it was derived for ---

def test_vram_figure_derived_for_another_drafter_fails():
    master = _master(drafters=TWO,
                     host_extra={"serving_shape": {"vram_non_kv_gib": 32.8,
                                                   "vram_non_kv_drafter": "mtp"}})
    res = resolve(master, ACTIVE, {"architect_critic": "dflash2"})
    assert any("vram_non_kv_drafter" in e and "'mtp'" in e for e in res.errors)


def test_vram_figure_derived_for_the_selected_drafter_passes():
    master = _master(drafters=TWO,
                     host_extra={"serving_shape": {"vram_non_kv_gib": 36.34,
                                                   "vram_non_kv_drafter": "dflash2"}})
    res = resolve(master, ACTIVE, {"architect_critic": "dflash2"})
    assert res.errors == []
    assert res.selected["architect_critic"]["drafter"] == "dflash2"


def test_vram_drafter_key_is_optional():
    master = _master(drafters=TWO, host_extra={"serving_shape": {"vram_non_kv_gib": 36.34}})
    assert resolve(master, ACTIVE, {"architect_critic": "dflash2"}).errors == []
