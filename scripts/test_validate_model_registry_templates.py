"""SEC-ART-5: validate_model_registry.py requires template provenance for deployable GGUF-serving roles."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_P = Path(__file__).resolve().parent / "validate_model_registry.py"
_S = importlib.util.spec_from_file_location("vmr", _P)
vmr = importlib.util.module_from_spec(_S)
_S.loader.exec_module(vmr)

H = "a" * 64


def _data(block, *, alias=False, model="/m/x.gguf"):
    cfg = {"model": model, "chat_template": block}
    if alias:
        cfg["alias_of"] = "other"
    return {"server_mode": {"frontdoor": cfg}}


def _good(tmp_path):
    (tmp_path / "d.diff").write_text("-a\n+b\n")
    return {"upstream_template_sha256": H, "served_template_sha256": "b" * 64, "template_diff": "d.diff"}


def test_complete_provenance_passes(tmp_path):
    assert vmr.check_template_provenance(_data(_good(tmp_path)), {"frontdoor"}, tmp_path) == []


def test_each_missing_field_is_an_error_naming_the_role(tmp_path):
    for field in vmr.TEMPLATE_FIELDS:
        block = _good(tmp_path)
        del block[field]
        errs = vmr.check_template_provenance(_data(block), {"frontdoor"}, tmp_path)
        assert errs and errs[0][0] == "MISSING_TEMPLATE_PROVENANCE" and field in errs[0][1] and "frontdoor" in errs[0][1]


def test_non_deployable_alias_and_non_gguf_roles_are_exempt(tmp_path):
    assert vmr.check_template_provenance(_data({}), set(), tmp_path) == []
    assert vmr.check_template_provenance(_data({}, alias=True), {"frontdoor"}, tmp_path) == []
    assert vmr.check_template_provenance(_data({}, model="large-v3-turbo"), {"frontdoor"}, tmp_path) == []


def test_bad_shape_prefix_mismatch_and_missing_or_empty_diff(tmp_path):
    block = _good(tmp_path)
    block["served_template_sha256"] = "XYZ"
    assert vmr.check_template_provenance(_data(block), {"frontdoor"}, tmp_path)[0][0] == "BAD_TEMPLATE_PIN"
    block = _good(tmp_path)
    block["served_template_sha256_12"] = "deadbeefcafe"
    assert vmr.check_template_provenance(_data(block), {"frontdoor"}, tmp_path)[0][0] == "TEMPLATE_PIN_PREFIX_MISMATCH"
    block = _good(tmp_path)
    block["template_diff"] = "gone.diff"
    assert vmr.check_template_provenance(_data(block), {"frontdoor"}, tmp_path)[0][0] == "TEMPLATE_DIFF_MISSING"
    (tmp_path / "empty.diff").write_text("")
    block["template_diff"] = "empty.diff"
    assert vmr.check_template_provenance(_data(block), {"frontdoor"}, tmp_path)[0][0] == "TEMPLATE_DIFF_MISSING"


def test_this_checkouts_master_registry_lineup_is_covered():
    path = str(Path(__file__).resolve().parent.parent / "orchestration" / "model_registry.yaml")
    reg = vmr.ModelRegistry(path).data
    errors, _ = vmr.validate(path)
    assert not [e for e in errors if e[0] in ("MISSING_TEMPLATE_PROVENANCE", "BAD_TEMPLATE_PIN", "TEMPLATE_DIFF_MISSING")]
    assert reg["server_mode"]["frontdoor"]["chat_template"]["served_template_sha256"]
