"""Offline admission tests: no GPU or server launched."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from . import longctx_c8_comparison as c8, serving


def _anchor():
    return SimpleNamespace(backend="gpu", template=SimpleNamespace(name="c8-long-context", np=8, recipe_hash="r" * 64),
                           execution_digest="a" * 64, build_dir="/build", port=12345)


# r4 correction: the anchor template has np=8 and serving.floor_path requires one frozen request per slot.
REQUESTS = tuple((f"p{i}", b"body") for i in range(8))


def _validated(monkeypatch, anchor):
    frozen = SimpleNamespace(prompts=[SimpleNamespace(prompt_id=pid) for pid, _ in REQUESTS],
                             requests=Mock(return_value=REQUESTS))
    validator = Mock(return_value=frozen)
    monkeypatch.setattr(c8.longctx_c8, "validate_manifest", validator)
    return validator


def test_compare_refuses_missing_floor_before_launch(tmp_path, monkeypatch):
    anchor = _anchor()
    validator = _validated(monkeypatch, anchor)
    monkeypatch.setattr(serving, "load_floor", lambda *a, **kw: serving.FloorReading(None, "absent", tmp_path, {}))
    invoke = Mock()
    monkeypatch.setattr(serving, "compare", invoke)
    with pytest.raises(serving.ServingFloorMismatch, match="own calibrated floor"):
        c8.compare(tmp_path, anchor, anchor, {}, {}, ["text"], lambda _: [], "b" * 64, pairs=5)
    validator.assert_called_once()
    invoke.assert_not_called()


def test_compare_uses_matched_instrument_and_exact_floor(tmp_path, monkeypatch):
    anchor = _anchor()
    _validated(monkeypatch, anchor)
    row = {"baseline_resolved_recipe": {"execution_digest": anchor.execution_digest},
           "unit": serving.CALIBRATION_UNIT, "request_digest": "digest"}
    reading = serving.FloorReading(1.2, "verified", tmp_path, row)
    load = Mock(return_value=reading)
    check = Mock()
    invoke = Mock(return_value={"ok": True})
    monkeypatch.setattr(serving, "load_floor", load)
    monkeypatch.setattr(serving, "_validate_matched_floor", check)
    monkeypatch.setattr(serving, "compare", invoke)
    assert c8.compare(tmp_path, anchor, anchor, {}, {}, [], None, "b" * 64, pairs=5) == {"ok": True}
    assert load.call_args.args[0] == tmp_path / "longctx-c8-floors" / anchor.template.recipe_hash / anchor.execution_digest
    assert invoke.call_args.kwargs["instrument"] == serving.MATCHED_INSTRUMENT
    assert invoke.call_args.kwargs["floor_record"] is row
    assert invoke.call_args.kwargs["frozen_requests"] == REQUESTS
    # This cell sends the complete frozen prompt through ordinary serving; the
    # restored-slot SurfaceLaunch protocol is a different measurement surface.
    assert "longctx" not in invoke.call_args.kwargs
    check.assert_called_once()


def test_wrong_anchor_identity_refuses_before_comparison(tmp_path, monkeypatch):
    anchor = _anchor()
    _validated(monkeypatch, anchor)
    row = {"baseline_resolved_recipe": {"execution_digest": "c" * 64}, "unit": "process"}
    monkeypatch.setattr(serving, "load_floor", lambda *a, **kw: serving.FloorReading(1.2, "verified", tmp_path, row))
    monkeypatch.setattr(serving, "_validate_matched_floor", Mock())
    invoke = Mock()
    monkeypatch.setattr(serving, "compare", invoke)
    with pytest.raises(serving.ServingFloorMismatch, match="anchor execution identity"):
        c8.compare(tmp_path, anchor, anchor, {}, {}, [], None, "b" * 64, pairs=5)
    invoke.assert_not_called()


def test_cpu_or_wrong_concurrency_refused(monkeypatch):
    anchor = _anchor()
    anchor.backend = "cpu"
    validator = _validated(monkeypatch, anchor)
    with pytest.raises(ValueError, match="GPU np=8"):
        c8._validated(anchor, {}, {}, [], None, "b" * 64, 5)
    validator.assert_not_called()


def test_calibration_is_request_bound_and_writes_matched_floor(tmp_path, monkeypatch):
    anchor = _anchor()
    _validated(monkeypatch, anchor)
    floor = {"baseline_resolved_recipe": {"execution_digest": anchor.execution_digest}}
    calibrate = Mock(return_value=floor)
    validate = Mock()
    write = Mock(return_value=tmp_path / "floor.json")
    monkeypatch.setattr(serving, "calibrate_floor", calibrate)
    monkeypatch.setattr(serving, "_validate_matched_floor", validate)
    monkeypatch.setattr(serving, "write_floor", write)
    assert c8.calibrate(tmp_path, anchor, {}, {}, [], None, "b" * 64, pairs=5) == tmp_path / "floor.json"
    assert calibrate.call_args.kwargs["samples"] == serving.MATCHED_CALIBRATION_PAIRS
    assert calibrate.call_args.kwargs["instrument"] == serving.MATCHED_INSTRUMENT
    assert calibrate.call_args.kwargs["frozen_requests"] == REQUESTS
    assert write.call_args.kwargs["unit"] == serving.CALIBRATION_UNIT
    validate.assert_called_once()


def test_calibration_refuses_when_same_floor_is_being_calibrated(tmp_path, monkeypatch):
    anchor = _anchor()
    _validated(monkeypatch, anchor)
    target = tmp_path / "serving-floor.json"
    monkeypatch.setattr(serving, "floor_path", lambda *a, **kw: target)
    calibrate = Mock()
    monkeypatch.setattr(serving, "calibrate_floor", calibrate)

    with c8._exclusive_floor_lock(target.with_name(target.name + ".lock")):
        with pytest.raises(serving.ServingFloorMismatch, match="already in progress"):
            c8.calibrate(tmp_path, anchor, {}, {}, [], None, "b" * 64, pairs=5)

    calibrate.assert_not_called()
    assert not target.exists()
