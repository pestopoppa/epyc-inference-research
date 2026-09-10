"""Original roster pins the source instrument; no claims, launches or file migration."""
import json
from pathlib import Path

import pytest

from . import planned_serving, resolved_recipe, serving, serial_run as sr
from .test_serial_roster import _inputs, _build
from .test_serial_run import CHILD


def _legacy(argv, tmp_path, *, corrupt=False):
    targets, _, _ = _build(argv)
    target = targets[0]
    launch = resolved_recipe.CanonicalResolvedRecipe.from_dict(
        json.loads(Path(sr.option(target, "--cpu-serving-launch")).read_text()))
    prompts = planned_serving.FrozenPromptManifest.from_dict(
        json.loads(Path(sr.option(target, "--frozen-prompts")).read_text()))
    requests = prompts.requests(tuple(row.prompt_id for row in prompts.prompts), launch.template)
    path = serving.write_floor(Path(sr.option(target, "--store")), launch.template,
        {"floor_pct": 7.801, "recipe_hash": launch.template.recipe_hash,
         "request_digest": serving.request_digest(launch.template, requests)}, frozen_requests=requests)
    if corrupt:
        path.write_text("{broken")
    return path


def test_fresh_cpu_selects_v2_gpu_unchanged_without_creating_state(tmp_path):
    _, _, argv = _inputs(tmp_path)
    targets, _, _ = _build(argv)
    cpu = next(row for row in targets if sr.option(row, "--cpu-serving-launch"))
    gpu = next(row for row in targets if sr.option(row, "--gpu-serving-launch"))
    assert sr.option(cpu, "--serving-instrument") == serving.MATCHED_INSTRUMENT
    assert sr.option(gpu, "--serving-instrument") is None
    assert not (tmp_path / "router").exists()


def test_applicable_original_v1_floor_stays_v1_and_original_bytes(tmp_path):
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    path = _legacy(argv, tmp_path)
    original = path.read_bytes()
    targets, _, _ = _build(argv)
    assert sr.option(targets[0], "--serving-instrument") is None
    assert path.read_bytes() == original


def test_invalid_original_v1_floor_does_not_authorize_migration(tmp_path):
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    _legacy(argv, tmp_path, corrupt=True)
    with pytest.raises(json.JSONDecodeError):
        _build(argv)


@pytest.mark.parametrize("original", [None, serving.LEGACY_INSTRUMENT, serving.MATCHED_INSTRUMENT])
def test_original_serial_state_pins_choice_before_a_floor_exists(tmp_path, original):
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    root = Path(sr.option(argv, "--state-dir"))
    root.mkdir()
    body = {"schema": sr.SERIAL_SCHEMA}
    if original is not None:
        body["serving_instruments"] = {"cpu-0": original}
    path = root / "serial-state.json"
    path.write_text(json.dumps(body))
    before = path.read_bytes()
    targets, _, _ = _build(argv)
    assert sr.option(targets[0], "--serving-instrument", serving.LEGACY_INSTRUMENT) == (
        original or serving.LEGACY_INSTRUMENT)
    assert path.read_bytes() == before


def test_explicit_old_calibration_count_is_not_silently_multiplied(tmp_path):
    _, owners, argv = _inputs(tmp_path, backends=("cpu",))
    owners["cpu-0"]["calibrate_serving"] = 5
    Path(sr.option(argv, "--owned-targets")).write_text(json.dumps(owners))
    with pytest.raises(sr.SerialRefused, match="24 A/A pairs"):
        _build(argv)


def test_actual_serial_writer_pins_before_first_child_and_reuses_on_restart(tmp_path, monkeypatch):
    _, _, argv = _inputs(tmp_path, backends=("cpu",))
    child = tmp_path / "tiny.py"
    child.write_text(CHILD)
    root = Path(sr.option(argv, "--state-dir"))
    captured = []
    def command(args):
        state = json.loads((root / "serial-state.json").read_text())
        assert state["serving_instruments"] == {"cpu-0": serving.MATCHED_INSTRUMENT}
        assert sr.option(args, "--serving-instrument") == serving.MATCHED_INSTRUMENT
        captured.append(args)
        return [sr.sys.executable, str(child), *args]
    monkeypatch.setattr(sr, "_child_command", command)
    monkeypatch.setenv("PYTHONPATH", str(Path(sr.__file__).resolve().parents[4]))
    assert sr.main(argv) == 0 and len(captured) == 2
    assert sr.main(argv) == 0 and len(captured) == 2
