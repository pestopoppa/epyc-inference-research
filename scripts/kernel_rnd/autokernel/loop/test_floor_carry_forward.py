"""DS41-C69: carried-forward lineage serving floor (operator directive 2026-09-28).

Synthetic rates only: `serving._measure_once` is replaced, no server is launched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import time

import pytest

from ..controller import experiments
from . import anchor, claim, run, serving
from .test_resolved_recipe import BUILD, _artifacts, _resolve

MODE = {"instrument": serving.MATCHED_INSTRUMENT, "pairs": 5}
TIP = "a1faab471e83be12398fc4da57a4c7a7e3f75d77"
OLD = "00d118d44b0000000000000000000000000000aa"
REQUESTS = (("p", b'{"prompt":[1],"n_predict":8}'),)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")
    calls = []

    def measure(recipe, build, port, *, evidence, **kwargs):
        calls.append(str(build))
        now = time.time()
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "window_start": now,
                         "window_end": now, "samples": 0})
        return 10 + (len(calls) % 7) / 100
    monkeypatch.setattr(serving, "_measure_once", measure)
    return calls


def _recipe(name="matched", np=1):
    return serving.Recipe(name=name, model="/fixture-model", device="none", ngl=0, np=np)


def _launch(recipe, build):
    """A resolved CPU launch whose executable/DSO digests are distinct per build: a
    source treatment moves the execution identity and nothing else in the frame."""
    import hashlib
    artifacts = _artifacts(recipe, build=Path(build))
    marker = hashlib.sha256(str(build).encode()).hexdigest()
    artifacts["executable"]["sha256"] = marker
    artifacts["dsos"][0]["sha256"] = marker
    return _resolve(recipe, backend="cpu", build=Path(build), artifacts=artifacts)


def _calibrated(store, recipe, build):
    """Calibrate and file one immutable floor under `build`'s execution identity."""
    launch = _launch(recipe, build)
    row = serving.calibrate_floor(recipe, Path(build), samples=24, resolved_recipe=launch,
                                  frozen_requests=REQUESTS, **MODE)
    floor_store = run._source_floor_store(store, recipe, launch,
                                          instrument=serving.MATCHED_INSTRUMENT)
    path = run._write_new_source_floor(floor_store, recipe, launch, row,
                                       frozen_requests=REQUESTS,
                                       instrument=serving.MATCHED_INSTRUMENT, pairs=5)
    return launch, json.loads(Path(path).read_text()), Path(path)


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _aa(store, recipe, *, effect_pct, recorded_at, commit=TIP, excursion=False,
        passed=True, recipe_hash=None):
    """File one anchor-guard verdict exactly as `run.verify_anchor` does."""
    comparison = {"schema": "epyc.autokernel.serving_ab.v2", "pairs": 5,
                  "recipe_hash": recipe_hash or recipe.recipe_hash,
                  "request_digest": serving.request_digest(recipe, REQUESTS),
                  "measurement_plan": serving._matched_plan(5), "effect_pct": effect_pct}
    verdict = anchor.AnchorVerdict(
        passed=passed, champion_commit=commit, anchor="/store/anchor-gen-004",
        effect_pct=effect_pct, noise_floor_pct=5.525,
        surface="serving:" + recipe.name, pairs=5,
        detail=f"fixture A/A {effect_pct:+.3f}% at {recorded_at}", excursion=excursion,
        evidence={"comparison": comparison})
    with experiments.ExperimentStore(store) as memory:
        assert memory.record(verdict.to_attempt(), epoch="fixture-epoch",
                             recorded_at=recorded_at, campaign_id="ak-loop")


def _select(store, recipe, launch, **kwargs):
    kwargs.setdefault("anchor_commits", (TIP,))
    return run._select_source_floor(store, recipe, launch, frozen_requests=REQUESTS,
                                    instrument=serving.MATCHED_INSTRUMENT, pairs=5, **kwargs)


def _later(row, seconds=60):
    return _iso(row["residency"]["window_end"] + seconds)


def test_carry_admitted_on_same_recipe_with_fresh_in_floor_aa_and_compare_admits_it(tmp_path):
    recipe = _recipe()
    _parent, row, parent_path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=-row["floor_pct"] / 2, recorded_at=_later(row))
    tip = _launch(recipe, BUILD)
    floor_store, reading, carry = _select(tmp_path, recipe, tip)

    assert reading.provenance == run.FLOOR_CARRY_PROVENANCE == "carried_forward"
    assert reading.row == row and reading.floor_pct == row["floor_pct"]
    assert reading.path == parent_path
    assert run._gate_floor(reading) == (row["floor_pct"], serving.UNIT_PROCESS)
    assert carry["schema"] == run.FLOOR_CARRY_SCHEMA
    assert carry["parent_floor_path"] == str(parent_path)
    assert carry["parent_content_sha256"] == row["content_sha256"]
    assert carry["parent_execution_digest"] == _parent.execution_digest
    import hashlib
    assert carry["parent_floor_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert carry["aa_status"] == "anchor_verified" and carry["aa_attempt_id"]
    assert carry["anchor_execution_digest"] == tip.execution_digest
    # Read-only: nothing is filed under the tip's own identity directory.
    assert floor_store == run._source_floor_store(tmp_path, recipe, tip,
                                                  instrument=serving.MATCHED_INSTRUMENT)
    assert not floor_store.exists()

    # serving.compare admits the carried floor under the same frame revalidation it
    # applies to the COR fallback: the tip arm as anchor, a new source build as candidate.
    out = serving.compare(recipe, BUILD, Path("/candidate"), pairs=5,
                          floor_pct=reading.floor_pct, floor_unit=reading.unit,
                          floor_record=reading.row, instrument=serving.MATCHED_INSTRUMENT,
                          anchor_resolved_recipe=tip,
                          candidate_resolved_recipe=_launch(recipe, "/candidate"),
                          frozen_requests=REQUESTS,
                          floor_request_digest=reading.request_digest)
    assert out["floor_sha256"] == row["content_sha256"]
    assert out["decisive"] == (abs(out["effect"]) * 100 >= row["floor_pct"])


def test_aa_outside_the_carried_floor_counts_as_an_excursion(tmp_path):
    """DS41-C79: an above-floor row is ONE excursion whatever its guard status -- a lone
    one admits the carry, a second refuses it."""
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=row["floor_pct"] + 0.5, recorded_at=_later(row))
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.provenance == run.FLOOR_CARRY_PROVENANCE and carry["aa_excursions"] == 1
    _aa(tmp_path, recipe, effect_pct=-(row["floor_pct"] + 0.5), recorded_at=_later(row, 90))
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.floor_pct is None and reading.provenance == "absent" and carry is None


def test_single_excursion_is_admitted(tmp_path):
    """DS41-C79, operator 2026-09-30: one excursion is a single sample (the live
    2026-09-29 case: a hash-proven -2.145% A/A against a 1.876% floor)."""
    recipe = _recipe()
    _parent, row, parent_path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=-(row["floor_pct"] + 0.3), recorded_at=_later(row),
        excursion=True)
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.provenance == run.FLOOR_CARRY_PROVENANCE
    assert reading.path == parent_path and reading.floor_pct == row["floor_pct"]
    assert carry["aa_status"] == "anchor_guard_excursion"
    assert carry["aa_rows_considered"] == 1 and carry["aa_excursions"] == 1


def test_two_excursions_refuse(tmp_path):
    """DS41-C79: two above-floor A/As on the current anchor persist -- refuse."""
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=row["floor_pct"] + 1, recorded_at=_later(row, 60),
        excursion=True)
    _aa(tmp_path, recipe, effect_pct=-(row["floor_pct"] + 1), recorded_at=_later(row, 120),
        excursion=True)
    carried, reason = run._carry_forward_floor(
        tmp_path, recipe, _launch(recipe, BUILD), frozen_requests=REQUESTS,
        instrument=serving.MATCHED_INSTRUMENT, pairs=5, anchor_commits=(TIP,))
    assert carried is None and "two excursions refuse" in reason and "DS41-C79" in reason
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.floor_pct is None and reading.provenance == "absent" and carry is None


def test_one_excursion_among_in_floor_rows_is_admitted(tmp_path):
    recipe = _recipe()
    _parent, row, parent_path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row, 60))
    _aa(tmp_path, recipe, effect_pct=row["floor_pct"] + 1, recorded_at=_later(row, 120),
        excursion=True)
    _aa(tmp_path, recipe, effect_pct=-row["floor_pct"] / 2, recorded_at=_later(row, 180))
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.provenance == run.FLOOR_CARRY_PROVENANCE and reading.path == parent_path
    assert carry["aa_rows_considered"] == 3 and carry["aa_excursions"] == 1
    # The newest row is still the one the carry record names.
    assert carry["aa_status"] == "anchor_verified"
    assert carry["aa_effect_pct"] == pytest.approx(-row["floor_pct"] / 2)


def test_a_failed_guard_row_still_refuses_outright(tmp_path):
    """C79 relaxes the noise rule only: an `anchor_mismatch` (the run-ending verdict)
    is no usable A/A sample and refuses on its own, even inside the floor."""
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row, 60))
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row, 120), passed=False)
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.floor_pct is None and carry is None


def test_refused_when_no_aa_is_newer_than_the_floor(tmp_path):
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row, -3600))
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.floor_pct is None and carry is None
    # ... and no experiment store at all is also no evidence.
    (tmp_path / "experiments.db").unlink()
    for side in ("experiments.db-wal", "experiments.db-shm"):
        (tmp_path / side).unlink(missing_ok=True)
    assert _select(tmp_path, recipe, _launch(recipe, BUILD))[1].floor_pct is None


def test_refused_when_the_aa_is_for_another_commit_or_another_recipe(tmp_path):
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row), commit=OLD)
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row, 90),
        recipe_hash="f" * 64)
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.floor_pct is None and carry is None


def test_refused_across_a_runtime_recipe_hash_change(tmp_path):
    before = _recipe()
    after = serving.Recipe(name="matched", model="/fixture-model", device="none", ngl=0,
                           np=1, threads=4)  # an adopted runtime recipe (cpu-half)
    assert after.recipe_hash != before.recipe_hash
    _parent, row, parent_path = _calibrated(tmp_path, before, "/parent")
    _aa(tmp_path, before, effect_pct=0.0, recorded_at=_later(row))
    _aa(tmp_path, after, effect_pct=0.0, recorded_at=_later(row, 90))
    tip_after = _launch(after, BUILD)
    # No lineage under the new recipe hash: nothing to carry.
    _store, reading, carry = _select(tmp_path, after, tip_after)
    assert reading.floor_pct is None and carry is None
    # A pre-adoption floor copied under the new hash is still refused on its sealed identity.
    smuggled = (tmp_path / "runtime-source-floors" / after.recipe_hash
                / parent_path.parent.name / serving.floor_path(
                    Path(), after, frozen_requests=REQUESTS, **MODE).name)
    smuggled.parent.mkdir(parents=True)
    smuggled.write_bytes(parent_path.read_bytes())
    _store, reading, carry = _select(tmp_path, after, tip_after)
    assert reading.floor_pct is None and carry is None


def test_refused_for_a_non_matched_instrument(tmp_path):
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row))
    tip = _launch(recipe, BUILD)
    carried, reason = run._carry_forward_floor(
        tmp_path, recipe, tip, frozen_requests=REQUESTS,
        instrument=serving.LEGACY_INSTRUMENT, pairs=5, anchor_commits=(TIP,))
    assert carried is None and "matched instrument" in reason
    _store, reading, carry = run._select_source_floor(
        tmp_path, recipe, tip, frozen_requests=REQUESTS,
        instrument=serving.LEGACY_INSTRUMENT, pairs=5, anchor_commits=(TIP,))
    assert reading.floor_pct is None and carry is None


def test_only_the_newest_lineage_floor_is_considered(tmp_path):
    recipe = _recipe()
    _older, older_row, _p1 = _calibrated(tmp_path, recipe, "/older")
    _newer, newer_row, newer_path = _calibrated(tmp_path, recipe, "/newer")
    assert newer_row["residency"]["window_end"] >= older_row["residency"]["window_end"]
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(newer_row))
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD))
    assert reading.path == newer_path and carry["parent_floor_path"] == str(newer_path)


def test_ordinary_exact_floor_path_is_unchanged(tmp_path):
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row))
    tip, exact_row, exact_path = _calibrated(tmp_path, recipe, BUILD)
    _store, reading, carry = _select(tmp_path, recipe, tip)
    assert reading.provenance == "verified" and carry is None
    assert reading.path == exact_path and reading.row == exact_row
    assert run._load_source_floor(tmp_path, recipe, tip, frozen_requests=REQUESTS,
                                  instrument=serving.MATCHED_INSTRUMENT,
                                  pairs=5)[1].row == reading.row


def test_cor_floor_still_precedes_the_carry(tmp_path, monkeypatch):
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row))
    cor, cor_row, cor_path = _calibrated(tmp_path, recipe, "/cor")
    monkeypatch.setattr(run, "_cpu_arm", lambda _launch, build: cor)
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD),
                                     cor_build=Path("/cor"))
    assert reading.provenance == "verified" and carry is None and reading.path == cor_path


def test_flag_disables_the_carry(tmp_path):
    recipe = _recipe()
    _parent, row, _path = _calibrated(tmp_path, recipe, "/parent")
    _aa(tmp_path, recipe, effect_pct=0.0, recorded_at=_later(row))
    _store, reading, carry = _select(tmp_path, recipe, _launch(recipe, BUILD),
                                     carry_forward=False)
    assert reading.floor_pct is None and reading.provenance == "absent" and carry is None
    # Both floor-selection sites in the loop honour the CLI escape hatch.
    source = inspect.getsource(run.main)
    assert source.count("carry_forward=args.floor_carry_forward") == 2


def test_cli_exposes_the_escape_hatch(capsys):
    with pytest.raises(SystemExit):
        run.main(["--help"])
    assert "--no-floor-carry-forward" in capsys.readouterr().out
