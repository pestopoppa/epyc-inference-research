"""INF-62 SL-2: estimated target steps/s is emitted BESIDE `aggregate_tok_s`, never instead of it.

Nothing here launches a server: the process, HTTP and residency sampler are stubbed and
every response is synthetic. The tok/s value, the verdict and the floor must be identical
to what they were before SL-2 -- steps/s is a reporting field until a run boundary moves
the keep gate onto it.
"""
import json
from pathlib import Path
from unittest import mock

import pytest

from autokernel.loop import serving
from autokernel.loop.test_serving_residency import _proof, _record, _sampler_class

RECIPE = serving.Recipe(name="sl2", model="/m/t.gguf", np=2, n_predict=8)


def _timings(predicted_n, predicted_ms, **extra):
    return {"predicted_n": predicted_n, "predicted_ms": predicted_ms,
            "predicted_per_second": predicted_n * 1000.0 / predicted_ms if predicted_ms else 0.0,
            **extra}


# --------------------------------------------------------------------------- slot


def test_slot_derives_target_steps_from_accepted_drafts():
    slot = serving._target_steps_slot(_timings(100, 2000.0, draft_n=90, draft_n_accepted=60))
    assert slot["status"] == "ok"
    assert slot["estimator"] == serving.TARGET_STEPS_ESTIMATOR_TARGET_STEPS
    assert slot["target_sample_steps_est"] == 40
    assert slot["target_sample_steps_per_second_est"] == pytest.approx(20.0)
    assert slot["tokens_per_step"] == pytest.approx(2.5)
    # The identity SL-2 rests on: tok/s = steps/s x tokens/step, exactly, per slot.
    assert slot["target_sample_steps_per_second_est"] * slot["tokens_per_step"] == pytest.approx(50.0)


def test_slot_without_draft_counters_is_plain_decode():
    slot = serving._target_steps_slot(_timings(64, 1000.0))
    assert slot["estimator"] == serving.TARGET_STEPS_ESTIMATOR_NO_SPEC
    assert slot["target_sample_steps_est"] == 64 and slot["tokens_per_step"] == 1.0


def test_slot_prefers_a_server_exported_counter():
    slot = serving._target_steps_slot(_timings(100, 2000.0, draft_n=90, draft_n_accepted=60,
                                           draft_verif_steps=39))
    assert slot["estimator"] == serving.TARGET_STEPS_ESTIMATOR_SERVER
    assert slot["target_sample_steps_est"] == 39


@pytest.mark.parametrize("timings", [
    None, [], {"predicted_n": 10},
    _timings(10, 0.0),
    {"predicted_n": True, "predicted_ms": 5.0},
    _timings(10, 100.0, draft_n=5),                              # one counter missing
    _timings(10, 100.0, draft_n=5, draft_n_accepted=6),          # accepted > drafted
    _timings(10, 100.0, draft_n=20, draft_n_accepted=11),        # accepted > decoded
    _timings(10, 100.0, draft_n=20, draft_n_accepted=10),        # zero steps
    _timings(10, 100.0, draft_n=5, draft_n_accepted=-1),
    {"predicted_n": 10, "predicted_ms": float("nan")},
])
def test_slot_never_raises_and_never_invents_a_count(timings):
    slot = serving._target_steps_slot(timings)
    assert slot["status"] == "unavailable" and slot["reason"]
    assert "target_sample_steps_est" not in slot


# --------------------------------------------------------------------------- launch


def test_aggregate_is_the_sum_over_slots_and_refuses_partial():
    ok = [serving._target_steps_slot(_timings(100, 2000.0, draft_n=90, draft_n_accepted=60)),
          serving._target_steps_slot(_timings(50, 1000.0, draft_n=40, draft_n_accepted=30))]
    agg = serving._target_steps_aggregate(ok, 100.0)
    assert agg["status"] == "ok" and agg["value"] == pytest.approx(40.0)
    assert agg["metric"] == "aggregate_target_sample_steps_s_est"
    assert agg["role"] == "reporting_only_not_gate"
    assert agg["tokens_per_step"] == pytest.approx(2.5)
    partial = serving._target_steps_aggregate([ok[0], {"status": "unavailable", "reason": "x"}], 50.0)
    assert partial["status"] == "unavailable" and partial["value"] is None
    assert serving._target_steps_aggregate([], 0.0)["value"] is None
    mixed = serving._target_steps_aggregate([ok[0], serving._target_steps_slot(_timings(8, 100.0))], 130.0)
    assert mixed["estimator"].startswith("mixed:")


def _launch(bodies, *, steps):
    """Drive the REAL `_measure_once` with a fake server returning `bodies` in order."""
    responses = iter(bodies)

    class _Proc:
        pid = 4321
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, _timeout=None):
            return 0

    class _Resp:
        def __init__(self, body):
            self.body = body

        def read(self, *_a):
            return json.dumps(self.body).encode()

    def _urlopen(req, timeout=None, **_kw):
        url = req if isinstance(req, str) else req.full_url
        return _Resp({} if url.endswith("/health") else next(responses))

    with mock.patch.object(serving.subprocess, "Popen", lambda argv, **kw: _Proc()), \
         mock.patch.object(serving.urllib.request, "urlopen", _urlopen), \
         mock.patch.object(serving.residency, "Sampler", _sampler_class(_proof())):
        return serving._measure_once(RECIPE, Path("/B"), 18311, evidence=[],
                                     target_sample_steps=steps)


def test_real_launch_emits_steps_beside_an_unchanged_tok_s():
    warm = {"timings": _timings(8, 400.0, draft_n=6, draft_n_accepted=4), "stop": True}
    measured = [{"timings": _timings(8, 200.0, draft_n=6, draft_n_accepted=4), "stop": True},
                {"timings": _timings(8, 100.0, draft_n=8, draft_n_accepted=6), "stop": True}]
    steps: list = []
    value = _launch([warm, warm, *measured], steps=steps)
    # tok/s is exactly what it was before SL-2: the sum of per-slot predicted_per_second.
    assert value == pytest.approx(40.0 + 80.0)
    assert len(steps) == 1
    (row,) = steps
    assert row["schema"] == serving.TARGET_STEPS_SCHEMA
    assert row["aggregate_tok_s"] == value
    # slot 0: 4 steps / 0.2 s = 20; slot 1: 2 steps / 0.1 s = 20. Warmup is excluded.
    assert row["value"] == pytest.approx(40.0)
    assert [s["target_sample_steps_est"] for s in row["slots"]] == [4, 2]
    assert row["tokens_per_step"] == pytest.approx(3.0)


def test_real_launch_without_a_sink_is_unchanged():
    body = {"timings": _timings(8, 100.0), "stop": True}
    assert _launch([body] * 4, steps=None) == pytest.approx(160.0)


def test_failed_launch_appends_nothing_so_the_sink_stays_aligned():
    bad = {"timings": _timings(2, 100.0), "stop": True}  # degenerate: < n_predict // 2
    steps: list = []
    with pytest.raises(serving.ServerDied):
        _launch([bad] * 4, steps=steps)
    assert steps == []


def test_request_rows_and_observation_shape_stay_closed():
    """Native capture seals these shapes by exact key set; SL-2 must not widen them."""
    observed: list = []
    body = {"timings": _timings(8, 100.0, draft_n=4, draft_n_accepted=2), "stop": True}
    with mock.patch.object(serving.subprocess, "Popen", lambda argv, **kw: type(
            "P", (), {"pid": 1, "returncode": 0, "poll": lambda s: None,
                      "terminate": lambda s: None, "wait": lambda s, t=None: 0})()), \
         mock.patch.object(serving.urllib.request, "urlopen", lambda req, timeout=None, **k:
                           type("R", (), {"read": lambda s, *a: json.dumps(body).encode()})()), \
         mock.patch.object(serving.residency, "Sampler", _sampler_class(_proof())):
        serving._measure_once(RECIPE, Path("/B"), 18311, observation=observed,
                              target_sample_steps=[])
    (exported,) = observed
    assert set(exported) == {"schema", "process_pid", "requests", "residency", "teardown",
                             "failure"}
    assert all(set(row) == {"phase", "slot_index", "prompt_id", "request_sha256",
                            "predicted_n", "predicted_per_second", "terminal", "error"}
               for row in exported["requests"])


# --------------------------------------------------------------------------- A/B + floor


def _stub(values, steps_values):
    """A `_measure_once` stand-in that feeds both sinks the way the real one does."""
    it, st = iter(values), iter(steps_values)

    def _measure(recipe, build_dir, port, boot_timeout_s=360, *, evidence=None,
                 target_sample_steps=None, **_kw):
        value = next(it)
        s = next(st)
        if evidence is not None:
            evidence.append(_record(_proof()))
        if target_sample_steps is not None:
            target_sample_steps.append({"schema": serving.TARGET_STEPS_SCHEMA,
                                   "metric": serving.TARGET_STEPS_METRIC,
                                   "status": "ok" if s else "unavailable", "value": s,
                                   "estimator": serving.TARGET_STEPS_ESTIMATOR_TARGET_STEPS,
                                   "tokens_per_step": value / s if s else None,
                                   "aggregate_tok_s": value, "slots": []})
        return value

    return _measure


def test_compare_reports_steps_effect_and_leaves_the_verdict_alone():
    # anchor, candidate alternating: tok/s +10%, steps/s +20% (acceptance fell).
    tok = [100.0, 110.0, 100.0, 110.0]
    stp = [40.0, 48.0, 40.0, 48.0]
    with mock.patch.object(serving, "_measure_once", side_effect=_stub(tok, stp)):
        row = serving.compare(RECIPE, Path("/A"), Path("/C"), pairs=2, floor_pct=5.0,
                              floor_unit=serving.COMPARE_EFFECT_UNIT)
    assert row["effect_pct"] == pytest.approx(10.0) and row["decisive"] is True
    vs = row["target_sample_steps_est"]
    assert vs["role"] == "reporting_only_not_gate"
    assert vs["anchor"]["median"] == 40.0 and vs["candidate"]["median"] == 48.0
    assert vs["effect_pct"] == pytest.approx(20.0)
    assert vs["anchor"]["median_tokens_per_step"] == pytest.approx(2.5)
    assert len(vs["anchor_launches"]) == len(row["anchor_samples"]) == 2


def test_compare_with_an_unavailable_launch_reports_no_steps_effect():
    with mock.patch.object(serving, "_measure_once",
                           side_effect=_stub([100.0] * 4, [40.0, None, 40.0, 41.0])):
        row = serving.compare(RECIPE, Path("/A"), Path("/C"), pairs=2, floor_pct=None)
    assert row["target_sample_steps_est"]["candidate"]["status"] == "unavailable"
    assert row["target_sample_steps_est"]["effect"] is None
    assert row["effect"] == 0.0


def test_calibrate_floor_recalibrates_both_metrics_from_the_same_launches():
    tok = [100.0, 101.0, 99.0, 100.0, 104.0]
    stp = [40.0, 40.0, 40.0, 40.0, 44.0]
    with mock.patch.object(serving, "_measure_once", side_effect=_stub(tok, stp)):
        row = serving.calibrate_floor(RECIPE, Path("/B"), samples=5)
    assert row["floor_pct"] == serving._spread(tok)["p95_dev_pct"]
    vs = row["target_sample_steps_est"]
    assert vs["floor_pct"] == pytest.approx(10.0)
    assert vs["samples"] == stp and vs["n"] == 5
    assert vs["floor_ci"]["use"] == "descriptive_only_not_gate_endpoint"
    # Written through the one writer untouched, next to the tok/s floor.
    import tempfile
    with tempfile.TemporaryDirectory() as store:
        path = serving.write_floor(store, RECIPE, row, unit=serving.CALIBRATION_UNIT)
        disk = json.loads(Path(path).read_text())
    assert disk["target_sample_steps_est"]["floor_pct"] == vs["floor_pct"]
    assert disk["floor_pct"] == row["floor_pct"]
