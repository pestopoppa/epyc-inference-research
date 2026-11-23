import json

import pytest

from . import stage_receipts as sr


def _fns(calls, stop_after=None, latch=None):
    def make(stage):
        def fn(inputs):
            calls.append(stage)
            if stage == stop_after:
                latch.request("operator stop")
            return {"stage": stage, "n": len(inputs)}
        return fn
    return {s: make(s) for s in sr.STAGES}


def _setup(tmp_path, version=1, deployment="dep-A"):
    store = sr.ReceiptStore(tmp_path / "state")
    latch = sr.StopLatch(tmp_path / "state" / "STOP")
    return store, latch, sr.StageGraph(version, deployment)


def test_stop_latch_finishes_inflight_stage_then_stops(tmp_path):
    store, latch, graph = _setup(tmp_path)
    calls = []
    out = sr.StageRunner(store, graph, latch).run("it1", _fns(calls, "build", latch))
    assert out.status == "stopped"
    assert out.executed == sr.STAGES[:5]
    assert out.stop_reason == "stop latched before correctness"
    assert len(store.all()) == 5


def test_same_graph_restart_never_repeats_completed_stages(tmp_path):
    store, latch, graph = _setup(tmp_path)
    first = []
    sr.StageRunner(store, graph, latch).run("it1", _fns(first, "critic", latch))
    latch.clear()
    second = []
    out = sr.StageRunner(store, graph, latch).run("it1", _fns(second))
    assert out.status == "complete"
    assert set(first).isdisjoint(second)
    assert out.replayed == sr.STAGES[:3]
    assert tuple(second) == sr.STAGES[3:]
    third = []
    again = sr.StageRunner(store, graph, latch).run("it1", _fns(third))
    assert third == [] and again.replayed == sr.STAGES
    assert again.outputs == out.outputs


def test_receipts_are_content_addressed_and_tamper_evident(tmp_path):
    store, latch, graph = _setup(tmp_path)
    sr.StageRunner(store, graph, latch).run("it1", _fns([]))
    name, body = store.all()[0]
    assert sr.digest(body) == name
    path = store.receipts / f"{name}.json"
    body["output"] = {"forged": True}
    path.write_text(json.dumps(body))
    with pytest.raises(sr.ReceiptRefused, match="does not match its address"):
        sr.StageRunner(store, graph, latch).run("it1", _fns([]))


def test_changed_seed_refuses_replay(tmp_path):
    store, latch, graph = _setup(tmp_path)
    sr.StageRunner(store, graph, latch).run("it1", _fns([]), seed={"a": 1})
    with pytest.raises(sr.ReceiptRefused, match="input does not match"):
        sr.StageRunner(store, graph, latch).run("it1", _fns([]), seed={"a": 2})


def test_graph_upgrade_without_carry_forward_reruns_everything(tmp_path):
    store, latch, old = _setup(tmp_path)
    sr.StageRunner(store, old, latch).run("it1", _fns([]))
    calls = []
    out = sr.StageRunner(store, sr.StageGraph(2, "dep-A"), latch).run("it1", _fns(calls))
    assert out.replayed == () and tuple(calls) == sr.STAGES


def test_graph_upgrade_with_versioned_carry_forward(tmp_path):
    store, latch, old = _setup(tmp_path)
    sr.StageRunner(store, old, latch).run("it1", _fns([], "critic", latch))
    latch.clear()
    new = sr.StageGraph(2, "dep-A")
    sr.carry_forward(store, old=old, new=new, iteration="it1",
                     stages=("assignment", "planner"), reason="graph v2 adds gate")
    calls = []
    out = sr.StageRunner(store, new, latch).run("it1", _fns(calls))
    assert out.replayed == ("assignment", "planner")
    assert tuple(calls) == sr.STAGES[2:]


def test_carry_forward_cannot_change_deployment_or_downgrade(tmp_path):
    store, latch, old = _setup(tmp_path)
    sr.StageRunner(store, old, latch).run("it1", _fns([]))
    with pytest.raises(sr.ReceiptRefused, match="deployment identity"):
        sr.carry_forward(store, old=old, new=sr.StageGraph(2, "dep-B"), iteration="it1",
                         stages=("assignment",), reason="x")
    with pytest.raises(sr.ReceiptRefused, match="strictly newer"):
        sr.carry_forward(store, old=old, new=sr.StageGraph(1, "dep-A"), iteration="it1",
                         stages=("assignment",), reason="x")
    with pytest.raises(sr.ReceiptRefused, match="no 'planner' receipt"):
        sr.carry_forward(store, old=old, new=sr.StageGraph(2, "dep-A"), iteration="it2",
                         stages=("planner",), reason="x")


def test_foreign_deployment_receipt_is_refused(tmp_path):
    store, latch, graph = _setup(tmp_path)
    sr.StageRunner(store, graph, latch).run("it1", _fns([]))
    with pytest.raises(sr.ReceiptRefused, match="foreign deployment"):
        sr.StageRunner(store, sr.StageGraph(1, "dep-B"), latch).run("it1", _fns([]))


def test_latch_is_a_stop_requested_callable(tmp_path):
    latch = sr.StopLatch(tmp_path / "STOP")
    assert latch() is False
    latch.request("x")
    assert latch() is True


def test_carry_cannot_import_another_iterations_receipt(tmp_path):
    store, latch, old = _setup(tmp_path)
    sr.StageRunner(store, old, latch).run("other", _fns([]))
    name, _ = next((name, body) for name, body in store.all()
                   if body["stage"] == "assignment")
    new = sr.StageGraph(2, "dep-A")
    store.put_carry({"schema": sr.CARRY_SCHEMA, "from_graph": old.digest,
                     "from_version": 1, "to_graph": new.digest, "to_version": 2,
                     "deployment_identity": "dep-A", "iteration": "it1",
                     "carried": {"assignment": name}, "reason": "test"})
    with pytest.raises(sr.ReceiptRefused, match="mismatched receipt"):
        sr.StageRunner(store, new, latch).run("it1", _fns([]))


def test_receipt_address_cannot_escape_store(tmp_path):
    store, _, _ = _setup(tmp_path)
    with pytest.raises(sr.ReceiptRefused, match="invalid receipt address"):
        store.get("../../outside")


def test_conflicting_completed_stage_receipts_refuse_replay(tmp_path):
    store, latch, graph = _setup(tmp_path)
    sr.StageRunner(store, graph, latch).run("it1", _fns([]))
    body = next(body for _, body in store.all() if body["stage"] == "assignment")
    body["output"] = {"different": True}
    body["output_digest"] = sr.digest(body["output"])
    store.put(body)
    with pytest.raises(sr.ReceiptRefused, match="conflicting receipts"):
        sr.StageRunner(store, graph, latch).run("it1", _fns([]))


def test_content_addressed_receipt_with_false_output_digest_is_refused(tmp_path):
    store, latch, graph = _setup(tmp_path)
    sr.StageRunner(store, graph, latch).run("it1", _fns([]))
    name, body = store.all()[0]
    (store.receipts / f"{name}.json").unlink()
    body["output_digest"] = "0" * 64
    store.put(body)
    with pytest.raises(sr.ReceiptRefused, match="output digest"):
        sr.StageRunner(store, graph, latch).run("it1", _fns([]))


def test_ambiguous_inflight_stage_is_not_reexecuted(tmp_path):
    store, latch, graph = _setup(tmp_path)
    calls = []
    fns = _fns(calls)
    def interrupted(_inputs):
        calls.append('assignment')
        raise RuntimeError('owner exited before receipt')
    fns['assignment'] = interrupted
    with pytest.raises(RuntimeError, match='before receipt'):
        sr.StageRunner(store, graph, latch).run('it1', fns)
    with pytest.raises(sr.ReceiptRefused, match='reconcile its owner'):
        sr.StageRunner(store, graph, latch).run('it1', _fns(calls))
    assert calls == ['assignment']


def test_carry_requires_matching_graph_versions(tmp_path):
    store, latch, old = _setup(tmp_path)
    sr.StageRunner(store, old, latch).run("it1", _fns([]))
    name, body = next((name, body) for name, body in store.all()
                      if body["stage"] == "assignment")
    new = sr.StageGraph(2, "dep-A")
    store.put_carry({"schema": sr.CARRY_SCHEMA, "from_graph": old.digest,
                     "from_version": 1, "to_graph": new.digest, "to_version": 99,
                     "deployment_identity": "dep-A", "iteration": "it1",
                     "carried": {"assignment": name}, "reason": "test"})
    with pytest.raises(sr.ReceiptRefused, match="version or schema"):
        sr.StageRunner(store, new, latch).run("it1", _fns([]))
