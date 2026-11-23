"""AK-TRIM-GATE-PLUMBING golden: every entry path ends up with the SAME gate chain.

Each case drives a real path (`loop.iterate`, `pipeline.run_pool`, `pool.drive`) with
fake gate features and records the ordered events of the gate the loop finally calls.
The expected traces were recorded from research main BEFORE the single `GateWrappers`
list replaced the per-feature kwargs; they must not change.
"""
from __future__ import annotations

import pytest

from . import exact_lane as el
from . import loop as _loop
from . import pipeline as _pipeline
from . import pool as _pool
from .gate_wrappers import GateWrappers

EV = []


def _hyp(lane=None):
    kw = {"lane": lane} if lane else {}
    return _loop.Hypothesis(mechanism_id="m", statement="s", falsifier="f",
                            target_surface="cpu", target_symbol="x", **kw)


def _base_gate(h, p):
    EV.append("gate")
    return True, []


def _make_exact(worker):
    EV.append(f"make_exact:{worker.name}")

    def judge(h, p):
        EV.append("judge")
        return el.ExactVerdict("pass", "ok")
    return judge


def _make_isa(worker):
    EV.append(f"make_isa:{worker.name}")

    def check(h, p):
        EV.append("isa")
        return None
    return check


def _demote(h):
    return True


def _feature_kwargs(exact, isa, demote=None):
    return _WRAP(_make_exact if exact else None, _make_isa if isa else None, demote)


def _WRAP(make_exact, make_isa, demote):
    return {"gate_wrappers": GateWrappers(make_exact, make_isa, demote)}


def _trace_from_gate(gate):
    out = {}
    for lane in (None, "exact"):
        del EV[:]
        result = gate(_hyp(lane), ())
        out[str(lane)] = (list(EV), result[0], [v.gate for v in result[1]])
    return out


def _capture(monkeypatch):
    seen = []

    def fake(**kw):
        seen.append(kw["gate"])
        raise _loop.RunAborted("stop")
    monkeypatch.setattr(_pipeline.loop_mod, "_iterate", fake)
    return seen


def _worker(tmp_path):
    return _pipeline.Worker("w0", tmp_path / "wt", tmp_path / "build")


def _pool_kwargs(tmp_path):
    return dict(
        workers=[_worker(tmp_path)], make_planner=lambda w: None, make_critic=lambda w: None,
        build_context=dict, make_gate=lambda w: _base_gate,
        make_measure=lambda w: (lambda h, p: None), record=lambda o: None,
        record_abandoned=lambda w, c: None, iterations=1)


COMBOS = [(False, False), (True, False), (False, True), (True, True)]


def _run_pipeline(monkeypatch, tmp_path, exact, isa):
    seen = _capture(monkeypatch)
    del EV[:]
    try:
        _pipeline.run_pool(commit=lambda *a: "c", champion_head=lambda: "h",
                           reset_to_champion=lambda w: "h",
                           **_pool_kwargs(tmp_path), **_feature_kwargs(exact, isa))
    except Exception:
        pass
    factories = list(EV)
    assert len(seen) == 1
    return {"factories": factories, **_trace_from_gate(seen[0])}


def _run_drive(monkeypatch, tmp_path, exact, isa):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(_pool, "champion_head", lambda *a, **k: "h")
    monkeypatch.setattr(_pool, "reset_to_champion", lambda *a, **k: "h")
    tails = []
    del EV[:]
    try:
        _pool.drive(commit=lambda *a: "c", on_tail_ready=tails.append,
                    **_pool_kwargs(tmp_path), **_feature_kwargs(exact, isa, _demote))
    except Exception:
        pass
    factories = list(EV)
    assert len(seen) == 1
    return {"factories": factories, "tail_demote": tails[0]._demote is _demote,
            **_trace_from_gate(seen[0])}


GOLDEN = {
    ('pipeline', False, False): {'factories': [], 'None': (['gate'], True, []), 'exact': ([], False, ['exact_token_identity'])},
    ('pipeline', True, False): {'factories': ['make_exact:w0'], 'None': (['gate'], True, []), 'exact': (['judge'], True, ['exact_token_identity'])},
    ('pipeline', False, True): {'factories': ['make_isa:w0'], 'None': (['gate', 'isa'], True, []), 'exact': ([], False, ['exact_token_identity'])},
    ('pipeline', True, True): {'factories': ['make_exact:w0', 'make_isa:w0'], 'None': (['gate', 'isa'], True, []), 'exact': (['judge', 'isa'], True, ['exact_token_identity'])},
    ('pool', False, False): {'factories': [], 'tail_demote': True, 'None': (['gate'], True, []), 'exact': ([], False, ['exact_token_identity'])},
    ('pool', True, False): {'factories': ['make_exact:w0'], 'tail_demote': True, 'None': (['gate'], True, []), 'exact': (['judge'], True, ['exact_token_identity'])},
    ('pool', False, True): {'factories': ['make_isa:w0'], 'tail_demote': True, 'None': (['gate', 'isa'], True, []), 'exact': ([], False, ['exact_token_identity'])},
    ('pool', True, True): {'factories': ['make_exact:w0', 'make_isa:w0'], 'tail_demote': True, 'None': (['gate', 'isa'], True, []), 'exact': (['judge', 'isa'], True, ['exact_token_identity'])},
}


@pytest.mark.parametrize("exact,isa", COMBOS)
def test_run_pool_gate_chain(monkeypatch, tmp_path, exact, isa):
    got = _run_pipeline(monkeypatch, tmp_path, exact, isa)
    assert got == GOLDEN[("pipeline", exact, isa)], got


@pytest.mark.parametrize("exact,isa", COMBOS)
def test_pool_drive_gate_chain(monkeypatch, tmp_path, exact, isa):
    got = _run_drive(monkeypatch, tmp_path, exact, isa)
    assert got == GOLDEN[("pool", exact, isa)], got


def test_run_builds_the_one_list_from_args():
    from types import SimpleNamespace
    from . import run as _run
    args = SimpleNamespace(exact_lane_config=None, isa_claims_gate="off", amdahl_prune="off",
                           cpu_serving_launch=None, gpu_serving_launch=None, out=None)
    assert _run._gate_wrappers(args, False) == GateWrappers()
    args.amdahl_prune = "demote"
    assert callable(_run._gate_wrappers(args, False).tail_demote)
    assert _run._gate_wrappers(args, True).tail_demote is None      # CPU launch: never demoted
    args.cpu_serving_launch, args.exact_lane_config = "/c.json", el.DEFAULT_CPU_CONFIG
    assert callable(_run._gate_wrappers(args, True).make_exact_judge)
    args.isa_claims_gate = "on"
    with pytest.raises(SystemExit):                                  # ISA claims are GPU-only
        _run._gate_wrappers(args, True)
