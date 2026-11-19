import random

import pytest

from . import cpu_aa_calibration as aa
from . import two_instance_floor as t

STORE = "/s/store"


class FakeServer:
    def __init__(self, tok=256, ms=100.0, fail=False, log=None, name=""):
        self.tok, self.ms, self.fail, self.log, self.name = tok, ms, fail, log if log is not None else [], name

    def start(self):
        self.log.append(("start", self.name))

    def stop(self):
        self.log.append(("stop", self.name))

    def complete(self, text, n):
        self.log.append(("complete", self.name))
        if self.fail:
            raise RuntimeError("boom")
        return {"predicted_n": self.tok, "predicted_ms": self.ms * self.tok, "draft_n_accepted": 0, "tok_s": 1000 / self.ms}


def test_recipes_valid_and_halves_disjoint():
    for r in t.RECIPES.values():
        t.validate_recipe(r)
    bad = t.Recipe("x", "W3", "m", (t.Instance("0-47", "0", 1), t.Instance("40-95", "1", 2)), 8, ())
    with pytest.raises(t.TwoInstanceError, match="overlap"):
        t.validate_recipe(bad)
    same_port = t.Recipe("x", "W3", "m", (t.Instance("0-47", "0", 1), t.Instance("48-95", "1", 1)), 8, ())
    with pytest.raises(t.TwoInstanceError, match="port"):
        t.validate_recipe(same_port)
    few = t.Recipe("x", "W3", "m", (t.Instance("0-3", "0", 1), t.Instance("48-51", "1", 2)), 48, ())
    with pytest.raises(t.TwoInstanceError, match="fewer"):
        t.validate_recipe(few)


def test_argv_is_pinned_literal_threads_per_half():
    a0 = t.instance_argv(t.W3, STORE, 0)
    a1 = t.instance_argv(t.W3, STORE, 1)
    assert a0[:3] == ["/usr/bin/taskset", "-c", t.HALF_A] and a1[2] == t.HALF_B
    assert "--interleave=0,1" in a0 and "--interleave=2,3" in a1
    assert a0[a0.index("--port") + 1] == "8080" and a1[a1.index("--port") + 1] == "8180"
    assert a0[a0.index("-t") + 1] == "48" and a0[a0.index("-tb") + 1] == "48"
    assert a0[a0.index("--spec-draft-n-max") + 1] == "4"
    assert t.instance_argv(t.W3, STORE, 0, port=19000)[a0.index("--port") + 1] == "19000"


def test_schedule_balanced_and_seeded():
    s = t.schedule(7, 24)
    assert len(s) == 48 and sum(1 for _, _, a in s if a == "A") == 24
    assert s == t.schedule(7, 24) and s != t.schedule(8, 24)
    assert sorted(v for v, _, _ in s) == list(range(48))
    with pytest.raises(t.TwoInstanceError):
        t.schedule(1, 23)


def test_visit_runs_both_concurrently_and_stops_both_even_on_failure():
    log: list = []
    r = t.run_visit([FakeServer(log=log, name="a"), FakeServer(ms=50, log=log, name="b")], "p", 256)
    assert r["valid"] and abs(r["aggregate_tok_s"] - (10 + 20)) < 1e-9
    assert [e for e in log if e[0] == "stop"] and len([e for e in log if e[0] == "start"]) == 2
    log2: list = []
    with pytest.raises(RuntimeError):
        t.run_visit([FakeServer(log=log2, name="a"), FakeServer(fail=True, log=log2, name="b")], "p", 256)
    assert sorted(n for k, n in log2 if k == "stop") == ["a", "b"]


def test_short_generation_makes_visit_invalid_not_one_sided():
    r = t.run_visit([FakeServer(), FakeServer(tok=100)], "p", 256)
    assert not r["valid"] and r["aggregate_tok_s"] is None


def rows(n, sd=0.01, seed=1, drop=()):
    rnd = random.Random(seed)
    out = []
    for p in range(n):
        for arm in "AB":
            ms = [100 * (1 + rnd.gauss(0, sd)), 120 * (1 + rnd.gauss(0, sd))]
            ok = (p, arm) not in drop
            out.append({"pair": p, "arm": arm, "valid": ok,
                        "instances": [{"step_ms": m} for m in ms],
                        "aggregate_tok_s": sum(1000 / m for m in ms)})
    return out


def test_record_n24_has_agg_gating_and_topology():
    rec = t.build_floor_record(t.W3, "d" * 64, rows(24), contention_label="quiet", foreign_busy_p95=[3.0, 6.0], source_row="r")
    assert rec["topology"] == "two-instance" and rec["unit"] == "process" and rec["workload"] == "W3"
    assert rec["surfaces"]["AGG"]["n"] == 24 and rec["surfaces"]["AGG"]["gating"] is True
    assert rec["surfaces"]["I0"]["gating"] is False and rec["recipe_hash"] == t.W3.digest()
    assert rec["host_state"]["cpu_list"] == f"{t.HALF_A};{t.HALF_B}"


def test_record_refuses_under_sampled_or_mislabelled():
    with pytest.raises(aa.CpuAACalibrationError, match="n=23"):
        t.build_floor_record(t.W3, "d" * 64, rows(24, drop={(3, "B")}), contention_label="quiet", foreign_busy_p95=[3.0],
                             source_row="r")
    with pytest.raises(aa.CpuAACalibrationError, match="quiet"):
        t.build_floor_record(t.W3, "d" * 64, rows(24), contention_label="quiet", foreign_busy_p95=[40.0], source_row="r")


def test_missing_inputs_flags_retired_models(tmp_path):
    assert t.W6.model in t.missing_inputs(t.W6, str(tmp_path))
    assert t.W6.retired and t.W4.retired and not t.W3.retired
