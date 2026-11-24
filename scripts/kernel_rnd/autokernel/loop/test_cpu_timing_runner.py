"""RSL-R6 live CPU timing runner: served topology from registry + stack_numa, tg128 pairs, per-topology floors (no binary runs)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autokernel.loop import census_coverage as cc
from autokernel.loop import cpu_timing_runner as ctr
from autokernel.loop import stage_bodies as sb
from autokernel.loop.op_perf_runner import RunnerRefused

BUILD = Path("/nonexistent-build")
FAKE_SN = SimpleNamespace(
    NUMA_CONFIG={"architect_general": {"instances": [("0-95", 8074, 96)]},
                 "frontdoor": {"instances": [("0-95", 8070, 96), ("0-47,96-143", 8080, 48), ("48-95,144-191", 8180, 48)]}},
    _numa_prefix=lambda role, i: ["numactl", "--interleave=all", "--", "taskset", "-c", {8074: "0-95"}.get(i, str(i))])
Q38 = {"key": "q38fn-ud-iq4xs-cpu|cpu", "host": "architect_general", "gguf": "/m/Flash-Next.gguf", "lane": "cpu", "surface": "tg128",
       "serving": {"port": 8074, "numa_instances": 1, "numa_ports": [8074], "serving_shape": {"kv_quant": {"k": "f16", "v": "f16"}},
                   "recipe": {"env": {"GGML_FA_SPLIT_KV": "0"}}}}
A3B = {"key": "a3b-q8_0-cpu|cpu", "host": "frontdoor", "gguf": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf", "lane": "cpu",
       "surface": "tg128", "serving": {"port": 8070, "numa_instances": 2, "numa_ports": [8080, 8180],
                                       "serving_shape": {"kv_quant": {"k": "q8_0", "v": "q8_0"}}}}
TOPO1 = ctr.topology(Q38, FAKE_SN)


def test_topology_from_registry_ports_and_stack_numa():
    assert [(t["threads"], t["cpus"]) for t in TOPO1] == [(96, "0-95")]
    two = ctr.topology(A3B, FAKE_SN)
    assert [(t["port"], t["threads"]) for t in two] == [(8080, 48), (8180, 48)]
    for bad in ({**A3B, "serving": {**A3B["serving"], "numa_ports": [8080, 9999]}}, {**A3B, "serving": None},
                {**A3B, "serving": {**A3B["serving"], "numa_instances": 3}}):
        with pytest.raises(RunnerRefused):
            ctr.topology(bad, FAKE_SN)


def test_topology_of_the_real_served_hosts():
    reg = cc.load_registry()
    if not reg or not ctr.STACK_NUMA.is_file():
        pytest.skip("registry / stack_numa not readable")
    specs = sb.path_specs(reg)
    q, a = ctr.topology(specs["q38fn-ud-iq4xs-cpu|cpu"]), ctr.topology(specs["a3b-q8_0-cpu|cpu"])
    assert [(t["port"], t["threads"], t["cpus"]) for t in q] == [(8074, 96, "0-95")] and q[0]["prefix"][:2] == ["numactl", "--interleave=all"]
    assert [(t["port"], t["threads"]) for t in a] == [(8080, 48), (8180, 48)] and a[0]["cpus"] != a[1]["cpus"]


def test_argv_comes_from_the_topology_not_typed():
    argv = ctr.bench_argv(BUILD / "bin" / "llama-bench", Q38, TOPO1[0], reps=10, seed=7)
    assert argv[:len(TOPO1[0]["prefix"])] == TOPO1[0]["prefix"] and argv[argv.index("-t") + 1] == "96"
    assert argv[argv.index("-ctk") + 1] == "f16" and argv[argv.index("-n") + 1] == "128" and "--autokernel-harden" in argv
    assert ctr.recipe_env(Q38) == {"GGML_FA_SPLIT_KV": "0"} and ctr.recipe_env(A3B)["GGML_IQK"] == "1"   # A3B: two_instance_floor W3 env


def _time(outcome="pass", floor=1.0, topo=TOPO1, **cand):
    runner = ctr.synth_runner(outcome)   # the RECORDED real llama-bench receipt, rescaled; the candidate arm's path has `cand`
    return ctr.time_path({"pairs": 5, "reps": 10, "test_topology": topo, **cand}, BUILD / "anchor", BUILD / "cand", Q38,
                         runner=runner, live=False, floor=floor)


def test_decision_paths():
    t = _time()
    assert t["pairs"] == 5 and t["effect_pct"] == pytest.approx(5.0, abs=1.0) and t["decisive"] is True and t["noise_floor_pct"] == 1.0
    assert len(t["anchor_samples"]) == len(t["candidate_samples"]) == 5
    assert _time("flat")["decisive"] is False                         # inside the floor: not a measurement of anything
    assert _time(floor=50.0)["decisive"] is False
    for outcome in ("harness", "bad-rc"):
        with pytest.raises(RunnerRefused) as e:
            _time(outcome)
        assert e.value.token == "FUTILITY_HARNESS_INSTRUMENT"
    with pytest.raises(RunnerRefused):                                # no floor for this model+topology -> refuse
        ctr.time_path({"pairs": 5, "reps": 10, "test_topology": TOPO1}, BUILD, BUILD, Q38, store=Path("/nonexistent-store"),
                      runner=ctr.synth_runner(), live=False)
    assert _time(pairs=3)["pairs"] == 5                               # raised to bench.MIN_PAIRS


def test_two_instances_sample_is_the_aggregate():
    two = ctr.topology(A3B, FAKE_SN)
    one = ctr.sample(BUILD / "cand" / "bin" / "llama-bench", Q38, TOPO1, reps=10, seed=1, runner=ctr.synth_runner("flat"))
    agg = ctr.sample(BUILD / "cand" / "bin" / "llama-bench", A3B, two, reps=10, seed=1, runner=ctr.synth_runner("flat"))
    assert agg == pytest.approx(2 * one, rel=0.03)


def test_calibrate_writes_per_topology_record_and_floor_is_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(ctr, "_stack_numa", lambda: FAKE_SN)
    runner = ctr.synth_runner("flat")
    with pytest.raises(RunnerRefused):                                # < 24 pairs: refused by cpu_aa_calibration
        ctr.calibrate(BUILD / "a", Q38, source_row="r", pairs=10, store=tmp_path, runner=runner, live=False, reps=10)
    assert ctr.floor_pct(Q38, tmp_path, 5) is None                    # nothing written: ungated
    p = ctr.calibrate(BUILD / "a", Q38, source_row="row-1", store=tmp_path, runner=runner, live=False, reps=10)
    rec = json.loads(p.read_text())
    assert p.parent == tmp_path / "calibration" and rec["schema"] == ctr.aa.SCHEMA and rec["recipe"] == "cpu-tg128.Flash-Next.1inst-96t"
    assert rec["surfaces"]["TG128"]["n"] == 24 and rec["contention_label"] == "quiet" and rec["source_row"] == "row-1"
    f = ctr.floor_pct(Q38, tmp_path, 5)
    assert f is not None and f > 0 and ctr.floor_pct(Q38, tmp_path, 7) is None
    assert ctr.floor_pct(A3B, tmp_path, 5) is None                    # another model/topology never borrows it
    cand = {"store": str(tmp_path), "pairs": 5}
    assert sb.timing_floor(Q38, cand) is True and sb.timing_floor({**Q38, "gguf": "/m/other.gguf"}, cand) is False


def test_stage_live_timing_cpu_refuses_without_floor_and_cli_runs_no_binary(tmp_path):
    with pytest.raises(sb.StageRefused) as e:
        sb._live_timing({"path_spec": Q38, "store": str(tmp_path)}, BUILD, BUILD)
    assert e.value.token == "FUTILITY_HARNESS_INSTRUMENT"
    reg = cc.load_registry()
    if not reg:
        pytest.skip("registry unreadable")
    out = tmp_path / "o.json"
    assert ctr.main(["--path", "q38fn-ud-iq4xs-cpu|cpu", "--anchor-build", str(BUILD / "anchor"), "--candidate-build", str(BUILD / "cand"),
                     "--test-mode", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["decisive"] is True
    assert ctr.main(["--path", "q38fn-ud-iq4xs-cpu|cpu", "--anchor-build", str(BUILD / "a"), "--test-mode", "--calibrate", "--row", "r",
                     "--store", str(tmp_path), "--out", str(out)]) == 0
    assert Path(json.loads(out.read_text())["floor_file"]).is_file()
