"""UFH-12 arms A0 (cap 0) and A3 (embedder OpenMP env): offline tests.

A0: the gate's policy override reaches the scheduler without touching the policy file.
A3: the one-shot embedder env override is refused outside its narrow shape, recorded, reported by
env_attestation as EXPECTED (never silent, never an error) while live and unexpired, reported as an
ERROR once expired or on a pid it does not name, and cleared by the restore. The arm's provenance
rides in the belief sidecar. Nothing here reads a live process or sends a request.
"""

from __future__ import annotations

import json
import time
import types
from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from scripts.server import embedder_env_override as eo
from scripts.server import embedder_placement_capture as cap
from scripts.server import embedder_placement_gate as gate
from scripts.server import env_attestation as ea
from scripts.server.stack_env import _CANONICAL_OMP_ENV, _LLVM20_LIBDIR
from src.embedding_pool.client import PooledEmbeddingClient
from src.embedding_pool.fake import FakeEmbeddingServers
from src.embedding_pool.policy import EmbeddingPoolPolicy, POLICY_PATH, load_policy
from src.embedding_pool.scheduler import EmbeddingScheduler
from tests.unit.test_embedder_placement_capture import GATE, T0, _iso, _record, _window
from tests.unit.test_embedding_pool_scheduler import NODES, make_topology

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
GOOD_ENV = {**_CANONICAL_OMP_ENV, "LD_LIBRARY_PATH": f"{_LLVM20_LIBDIR}:/x"}


# ------------------------------------------------------------------ A0: policy override


def test_policy_override_cap_zero_leaves_the_file_alone():
    before = POLICY_PATH.read_bytes()
    base = load_policy()
    eff, parsed = gate.apply_policy_overrides(base, ["neighbour_cap.max_in_flight=0"])
    assert eff.neighbour_cap.max_in_flight == 0
    assert parsed == {"neighbour_cap.max_in_flight": 0}
    assert eff.client == base.client
    assert eff.neighbour_cap.guarded_roles == base.neighbour_cap.guarded_roles
    assert POLICY_PATH.read_bytes() == before


@pytest.mark.parametrize("item", ["neighbour_cap.nope=1", "max_in_flight=0", "neighbour_cap.max_in_flight=-1",
                                  "neighbour_cap.enabled=1", "other.max_in_flight=0"])
def test_policy_override_refuses_what_the_file_could_not_say(item):
    with pytest.raises(ValueError):
        gate.apply_policy_overrides(load_policy(), [item])


def test_policy_override_needs_the_scheduler():
    pol = EmbeddingPoolPolicy().with_cap(max_in_flight=0)
    with pytest.raises(ValueError):
        gate._load_for("raw", policy=pol)
    assert gate._load_for("raw") is gate._Load


def test_cap_zero_starves_the_busy_neighbours_and_nothing_else():
    fake = FakeEmbeddingServers(NODES, dim=16, delay_s=0.01)
    fake.processing = {8180: 1}  # the frontdoor half on nodes 2-3 decoding
    # slots only (as test_embedding_pool_client): the default ledger source reads the HOST's
    # region locks, so a live frontdoor lock would cap (here: starve) the idle half too.
    pol = (EmbeddingPoolPolicy().with_client(embedding_dim=None)
           .with_cap(max_in_flight=0, busy_sources=("slots",)))

    def build(ports):
        sched = EmbeddingScheduler(make_topology().restricted_to(ports), pol)
        return PooledEmbeddingClient(sched, policy=pol, transport=fake.transport())

    load = gate._SchedulerLoad(list(NODES), 4, client_factory=build, warmup_s=0.1, admission_wait_s=0.05)
    with load:
        time.sleep(0.4)
    summary = load.summary()
    assert load.errors == 0 and load.done > 0
    assert summary["effective_caps_at_end"] == {8090: 4, 8091: 4, 8092: 0, 8093: 0, 8094: 0, 8095: 0}
    for port in (8092, 8093, 8094, 8095):
        assert fake.peak.get(port, 0) == 0
    assert all(v == 0 for v in summary["peak_in_flight_capped"].values())


def test_paced_raw_load_waits_between_requests(monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "_post", lambda url, body, timeout=0: calls.append(time.monotonic()) or {})
    monkeypatch.setattr(gate, "LOAD_WARMUP_S", 0.0)
    with gate._Load([8090], 1, pace_s=0.2):
        time.sleep(0.5)
    assert 1 <= len(calls) <= 4  # back-to-back would be thousands


# ------------------------------------------------------------------ A3: the override record


def test_parse_overrides_allows_only_openmp_knobs():
    assert eo.parse_overrides(["OMP_WAIT_POLICY=passive", "KMP_BLOCKTIME=0"]) == {
        "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"}
    for bad in (["GGML_IQK=0"], ["OMP_WAIT_POLICY=sleepy"], ["KMP_BLOCKTIME=-1"], ["OMP_WAIT_POLICY"],
                ["OMP_WAIT_POLICY=active", "OMP_WAIT_POLICY=passive"], ["LD_LIBRARY_PATH=/tmp"]):
        with pytest.raises(eo.OverrideError):
            eo.parse_overrides(bad)


def test_record_lifecycle(tmp_path):
    rec = eo.write_active(experiment_id="ufh12-a3", env={"OMP_WAIT_POLICY": "passive"},
                          pids={8090: 11, 8091: 12}, ttl_s=600, log_dir=tmp_path, now=NOW)
    assert eo.read_record(tmp_path) == rec
    assert eo.covers(rec, pid=11, key="OMP_WAIT_POLICY", live_value="passive", now=NOW)
    assert not eo.covers(rec, pid=99, key="OMP_WAIT_POLICY", live_value="passive", now=NOW)
    assert not eo.covers(rec, pid=11, key="OMP_WAIT_POLICY", live_value="active", now=NOW)
    assert not eo.covers(rec, pid=11, key="KMP_BLOCKTIME", live_value="0", now=NOW)
    assert not eo.covers(rec, pid=11, key="OMP_WAIT_POLICY", live_value="passive",
                         now=NOW + timedelta(seconds=601))
    entry = eo.clear(reason="restored", readback_after={"8090": {"pid": 21, "env": {}}}, log_dir=tmp_path)
    assert entry["record"]["experiment_id"] == "ufh12-a3"
    assert eo.read_record(tmp_path) is None
    hist = [json.loads(line) for line in eo.history_path(tmp_path).read_text().splitlines()]
    assert hist[0]["reason"] == "restored" and hist[0]["readback_after"]["8090"]["pid"] == 21
    assert eo.clear(reason="again", log_dir=tmp_path) is None


def test_record_refuses_unbounded_or_anonymous(tmp_path):
    with pytest.raises(eo.OverrideError):
        eo.write_active(experiment_id=" ", env={"OMP_WAIT_POLICY": "passive"}, pids={8090: 1}, log_dir=tmp_path)
    with pytest.raises(eo.OverrideError):
        eo.write_active(experiment_id="x", env={"OMP_WAIT_POLICY": "passive"}, pids={8090: 1},
                        ttl_s=eo.MAX_TTL_S + 1, log_dir=tmp_path)


def test_restore_problems_name_every_leftover():
    rb = {"8090": {"pid": 1, "env": {"OMP_WAIT_POLICY": "active", "KMP_BLOCKTIME": "10", "KMP_LIBRARY": None}},
          "8091": {"pid": 2, "env": {"OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "10", "KMP_LIBRARY": None}},
          "8092": {"pid": None, "error": "no process on port"}}
    declared = eo.declared_values()
    assert declared["OMP_WAIT_POLICY"] == "active" and declared["KMP_LIBRARY"] is None
    problems = eo.restore_problems(rb, declared)
    assert len(problems) == 2 and any(":8091" in p for p in problems) and any(":8092" in p for p in problems)


# ------------------------------------------------------------------ A3: attestation


@pytest.fixture
def fake_proc(monkeypatch):
    procs: dict[int, tuple[str, dict[str, str]]] = {}
    monkeypatch.setattr(ea, "_alive", lambda pid: pid in procs)
    monkeypatch.setattr(ea, "_cmdline", lambda pid: procs[pid][0])
    monkeypatch.setattr(ea, "read_environ", lambda pid: dict(procs[pid][1]))
    return procs


def _emb(pid, port):
    return types.SimpleNamespace(role="embedder", pid=pid, port=port)


def _record_for(pids, *, applied=None, env=None):
    applied = applied or datetime.now(UTC)
    return {"schema": eo.RECORD_SCHEMA, "experiment_id": "ufh12-a3",
            "env": env or {"OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"},
            "applied_at": applied.isoformat(), "expires_at": (applied + timedelta(hours=1)).isoformat(),
            "pids": {str(8090 + i): p for i, p in enumerate(pids)}}


def test_covered_deviation_is_expected_not_an_error(fake_proc):
    fake_proc[31] = ("llama-server --embeddings", {**GOOD_ENV, "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"})
    r = ea.attest({"embedder": _emb(31, 8090)}, aux_services={}, override_record=_record_for([31]))
    assert r.verdict == "ok" and r.errors == []
    assert len(r.expected) == 2 and all("ufh12-a3" in line for line in r.expected)


def test_expired_or_foreign_pid_deviation_stays_an_error(fake_proc):
    fake_proc[31] = ("llama-server --embeddings", {**GOOD_ENV, "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"})
    old = _record_for([31], applied=datetime.now(UTC) - timedelta(hours=2))
    r = ea.attest({"embedder": _emb(31, 8090)}, aux_services={}, override_record=old)
    assert r.verdict == "failed" and r.expected == []
    assert all("EXPIRED" in e for e in r.errors)
    r = ea.attest({"embedder": _emb(31, 8090)}, aux_services={}, override_record=_record_for([32]))
    assert r.verdict == "failed" and r.expected == []


def test_value_the_record_does_not_name_stays_an_error(fake_proc):
    fake_proc[31] = ("llama-server --embeddings", {**GOOD_ENV, "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "5"})
    r = ea.attest({"embedder": _emb(31, 8090)}, aux_services={}, override_record=_record_for([31]))
    assert r.verdict == "failed"
    assert len(r.expected) == 1 and len(r.errors) == 1 and "KMP_BLOCKTIME" in r.errors[0]


def test_no_record_means_plain_drift(fake_proc):
    fake_proc[31] = ("llama-server --embeddings", {**GOOD_ENV, "OMP_WAIT_POLICY": "passive"})
    r = ea.attest({"embedder": _emb(31, 8090)}, aux_services={}, override_record=None)
    assert r.verdict == "failed" and r.expected == []


def test_pipeline_step_warns_on_expected_deviation(monkeypatch):
    from scripts.registry import stack_change_pipeline as scp

    result = ea.EnvAttestation(compared=["embedder:8090 pid 31"], expected=["x -- EXPECTED under experiment 'a3'"])
    monkeypatch.setattr(scp, "_declared_env_attestation_result", lambda: result)
    step = scp._declared_env_attestation_step()
    assert step.status == "ok" and step.ok
    assert step.warnings == ["EXPECTED env deviation: x -- EXPECTED under experiment 'a3'"]


# ------------------------------------------------------------------ A3: reload refusals


def _reload_args(**kw):
    base = {"components": ["embedders"], "profile": None, "allow_during_bench": False,
            "embedder_env_override": [], "experiment_id": None, "override_ttl_s": 5400.0}
    base.update(kw)
    return Namespace(**base)


@pytest.mark.parametrize("kw", [
    {"embedder_env_override": ["OMP_WAIT_POLICY=passive"]},                                   # no id
    {"embedder_env_override": ["OMP_WAIT_POLICY=passive"], "experiment_id": "a3",
     "components": ["embedders", "orchestrator"]},                                            # widened
    {"embedder_env_override": ["OMP_WAIT_POLICY=passive"], "experiment_id": "a3",
     "components": ["frontdoor"]},                                                            # not embedders
    {"embedder_env_override": ["GGML_IQK=0"], "experiment_id": "a3"},                         # not OpenMP
    {"experiment_id": "a3"},                                                                  # id alone
    {"embedder_env_override": ["OMP_WAIT_POLICY=passive"], "experiment_id": "a3",
     "override_ttl_s": 0.0},                                                                  # unbounded
])
def test_reload_refuses_before_touching_anything(monkeypatch, kw):
    from scripts.server import stack_commands

    touched = []
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "kill_process", lambda pid: touched.append(pid))
    monkeypatch.setattr(stack_commands, "start_server", lambda *a, **k: touched.append("start"))
    assert stack_commands.cmd_reload(_reload_args(**kw)) == 2
    assert touched == []


def test_start_server_refuses_override_outside_embedders():
    from scripts.server import orchestrator_stack as stack

    with pytest.raises(ValueError):
        stack.start_server(8070, ["frontdoor"], None, env_override={"OMP_WAIT_POLICY": "passive"})


# ------------------------------------------------------------------ provenance in the sidecar


def test_arm_labels_and_provenance_reach_the_sidecar(tmp_path):
    record = _record(label="arm-candidate")
    record["params"].update({
        "arm": "A3-passive", "policy_overrides": {}, "load_pace_s": 0.0,
        "expect_embedder_env": {"OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0"},
        "embedder_env": {"8090": {"pid": 1, "env": {"OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0",
                                                     "KMP_LIBRARY": None}}},
        "embedder_env_override_record": {"experiment_id": "ufh12-a3"}})
    out = tmp_path / "arm-a3-candidate.json"
    out.write_text(json.dumps(record))
    sidecar = cap.write_belief_measurements(out, record, window=_window(), producer="test", gate_path=GATE,
                                            emitted_at=_iso(T0 + timedelta(minutes=21)))
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert rows and all(cap.validate_row(r) == [] for r in rows)
    for row in rows:
        assert row["category"] == "CANDIDATE" and row["extra"]["load_mode"] == "scheduler"
        method = row["extra"]["provenance"]["instrument"]["method"]
        assert method["arm"] == "A3-passive"
        assert method["embedder_env"]["8090"]["env"]["OMP_WAIT_POLICY"] == "passive"
        assert method["embedder_env_override_record"]["experiment_id"] == "ufh12-a3"
        assert "A3-passive" in row["claim"]
    base = _record(label="arm-baseline")
    out_b = tmp_path / "arm-a3-baseline.json"
    out_b.write_text(json.dumps(base))
    sidecar_b = cap.write_belief_measurements(out_b, base, window=_window(), producer="test", gate_path=GATE,
                                              emitted_at=_iso(T0 + timedelta(minutes=21)))
    assert {json.loads(line)["category"] for line in sidecar_b.read_text().splitlines()} == {"BASELINE"}


def test_gate_main_records_arm_provenance_and_refuses_the_wrong_env(tmp_path, monkeypatch):
    import sys

    from tests.unit.test_embedder_placement_capture import _snapshots

    base = _record(load_mode="scheduler")
    seen = {}
    live = {str(p): {"pid": p, "env": {"OMP_WAIT_POLICY": "active", "KMP_BLOCKTIME": "10", "KMP_LIBRARY": None}}
            for p in gate.POOL_PORTS}
    monkeypatch.setattr(gate, "_healthy", lambda port: True)
    monkeypatch.setattr(gate, "embedder_env_readback", lambda ports: json.loads(json.dumps(live)))
    monkeypatch.setattr(eo, "read_record", lambda log_dir=None: None)
    monkeypatch.setattr(gate, "g2", lambda window_s, load_factory: base["g2_pool_scaling"])

    def fake_g1(fd, pairs, n, load_factory):
        seen["load"] = load_factory(gate.POOL_PORTS, gate.PER_PORT_IN_FLIGHT)
        return base["g1_frontdoor_decode"]

    monkeypatch.setattr(gate, "g1", fake_g1)
    monkeypatch.setattr(gate, "_capture_window",
                        lambda fd: cap.CaptureWindow(gate.POOL_PORTS + fd, fd, snapshot_fn=_snapshots()))
    out = tmp_path / "arm-a0-candidate.json"
    argv = ["gate", "--label", "arm-candidate", "--arm", "A0-cap0", "--out", str(out), "--fd-ports", "8070,8080",
            "--load-mode", "scheduler", "--policy-override", "neighbour_cap.max_in_flight=0",
            "--expect-embedder-env", "OMP_WAIT_POLICY=active"]
    monkeypatch.setattr(sys, "argv", argv)
    assert gate.main() == 0
    record = json.loads(out.read_text())
    assert record["scheduler_policy"]["neighbour_cap"]["max_in_flight"] == 0
    assert record["params"]["policy_overrides"] == {"neighbour_cap.max_in_flight": 0}
    assert record["params"]["arm"] == "A0-cap0" and record["params"]["embedder_env"] == live
    assert seen["load"].client_factory is not gate._default_pool_client  # the effective policy's client
    rows = [json.loads(line) for line in cap.sidecar_path(out).read_text().splitlines()]
    assert rows and all(cap.validate_row(r) == [] for r in rows)
    assert rows[0]["extra"]["provenance"]["load"]["scheduler_policy_sha256"] == cap.content_hash(
        record["scheduler_policy"])

    monkeypatch.setattr(sys, "argv", argv[:-1] + ["OMP_WAIT_POLICY=passive"])
    with pytest.raises(SystemExit, match="not the arm's"):
        gate.main()
    monkeypatch.setattr(sys, "argv", ["gate", "--label", "arm-candidate", "--out", str(out)])
    with pytest.raises(SystemExit, match="needs --arm"):
        gate.main()
