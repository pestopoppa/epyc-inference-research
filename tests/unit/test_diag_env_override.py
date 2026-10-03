"""Per-component DIAGNOSTIC env override (UFH-14; scripts/server/env_override.py): offline tests.

`reload <component> --diag-env-override LLAMA_SERVER_SLOTS_DEBUG=1 --experiment-id ID` relaunches ONE
llama-server model component with an allow-listed diagnostic key. These tests pin: the allow-list
refusal; component scoping; the record / expiry / clear lifecycle; attestation severity before and
after expiry; that the embedder OpenMP path is unchanged; and that the key reaches the env handed to
Popen (and a plain launch strips it). Nothing here reads a live process or launches anything.
"""

from __future__ import annotations

import json
import types
from argparse import Namespace
from datetime import UTC, datetime, timedelta

import pytest

from scripts.server import embedder_env_override as eo
from scripts.server import env_attestation as ea
from scripts.server import env_override as dov
from scripts.server import orchestrator_stack as osk
from scripts.server import stack_commands
from scripts.server.stack_env import _CANONICAL_OMP_ENV, _LLVM20_LIBDIR
from tests.unit.test_launch_env_every_branch import harness  # noqa: F401  (pytest fixture)

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
GOOD_ENV = {**_CANONICAL_OMP_ENV, "LD_LIBRARY_PATH": f"{_LLVM20_LIBDIR}:/x"}
KEY = "LLAMA_SERVER_SLOTS_DEBUG"


# ------------------------------------------------------------------ the allow-list


def test_allowlist_is_one_table_with_exactly_the_diag_key():
    assert dov.ALLOWLISTS[dov.LLAMA_SERVER] == {KEY: frozenset({"0", "1"})}
    # The embedder row IS the embedder module's ALLOWED (one place), unchanged in content.
    assert eo.ALLOWED is dov.ALLOWLISTS[dov.EMBEDDER]
    assert set(eo.ALLOWED) == {"OMP_WAIT_POLICY", "KMP_BLOCKTIME", "KMP_LIBRARY"}


@pytest.mark.parametrize("bad", [
    [f"{KEY}=2"], [f"{KEY}=true"], [KEY], ["GGML_IQK=0"], ["OMP_WAIT_POLICY=passive"],
    ["LD_LIBRARY_PATH=/tmp"], [f"{KEY}=1", f"{KEY}=0"],
])
def test_diag_parse_refuses_anything_not_allow_listed(bad):
    with pytest.raises(dov.OverrideError):
        dov.parse_overrides(bad, dov.LLAMA_SERVER)


def test_diag_parse_accepts_the_allow_list():
    assert dov.parse_overrides([f"{KEY}=1"], dov.LLAMA_SERVER) == {KEY: "1"}
    assert dov.parse_overrides([f" {KEY} = 0 "], dov.LLAMA_SERVER) == {KEY: "0"}


def test_embedder_parse_does_not_take_the_diag_key():
    with pytest.raises(eo.OverrideError):
        eo.parse_overrides([f"{KEY}=1"])
    # and its messages are the embedder ones, unchanged
    with pytest.raises(eo.OverrideError, match="^embedder env override key 'GGML_IQK' is not allowed"):
        eo.parse_overrides(["GGML_IQK=0"])


# ------------------------------------------------------------------ record lifecycle


def test_record_lifecycle(tmp_path):
    rec = dov.write_active(component="architect_critic", component_class=dov.LLAMA_SERVER,
                           experiment_id="UFH14-A7", env={KEY: "1"}, pids={8083: 77}, port=8083,
                           ttl_s=600, log_dir=tmp_path, now=NOW)
    assert dov.record_path("architect_critic", tmp_path) == tmp_path / "env_overrides" / "architect_critic.json"
    assert dov.read_record("architect_critic", tmp_path) == rec
    assert dov.read_all_records(tmp_path) == [rec]
    assert rec["restore"].startswith("orchestrator_stack.py reload architect_critic")
    assert dov.covers(rec, pid=77, key=KEY, live_value="1", now=NOW)
    assert not dov.covers(rec, pid=78, key=KEY, live_value="1", now=NOW)
    assert not dov.covers(rec, pid=77, key=KEY, live_value="0", now=NOW)
    assert not dov.covers(rec, pid=77, key=KEY, live_value="1", now=NOW + timedelta(seconds=601))
    entry = dov.clear("architect_critic", reason="restored", log_dir=tmp_path,
                      readback_after={"8083": {"pid": 88, "env": {KEY: None}}})
    assert entry["record"]["experiment_id"] == "UFH14-A7"
    assert dov.read_record("architect_critic", tmp_path) is None and dov.read_all_records(tmp_path) == []
    hist = [json.loads(line) for line in dov.history_path("architect_critic", tmp_path).read_text().splitlines()]
    assert hist[0]["reason"] == "restored" and hist[0]["readback_after"]["8083"]["env"][KEY] is None
    assert dov.clear("architect_critic", reason="again", log_dir=tmp_path) is None


@pytest.mark.parametrize("kw", [
    {"experiment_id": " "},
    {"ttl_s": dov.MAX_TTL_S + 1},
    {"ttl_s": 0},
    {"env": {"GGML_IQK": "0"}},
    {"env": {}},
    {"pids": {}},
    {"component": "../escape"},
])
def test_record_refuses_unbounded_anonymous_or_unlisted(tmp_path, kw):
    base = {"component": "architect_critic", "component_class": dov.LLAMA_SERVER,
            "experiment_id": "x", "env": {KEY: "1"}, "pids": {8083: 1}, "log_dir": tmp_path}
    base.update(kw)
    with pytest.raises(dov.OverrideError):
        dov.write_active(**base)
    assert not (tmp_path / "env_overrides" / "architect_critic.json").exists()


def test_unreadable_record_raises(tmp_path):
    rdir = tmp_path / "env_overrides"
    rdir.mkdir()
    (rdir / "architect_critic.json").write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(dov.OverrideError):
        dov.read_all_records(tmp_path)


# ------------------------------------------------------------------ attestation severity


@pytest.fixture
def fake_proc(monkeypatch):
    procs: dict[int, tuple[str, dict[str, str]]] = {}
    monkeypatch.setattr(ea, "_alive", lambda pid: pid in procs)
    monkeypatch.setattr(ea, "_cmdline", lambda pid: procs[pid][0])
    monkeypatch.setattr(ea, "read_environ", lambda pid: dict(procs[pid][1]))
    return procs


def _ac(pid):
    return types.SimpleNamespace(role="architect_critic", pid=pid, port=8083)


def _diag_record(pid, *, applied=None, value="1"):
    applied = applied or datetime.now(UTC)
    return {"schema": dov.RECORD_SCHEMA, "component": "architect_critic", "component_class": dov.LLAMA_SERVER,
            "port": 8083, "experiment_id": "UFH14-A7", "env": {KEY: value},
            "applied_at": applied.isoformat(), "expires_at": (applied + timedelta(hours=1)).isoformat(),
            "pids": {"8083": pid}, "restore": "orchestrator_stack.py reload architect_critic (no override flags)"}


def _attest(records):
    return ea.attest({"architect_critic": _ac(41)}, aux_services={}, override_record=None, diag_records=records)


def test_unexpired_record_is_a_declared_time_bound_warning(fake_proc):
    fake_proc[41] = ("/k/llama-server --port 8083", {**GOOD_ENV, KEY: "1"})
    r = _attest([_diag_record(41)])
    assert r.verdict == "ok" and r.errors == []
    assert len(r.expected) == 1
    assert "UFH14-A7" in r.expected[0] and "declared, time-bound" in r.expected[0] and KEY in r.expected[0]


def test_expired_record_is_an_error_naming_the_restore(fake_proc):
    fake_proc[41] = ("/k/llama-server --port 8083", {**GOOD_ENV, KEY: "1"})
    r = _attest([_diag_record(41, applied=datetime.now(UTC) - timedelta(hours=2))])
    assert r.verdict == "failed" and r.expected == []
    assert len(r.errors) == 1 and "EXPIRED" in r.errors[0] and "reload architect_critic" in r.errors[0]


@pytest.mark.parametrize("records", [
    [],                                   # no record at all: unexplained drift
    [_diag_record(99)],                   # a record for another pid
    [_diag_record(41, value="0")],        # a record for another value
])
def test_unexplained_diag_key_is_an_error(fake_proc, records):
    fake_proc[41] = ("/k/llama-server --port 8083", {**GOOD_ENV, KEY: "1"})
    r = _attest(records)
    assert r.verdict == "failed" and r.expected == []
    assert any(KEY in e and "no covering override record" in e for e in r.errors)


def test_restored_process_is_clean(fake_proc):
    fake_proc[41] = ("/k/llama-server --port 8083", dict(GOOD_ENV))
    r = _attest([])
    assert r.verdict == "ok" and r.errors == [] and r.expected == []


def test_unreadable_diag_records_are_an_error(fake_proc, monkeypatch):
    fake_proc[41] = ("/k/llama-server --port 8083", dict(GOOD_ENV))

    def boom():
        raise dov.OverrideError("bad record")

    monkeypatch.setattr(ea, "_read_diag_records", boom)
    r = ea.attest({"architect_critic": _ac(41)}, aux_services={}, override_record=None)
    assert r.verdict == "failed" and any("diagnostic env override records unreadable" in e for e in r.errors)


def test_pipeline_step_warns_not_fails_on_a_live_diag_override(fake_proc, monkeypatch):
    from scripts.registry import stack_change_pipeline as scp

    fake_proc[41] = ("/k/llama-server --port 8083", {**GOOD_ENV, KEY: "1"})
    monkeypatch.setattr(scp, "_declared_env_attestation_result", lambda: _attest([_diag_record(41)]))
    step = scp._declared_env_attestation_step()
    assert step.status == "ok" and step.ok
    assert len(step.warnings) == 1 and step.warnings[0].startswith("EXPECTED env deviation:")
    monkeypatch.setattr(scp, "_declared_env_attestation_result",
                        lambda: _attest([_diag_record(41, applied=datetime.now(UTC) - timedelta(hours=2))]))
    step = scp._declared_env_attestation_step()
    assert step.status == "failed" and any("EXPIRED" in e for e in step.errors)


def test_runtime_attestation_does_not_report_the_diag_key_as_drift(monkeypatch):
    """runtime_attestation compares argv plus the env fields in _RUNTIME_FIELD_CHECKS. The diag
    override changes neither the argv nor any of those fields, so a recorded override can never
    surface there as unexplained drift -- this pins that the allow-list stays disjoint from them."""
    env_attested = {flags[0] for mode, flags, _ in stack_commands._RUNTIME_FIELD_CHECKS.values()
                    if mode in ("env_prefix", "env_value")}
    assert env_attested and not (env_attested & dov.diagnostic_keys())

    info = stack_commands.ProcessInfo(role="architect_critic", pid=41, port=8083, started_at="t",
                                      model_path="/m/a.gguf", log_file="l")
    cmdline = ["/k/llama-server", "-m", "/m/a.gguf", "--port", "8083"]
    contract = {"requirements": {"model_path": "/m/a.gguf"},
                "runtime": {"binary_path": "/k/llama-server", "kmp_blocktime": 10,
                            "ld_library_path": [_LLVM20_LIBDIR]}}
    live = {**GOOD_ENV, "KMP_BLOCKTIME": "10", "LD_LIBRARY_PATH": _LLVM20_LIBDIR, KEY: "1"}
    monkeypatch.setattr(stack_commands, "_process_environ", lambda pid: dict(live))
    assert stack_commands._runtime_attestation_warnings("architect_critic", info, cmdline, contract) == []


def test_embedder_attestation_path_is_unchanged(fake_proc):
    """The embedder record still covers its OpenMP deviation, with diag records present."""
    fake_proc[31] = ("llama-server --embeddings", {**GOOD_ENV, "OMP_WAIT_POLICY": "passive"})
    applied = datetime.now(UTC)
    emb = {"schema": eo.RECORD_SCHEMA, "experiment_id": "ufh12-a3", "env": {"OMP_WAIT_POLICY": "passive"},
           "applied_at": applied.isoformat(), "expires_at": (applied + timedelta(hours=1)).isoformat(),
           "pids": {"8090": 31}}
    r = ea.attest({"embedder": types.SimpleNamespace(role="embedder", pid=31, port=8090)}, aux_services={},
                  override_record=emb, diag_records=[_diag_record(41)])
    assert r.verdict == "ok" and len(r.expected) == 1 and "ufh12-a3" in r.expected[0]


# ------------------------------------------------------------------ reload: scoping + refusals


def _reload_args(**kw):
    base = {"components": ["architect_critic"], "profile": None, "allow_during_bench": False,
            "embedder_env_override": [], "diag_env_override": [f"{KEY}=1"], "experiment_id": "UFH14-A7",
            "override_ttl_s": 5400.0}
    base.update(kw)
    return Namespace(**base)


@pytest.mark.parametrize("kw", [
    {"experiment_id": None},                                               # anonymous
    {"components": ["architect_critic", "architect_general"]},             # multi-component
    {"components": ["embedders"]},                                         # not llama model component
    {"components": ["embedder_1"]},                                        # an embedder by role
    {"components": ["orchestrator"]},                                      # the API
    {"components": ["whisper"]},                                           # an aux service
    {"components": ["no_such_role"]},                                      # unresolvable
    {"diag_env_override": ["GGML_IQK=0"]},                                 # not allow-listed
    {"diag_env_override": [f"{KEY}=2"]},                                   # bad value
    {"diag_env_override": ["OMP_WAIT_POLICY=passive"]},                    # embedder key, wrong class
    {"override_ttl_s": 0.0},                                               # unbounded
    {"override_ttl_s": dov.MAX_TTL_S + 1},                                 # past the max
    {"embedder_env_override": ["OMP_WAIT_POLICY=passive"]},                # both flags
])
def test_reload_refuses_before_touching_anything(monkeypatch, kw):
    touched = []
    monkeypatch.setattr(stack_commands, "load_state", lambda: {})
    monkeypatch.setattr(stack_commands, "kill_process", lambda pid: touched.append(pid))
    monkeypatch.setattr(stack_commands, "start_server", lambda *a, **k: touched.append("start"))
    monkeypatch.setattr(stack_commands, "_pids_on_port", lambda port: touched.append(port) or [])
    assert stack_commands.cmd_reload(_reload_args(**kw)) == 2
    assert touched == []


def test_diag_target_resolution():
    port, roles, key = stack_commands._diag_override_target("architect_critic")
    assert port == 8083 and roles[0] == "architect_critic" and key == "architect_critic"
    # a sub-full instance shares its role name with the full one: keyed by port
    assert stack_commands._diag_record_key(8080, ["frontdoor"]) == "server_8080"
    assert stack_commands._diag_record_key(8070, ["frontdoor"]) == "frontdoor"


@pytest.fixture
def reload_env(monkeypatch, tmp_path):
    started: list[dict] = []
    killed: list[int] = []
    saved: list[dict] = []
    pid = {"next": 500}

    def fake_start(port, roles, registry, **kwargs):
        started.append({"port": port, "roles": roles, **kwargs})
        pid["next"] += 1
        return stack_commands.ProcessInfo(role=roles[0], pid=pid["next"], port=port, started_at="t",
                                          model_path="m", log_file="l")

    monkeypatch.setattr(dov, "_log_dir", lambda: tmp_path)
    monkeypatch.setattr(osk, "load_state", lambda: {})
    monkeypatch.setattr(osk, "save_state", lambda value: saved.append(dict(value)))
    monkeypatch.setattr(stack_commands, "_refresh_runtime_facts_manifest", lambda *a, **k: None)
    monkeypatch.setattr(osk, "RegistryLoader", lambda: object())
    monkeypatch.setattr(stack_commands, "RegistryLoader", lambda: object())
    monkeypatch.setattr(osk, "kill_process", lambda p: killed.append(p))
    monkeypatch.setattr(osk, "_pids_on_port", lambda port: [400] if port == 8083 else [])
    monkeypatch.setattr(osk.time, "sleep", lambda _s: None)
    monkeypatch.setattr(osk, "start_server", fake_start)
    monkeypatch.setattr(dov, "read_environ", lambda p: {KEY: "1"} if p == 501 else {})
    return types.SimpleNamespace(started=started, killed=killed, saved=saved, log_dir=tmp_path)


def test_reload_records_then_plain_reload_restores_and_archives(reload_env):
    assert stack_commands.cmd_reload(_reload_args()) == 0
    first = reload_env.started[-1]
    assert first["port"] == 8083 and first["diag_env_override"] == {KEY: "1"}
    rec = dov.read_record("architect_critic", reload_env.log_dir)
    assert rec["experiment_id"] == "UFH14-A7" and rec["env"] == {KEY: "1"}
    assert rec["pids"] == {"8083": 501} and rec["port"] == 8083 and rec["component_class"] == dov.LLAMA_SERVER
    assert rec["ttl_s"] == 5400.0 and reload_env.killed == [400]

    plain = _reload_args(diag_env_override=[], experiment_id=None)
    assert stack_commands.cmd_reload(plain) == 0
    assert "diag_env_override" not in reload_env.started[-1]
    assert dov.read_record("architect_critic", reload_env.log_dir) is None
    hist = [json.loads(line) for line in
            dov.history_path("architect_critic", reload_env.log_dir).read_text().splitlines()]
    assert hist[-1]["record"]["experiment_id"] == "UFH14-A7"
    assert hist[-1]["readback_after"]["8083"]["pid"] == 502
    assert hist[-1]["readback_after"]["8083"]["env"][KEY] is None


def test_plain_reload_without_a_record_writes_nothing(reload_env):
    assert stack_commands.cmd_reload(_reload_args(diag_env_override=[], experiment_id=None)) == 0
    assert not (reload_env.log_dir / "env_overrides").exists()


# ------------------------------------------------------------------ the env reaching Popen


def test_diag_key_reaches_the_popen_env(harness):  # noqa: F811
    info = osk.start_server(8083, ["architect_critic"], harness.registry, diag_env_override={KEY: "1"})
    assert info is not None
    env = harness.captured[-1]
    assert env[KEY] == "1"
    assert env["OMP_WAIT_POLICY"] == _CANONICAL_OMP_ENV["OMP_WAIT_POLICY"]   # declared env intact


def test_plain_launch_strips_an_ambient_diag_key(harness, monkeypatch):  # noqa: F811
    monkeypatch.setenv(KEY, "1")
    assert osk.start_server(8083, ["architect_critic"], harness.registry) is not None
    assert KEY not in harness.captured[-1]


@pytest.mark.parametrize("mode", [{"vision_mode": True}, {"eval_batch_frontdoor_mode": True},
                                  {"gpu_shadow_lane_mode": True}])
def test_every_non_embedder_branch_applies_it(harness, mode):  # noqa: F811
    assert osk.start_server(1, ["worker_vision"], harness.registry, diag_env_override={KEY: "1"}, **mode)
    assert harness.captured[-1][KEY] == "1"


def test_start_server_refuses_diag_override_for_embedders_or_unlisted_keys(harness):  # noqa: F811
    with pytest.raises(ValueError):
        osk.start_server(8090, ["embedder"], harness.registry, embedding_mode=True,
                         diag_env_override={KEY: "1"})
    with pytest.raises(ValueError):
        osk.start_server(8083, ["architect_critic"], harness.registry, diag_env_override={"GGML_IQK": "0"})
    assert harness.captured == []
