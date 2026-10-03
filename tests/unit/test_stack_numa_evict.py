"""Unit tests for scripts/server/stack_numa_evict.py (INF-70/C7, ENABLED).

Pins four things:
  * the config gate — every CPU llama-server role declares 40, no GPU
    host-lane role does, and the loader refuses a GPU role even if it did;
  * the FORCING sizing (TARGET+2 whenever free < TARGET, nothing at/above,
    verify per node, a second pass on concurrent growth). MUTATION HARNESS:
    `NUMA_EVICT_TEST_FORM=weak pytest tests/unit/test_stack_numa_evict.py`
    swaps in the 2026-09-02 weak formula and every `@teeth` test must fail;
  * that the research helper is reused only when it IS the forcing form, and
    every failure mode degrades to a message rather than an exception;
  * the launch path: a CPU role's start_server runs the pre-evict before
    Popen and logs `[numa-placement] pid=<pid> ...` after health; a GPU role
    through the same branch runs no eviction.
No subprocess is ever really run — `subprocess.run` / `Popen` are patched.
"""

from __future__ import annotations

import os
import subprocess
import types

import pytest

from scripts.server import stack_numa_evict as ev

TARGET = 40
GIB_MB = 1024

# The EXACT role set is deliberately a decision list, not a derivation: adding
# or removing a NUMA role must force someone to decide its numa_pre_evict_gib.
# 2026-09-26: worker_general and ingest_long_context left this set with the
# operator-signed 2026-09-22 lineup cutover (orchestrator 860b0b2d), which made
# them ALIASES (of frontdoor's :8070 CPU fleet and architect_general's :8083
# GPU process) and deleted their NUMA_CONFIG entries — an alias launches no
# process, so it has no pre-evict decision of its own. The test below now
# cross-checks that claim against the registry (every `alias_of` row is absent
# from NUMA_CONFIG, and every host it names is present).
# 2026-09-27 ARCHITECT SWAP (operator-decided): the full-CPU Flash-Next process
# (:8074, pre-evicts 40 GiB) is architect_general; the MI210 27B GPU host lane
# (:8083, never evicts) is architect_critic.
# STACKCHG-DFLASH2-20261003: worker_vision moved GPU host lane -> cold CPU (NUMA_HALF_A): it is now a CPU
# llama-server role and pre-evicts like every other one.
CPU_LLAMA_SERVER_ROLES = {"frontdoor", "eval_batch_frontdoor", "architect_general", "worker_vision"}
GPU_HOST_LANE_ROLES = {"architect_critic"}


def _registry_alias_rows() -> dict[str, str]:
    """``{alias: host}`` for every registry server_mode row declaring alias_of."""
    from pathlib import Path

    import yaml

    registry = Path(__file__).resolve().parents[2] / "orchestration" / "model_registry.yaml"
    server_mode = yaml.safe_load(registry.read_text(encoding="utf-8"))["server_mode"]
    return {
        role: row["alias_of"]
        for role, row in server_mode.items()
        if isinstance(row, dict) and row.get("alias_of")
    }


def teeth(fn):
    """Marks a test that MUST fail under NUMA_EVICT_TEST_FORM=weak."""
    fn.teeth = True
    return fn


def _weak_plan(free_mb, target_gib, headroom_gib=2):
    return max(0, target_gib - free_mb // GIB_MB + 1)


@pytest.fixture(autouse=True)
def _maybe_mutate_to_weak_form(monkeypatch):
    if os.environ.get("NUMA_EVICT_TEST_FORM") == "weak":
        monkeypatch.setattr(ev, "plan_allocation_gib", _weak_plan)
    yield


class _FakeBox:
    """allocate-touch-release model: after evict(G) on a node with F free,
    free == max(F, G) — G <= F reclaims nothing, which is the weak form's bug."""

    def __init__(self, free_gib):
        self.free_mb = {n: int(g * GIB_MB) for n, g in free_gib.items()}
        self.calls: list[tuple[int, int]] = []

    def query(self):
        return dict(self.free_mb)

    def evict(self, node, gib):
        self.calls.append((node, gib))
        self.free_mb[node] = max(self.free_mb[node], gib * GIB_MB)
        return True


# --- the config gate ---------------------------------------------------------

@pytest.mark.parametrize(
    "cfg,expected",
    [
        (None, 0),
        ({}, 0),
        ({"instances": []}, 0),
        ({"numa_pre_evict_gib": 0}, 0),
        ({"numa_pre_evict_gib": -5}, 0),
        ({"numa_pre_evict_gib": "nonsense"}, 0),
        ({"numa_pre_evict_gib": None}, 0),
        ({"numa_pre_evict_gib": 40}, 40),
        ({"numa_pre_evict_gib": "40"}, 40),
        ({"numa_pre_evict_gib": 10_000}, ev.MAX_PRE_EVICT_GIB),
        ({"numa_pre_evict_gib": 40, "gpu_host_lane": True}, 0),
    ],
)
def test_pre_evict_gib_for_role(cfg, expected):
    assert ev.pre_evict_gib_for_role(cfg) == expected


def test_every_cpu_llama_server_role_is_enabled_and_no_gpu_role_is():
    """ENABLED 2026-09-03: the exact role set, so an added role is a decision."""
    from scripts.server.stack_numa import NUMA_CONFIG

    effective = {r: ev.pre_evict_gib_for_role(c) for r, c in NUMA_CONFIG.items()}
    gpu = {r for r, c in NUMA_CONFIG.items() if isinstance(c, dict) and c.get("gpu_host_lane")}
    assert gpu == GPU_HOST_LANE_ROLES
    assert set(NUMA_CONFIG) - gpu == CPU_LLAMA_SERVER_ROLES, (
        "a NUMA role was added or removed: decide its numa_pre_evict_gib and update this set"
    )
    for role in CPU_LLAMA_SERVER_ROLES:
        assert effective[role] == 40, f"{role} must pre-evict 40 GiB/node"
        assert NUMA_CONFIG[role].get("numa_pre_evict_gib") == 40
    for role in GPU_HOST_LANE_ROLES:
        assert effective[role] == 0, f"GPU host-lane role {role} must never pre-evict"
        assert "numa_pre_evict_gib" not in NUMA_CONFIG[role]

    # Aliases launch nothing: none may carry a NUMA entry (and so a pre-evict
    # decision) of its own, and each must ride a host this gate does cover.
    aliases = _registry_alias_rows()
    assert {"worker", "ingest_long_context"} <= set(aliases), aliases  # non-vacuity
    for alias, host in aliases.items():
        assert alias not in NUMA_CONFIG, f"alias {alias!r} has its own NUMA entry"
        assert host in CPU_LLAMA_SERVER_ROLES | GPU_HOST_LANE_ROLES, (alias, host)


def test_role_field_is_accepted_by_the_topology_allowlist():
    from scripts.server import stack_numa

    assert "numa_pre_evict_gib" in stack_numa._ROLE_FIELDS


# --- the FORCING sizing (mutation-tested) ------------------------------------

@teeth
def test_node_just_below_target_forces_target_plus_headroom():
    assert ev.plan_allocation_gib((TARGET - 1) * GIB_MB, TARGET) == TARGET + 2


@teeth
@pytest.mark.parametrize("free_mb", [TARGET * GIB_MB, TARGET * GIB_MB + 1])
def test_node_at_or_above_target_allocates_nothing(free_mb):
    assert ev.plan_allocation_gib(free_mb, TARGET) == 0


@teeth
@pytest.mark.parametrize("free_gib", [0, 10, 30, 39])
def test_forced_allocation_always_exceeds_what_is_free(free_gib):
    assert ev.plan_allocation_gib(free_gib * GIB_MB, TARGET) > free_gib


@teeth
def test_near_target_nodes_reach_target_in_one_pass():
    box = _FakeBox({0: 39, 1: 20, 2: 45, 3: 39.9})
    short, allocs = ev.run_eviction([0, 1, 2, 3], TARGET, query_free_mb=box.query, evict=box.evict)
    assert short == []
    assert allocs == [(1, 0, 42), (1, 1, 42), (1, 3, 42)]
    assert box.free_mb[2] == 45 * GIB_MB


@teeth
def test_concurrent_cache_growth_after_pass_one_triggers_pass_two():
    box = _FakeBox({0: 39, 1: 39, 2: 39, 3: 39})
    n = {"q": 0}

    def query():
        n["q"] += 1
        if n["q"] == 2:  # right after pass 1: a writer refills node 3
            box.free_mb[3] = 35 * GIB_MB
        return box.query()

    short, allocs = ev.run_eviction([0, 1, 2, 3], TARGET, query_free_mb=query, evict=box.evict)
    assert short == []
    assert [a for a in allocs if a[0] == 2] == [(2, 3, 42)]


def test_gives_up_after_passes_and_reports_short():
    box = _FakeBox({0: 39, 1: 50})
    box.evict = lambda node, gib: box.calls.append((node, gib)) or True  # frees nothing
    short, allocs = ev.run_eviction([0, 1], TARGET, query_free_mb=box.query, evict=box.evict, passes=2)
    assert short == [0]
    assert allocs == [(1, 0, 42), (2, 0, 42)]


def test_parse_free_mb():
    assert ev.parse_free_mb("node 0 free: 23001 MB\nnode 1 free: 2649 MB\nnode 0 size: 1 MB\n") == {0: 23001, 1: 2649}


# --- pre_evict_nodes: which implementation, and never raising ----------------

def _four_nodes(monkeypatch):
    monkeypatch.setattr(ev, "_node_ids", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/numactl")


def test_zero_target_is_a_noop(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("pre_evict_nodes(0) must not spawn anything")

    monkeypatch.setattr(ev.subprocess, "run", boom)
    ok, msg = ev.pre_evict_nodes(0)
    assert ok is True and "disabled" in msg


def test_missing_numactl_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is False and "numactl" in msg


def test_single_node_host_is_a_noop(monkeypatch):
    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/numactl")
    monkeypatch.setattr(ev, "_node_ids", lambda: [0])
    monkeypatch.setattr(ev.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no spawn")))
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is True and "single-node" in msg


def test_research_helper_probe_distinguishes_forcing_from_weak(tmp_path):
    forcing = tmp_path / "forcing.py"
    forcing.write_text("def plan_allocation_gib(free_mb, target_gib, headroom_gib=2):\n    return 0\n")
    weak = tmp_path / "weak.py"
    weak.write_text("need = args.target_gib - before[node] // 1024 + 1\n")
    assert ev.research_helper_is_forcing(str(forcing)) is True
    assert ev.research_helper_is_forcing(str(weak)) is False
    assert ev.research_helper_is_forcing(str(tmp_path / "absent.py")) is False


def test_uses_the_research_helper_only_when_it_is_the_forcing_form(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev, "research_helper_is_forcing", lambda path=None: True)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "  OK: every requested node is at or above target.\n", "")

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is True
    assert seen["cmd"] == ["python3", ev.RESEARCH_EVICT, "--target-gib", "40", "--passes", "2"]
    assert "forcing form" in msg


@teeth
def test_weak_research_helper_on_disk_is_bypassed_for_inline_forcing(monkeypatch):
    """The shared research clone can lag origin/main: the stack must not run
    the weak helper just because a file exists at that path."""
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev, "research_helper_is_forcing", lambda path=None: False)
    monkeypatch.setattr(ev.os.path, "isfile", lambda p: p == ev.RESEARCH_EVICT)
    box = _FakeBox({0: 39, 1: 39, 2: 45, 3: 10})
    monkeypatch.setattr(ev, "_query_free_mb", box.query)
    monkeypatch.setattr(ev, "_touch_node", lambda node, gib, timeout_s=0: box.evict(node, gib))

    def never(cmd, **kw):  # pragma: no cover
        raise AssertionError(f"research helper must not be spawned: {cmd}")

    monkeypatch.setattr(ev.subprocess, "run", never)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is True, msg
    assert "inline forcing form" in msg and "weak form" in msg
    assert box.calls == [(0, 42), (1, 42), (3, 42)]
    assert all(v >= 40 * GIB_MB for v in box.free_mb.values())


def test_inline_form_reports_short_nodes_without_raising(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev, "research_helper_is_forcing", lambda path=None: False)
    monkeypatch.setattr(ev.os.path, "isfile", lambda p: False)
    monkeypatch.setattr(ev, "_query_free_mb", lambda: {0: 39 * GIB_MB, 1: 50 * GIB_MB, 2: 50 * GIB_MB, 3: 50 * GIB_MB})
    monkeypatch.setattr(ev, "_touch_node", lambda node, gib, timeout_s=0: True)
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is False
    assert "still below 40 GiB" in msg and "[0]" in msg and "absent" in msg


def test_inline_form_query_failure_degrades_to_message(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev, "research_helper_is_forcing", lambda path=None: False)
    monkeypatch.setattr(ev, "_query_free_mb", lambda: (_ for _ in ()).throw(OSError("numactl exploded")))
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is False and "OSError" in msg


def test_touch_child_source_faults_every_page():
    assert "np.empty" in ev._TOUCH_CHILD and "b[:] = 1" in ev._TOUCH_CHILD
    assert "mmap.MAP_ANONYMOUS" in ev._TOUCH_CHILD and "range(0, n, 4096)" in ev._TOUCH_CHILD


def test_touch_node_binds_the_node(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    assert ev._touch_node(2, 42) is True
    assert seen["cmd"][:2] == ["numactl", "--membind=2"] and seen["cmd"][-1] == "42"


def test_research_nonzero_exit_degrades_but_does_not_raise(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev, "research_helper_is_forcing", lambda path=None: True)
    monkeypatch.setattr(ev.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "node 2 short\n", ""))
    ok, msg = ev.pre_evict_nodes(40)
    assert ok is False and "exit 1" in msg


def test_research_timeout_is_caught(monkeypatch):
    _four_nodes(monkeypatch)
    monkeypatch.setattr(ev, "research_helper_is_forcing", lambda path=None: True)

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))

    monkeypatch.setattr(ev.subprocess, "run", slow)
    ok, msg = ev.pre_evict_nodes(40, timeout_s=7)
    assert ok is False and "timed out after 7s" in msg


# --- the placement fold ------------------------------------------------------

def _numa_maps_line(addr, per_node_mb, page_kb=4):
    pages = {n: int(mb * 1024 / page_kb) for n, mb in per_node_mb.items()}
    return (f"{addr} interleave:0-3 anon=1 dirty=1 "
            + " ".join(f"N{n}={p}" for n, p in sorted(pages.items()))
            + f" kernelpagesize_kB={page_kb}")


def test_summary_reports_the_measured_skew():
    text = _numa_maps_line("7f0000000000", {0: 57.7 * 1024, 1: 10.7 * 1024, 2: 8.0 * 1024, 3: 17.7 * 1024})
    out = ev.summarize_numa_maps(text)
    assert "max=node0@61." in out and "(even=25%)" in out


def test_summary_reports_clean_placement():
    text = _numa_maps_line("7f0000000000", {0: 24_000, 1: 24_000, 2: 24_000, 3: 24_000})
    assert "max=node0@25.0%" in ev.summarize_numa_maps(text)


def test_huge_pages_are_not_understated():
    small = _numa_maps_line("7f0000000000", {0: 2048}, page_kb=4)
    huge = _numa_maps_line("7f0000000000", {0: 2048}, page_kb=2048)
    assert ev.summarize_numa_maps(small).split()[0] == ev.summarize_numa_maps(huge).split()[0]


def test_summary_handles_garbage_and_empty():
    assert ev.summarize_numa_maps("") == "no NUMA-resident mappings"
    assert ev.summarize_numa_maps("garbage N=x Nq=3\n") == "no NUMA-resident mappings"


def test_placement_summary_on_a_missing_pid_does_not_raise():
    assert "unreadable" in ev.placement_summary(2**22 + 12345)


def test_placement_summary_reads_this_process():
    out = ev.placement_summary(os.getpid())
    assert "total=" in out and "max=node" in out


# --- the launch path: start_server ------------------------------------------

def _drive_start_server(monkeypatch, tmp_path, role: str):
    """Run start_server's default llama-server branch with every side effect
    faked; return (pre_evict calls, printed lines)."""
    from scripts.server import orchestrator_stack as osk

    calls: list[int] = []
    monkeypatch.setattr(osk, "pre_evict_nodes", lambda gib, **kw: calls.append(gib) or (True, f"faked {gib}"))
    monkeypatch.setattr(osk, "placement_summary", lambda pid: f"n0=100MB n1=100MB n2=100MB n3=100MB total=400MB max=node0@25.0% (even=25%) [pid {pid}]")
    monkeypatch.setattr(osk, "_bench_guarded_numa_prefix", lambda *a, **k: [])
    monkeypatch.setattr(osk, "build_server_command", lambda *a, **k: ["llama-server", "--port", "1"])
    monkeypatch.setattr(osk, "build_launch_env", lambda role, env: dict(env))
    monkeypatch.setattr(osk, "_stack_prior_runtime_overrides", lambda role: (None, []))
    monkeypatch.setattr(osk, "_apply_runtime_requirements_env", lambda env, **kw: None)
    monkeypatch.setattr(osk, "_write_llama_marker", lambda *a, **k: None)
    monkeypatch.setattr(osk, "wait_for_health", lambda port, timeout=0: True)
    monkeypatch.setattr(osk, "LOG_DIR", tmp_path)
    monkeypatch.setattr(osk.subprocess, "Popen", lambda *a, **k: types.SimpleNamespace(pid=4242))

    model = types.SimpleNamespace(name="fake-model", full_path="/nonexistent/fake.gguf")
    registry = types.SimpleNamespace(get_role=lambda r: types.SimpleNamespace(model=model))
    info = osk.start_server(1, [role], registry)
    assert info is not None and info.pid == 4242
    return calls


def test_cpu_role_launch_pre_evicts_then_logs_placement(monkeypatch, tmp_path, capsys):
    calls = _drive_start_server(monkeypatch, tmp_path, "frontdoor")
    out = capsys.readouterr().out
    assert calls == [40]
    assert "[numa-pre-evict] forcing 40 GiB free per node" in out
    assert "[numa-placement] pid=4242 " in out and "max=node0@25.0%" in out
    # ordering: eviction is logged before the PID (i.e. before Popen), placement after health
    assert out.index("[numa-pre-evict]") < out.index("PID: 4242") < out.index("[numa-placement]")


def test_gpu_role_launch_runs_no_eviction_but_still_logs_placement(monkeypatch, tmp_path, capsys):
    """The MI210 27B (architect_critic since the 2026-09-27 ARCHITECT SWAP) takes
    the same default branch as the CPU roles; it is a gpu_host_lane role and must
    never evict (VRAM weights, lane on node 3)."""
    calls = _drive_start_server(monkeypatch, tmp_path, "architect_critic")
    out = capsys.readouterr().out
    assert calls == []
    assert "[numa-pre-evict]" not in out
    assert "[numa-placement] pid=4242 " in out


@pytest.mark.parametrize("role", sorted(CPU_LLAMA_SERVER_ROLES))
def test_every_cpu_role_launch_pre_evicts(monkeypatch, tmp_path, role):
    assert _drive_start_server(monkeypatch, tmp_path, role) == [40]
