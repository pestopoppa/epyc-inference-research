"""Per-instance thread-count resolution for the orchestrator stack launcher.

Regression test for the 2026-05-24 launcher bug: `_resolve_thread_count` used
to always return `NUMA_CONFIG[role]["instances"][0][2]` regardless of which
instance was being launched. Quarters got `-t 96` instead of `-t 48`. This
test asserts that each instance's thread count matches its NUMA_CONFIG entry.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = ROOT / "scripts" / "server"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SERVER_DIR))


orchestrator_stack = importlib.import_module("orchestrator_stack")
stack_numa = importlib.import_module("stack_numa")


def _instances(role: str) -> list[tuple[str, int, int]]:
    return stack_numa.NUMA_CONFIG.get(role, {}).get("instances", [])


def _registry_server_mode() -> dict:
    return yaml.safe_load((ROOT / "orchestration" / "model_registry.yaml").read_text())[
        "server_mode"
    ]


def _host_of(role: str) -> str:
    """The registry row whose process serves ``role`` (shared_with / alias_of).

    2026-09-26: the operator-signed 2026-09-22 lineup cutover (orchestrator
    860b0b2d) made worker_general an alias of frontdoor's :8070 CPU fleet and
    ingest_long_context an alias of architect_general's :8083 GPU process, and
    deleted both NUMA_CONFIG entries. Aliases launch no process, so the thread
    count that matters is their HOST's; it is read from the registry here.
    """
    server_mode = _registry_server_mode()
    row = server_mode.get(role) or {}
    if row.get("alias_of"):
        return row["alias_of"]
    for host, host_row in server_mode.items():
        if host_row.get("alias_of"):
            continue
        if host == role or role in (host_row.get("shared_with") or []):
            return host
    raise AssertionError(f"registry declares no host for {role!r}")


def _assert_per_instance(role: str) -> int:
    instances = _instances(role)
    for idx, (_cpus, _port, expected_threads) in enumerate(instances):
        resolved = orchestrator_stack._resolve_thread_count(role, idx)
        assert int(resolved) == expected_threads, (
            f"{role} instance[{idx}] expected -t {expected_threads}, got {resolved}"
        )
    return len(instances)


def test_frontdoor_per_instance_thread_count() -> None:
    instances = _instances("frontdoor")
    assert len(instances) >= 2, "frontdoor should have a full instance + at least 1 quarter"
    for idx, (_cpus, _port, expected_threads) in enumerate(instances):
        resolved = orchestrator_stack._resolve_thread_count("frontdoor", idx)
        assert int(resolved) == expected_threads, (
            f"frontdoor instance[{idx}] expected -t {expected_threads}, got {resolved}"
        )


def test_worker_general_per_instance_thread_count() -> None:
    """worker_general previously had a manual port-matching workaround; with the
    generic fix it should give identical results without the workaround.

    Since the 2026-09-22 cutover worker_general is an alias with no NUMA entry
    of its own: the process launched for it is its registry host's (frontdoor's
    full + halves), so that host's per-instance counts are what must resolve."""
    assert "worker_general" not in stack_numa.NUMA_CONFIG, (
        "worker_general regained a NUMA entry — it is a process again; test it directly"
    )
    host = _host_of("worker_general")
    assert _assert_per_instance(host) >= 2, f"{host} should be full + at least 1 sibling"


def test_single_instance_roles_use_first_entry() -> None:
    """Roles with only one instance (architect_general only post-2026-05-24 fix)
    should resolve to that instance's thread count regardless of numa_instance
    argument (defensive fallback)."""
    for role in ("architect_general",):
        instances = _instances(role)
        if not instances:
            continue
        expected = instances[0][2]
        assert int(orchestrator_stack._resolve_thread_count(role)) == expected
        assert int(orchestrator_stack._resolve_thread_count(role, 0)) == expected
        # out-of-range falls back defensively
        assert int(orchestrator_stack._resolve_thread_count(role, 99)) == expected


def test_quartered_roles_per_instance_thread_count() -> None:
    """Per-instance thread count must resolve correctly for the quartered roles.

    Scope corrected 2026-07-30: this previously asserted `vision_escalation`
    had "full + at least 1 quarter post-Phase-1b" and had been failing against
    the live config, which gives it a SINGLE instance on `72-95,168-191` @24t
    (NUMA_Q1B, = NPS4 node3). The Phase-1b claim in the old docstring does not
    match `stack_numa.NUMA_CONFIG`; the config is the source of truth, so the
    role moved to its own shape assertion below.

    Re-derived 2026-09-26: this named ``ingest_long_context`` as the quartered
    role. The 2026-09-22 cutover made it an alias of architect_general's
    single-instance :8083 GPU process (no NUMA entry of its own), so the set of
    multi-instance roles is now DERIVED from NUMA_CONFIG, and ingest's host is
    checked to resolve its own (single) instance's count.
    """
    multi = sorted(r for r, c in stack_numa.NUMA_CONFIG.items() if len(c["instances"]) >= 2)
    assert multi, "no multi-instance role in NUMA_CONFIG — nothing per-instance to check"
    for role in multi:
        assert _assert_per_instance(role) >= 2

    assert "ingest_long_context" not in stack_numa.NUMA_CONFIG
    ingest_host = _host_of("ingest_long_context")
    assert _assert_per_instance(ingest_host) >= 1, ingest_host


def test_vision_escalation_stays_single_instance() -> None:
    """`vision_escalation` is served by ONE instance — worker_vision's :8086.

    Rewritten 2026-08-04. It used to assert its own process on :8087 @24t on
    NUMA_Q1B. The 2026-08-01 W1 cutover deleted that entry, with the reason
    committed in stack_topology.yaml: worker_vision and vision_escalation are ONE
    MI210 server on :8086 serving both role names, bound by
    `server_mode.worker_vision.shared_with`. Two processes would also mean two
    loads of one 17 GB VL GGUF. So the coverage is not deleted, it is
    re-pointed: the role must have NO instance of its own (it must not silently
    become a process again) and must resolve to its host's single instance.
    Every expectation is read from the registry / topology, not restated.
    """
    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]

    # Master says vision_escalation rides worker_vision's process.
    assert "vision_escalation" in registry["worker_vision"]["shared_with"]

    assert _instances("vision_escalation") == [], (
        "vision_escalation must not have a NUMA_CONFIG instance of its own — it is "
        "an alias on worker_vision's server, not a process"
    )
    assert "vision_escalation" not in stack_numa.NUMA_CONFIG

    host = _instances("worker_vision")
    assert len(host) == 1, "the process serving both vision roles is single-instance"
    assert orchestrator_stack.PORT_MAP["vision_escalation"] == host[0][1]
    assert (
        orchestrator_stack.PORT_MAP["vision_escalation"]
        == orchestrator_stack.PORT_MAP["worker_vision"]
        == registry["worker_vision"]["port"]
    )
    assert "vision_escalation" in orchestrator_stack.ROLE_LAUNCH_META["worker_vision"][
        "shared_with_first_n"
    ]


def test_worker_vision_stays_single_instance() -> None:
    """`worker_vision` is single-instance on the shared GPU host lane.

    Rewritten 2026-08-04. The old docstring's premise — "Phase 0.5 bench showed
    Qwen2.5-VL-7B is flat between 24t/48t" — retired with the model: the W1
    cutover moved the role to Qwen3-VL-30B-A3B Q4_K_M on MI210 (device ROCm0,
    ngl all, MMMU-250 63.6% vs 52.4%, +11.2 pp, paired exact McNemar p=0.0011),
    so its host threads now sit on GPU_HOST_LANE (184-191 @8t, the SMT siblings
    of physical 88-95) instead of a 24-thread NPS4 quarter. Host threads on a
    fully-offloaded role serve tokenization/dispatch; applying a CPU-inference
    thread count to it would be the defect. The shape is DERIVED from
    stack_numa's shape table so a lane change fails here instead of a literal
    going stale.
    """
    registry = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text()
    )["server_mode"]
    # STACKCHG-DFLASH2-20261003: COLD CPU role while the MI210 carries DFlash2 (was ROCm0).
    assert registry["worker_vision"]["device"] == "none"

    instances = _instances("worker_vision")
    assert len(instances) == 1, "worker_vision should be single-instance (one GPU server)"
    cpus, port, threads = instances[0]

    assert port == registry["worker_vision"]["port"]
    # STACKCHG-DFLASH2-20261003: one CPU half (NPS4 nodes 0,1), GPU-disjoint from the 27B's lane on node 3.
    assert stack_numa.NUMA_INSTANCE_SHAPE_CLASSES["worker_vision"] == ("half",)
    assert (cpus, threads) == stack_numa.NUMA_HALF_A
    assert "gpu_host_lane" not in stack_numa.NUMA_CONFIG["worker_vision"]
    assert _nodes_spanned(cpus) == {0, 1}, "NUMA_HALF_A spans exactly NPS4 nodes 0 and 1"
    assert stack_numa.NUMA_CONFIG["worker_vision"]["numactl_policy"] == "interleave=0,1"


# Live NPS4 topology, verified 2026-07-30 via `numactl --hardware`.
NPS4_NODES = {
    0: "0-23,96-119",
    1: "24-47,120-143",
    2: "48-71,144-167",
    3: "72-95,168-191",
}


def _parse_cpuset(cpuset: str) -> set[int]:
    cpus: set[int] = set()
    for part in cpuset.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _nodes_spanned(cpuset: str) -> set[int]:
    cpus = _parse_cpuset(cpuset)
    return {n for n, cs in NPS4_NODES.items() if cpus & _parse_cpuset(cs)}


# 2026-08-03: the `xfail(strict=True)` that stood here is GONE, as its own reason
# string instructed. It documented the NUMA placement defect (frontdoor :8070 and
# ingest_long_context :8085 straddling NPS4 nodes with no memory policy, measured
# 2.16x / 1.85x) and was strict precisely so that the fix landing would fail the
# suite instead of letting the marker outlive the defect. That is what happened:
# every straddling instance in NUMA_CONFIG now carries a policy —
#   frontdoor           :8070/:8080/:8180 -> {0: interleave=all, 1: interleave=0,1, 2: interleave=2,3}
#   ingest_long_context :8085/:8185/:8285 -> same per-instance table
#   worker_general      :8072/:8082/:8182 -> same per-instance table
#   architect_critic    :8074             -> interleave=all
#   eval_batch_frontdoor:18070            -> interleave=0,1
# so `offenders` is empty. The assertion below is unchanged and still has teeth: a
# new straddling instance added without a policy fails it. (2026-09-26: the
# ingest_long_context and worker_general rows above are historical — the
# 2026-09-22 cutover made both aliases and deleted their NUMA entries.)
def test_straddling_cpusets_declare_a_numa_policy() -> None:
    """Any instance whose cpuset spans more than one NPS4 node must declare a
    memory policy.

    Replaces `test_frontdoor_full_uses_numa_node0_not_full`, removed 2026-07-30.
    That test asserted `full[0] == "0-47,96-143"` AND that frontdoor carries no
    `numactl_policy` — so it did not merely document the defect, it *forbade the
    fix*: adding `interleave=all` to frontdoor would have failed it. Its stated
    authority was an April 2026-04-17 head-to-head that is invalid twice over —
    it predates the 2026-04-24 NPS4 reboot (when `0-47,96-143` genuinely was one
    node), and its source CSV records `spec == "baseline"`.

    Without a policy, every weight page lands on whichever node faults it first
    and the rest of the thread team reads cross-node for the model's lifetime.
    `numactl --interleave` binds at FIRST TOUCH only, so a warm-cache A/B cannot
    detect this — which is exactly how the 1.7% result that justified the old
    test was produced.
    """
    offenders: list[str] = []
    for role, cfg in stack_numa.NUMA_CONFIG.items():
        policy = cfg.get("numactl_policy") or cfg.get("numactl_policy_instances")
        if policy:
            continue
        for cpuset, port, _threads in cfg.get("instances", []):
            spanned = _nodes_spanned(cpuset)
            if len(spanned) > 1:
                offenders.append(
                    f"{role}:{port} cpuset={cpuset!r} spans NPS4 nodes "
                    f"{sorted(spanned)} with no numactl policy"
                )
    assert not offenders, "straddling cpuset without a memory policy:\n  " + "\n  ".join(
        offenders
    )


def test_unknown_role_falls_back_to_96() -> None:
    assert orchestrator_stack._resolve_thread_count("not_a_real_role") == "96"
    assert orchestrator_stack._resolve_thread_count("not_a_real_role", 3) == "96"


def test_pre_fix_quarter_bug_is_fixed() -> None:
    """The exact regression: pre-fix, frontdoor quarter 1 (port 8080) reported
    96 threads when its NUMA_CONFIG entry specifies 48. Post-fix it must
    report 48."""
    resolved = orchestrator_stack._resolve_thread_count("frontdoor", 1)
    assert int(resolved) == 48, (
        f"frontdoor quarter 0 should be -t 48 (NPS4 single-quarter spec), "
        f"got {resolved} — this is the 2026-05-24 regression returning"
    )
