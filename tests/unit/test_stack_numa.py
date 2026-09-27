"""Tests for orchestrator stack NUMA topology + pinning helper."""

from __future__ import annotations


from unittest.mock import patch

from scripts.server.stack_numa import (
    MLOCK_ROLES,
    NUMA_CONFIG,
    NUMA_FULL,
    _numa_prefix,
)


def test_numa_prefix_taskset_only_when_no_policy() -> None:
    """`_numa_prefix` emits a bare taskset when the entry declares no policy.

    Rewritten 2026-07-30 against a SYNTHETIC entry. It previously asserted this
    using `frontdoor` instance 1, on the premise "frontdoor instance 1 is NUMA_Q0A
    with no numactl_policy". That premise is gone: after the topology correction
    every live role declares a memory policy, because an instance whose cpuset
    spans more than one NPS4 node and does not declare one gets its pages placed
    by whichever thread faults first (measured cost: up to 2.9x).

    So there is deliberately no policy-less role left to point at. Testing the
    helper against a fixture keeps the branch covered without requiring the live
    config to keep a defective entry alive purely so a unit test has something to
    assert — which is how the previous version of this file came to pin the
    defect in place.
    """
    synthetic = dict(NUMA_CONFIG)
    synthetic["_fixture_no_policy"] = {"instances": [("0-23,96-119", 9999, 24)]}
    with patch.dict("scripts.server.stack_numa.NUMA_CONFIG", synthetic, clear=True):
        assert _numa_prefix("_fixture_no_policy", instance_idx=0) == [
            "taskset",
            "-c",
            "0-23,96-119",
        ]


def test_every_live_role_declares_a_memory_policy() -> None:
    """The counterpart to the fixture above: no LIVE entry may omit a policy.

    This is the assertion the old test displaced. Node-aligned entries use
    `membind`; multi-node entries use `interleave`. Either satisfies it.
    """
    missing = [
        f"{role}:{port}"
        for role, cfg in NUMA_CONFIG.items()
        if not (cfg.get("numactl_policy") or cfg.get("numactl_policy_instances"))
        for _cpus, port, _t in cfg.get("instances", [])
    ]
    assert not missing, "instances with no memory policy: " + ", ".join(missing)


def test_numa_prefix_wraps_with_numactl_when_policy_present() -> None:
    """A role-level `numactl_policy` becomes `numactl --<policy> --` ahead of taskset.

    Retargeted 2026-08-04. This used to name `architect_general` as "the role
    with numactl_policy=interleave=all" and restate `["numactl",
    "--interleave=all", "--", "taskset", "-c", "0-95"]`. Both halves of that went
    stale together in the 2026-07-31 W1 cutover: the 122B moved to
    architect_critic (which now holds the full-machine interleave placement) and
    architect_general became a ROCm role whose HOST threads sit on the GPU lane
    (membind=3, taskset -c 184-191). Nothing was lost — the interleave placement
    followed the model. Both forms are asserted below, and both expectations are
    read out of NUMA_CONFIG rather than restated, so a genuine placement
    regression still fails here while a role reshuffle does not.
    """
    for role in ("architect_critic", "architect_general"):
        cfg = NUMA_CONFIG[role]
        cpus, _port, _threads = cfg["instances"][0]
        policy = cfg["numactl_policy"]
        assert _numa_prefix(role, instance_idx=0) == [
            "numactl",
            f"--{policy}",
            "--",
            "taskset",
            "-c",
            cpus,
        ]

    # The two live forms are genuinely different: a full-machine CPU role
    # interleaves across every node, a GPU host lane binds to the card's node.
    # 2026-09-27 ARCHITECT SWAP: architect_general = the full-machine CPU
    # Flash-Next (:8074), architect_critic = the MI210 27B's GPU host lane (:8083).
    assert NUMA_CONFIG["architect_general"]["numactl_policy"].startswith("interleave")
    assert NUMA_CONFIG["architect_critic"]["numactl_policy"].startswith("membind")
    assert NUMA_CONFIG["architect_critic"].get("gpu_host_lane") is True


def test_numa_prefix_returns_empty_for_unknown_role() -> None:
    assert _numa_prefix("nonexistent_role") == []


def test_numa_prefix_returns_empty_for_out_of_range_instance() -> None:
    # frontdoor has 5 instances (indices 0-4); instance 10 is out of range
    assert _numa_prefix("frontdoor", instance_idx=10) == []


def test_numa_prefix_defaults_to_instance_zero() -> None:
    assert _numa_prefix("frontdoor") == _numa_prefix("frontdoor", instance_idx=0)


def test_mlock_roles_derived_from_numa_config() -> None:
    """Every NUMA_CONFIG entry with mlock=True must show up in MLOCK_ROLES."""
    expected = {role for role, cfg in NUMA_CONFIG.items() if cfg.get("mlock")}
    assert MLOCK_ROLES == expected
    # Non-vacuity: the derivation above is trivially true over an empty set, so the
    # sanity arm must prove SOMETHING declares mlock. It used to name frontdoor and
    # worker_general as two literals; the 2026-09-22 cutover made worker_general an
    # alias with no NUMA_CONFIG entry at all, and the literal then asserted mlock on
    # a role that declares no topology. A count, not a name.
    assert MLOCK_ROLES, "no NUMA_CONFIG entry declares mlock — the check is vacuous"


def test_numa_config_schema_all_instances_are_three_tuples() -> None:
    """Every instance entry must be (cpu_list:str, port:int, threads:int)."""
    for role, cfg in NUMA_CONFIG.items():
        assert "instances" in cfg, f"{role} missing instances"
        for inst in cfg["instances"]:
            assert len(inst) == 3, f"{role}: instance {inst} is not 3-tuple"
            cpu_list, port, threads = inst
            assert isinstance(cpu_list, str), f"{role}: cpu_list not str"
            assert isinstance(port, int), f"{role}: port not int"
            assert isinstance(threads, int), f"{role}: threads not int"
            assert 1 <= threads <= 192


def test_numa_config_ports_are_unique_within_role() -> None:
    for role, cfg in NUMA_CONFIG.items():
        ports = [p for _, p, _ in cfg["instances"]]
        assert len(ports) == len(set(ports)), f"{role}: duplicate ports {ports}"


def test_frontdoor_first_instance_is_the_full_machine() -> None:
    """frontdoor[0] is the FULL instance: all 96 physical cores, interleaved.

    Replaces `test_numa_node0_constant_matches_first_frontdoor_instance`, which
    asserted `frontdoor[0] == NUMA_NODE0`. `NUMA_NODE0` is an NPS2-era name: it is
    `("0-47,96-143", 96)`, which on the live NPS4 topology spans TWO nodes and
    asks for 96 threads on 48 physical cores. Binding frontdoor's primary
    instance to it was the placement defect, measured at 10.83 vs 23.36 tok/s
    (tg128, n=10) against the canonical full-machine recipe.

    The old test did not merely describe that wiring, it REQUIRED it — correcting
    frontdoor[0] necessarily failed it. A guard whose passing condition is the
    defect cannot survive the fix, so it is replaced rather than adjusted.

    `NUMA_NODE0`/`NUMA_Q0A` still exist in stack_numa but are now referenced only
    by tests and comments; no production path reads them. They are deletion
    candidates once the remaining test fixtures stop naming them.
    """
    cpu_list, _, threads = NUMA_CONFIG["frontdoor"]["instances"][0]
    assert (cpu_list, threads) == NUMA_FULL
    assert NUMA_CONFIG["frontdoor"]["numactl_policy_instances"][0] == "interleave=all"
