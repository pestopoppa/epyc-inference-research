"""Tests for the gpu_shadow_lane guarded preflight probe (pure logic only).

The gather layer (rocm-smi / /proc / subprocess / hashing) is not exercised;
tests feed synthetic PreflightFacts into the pure evaluators, mirroring the
eval_batch_serving probe's testing style. No process is started or signalled.

P2-6 punch-list coverage: the P1-1 overlap taxonomy (unpinned / static-co-tenant
/ unexplained), the P1-2 SMT-sibling fold (cpu N <-> N+96), the P2-1 port +
tenant-sha checks, the P2-7 card-0 pin, and the P2-8 expected-PID allowlist.
"""

from __future__ import annotations

import pytest

from scripts.server import gpu_shadow_lane_preflight as probe
from scripts.server.gpu_shadow_lane import LANE_BINARY_COMMIT, LANE_BINARY_VERSION

STOCK_SHA = "5927dc06c2b19f732fb6e2a6546dff4c130b552f2ab5f91feb3daafe43897b2a"


# ---------------------------------------------------------------------------
# Topology expectations, DERIVED. 2026-08-04: several tests below restated live
# ports and role names (frontdoor's 8380 quarter, vision_escalation's :8087,
# architect_general's "0-95") that the 2026-07-31 quarter retirement and the
# 2026-08-01 W1 cutover moved. The mechanisms they cover — the SMT-sibling fold
# and the static-cpuset explanation — did not move. These helpers re-derive the
# expectations from stack_numa with their own cpuset arithmetic, so a topology
# reshuffle no longer looks like a probe regression while a real fold/explain
# regression still fails.
# ---------------------------------------------------------------------------
def _expand(spec: str) -> set[int]:
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return out


def _fold(cpus: set[int]) -> set[int]:
    """cpu N and N+96 are SMT siblings on this host — fold to physical cores."""
    return {cpu % 96 for cpu in cpus}


def _numa_instances() -> list[tuple[str, int, set[int]]]:
    """(role, port, cpus) for every declared production instance."""
    from scripts.server.stack_numa import NUMA_CONFIG

    return [
        (role, port, _expand(cpu_list))
        for role, cfg in NUMA_CONFIG.items()
        for cpu_list, port, _threads in cfg.get("instances", [])
    ]


def _expected_lane_overlaps(lane: str = "184-191") -> dict[str, list[int]]:
    lane_folded = _fold(_expand(lane))
    out: dict[str, list[int]] = {}
    for role, port, cpus in _numa_instances():
        if _fold(cpus) & lane_folded:
            out.setdefault(role, []).append(port)
    return out


def _expected_static_matches(mask: str) -> set[str]:
    """`role[port]` labels whose instance cpuset explains this affinity mask."""
    proc = _expand(mask)
    return {
        f"{role}[{port}]"
        for role, port, cpus in _numa_instances()
        if proc and proc <= cpus
    }


def make_facts(**overrides) -> probe.PreflightFacts:
    base = dict(
        flag_enabled=False,
        binary_exists=True,
        binary_version_output=f"version: {LANE_BINARY_VERSION} ({LANE_BINARY_COMMIT})",
        kfd_dev_present=True,
        kfd_topology_present=True,
        vram_total_gib=64.0,
        vram_used_gib=0.5,
        gpu_compute_pids=[],
        live_llama_processes=[],
        model_file_exists=True,
        static_smt_overlaps={},
        errors=[],
        lane_port_in_use=False,
        model_sha256=STOCK_SHA,
        model_sha256_skipped=False,
    )
    base.update(overrides)
    return probe.PreflightFacts(**base)


def make_plan(**overrides) -> probe.LanePlan:
    base = dict(
        tenant_id="qwen36_27b_stock_q8",
        model_path="/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf",
        model_vram_gib=26.7,
        np_slots=8,
        slot_context_tokens=8192,
        port=18100,
        host_cpuset="184-191",
        dynamic_budget_gib=27.0,
        np_ceiling=16,
        estimated_dynamic_gib=7.2,
        launch_argv=["taskset", "-c", "184-191"],
        expected_model_sha256=STOCK_SHA,
    )
    base.update(overrides)
    return probe.LanePlan(**base)


def evaluate(facts=None, plan=None, **kwargs):
    return probe.evaluate_preflight(
        facts if facts is not None else make_facts(),
        plan if plan is not None else make_plan(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------
class TestParsers:
    def test_parse_cpu_list_ranges_and_singles(self):
        assert probe.parse_cpu_list("184-191") == set(range(184, 192))
        assert probe.parse_cpu_list("72-95,168-191") == set(range(72, 96)) | set(
            range(168, 192)
        )
        assert probe.parse_cpu_list("3") == {3}
        assert probe.parse_cpu_list("") == set()

    def test_cpuset_overlap(self):
        assert probe.cpuset_overlap("184-191", "72-95,168-191") == set(range(184, 192))
        assert probe.cpuset_overlap("184-191", "0-47,96-143") == set()

    def test_smt_fold_maps_siblings_both_ways(self):
        # P1-2: cpu N <-> N+96 on the 192-cpu host.
        assert probe.smt_fold({88}) == {88, 184}
        assert probe.smt_fold({184}) == {88, 184}
        assert probe.smt_fold(set(range(184, 192))) == set(range(88, 96)) | set(
            range(184, 192)
        )

    def test_folded_overlap_sees_physical_co_tenancy(self):
        # architect_general/worker_general full masks (0-95) share PHYSICAL
        # cores 88-95 with the lane's SMT slice 184-191 despite disjoint masks.
        assert probe.cpuset_overlap("184-191", "0-95") == set()
        assert probe.folded_overlap("184-191", "0-95") == set(range(88, 96)) | set(
            range(184, 192)
        )
        assert probe.folded_overlap("184-191", "0-47,96-143") == set()

    def test_static_smt_overlap_includes_q1b_roles(self):
        """Every instance sharing the lane's physical cores is reported, with its port.

        Derived 2026-08-04. The old form asserted `vision_escalation[8087]`;
        vision_escalation has had no NUMA_CONFIG instance since the W1 cutover
        made it an alias on worker_vision's :8086 process, so a function that
        iterates NUMA_CONFIG instances cannot report it and no live vision
        process is missed — worker_vision[8086] IS reported. Asserting the whole
        derived mapping is stricter than the two spot-checks it replaces.

        STACKCHG-DFLASH2-20261003 (operator-signed): worker_vision left the MI210
        host lane (GPU_HOST_LANE 184-191) for NUMA_HALF_A (0-47,96-143), which is
        disjoint from the lane even after the SMT fold (184-191 folds onto 88-95),
        so the vision arm is now correctly ABSENT. The spot-checks pin the lane's
        real co-tenants instead: its own GPU host role and the half-B frontdoor
        instance.
        """
        expected = _expected_lane_overlaps("184-191")
        assert expected, "no instance overlaps the lane — check the topology"

        overlaps = probe.static_smt_overlap_roles("184-191")
        assert {role: sorted(ports) for role, ports in overlaps.items()} == {
            role: sorted(ports) for role, ports in expected.items()
        }
        assert 8083 in overlaps["architect_critic"]  # the lane's own MI210 27B
        assert 8180 in overlaps["frontdoor"]  # half-B (48-95,144-191) instance
        # Vision is off the lane: neither the host role nor its alias is reported.
        assert "worker_vision" not in overlaps
        assert "vision_escalation" not in overlaps

    def test_static_smt_overlap_folds_in_full_instances(self):
        """P1-2: sibling fold surfaces every declared full-host (0-95) instance as
        a co-tenant of the 184-191 lane, even though their masks are DISJOINT
        from it before folding.

        Derived 2026-09-24 (2026-09-22 lineup cutover): the old form hardcoded
        `architect_general[8083]` / `worker_general[8072]`. worker_general has
        had no NUMA_CONFIG instance of its own since that cutover — it folded
        into frontdoor's :8070 process (stack_templates/default.yaml) — and
        architect_general's own instance moved onto the lane's own cpuset
        (184-191 itself), so it now overlaps trivially without exercising the
        fold at all. The mechanism this test targets — a mask DISJOINT from the
        lane becoming a co-tenant only after the SMT-sibling fold — is exercised
        today by whichever roles hold the genuine full-host 0-95 mask. Derive
        them from NUMA_CONFIG rather than repin literals a second time.
        """
        full_host = _expand("0-95")
        full_instances = [
            (role, port) for role, port, cpus in _numa_instances() if cpus == full_host
        ]
        assert full_instances, (
            "no declared instance owns the full 0-95 mask — check the topology"
        )

        overlaps = probe.static_smt_overlap_roles("184-191")
        for role, port in full_instances:
            assert role in overlaps, (
                f"{role} (full 0-95 instance) should fold into the lane overlap"
            )
            assert port in overlaps[role]

    def test_parse_rocm_meminfo_card0(self):
        payload = {
            "card0": {
                "VRAM Total Memory (B)": str(64 * (1 << 30)),
                "VRAM Total Used Memory (B)": str(1 << 30),
            }
        }
        parsed = probe.parse_rocm_meminfo(payload)
        assert parsed is not None
        total, used = parsed
        assert total == pytest.approx(64.0)
        assert used == pytest.approx(1.0)

    def test_parse_rocm_meminfo_ignores_other_cards(self):
        # P2-7: pinned to card 0 — a card1-only payload must NOT be attested.
        payload = {
            "card1": {
                "VRAM Total Memory (B)": str(64 * (1 << 30)),
                "VRAM Total Used Memory (B)": str(1 << 30),
            }
        }
        assert probe.parse_rocm_meminfo(payload) is None

    def test_parse_rocm_meminfo_malformed(self):
        assert probe.parse_rocm_meminfo(None) is None
        assert probe.parse_rocm_meminfo({"card0": {"foo": "1"}}) is None

    def test_parse_rocm_pids_from_keys(self):
        payload = {"system": {"PID 12345": ["comm", "1", "2"], "PID678": "x"}}
        assert probe.parse_rocm_pids(payload) == [678, 12345]
        assert probe.parse_rocm_pids({"system": {}}) == []
        assert probe.parse_rocm_pids(None) == []

    def test_parse_rocm_pids_nested_values(self):
        payload = {"process": [{"pid": 42}, {"pid": "77"}]}
        assert probe.parse_rocm_pids(payload) == [42, 77]


# ---------------------------------------------------------------------------
# Overlap classification (P1-1 taxonomy + P1-2 fold)
# ---------------------------------------------------------------------------
class TestClassifyLiveOverlap:
    def test_disjoint_mask_is_none(self):
        cls, overlap, matches = probe.classify_live_overlap("0-47,96-143")
        assert cls == "none" and overlap == [] and matches == []

    def test_full_host_mask_is_unpinned(self):
        cls, overlap, _ = probe.classify_live_overlap("0-191")
        assert cls == "unpinned"
        assert set(overlap) == set(range(88, 96)) | set(range(184, 192))

    def test_q1b_quarter_is_static_co_tenant(self):
        """A mask contained in a declared instance cpuset is EXPLAINED, not blocked.

        Derived 2026-08-04: the `vision_escalation[...]` arm named a role that no
        longer has an instance (alias on worker_vision's :8086 since the W1
        cutover). The mask is a subset of the half-B cpuset, so the explaining
        matches are now the three half-B instances; asserting the full derived
        set is stricter than the two spot-checks and cannot go stale the same way.
        """
        expected = _expected_static_matches("72-95,168-191")
        assert expected, "no instance explains this mask — check the topology"

        cls, _overlap, matches = probe.classify_live_overlap("72-95,168-191")
        assert cls == "static-co-tenant"
        assert set(matches) == expected

    def test_full_0_95_instance_is_static_co_tenant_via_fold(self):
        """P1-2: a full-machine instance (0-95) — a mask literally DISJOINT from
        184-191 — is a physical co-tenant through siblings 88-95.

        Derived 2026-08-04: the `architect_general[...]` arm went stale when that
        role moved onto the GPU host lane; architect_critic now holds the 122B on
        the full machine. The fold is demonstrably intact — every full-machine
        instance is matched, and the folded overlap is still the 88-95 slice.
        """
        expected = _expected_static_matches("0-95")
        assert expected, "no instance explains a full-machine mask"

        cls, overlap, matches = probe.classify_live_overlap("0-95")
        assert cls == "static-co-tenant"
        assert set(overlap) >= set(range(88, 96))
        assert set(matches) == expected

    def test_unmatched_pinned_overlap_is_unexplained(self):
        cls, overlap, matches = probe.classify_live_overlap("90-100")
        assert cls == "unexplained"
        assert matches == []
        # Folded overlap with the lane slice: physical 90-95 + siblings 186-191.
        assert set(overlap) == set(range(90, 96)) | set(range(186, 192))

    def test_missing_mask_is_none(self):
        assert probe.classify_live_overlap(None) == ("none", [], [])


# ---------------------------------------------------------------------------
# Blocker evaluation
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_clean_plan_has_no_blockers(self):
        blockers, warnings, _infos = evaluate()
        assert blockers == []
        # flag-off is a warning pre-activation, never a blocker by default
        assert any("feature flag is off" in w for w in warnings)

    def test_require_enabled_blocks_when_flag_off(self):
        blockers, _, _ = evaluate(require_enabled=True)
        assert any("feature flag" in b for b in blockers)

    def test_binary_version_mismatch_blocks(self):
        facts = make_facts(binary_version_output="version: 9999 (deadbeef)")
        blockers, _, _ = evaluate(facts)
        assert any("version mismatch" in b for b in blockers)

    def test_binary_missing_blocks(self):
        blockers, _, _ = evaluate(make_facts(binary_exists=False))
        assert any("binary missing" in b for b in blockers)

    def test_kfd_missing_blocks(self):
        blockers, _, _ = evaluate(make_facts(kfd_dev_present=False))
        assert any("KFD" in b for b in blockers)

    def test_vram_unattestable_blocks(self):
        blockers, _, _ = evaluate(
            make_facts(vram_total_gib=None, vram_used_gib=None)
        )
        assert any("attest VRAM" in b for b in blockers)

    def test_vram_short_blocks(self):
        # 64 total, 40 used -> 24 free < 26.7 model + 7.2 dynamic
        blockers, _, _ = evaluate(make_facts(vram_used_gib=40.0))
        assert any("VRAM budget short" in b for b in blockers)

    def test_foreign_gpu_pids_block_by_default(self):
        blockers, _, _ = evaluate(make_facts(gpu_compute_pids=[4242]))
        assert any("foreign GPU compute PIDs" in b for b in blockers)

    def test_foreign_gpu_pids_downgrade_with_flag(self):
        blockers, warnings, _ = evaluate(
            make_facts(gpu_compute_pids=[4242]), allow_existing_gpu_pids=True
        )
        assert not any("foreign GPU" in b for b in blockers)
        assert any("foreign GPU" in w for w in warnings)

    def test_expected_gpu_pid_allowlist(self):
        # P2-8: allowlisted PIDs are informational; others still block.
        blockers, _, infos = evaluate(
            make_facts(gpu_compute_pids=[4242, 5555]),
            expected_gpu_pids={4242},
        )
        assert any("allowlisted" in i and "4242" in i for i in infos)
        assert any("5555" in b for b in blockers)
        assert not any("4242" in b for b in blockers)

    def test_expected_gpu_pid_allowlist_alone_is_clean(self):
        blockers, _, infos = evaluate(
            make_facts(gpu_compute_pids=[4242]), expected_gpu_pids={4242}
        )
        assert blockers == []
        assert any("allowlisted" in i for i in infos)

    # ── P1-1 overlap taxonomy ────────────────────────────────────────────
    def test_static_co_tenant_overlap_is_warning_not_blocker(self):
        facts = make_facts(
            live_llama_processes=[
                {"pid": 111, "cpus_allowed_list": "72-95,168-191"}
            ]
        )
        blockers, warnings, _ = evaluate(facts)
        assert blockers == []
        assert any("static-co-tenant" in w and "pid 111" in w for w in warnings)

    def test_folded_full_instance_is_warning_not_blocker(self):
        """P1-2: a full-machine (0-95) process is a WARNING, never a blocker.

        Derived 2026-08-04: the assertion named architect_general, which left the
        full machine for the GPU host lane in the W1 cutover. The property is
        unchanged — the process is explained by the static topology and therefore
        warns instead of blocking — and the warning must still name every role
        that explains it.
        """
        expected = _expected_static_matches("0-95")
        assert expected

        facts = make_facts(
            live_llama_processes=[{"pid": 222, "cpus_allowed_list": "0-95"}]
        )
        blockers, warnings, _ = evaluate(facts)
        assert blockers == []
        matching = [w for w in warnings if "pid 222" in w and "static-co-tenant" in w]
        assert len(matching) == 1
        for label in expected:
            assert label in matching[0]

    def test_unpinned_process_is_informational(self):
        facts = make_facts(
            live_llama_processes=[{"pid": 333, "cpus_allowed_list": "0-191"}]
        )
        blockers, warnings, infos = evaluate(facts)
        assert blockers == []
        assert not any("pid 333" in w for w in warnings)
        assert any("unpinned" in i and "pid 333" in i for i in infos)

    def test_unexplained_pinned_overlap_blocks(self):
        facts = make_facts(
            live_llama_processes=[{"pid": 444, "cpus_allowed_list": "90-100"}]
        )
        blockers, _, _ = evaluate(facts)
        assert any("UNEXPLAINED" in b and "pid 444" in b for b in blockers)

    def test_allow_smt_overlap_cannot_excuse_unexplained(self):
        # P1-1: the flag's effect is narrowed to the static-co-tenant class.
        facts = make_facts(
            live_llama_processes=[{"pid": 444, "cpus_allowed_list": "90-100"}]
        )
        blockers, _, _ = evaluate(facts, allow_smt_overlap=True)
        assert any("UNEXPLAINED" in b for b in blockers)

    def test_allow_smt_overlap_acknowledges_static_class(self):
        facts = make_facts(
            live_llama_processes=[
                {"pid": 111, "cpus_allowed_list": "72-95,168-191"}
            ]
        )
        blockers, warnings, infos = evaluate(facts, allow_smt_overlap=True)
        assert blockers == []
        assert not any("static-co-tenant" in w for w in warnings)
        assert any("acknowledged via --allow-smt-overlap" in i for i in infos)

    def test_non_overlapping_live_process_is_fine(self):
        facts = make_facts(
            live_llama_processes=[
                {"pid": 222, "cpus_allowed_list": "0-47,96-143"}
            ]
        )
        blockers, _, _ = evaluate(facts)
        assert blockers == []

    def test_static_overlap_summary_is_warning_only(self):
        facts = make_facts(static_smt_overlaps={"vision_escalation": [8087]})
        blockers, warnings, _ = evaluate(facts)
        assert blockers == []
        assert any("static NUMA_CONFIG co-tenants" in w for w in warnings)

    # ── P2-1 port + tenant identity ──────────────────────────────────────
    def test_lane_port_in_use_blocks(self):
        blockers, _, _ = evaluate(make_facts(lane_port_in_use=True))
        assert any("port 18100 is already in use" in b for b in blockers)

    def test_lane_port_unprobeable_warns(self):
        blockers, warnings, _ = evaluate(make_facts(lane_port_in_use=None))
        assert blockers == []
        assert any("could not be probed" in w for w in warnings)

    def test_sha256_mismatch_blocks(self):
        blockers, _, _ = evaluate(make_facts(model_sha256="deadbeef"))
        assert any("sha256 mismatch" in b for b in blockers)

    def test_sha256_match_is_clean(self):
        blockers, warnings, _ = evaluate()
        assert not any("sha256" in b for b in blockers)
        assert not any("sha256" in w for w in warnings)

    def test_sha256_skip_recorded_as_warning(self):
        blockers, warnings, _ = evaluate(
            make_facts(model_sha256=None, model_sha256_skipped=True)
        )
        assert blockers == []
        assert any("SKIPPED" in w for w in warnings)

    def test_sha256_unreadable_blocks(self):
        blockers, _, _ = evaluate(make_facts(model_sha256=None))
        assert any("could not be computed" in b for b in blockers)

    def test_sha256_unpinned_tenant_warns(self):
        blockers, warnings, _ = evaluate(
            make_facts(model_sha256=None),
            make_plan(expected_model_sha256=None),
        )
        assert blockers == []
        assert any("pins no sha256" in w for w in warnings)

    def test_missing_model_blocks(self):
        blockers, _, _ = evaluate(
            make_facts(model_file_exists=False, model_sha256=None)
        )
        assert any("model file missing" in b for b in blockers)

    def test_null_ceiling_blocks(self):
        blockers, _, _ = evaluate(plan=make_plan(np_ceiling=None))
        assert any("no validated operating point" in b for b in blockers)

    def test_np_above_ceiling_blocks(self):
        blockers, _, _ = evaluate(plan=make_plan(np_slots=32, np_ceiling=16))
        assert any("exceeds validated ceiling" in b for b in blockers)

    def test_gather_errors_propagate_as_blockers(self):
        blockers, _, _ = evaluate(make_facts(errors=["rocm-smi: not found"]))
        assert "rocm-smi: not found" in blockers


# ---------------------------------------------------------------------------
# CLI defaults (no execution beyond argparse)
# ---------------------------------------------------------------------------
class TestCli:
    def test_defaults_are_plan_only(self):
        args = probe.parse_args([])
        assert args.apply is False
        assert args.require_enabled is False
        assert args.allow_existing_gpu_pids is False
        assert args.allow_smt_overlap is False
        assert args.skip_tenant_hash is False
        assert args.expected_gpu_pid == []
        assert args.tenant == "qwen36_27b_stock_q8"
        assert args.budget_profile == "phase2_resident_set"
        assert args.np == 8
        assert args.slot_context == 8192
        assert args.port == 18100

    def test_expected_gpu_pid_repeatable(self):
        args = probe.parse_args(["--expected-gpu-pid", "42", "--expected-gpu-pid", "43"])
        assert args.expected_gpu_pid == [42, 43]

    def test_build_plan_resolves_ceiling_and_estimate(self):
        from scripts.server.gpu_shadow_lane import load_np_ceiling_policy
        from src.features import Features

        policy = load_np_ceiling_policy(feats=Features(gpu_shadow_lane=True))
        args = probe.parse_args(["--np", "8", "--slot-context", "32768"])
        plan = probe.build_plan(policy, args)
        assert plan.np_ceiling == 8
        assert plan.model_vram_gib == pytest.approx(26.7)
        assert plan.estimated_dynamic_gib == pytest.approx(19.2, abs=0.01)
        assert "-np" in plan.launch_argv
        # P2-1: the policy's pinned tenant hash rides into the plan.
        assert plan.expected_model_sha256 == STOCK_SHA

    def test_build_plan_rejects_unknown_budget_profile(self):
        from scripts.server.gpu_shadow_lane import load_np_ceiling_policy
        from src.features import Features

        policy = load_np_ceiling_policy(feats=Features(gpu_shadow_lane=True))
        args = probe.parse_args(["--budget-profile", "nope"])
        with pytest.raises(KeyError, match="budget profile"):
            probe.build_plan(policy, args)
