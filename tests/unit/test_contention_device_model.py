"""Artifact 1 + 2: the placement-feasibility model has a DEVICE dimension.

Rider: `handoffs/active/contention-model-device-and-load-axes-rider.md` §2, §4.

Before this, feasibility was computed purely from NUMA cpusets, so a role whose
weights are VRAM-resident under `-ngl` — whose cpuset exists only to give it
host threads for tokenising and sampling — was accounted IDENTICALLY to a CPU
decode holding the same cpuset. These tests pin down the two answers that got
wrong, in both directions:

  * FALSE EXCLUSION — a GPU lane overlapping a full CPU instance was called a
    conflict although its draw on those regions' DRAM bandwidth is ~nil.
  * UNMODELLED CONTENTION — two GPU roles sharing no cpuset at all still
    contend for VRAM capacity, which the model could not see.

The synthetic topologies below use `72-95,168-191` for the GPU lane rather than
the live `184-191`. That is deliberate and load-bearing: `184-191` is
SMT-siblings-only, so `parse_cpu_list` strips it to the empty region set and the
old cpuset-only model reached the right answer BY ACCIDENT. `72-95,168-191` —
the 24-thread node-3 quarter W1 briefly wired for `architect_general` — maps to
a real region `q3`, so it exercises the actual defect.
"""

from __future__ import annotations

import json

import pytest

from scripts.server import contention_matrix as cm
from src.scheduling.contention import (
    ContentionMatrix,
    Nway,
    PairDecision,
    Placement,
    TrafficClass,
    admit_set,
    claims_cpu_regions,
    placements_conflict,
    placements_overlap,
    seam_admit,
)
from src.scheduling.device_model import (
    DeviceClass,
    DeviceResolutionError,
    resolve_role_device,
    vram_fit,
)


# ── Synthetic declarations ───────────────────────────────────────────

def _prior(role: str, device, vram_gib=None, server_role=None) -> dict:
    """One compiled-priors role record, shaped like the real artifact."""
    record: dict = {
        "serving": {
            "server_role": server_role or role,
            "launch": {"runtime": {"flags": {"device": device}}},
        }
    }
    if vram_gib is not None:
        record["evidence"] = {"quality": [{"value": {"vram_gib": vram_gib}}]}
    return record


GPU_LANE = "72-95,168-191"  # region q3 — a REAL region, unlike 184-191
FULL = "0-95"  # {q0,q1,q2,q3}
HALF_A = "0-47,96-143"  # {q0,q1}
HALF_B = "48-95,144-191"  # {q2,q3}


@pytest.fixture
def gpu_plus_cpu_full():
    """A GPU host lane on q3 alongside a whole-machine CPU decode."""
    numa = {
        "architect_general": {
            "instances": [(GPU_LANE, 8083, 24)],
            "gpu_host_lane": True,
        },
        "worker_general": {"instances": [(FULL, 8072, 96)]},
    }
    priors = {
        "architect_general": _prior("architect_general", "ROCm0", 36.7),
        "worker_general": _prior("worker_general", None),
    }
    return numa, priors


@pytest.fixture
def two_cpu_overlapping():
    """Two CPU decodes that both want the whole machine."""
    numa = {
        "architect_critic": {"instances": [(FULL, 8074, 96)]},
        "worker_general": {"instances": [(FULL, 8072, 96)]},
    }
    priors = {
        "architect_critic": _prior("architect_critic", None),
        "worker_general": _prior("worker_general", None),
    }
    return numa, priors


def _two_gpu(vram_a: float, vram_b: float):
    """Two GPU roles on the SHARED host lane — no cpuset conflict by design,
    so the only thing that can exclude them is VRAM."""
    numa = {
        "architect_general": {"instances": [("184-191", 8083, 8)], "gpu_host_lane": True},
        "worker_vision": {"instances": [("184-191", 8086, 8)], "gpu_host_lane": True},
    }
    priors = {
        "architect_general": _prior("architect_general", "ROCm0", vram_a),
        "worker_vision": _prior("worker_vision", "ROCm0", vram_b),
    }
    return numa, priors


# Declared capacity used throughout: the MI210's 64 GiB, passed explicitly so
# the tests never depend on the host or on the declared artifact drifting.
CAPACITY = 64.0
HEADROOM = 2.0


def _assess(roleset, numa, priors, **kw):
    kw.setdefault("vram_capacity_gib", CAPACITY)
    kw.setdefault("vram_headroom_gib", HEADROOM)
    return cm.assess_feasibility(roleset, numa, priors=priors, **kw)


# ── (a) FALSE EXCLUSION — the headline fix ───────────────────────────

def test_gpu_lane_plus_full_cpu_instance_is_feasible(gpu_plus_cpu_full) -> None:
    """A GPU host lane and a whole-machine CPU instance CAN coexist.

    The GPU role's claim is its host cores, not the DRAM bandwidth of every
    region its cpuset spans. This is the operator's stated example, and it
    fails against the pre-device model.
    """
    numa, priors = gpu_plus_cpu_full
    verdict = _assess(("architect_general", "worker_general"), numa, priors)
    assert verdict.feasible, verdict.evidence
    assert verdict.reason == ""
    assert verdict.device_classes["architect_general"] == DeviceClass.GPU.value
    assert verdict.device_classes["worker_general"] == DeviceClass.CPU.value
    # The CPU instance keeps its whole-machine placement; it is not demoted.
    assert set(verdict.assignment["worker_general"]["regions"]) == {"q0", "q1", "q2", "q3"}


def test_enumerate_feasible_admits_gpu_plus_cpu_set(gpu_plus_cpu_full) -> None:
    """The same fix, through the enumeration entry point."""
    numa, priors = gpu_plus_cpu_full
    body = cm.enumerate_feasible(
        numa, priors=priors, vram_capacity_gib=CAPACITY, vram_headroom_gib=HEADROOM
    )
    assert body["summary"]["n_candidates"] == 1
    assert body["summary"]["n_excluded"] == 0
    assert body["candidate_sets"][0]["roles"] == ["architect_general", "worker_general"]


# ── (b) CPU exclusivity is UNCHANGED, and the reason is specific ─────

def test_two_overlapping_cpu_instances_conflict_with_specific_reason(
    two_cpu_overlapping,
) -> None:
    numa, priors = two_cpu_overlapping
    verdict = _assess(("architect_critic", "worker_general"), numa, priors)
    assert not verdict.feasible
    assert verdict.reason == "cpu_region_conflict"
    # Never a bare `topology_infeasible` that conflates CPU and VRAM exclusion.
    assert verdict.reason != "topology_infeasible"


def test_enumerate_marks_cpu_conflicts_specifically(two_cpu_overlapping) -> None:
    numa, priors = two_cpu_overlapping
    body = cm.enumerate_feasible(
        numa, priors=priors, vram_capacity_gib=CAPACITY, vram_headroom_gib=HEADROOM
    )
    assert body["summary"]["n_candidates"] == 0
    assert [e["reason"] for e in body["excluded_sets"]] == ["cpu_region_conflict"]


def test_disjoint_cpu_halves_stay_feasible() -> None:
    """Regression guard: the device axis must not loosen CPU exclusivity."""
    numa = {
        "frontdoor": {"instances": [(HALF_A, 8080, 48)]},
        "worker_general": {"instances": [(HALF_B, 8182, 48)]},
    }
    priors = {
        "frontdoor": _prior("frontdoor", None),
        "worker_general": _prior("worker_general", None),
    }
    verdict = _assess(("frontdoor", "worker_general"), numa, priors)
    assert verdict.feasible, verdict.evidence


# ── (c) UNMODELLED CONTENTION — VRAM is now a first-class capacity ───

def test_two_gpu_roles_that_fit_are_feasible() -> None:
    """The live pair: 36.7 + 20.56 GiB against a 64 GiB card."""
    numa, priors = _two_gpu(36.7, 21049 / 1024.0)
    verdict = _assess(("architect_general", "worker_vision"), numa, priors)
    assert verdict.feasible, verdict.evidence
    assert verdict.vram["required_gib"] == pytest.approx(57.26, abs=0.01)
    assert verdict.vram["budget_gib"] == pytest.approx(62.0, abs=0.01)


def test_two_gpu_roles_over_capacity_are_infeasible() -> None:
    """Sharing no cpuset is not enough — they still share the card."""
    numa, priors = _two_gpu(36.7, 30.0)
    verdict = _assess(("architect_general", "worker_vision"), numa, priors)
    assert not verdict.feasible
    assert verdict.reason == "vram_capacity_exceeded"
    assert verdict.vram["required_gib"] == pytest.approx(66.7, abs=0.01)


def test_enumerate_marks_vram_exhaustion_specifically() -> None:
    numa, priors = _two_gpu(36.7, 30.0)
    body = cm.enumerate_feasible(
        numa, priors=priors, vram_capacity_gib=CAPACITY, vram_headroom_gib=HEADROOM
    )
    assert [e["reason"] for e in body["excluded_sets"]] == ["vram_capacity_exceeded"]


def test_undeclared_gpu_vram_fails_closed() -> None:
    """A GPU role with no declared footprint cannot be shown to fit, so the
    hard constraint fails CLOSED — with its own reason, not the VRAM-exceeded
    one, because 'unknown' and 'too big' are different facts."""
    numa, priors = _two_gpu(36.7, 20.0)
    priors["worker_vision"].pop("evidence")
    verdict = _assess(("architect_general", "worker_vision"), numa, priors)
    assert not verdict.feasible
    assert verdict.reason == "vram_declaration_missing"


def test_vram_fit_collapses_alias_roles_onto_one_process() -> None:
    """`vision_escalation` is an alias on `worker_vision`'s :8086 process.
    Billing it twice would invent ~20.6 GiB of pressure that does not exist."""
    priors = {
        "worker_vision": _prior("worker_vision", "ROCm0", 20.56),
        "vision_escalation": _prior(
            "vision_escalation", "ROCm0", 20.56, server_role="worker_vision"
        ),
    }
    fit = vram_fit(
        ["worker_vision", "vision_escalation"],
        priors=priors,
        capacity_gib=CAPACITY,
        headroom_gib=HEADROOM,
    )
    assert fit.ok
    assert fit.required_gib == pytest.approx(20.56, abs=0.01)


# ── (d) Disagreement RAISES; it is never a vote ──────────────────────

def test_device_declaration_disagreement_raises() -> None:
    """`gpu_host_lane` says GPU, the compiled priors say no accelerator."""
    numa = {"worker_vision": {"instances": [("184-191", 8086, 8)], "gpu_host_lane": True}}
    priors = {"worker_vision": _prior("worker_vision", None)}
    with pytest.raises(DeviceResolutionError, match="DISAGREE"):
        resolve_role_device("worker_vision", numa_config=numa, priors=priors)


def test_device_declaration_disagreement_other_direction_raises() -> None:
    """Priors say ROCm0, NUMA_CONFIG carries no `gpu_host_lane`."""
    numa = {"architect_general": {"instances": [(GPU_LANE, 8083, 24)]}}
    priors = {"architect_general": _prior("architect_general", "ROCm0", 36.7)}
    with pytest.raises(DeviceResolutionError, match="DISAGREE"):
        resolve_role_device("architect_general", numa_config=numa, priors=priors)


def test_unknown_role_raises_rather_than_defaulting_to_cpu() -> None:
    with pytest.raises(DeviceResolutionError, match="Refusing to default to CPU"):
        resolve_role_device("nonexistent_role", numa_config={}, priors={})


def test_numa_config_only_role_resolves_from_gpu_host_lane() -> None:
    """The permitted fallback: NUMA_CONFIG alone, when priors have no record.
    Absence of the key inside a role that IS in NUMA_CONFIG means False — the
    same reading `stack_numa._assert_instance_invariants` already relies on."""
    numa = {"eval_batch_frontdoor": {"instances": [(HALF_A, 18070, 48)]}}
    rd = resolve_role_device("eval_batch_frontdoor", numa_config=numa, priors={})
    assert rd.device_class is DeviceClass.CPU
    assert rd.source == "numa_config"
    assert rd.corroborated is False


def test_agreeing_sources_are_marked_corroborated() -> None:
    numa = {"worker_vision": {"instances": [("184-191", 8086, 8)], "gpu_host_lane": True}}
    priors = {"worker_vision": _prior("worker_vision", "ROCm0", 20.56)}
    rd = resolve_role_device("worker_vision", numa_config=numa, priors=priors)
    assert rd.device_class is DeviceClass.GPU
    assert rd.device == "ROCm0"
    assert rd.corroborated is True


def test_unrecognised_device_token_raises() -> None:
    numa = {"worker_vision": {"instances": [("184-191", 8086, 8)], "gpu_host_lane": True}}
    priors = {"worker_vision": _prior("worker_vision", "quantum7", 20.56)}
    with pytest.raises(DeviceResolutionError, match="not a recognised device token"):
        resolve_role_device("worker_vision", numa_config=numa, priors=priors)


# ── (e) The LIVE topology resolves cleanly ───────────────────────────

def _empty_matrix() -> ContentionMatrix:
    """A matrix with nothing measured — so every verdict below comes from the
    device/region logic under test, not from a matrix cell."""
    return ContentionMatrix(
        version=1, measured_at="", host="", topology_hash="synthetic",
        default_floor=0.85,
    )


def _matrix_allowing(*rolesets) -> ContentionMatrix:
    """Matrix whose N-way verdict for these sets is a measured ALLOW.

    Needed to isolate the DEVICE axis: `admit_set`'s disjoint branch delegates
    to `nway_policy`, which fails closed for BACKGROUND on an unmeasured set.
    That is pre-existing, correct, and orthogonal to what these tests assert —
    without pinning it, a QUEUE would be ambiguous between "region veto" (the
    thing being fixed) and "unmeasured set" (the thing being preserved).
    """
    m = _empty_matrix()
    for roles in rolesets:
        key = tuple(sorted(roles))
        m.n_way[key] = Nway(roles=key, ratio=1.0, verdict="allow")
    return m


def test_placements_conflict_is_device_aware_while_overlap_stays_physical() -> None:
    """`placements_overlap` must stay a pure physical predicate; only
    `placements_conflict` consults the device."""
    gpu = Placement("architect_general", frozenset({"q3"}), DeviceClass.GPU)
    cpu = Placement("worker_general", frozenset({"q0", "q1", "q2", "q3"}), DeviceClass.CPU)
    assert placements_overlap(gpu, cpu) is True
    assert placements_conflict(gpu, cpu) is False
    assert claims_cpu_regions(gpu) is False
    assert claims_cpu_regions(cpu) is True


def test_unresolved_placement_stays_cpu_exclusive() -> None:
    """No device supplied → today's exclusive semantics, unchanged. This is
    what keeps every pre-existing caller and test byte-identical."""
    a = Placement("frontdoor", frozenset({"q0"}))
    b = Placement("ingest_long_context", frozenset({"q0"}))
    assert a.device_class is None
    assert placements_conflict(a, b) is True


def test_admit_set_does_not_queue_gpu_candidate_behind_cpu_holder() -> None:
    """The scheduling-module half of the headline fix."""
    holder = Placement("worker_general", frozenset({"q0", "q1", "q2", "q3"}), DeviceClass.CPU)
    gpu = Placement("architect_general", frozenset({"q3"}), DeviceClass.GPU)
    allowed = _matrix_allowing(("worker_general", "architect_general"))
    assert admit_set([holder], gpu, TrafficClass.BACKGROUND,
                     matrix=allowed) is PairDecision.ALLOW
    # Same shapes, same matrix, device unresolved → still serialized on the
    # region veto, exactly as before.
    assert admit_set(
        [Placement("worker_general", holder.regions)],
        Placement("architect_general", gpu.regions),
        TrafficClass.BACKGROUND, matrix=allowed,
    ) is PairDecision.QUEUE


def test_admit_set_treats_empty_gpu_regions_as_answer_not_unknown() -> None:
    """The live lane `184-191` is SMT-siblings-only, so it maps to NO atomic
    region. An empty region set on a RESOLVED GPU placement means "claims no
    CPU region" — not "placement unknown", which fails closed for background."""
    holder = Placement("worker_general", frozenset({"q0", "q1"}), DeviceClass.CPU)
    gpu = Placement("worker_vision", frozenset(), DeviceClass.GPU)
    allowed = _matrix_allowing(("worker_general", "worker_vision"))
    assert admit_set([holder], gpu, TrafficClass.BACKGROUND,
                     matrix=allowed) is PairDecision.ALLOW
    # Unresolved + empty regions → the pre-existing fail-closed path.
    assert admit_set([holder], Placement("worker_vision", frozenset()),
                     TrafficClass.BACKGROUND, matrix=allowed) is PairDecision.QUEUE


def test_admit_set_still_queues_two_overlapping_cpu_placements() -> None:
    holder = Placement("worker_general", frozenset({"q0", "q1"}), DeviceClass.CPU)
    cand = Placement("frontdoor", frozenset({"q1"}), DeviceClass.CPU)
    assert admit_set([holder], cand, TrafficClass.BACKGROUND,
                     matrix=_empty_matrix()) is PairDecision.QUEUE


def test_seam_stays_inert_with_flags_off(monkeypatch) -> None:
    """No live behaviour change: both flags off, the seam returns None BEFORE
    any device resolution happens, so nothing new can even be consulted."""
    monkeypatch.delenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", raising=False)
    assert seam_admit(
        "architect_critic", 0, {"worker_general": frozenset({"q0", "q1", "q2", "q3"})},
        traffic_class=TrafficClass.BACKGROUND, matrix=_empty_matrix(),
    ) is None


def test_seam_with_flags_on_admits_gpu_beside_cpu_holder(monkeypatch) -> None:
    """Flag-on behaviour, exercised through the exact dual-flag runtime gate.

    The GPU role is resolved from the LIVE topology/priors: since the 2026-09-27
    ARCHITECT SWAP the MI210 27B (:8083, GPU host lane) is architect_critic;
    architect_general is the full-CPU Flash-Next process."""
    monkeypatch.setenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    regions = {
        ("architect_critic", 0): frozenset(),  # live lane 184-191 → no region
        ("worker_general", 0): frozenset({"q0", "q1", "q2", "q3"}),
    }
    out = seam_admit(
        "architect_critic", 0,
        {"worker_general": frozenset({"q0", "q1", "q2", "q3"})},
        traffic_class=TrafficClass.BACKGROUND,
        instance_regions=regions,
        matrix=_matrix_allowing(("worker_general", "architect_critic")),
    )
    assert out is PairDecision.ALLOW


def test_seam_fails_closed_when_device_is_unresolvable(monkeypatch) -> None:
    """An undeclared holder must never be silently treated as CPU."""
    monkeypatch.setenv("ORCHESTRATOR_SHAPE_AWARE_CONTENTION", "1")
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    out = seam_admit(
        "frontdoor", 0, {"a_role_that_does_not_exist": frozenset({"q0"})},
        traffic_class=TrafficClass.BACKGROUND,
        instance_regions={("frontdoor", 0): frozenset({"q0", "q1"})},
        matrix=_empty_matrix(),
    )
    assert out is PairDecision.QUEUE
    out_fg = seam_admit(
        "frontdoor", 0, {"a_role_that_does_not_exist": frozenset({"q0"})},
        traffic_class=TrafficClass.FOREGROUND_INTERACTIVE,
        instance_regions={("frontdoor", 0): frozenset({"q0", "q1"})},
        matrix=_empty_matrix(),
    )
    assert out_fg is None  # caller keeps its legacy path


def test_live_topology_device_map_is_consistent() -> None:
    """Every live role's two device declarations agree. If this ever fails,
    an artifact drifted and the feasibility model is reasoning on a lie."""
    from scripts.server.stack_numa import NUMA_CONFIG
    from src.scheduling.device_model import resolve_device_classes

    resolved = resolve_device_classes(NUMA_CONFIG.keys(), numa_config=NUMA_CONFIG)
    gpu = {r for r, rd in resolved.items() if rd.is_gpu}
    # 2026-09-27 ARCHITECT SWAP: the MI210 27B is architect_critic.
    # STACKCHG-DFLASH2-20261003 (operator-signed): the VL-30B (worker_vision, :8086)
    # left the MI210 for a cold CPU placement (NUMA_HALF_A, device none) so the 27B
    # can carry DFlash2; the 27B is the ONLY GPU role.
    assert gpu == {"architect_critic"}
    for role in gpu:
        assert resolved[role].corroborated, f"{role} device is uncorroborated"
    # The moved role's two declarations must agree on CPU, not merely drop off the GPU.
    vision = resolved["worker_vision"]
    assert vision.device_class is DeviceClass.CPU and vision.device is None, vision
    assert vision.corroborated, "worker_vision CPU placement is uncorroborated"


# ── RTG-35: a stale matrix must not refuse the feasibility enumeration ──
#
# The feasibility model reads no measured cell, so refusing it on matrix
# freshness was a category error (landed in a517793c). The N-way path consumes
# measured ratios and must still refuse. Pinned here so the asymmetry cannot
# silently collapse in either direction.


def _stale_matrix(tmp_path):
    import scripts.server.contention_matrix as cm_mod

    path = tmp_path / "stale_matrix.yaml"
    path.write_text(
        cm_mod._emit_yaml(
            [],
            topology_hash="0000stale0000",
            binary={"git_commit": "abc1234"},
            host="TestHost",
            host_health=None,
        )
    )
    return path


def _enum_args(matrix, out_dir, *, feasibility):
    import argparse

    return argparse.Namespace(
        matrix=str(matrix),
        output=str(out_dir),
        floor=None,
        max_size=None,
        run_id="test-run",
        feasibility=feasibility,
        vram_headroom=None,
        allow_host_query=False,
    )


def test_stale_matrix_does_not_refuse_feasibility_enumeration(tmp_path) -> None:
    matrix = _stale_matrix(tmp_path)
    rc = cm.cmd_enumerate(_enum_args(matrix, tmp_path, feasibility=True))
    assert rc == 0
    manifest = json.loads((tmp_path / "j4a_feasible_manifest.json").read_text())
    src = manifest["matrix_source"]
    assert src["status"] != "ok", "fixture must actually be stale"
    assert src["consumed_by_this_model"] is False
    assert manifest["task_id"] == "J4a-feasible"


def test_stale_matrix_still_refuses_nway_enumeration(tmp_path) -> None:
    matrix = _stale_matrix(tmp_path)
    rc = cm.cmd_enumerate(_enum_args(matrix, tmp_path, feasibility=False))
    assert rc == 2
    assert not (tmp_path / "j4a_candidate_manifest.json").exists()
