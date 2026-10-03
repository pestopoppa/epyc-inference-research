"""Phase A — contention matrix loader + policy decisions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

contention = importlib.import_module("src.scheduling.contention")


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def real_matrix_path() -> Path:
    """The actual committed contention matrix."""
    return ROOT / "orchestration" / "contention_matrix.yaml"


@pytest.fixture
def minimal_matrix_yaml(tmp_path: Path) -> Path:
    """Tiny synthetic matrix for isolated testing."""
    p = tmp_path / "contention_matrix.yaml"
    p.write_text(
        """
version: 1
measured_at: "2026-05-24T12:00:00Z"
host: "test"
topology_hash: "TEST_HASH"
default_floor: 0.85
pairs:
  - roles: ["frontdoor", "architect_general"]
    instance_a: {port: 8070}
    instance_b: {port: 8083}
    seq_aggregate_tps: 17.0
    parallel_aggregate_tps: 8.5
    ratio: 0.50
    verdict: "block"
  - roles: ["frontdoor", "worker_general"]
    instance_a: {port: 8070}
    instance_b: {port: 8072}
    seq_aggregate_tps: 35.0
    parallel_aggregate_tps: 45.0
    ratio: 1.28
    verdict: "allow"
  - roles: ["frontdoor", "vision_escalation"]
    instance_a: {port: 8070}
    instance_b: {port: 8087}
    seq_aggregate_tps: 26.6
    parallel_aggregate_tps: 22.3
    ratio: 0.84
    verdict: "borderline"
same_role:
  - role: "frontdoor"
    verdict: "allow"
  - role: "vision_escalation"
    verdict: "block"
    note: "4-quarter anomaly"
same_role_certifications:
  - role: "frontdoor"
    mode: "live_only"
    measured_at: "2026-07-20T00:00:00+00:00"
    topology_hash: "ROLE_HASH"
    live_ports: [8080, 8180]
    verdict: "allow"
    samples: 3
    min_ratio: 1.2
    max_cv: 0.04
    artifact: "data/contention_matrix/test/j5_within_role_results.json"
    affinity_artifact: "data/contention_matrix/test/affinity_preflight.json"
unknown_pairs:
  - roles: ["ingest_long_context", "worker_vision"]
    reason: "not_measured"
"""
    )
    return p


# ── Loader ───────────────────────────────────────────────────────────


def test_load_real_matrix(real_matrix_path: Path) -> None:
    """The committed matrix should parse cleanly and COVER the live lineup.

    Lineup-independence (2026-09-26): this used to pin the v6 refresh's shape
    (15 measured pairs, 0 unknown, an ingest_long_context row). The
    operator-signed 2026-09-22 cutover (orchestrator 860b0b2d) made
    ingest_long_context an alias of architect_general, and the OP-54 re-bench
    (0a564a1f, topology 4893e37e) regenerated the matrix over the physical
    fleets that remain. The expectation is now DERIVED from the same enumerator
    the bench uses (scripts/server/contention_matrix._enumerate_full_pairs over
    NUMA_CONFIG): every cross-role pair of the live lineup is either measured or
    explicitly declared unknown — never silently missing, never both."""
    import importlib as _il

    sys.path.insert(0, str(ROOT / "scripts" / "server"))
    cm = _il.import_module("contention_matrix")
    from scripts.server.stack_numa import NUMA_CONFIG

    m = contention.load_contention_matrix(real_matrix_path)
    assert m.version == 1
    assert m.host == "Beelzebub"
    assert m.default_floor == 0.85

    expected = {tuple(sorted(p)) for p in cm._enumerate_full_pairs(NUMA_CONFIG)}
    measured = set(m.pairs)
    unknown = {tuple(sorted(p)) for p in m.unknown_pairs}
    assert len(expected) >= 2, expected  # non-vacuity: a real multi-fleet lineup
    assert measured, "matrix measures no pair at all"
    assert not measured & unknown, measured & unknown
    assert measured | unknown == expected, {
        "missing": sorted(expected - measured - unknown),
        "stale": sorted((measured | unknown) - expected),
    }
    # STACKCHG-DFLASH2-20261003: worker_vision moved GPU_HOST_LANE -> a COLD CPU half
    # (NUMA_HALF_A). Its measured pairs described the old placement and were
    # WITHDRAWN into `unknown_pairs` (declared, never silently missing): no stale
    # number may gate the new placement until it is re-benched.
    assert m.get_pair("architect_general", "worker_vision") is None
    assert ("architect_general", "worker_vision") in unknown
    # ingest_long_context serves on its registry host's process; that host's
    # pair with worker_vision is the row that now carries its contention.
    import yaml

    mode = yaml.safe_load(
        (ROOT / "orchestration" / "model_registry.yaml").read_text(encoding="utf-8")
    )["server_mode"]
    ingest_host = mode["ingest_long_context"].get("alias_of") or "ingest_long_context"
    assert m.get_pair(ingest_host, "worker_vision") is None
    assert tuple(sorted((ingest_host, "worker_vision"))) in unknown


def test_real_matrix_declares_current_nway_role_classes(real_matrix_path: Path) -> None:
    """The committed matrix should expose current light/heavy role classes."""
    m = contention.load_contention_matrix(real_matrix_path)

    assert {
        "frontdoor",
        "vision_escalation",
        "worker_general",
        "worker_vision",
    } <= m.light_roles
    assert {"ingest_long_context", "architect_general"} <= m.heavy_roles
    assert "worker_fast" not in m.light_roles


def test_real_matrix_vision_escalation_same_role_allows(real_matrix_path: Path) -> None:
    """Certified matrix no longer treats same-role vision_escalation as blocked."""
    m = contention.load_contention_matrix(real_matrix_path)

    same_role = m.get_same_role("vision_escalation")

    assert same_role is not None
    assert same_role.verdict == "allow"
    assert (
        contention.pair_policy(
            "vision_escalation",
            "vision_escalation",
            contention.TrafficClass.BACKGROUND,
            matrix=m,
        )
        == contention.PairDecision.ALLOW
    )


def test_load_minimal_matrix(minimal_matrix_yaml: Path) -> None:
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert m.topology_hash == "TEST_HASH"
    assert len(m.pairs) == 3
    assert len(m.same_role) == 2


def test_pair_key_is_sorted(minimal_matrix_yaml: Path) -> None:
    """(A, B) and (B, A) should resolve to the same pair."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    p1 = m.get_pair("frontdoor", "architect_general")
    p2 = m.get_pair("architect_general", "frontdoor")
    assert p1 is not None and p2 is not None
    assert p1.ratio == p2.ratio == 0.50


def test_unknown_pair_detection(minimal_matrix_yaml: Path) -> None:
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert m.is_unknown_pair("ingest_long_context", "worker_vision")
    assert m.is_unknown_pair("worker_vision", "ingest_long_context")  # order-agnostic
    assert not m.is_unknown_pair("frontdoor", "architect_general")  # known


def test_same_role_certification_parses(minimal_matrix_yaml: Path) -> None:
    m = contention.load_contention_matrix(minimal_matrix_yaml)

    cert = m.get_same_role_certification("frontdoor")

    assert cert is not None
    assert cert.verdict == "allow"
    assert cert.mode == "live_only"
    assert cert.topology_hash == "ROLE_HASH"
    assert cert.live_ports == (8080, 8180)
    assert cert.samples == 3
    assert cert.min_ratio == 1.2
    assert cert.max_cv == 0.04
    assert cert.artifact.endswith("j5_within_role_results.json")


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        contention.load_contention_matrix(Path("/tmp/does_not_exist_xyz.yaml"))


# ── matrix_status ────────────────────────────────────────────────────


def test_matrix_status_missing(tmp_path: Path) -> None:
    assert contention.matrix_status(tmp_path / "missing.yaml") == contention.MatrixStatus.MISSING


def test_matrix_status_ok(minimal_matrix_yaml: Path) -> None:
    status = contention.matrix_status(minimal_matrix_yaml, current_topology_hash="TEST_HASH")
    assert status == contention.MatrixStatus.OK


def test_matrix_status_stale_topology(minimal_matrix_yaml: Path) -> None:
    status = contention.matrix_status(minimal_matrix_yaml, current_topology_hash="DIFFERENT_HASH")
    assert status == contention.MatrixStatus.STALE


def test_matrix_status_stale_age(minimal_matrix_yaml: Path) -> None:
    # File just-created; with max_age_days=-1 anything is stale
    status = contention.matrix_status(
        minimal_matrix_yaml, current_topology_hash="TEST_HASH", max_age_days=-1
    )
    assert status == contention.MatrixStatus.STALE


def test_matrix_status_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [a, valid\n  yaml mapping at root")
    assert contention.matrix_status(bad) == contention.MatrixStatus.INVALID


# ── pair_policy ──────────────────────────────────────────────────────


def test_policy_block_below_floor(minimal_matrix_yaml: Path) -> None:
    """frontdoor + architect at 0.50 → QUEUE for all classes."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    for tc in [
        contention.TrafficClass.FOREGROUND_INTERACTIVE,
        contention.TrafficClass.FOREGROUND_SPECIALIST,
        contention.TrafficClass.BACKGROUND,
    ]:
        assert (
            contention.pair_policy("frontdoor", "architect_general", tc, matrix=m)
            == contention.PairDecision.QUEUE
        )


def test_policy_allow_above_one(minimal_matrix_yaml: Path) -> None:
    """frontdoor + worker_general at 1.28 → ALLOW for all classes."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    for tc in [
        contention.TrafficClass.FOREGROUND_INTERACTIVE,
        contention.TrafficClass.BACKGROUND,
    ]:
        assert (
            contention.pair_policy("frontdoor", "worker_general", tc, matrix=m)
            == contention.PairDecision.ALLOW
        )


def test_policy_borderline_below_floor_queues_all(minimal_matrix_yaml: Path) -> None:
    """frontdoor + vision_escalation at 0.84 is BELOW the 0.85 floor → QUEUE for
    all classes. Foreground caller can promote to DEGRADED_ALLOW based on SLO."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert (
        contention.pair_policy(
            "frontdoor",
            "vision_escalation",
            contention.TrafficClass.FOREGROUND_INTERACTIVE,
            matrix=m,
        )
        == contention.PairDecision.QUEUE
    )
    assert (
        contention.pair_policy(
            "frontdoor",
            "vision_escalation",
            contention.TrafficClass.BACKGROUND,
            matrix=m,
        )
        == contention.PairDecision.QUEUE
    )


def test_policy_at_floor_allows_foreground(minimal_matrix_yaml: Path) -> None:
    """A pair exactly at the floor (0.85) should ALLOW foreground, QUEUE background.
    Test by passing a custom floor that puts vision_escalation at-or-just-below."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    # With floor=0.80, the 0.84 pair becomes "at-or-above" → foreground ALLOW
    assert (
        contention.pair_policy(
            "frontdoor",
            "vision_escalation",
            contention.TrafficClass.FOREGROUND_INTERACTIVE,
            matrix=m,
            floor=0.80,
        )
        == contention.PairDecision.ALLOW
    )
    assert (
        contention.pair_policy(
            "frontdoor",
            "vision_escalation",
            contention.TrafficClass.BACKGROUND,
            matrix=m,
            floor=0.80,
        )
        == contention.PairDecision.QUEUE
    )


def test_policy_unknown_pair(minimal_matrix_yaml: Path) -> None:
    """Unknown pair → ALLOW foreground, QUEUE background."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert (
        contention.pair_policy(
            "ingest_long_context",
            "worker_vision",
            contention.TrafficClass.BACKGROUND,
            matrix=m,
        )
        == contention.PairDecision.QUEUE
    )
    assert (
        contention.pair_policy(
            "ingest_long_context",
            "worker_vision",
            contention.TrafficClass.FOREGROUND_INTERACTIVE,
            matrix=m,
        )
        == contention.PairDecision.ALLOW
    )


def test_policy_same_role_allowed(minimal_matrix_yaml: Path) -> None:
    """frontdoor + frontdoor → ALLOW (matrix says same-role frontdoor is allow)."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert (
        contention.pair_policy(
            "frontdoor",
            "frontdoor",
            contention.TrafficClass.FOREGROUND_INTERACTIVE,
            matrix=m,
        )
        == contention.PairDecision.ALLOW
    )


def test_policy_same_role_blocked(minimal_matrix_yaml: Path) -> None:
    """vision_escalation + vision_escalation → QUEUE background, DEGRADED_ALLOW foreground."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    assert (
        contention.pair_policy(
            "vision_escalation",
            "vision_escalation",
            contention.TrafficClass.BACKGROUND,
            matrix=m,
        )
        == contention.PairDecision.QUEUE
    )
    assert (
        contention.pair_policy(
            "vision_escalation",
            "vision_escalation",
            contention.TrafficClass.FOREGROUND_INTERACTIVE,
            matrix=m,
        )
        == contention.PairDecision.DEGRADED_ALLOW
    )


def test_policy_fail_open_on_missing_matrix(monkeypatch, tmp_path: Path) -> None:
    """When matrix is missing, foreground ALLOW, background QUEUE."""
    monkeypatch.setattr(contention, "DEFAULT_MATRIX_PATH", tmp_path / "absent.yaml")
    assert (
        contention.pair_policy(
            "frontdoor",
            "architect_general",
            contention.TrafficClass.FOREGROUND_INTERACTIVE,
        )
        == contention.PairDecision.ALLOW
    )
    assert (
        contention.pair_policy(
            "frontdoor",
            "architect_general",
            contention.TrafficClass.BACKGROUND,
        )
        == contention.PairDecision.QUEUE
    )


def test_traffic_class_string_coerces(minimal_matrix_yaml: Path) -> None:
    """Passing a string for traffic_class should be coerced to TrafficClass."""
    m = contention.load_contention_matrix(minimal_matrix_yaml)
    # "background" string should work
    decision = contention.pair_policy("frontdoor", "architect_general", "background", matrix=m)
    assert decision == contention.PairDecision.QUEUE
    # "foreground_interactive" string should work
    decision = contention.pair_policy(
        "frontdoor", "architect_general", "foreground_interactive", matrix=m
    )
    assert decision == contention.PairDecision.QUEUE


# ── topology_fingerprint ─────────────────────────────────────────────


def test_topology_fingerprint_deterministic() -> None:
    config = {
        "frontdoor": {"instances": [("0-47,96-143", 8070, 96), ("0-23,96-119", 8080, 48)]},
        "worker_general": {"instances": [("0-95", 8072, 96)]},
    }
    h1 = contention.topology_fingerprint(config)
    h2 = contention.topology_fingerprint(config)
    assert h1 == h2 and len(h1) == 16


def test_topology_fingerprint_order_independent() -> None:
    """Same content in different insertion order should hash identically."""
    a = {
        "frontdoor": {"instances": [("0-47", 8070, 96)]},
        "worker_general": {"instances": [("0-95", 8072, 96)]},
    }
    b = {
        "worker_general": {"instances": [("0-95", 8072, 96)]},
        "frontdoor": {"instances": [("0-47", 8070, 96)]},
    }
    assert contention.topology_fingerprint(a) == contention.topology_fingerprint(b)


def test_topology_fingerprint_detects_change() -> None:
    base = {"frontdoor": {"instances": [("0-47", 8070, 96)]}}
    # Different thread count
    changed_threads = {"frontdoor": {"instances": [("0-47", 8070, 48)]}}
    assert contention.topology_fingerprint(base) != contention.topology_fingerprint(changed_threads)
    # Different cpu_list
    changed_cpus = {"frontdoor": {"instances": [("0-95", 8070, 96)]}}
    assert contention.topology_fingerprint(base) != contention.topology_fingerprint(changed_cpus)
    # New instance
    added = {"frontdoor": {"instances": [("0-47", 8070, 96), ("48-95", 8080, 96)]}}
    assert contention.topology_fingerprint(base) != contention.topology_fingerprint(added)


def test_topology_fingerprint_ignores_metadata() -> None:
    """mlock and numactl_policy should NOT change the topology fingerprint."""
    a = {"frontdoor": {"instances": [("0-47", 8070, 96)], "mlock": True}}
    b = {
        "frontdoor": {
            "instances": [("0-47", 8070, 96)],
            "mlock": False,
            "numactl_policy": "interleave=all",
        }
    }
    assert contention.topology_fingerprint(a) == contention.topology_fingerprint(b)


def test_role_topology_fingerprint_filters_to_live_ports() -> None:
    config = {
        "frontdoor": {
            "instances": [
                ("0-47,96-143", 8070, 96),
                ("0-23,96-119", 8080, 48),
                ("24-47,120-143", 8180, 48),
            ],
        },
        "worker_general": {"instances": [("0-95", 8072, 96)]},
    }

    live_hash = contention.role_topology_fingerprint(
        config,
        "frontdoor",
        live_ports={8080, 8180},
    )

    assert len(live_hash) == 16
    assert live_hash == contention.role_topology_fingerprint(
        config,
        "frontdoor",
        live_ports={8180, 8080},
    )
    assert live_hash != contention.role_topology_fingerprint(config, "frontdoor")
    assert live_hash != contention.role_topology_fingerprint(config, "worker_general")


def test_topology_fingerprint_for_matrix_excludes_unmeasured_auxiliary_role() -> None:
    measured = {
        "frontdoor": {"instances": [("0-1", 8070, 2)]},
        "worker_general": {"instances": [("2-3", 8072, 2)]},
    }
    live = {
        **measured,
        "eval_batch_frontdoor": {"instances": [("0-1", 18070, 2)]},
    }
    matrix = contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="",
        topology_hash=contention.topology_fingerprint(measured),
        default_floor=0.85,
        pairs={
            ("frontdoor", "worker_general"): contention.Pair(
                roles=("frontdoor", "worker_general"),
                ratio=1.0,
                verdict="allow",
            )
        },
    )

    assert contention.topology_fingerprint_for_matrix(live, matrix) == (
        contention.topology_fingerprint(measured)
    )
    assert contention.topology_fingerprint_for_matrix(live, matrix) != (
        contention.topology_fingerprint(live)
    )


def test_nway_policy_maps_eval_batch_lane_to_measured_frontdoor_identity() -> None:
    matrix = contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="",
        topology_hash="synthetic",
        default_floor=0.85,
        pairs={},
        n_way={
            ("frontdoor", "worker_general"): contention.Nway(
                roles=("frontdoor", "worker_general"),
                ratio=1.2,
                verdict="allow",
            )
        },
    )

    assert (
        contention.nway_policy(
            ["eval_batch_frontdoor", "worker_general"],
            contention.TrafficClass.BACKGROUND,
            matrix=matrix,
        )
        == contention.PairDecision.ALLOW
    )


def test_topology_fingerprint_scoped_to_the_stamp_but_still_catches_real_drift() -> None:
    """The comparable subset is the PAIRWISE-stamped set, and only that.

    `topology_hash` is emitted by `contention_matrix.py::cmd_run` over the roles of
    the pairwise run it is writing. A carried-forward N-way / same-role fragment
    may legitimately name a role that has since been folded onto another server's
    process, and that must not make a freshly re-benched `pairs` block read as
    uncertified. Everything the guard is FOR must still fire, so all three
    genuine-drift cases are pinned here too.
    """
    measured = {
        "frontdoor": {"instances": [("0-1", 8070, 2)]},
        "worker_general": {"instances": [("2-3", 8072, 2)]},
    }
    stamp = contention.topology_fingerprint(measured)

    def _matrix(**kwargs):
        return contention.ContentionMatrix(
            version=1,
            measured_at="",
            host="",
            topology_hash=stamp,
            default_floor=0.85,
            pairs={
                ("frontdoor", "worker_general"): contention.Pair(
                    roles=("frontdoor", "worker_general"), ratio=1.0, verdict="allow"
                )
            },
            **kwargs,
        )

    retired_fragment = _matrix(
        same_role={"retired_role": contention.SameRole(role="retired_role", verdict="allow")},
        n_way={
            ("frontdoor", "retired_role", "worker_general"): contention.Nway(
                roles=("frontdoor", "retired_role", "worker_general"),
                ratio=1.5,
                verdict="allow",
            )
        },
    )
    # The broad view still reports the fragment's role; only the STAMPED view drops it.
    assert "retired_role" in contention.matrix_measured_roles(retired_fragment)
    assert "retired_role" not in contention.matrix_stamped_roles(retired_fragment)
    assert contention.topology_fingerprint_for_matrix(measured, retired_fragment) == stamp

    # 1. A stamped role whose topology CHANGED is still caught.
    moved = {**measured, "worker_general": {"instances": [("2-3", 8072, 4)]}}
    assert contention.topology_fingerprint_for_matrix(moved, retired_fragment) != stamp

    # 2. A stamped role that VANISHED still forces the conservative full-topology
    #    fallback rather than silently fingerprinting a smaller set.
    without_stamped = {"frontdoor": measured["frontdoor"]}
    assert contention.topology_fingerprint_for_matrix(
        without_stamped, retired_fragment
    ) == contention.topology_fingerprint(without_stamped)
    assert contention.topology_fingerprint_for_matrix(without_stamped, retired_fragment) != stamp

    # 3. A role named only by `unknown_pairs` is part of the pairwise run, so it
    #    stays inside the stamped set.
    with_unknown = _matrix(unknown_pairs=[("frontdoor", "ingest_long_context")])
    assert "ingest_long_context" in contention.matrix_stamped_roles(with_unknown)
    assert contention.topology_fingerprint_for_matrix(measured, with_unknown) == (
        contention.topology_fingerprint(measured)
    )


def test_real_matrix_against_live_numa_config() -> None:
    """The committed matrix must be certified against the live measured topology."""
    sys.path.insert(0, str(ROOT / "scripts" / "server"))
    import stack_numa

    matrix = contention.load_contention_matrix(ROOT / "orchestration" / "contention_matrix.yaml")
    live_hash = contention.topology_fingerprint_for_matrix(stack_numa.NUMA_CONFIG, matrix)
    assert isinstance(live_hash, str) and len(live_hash) == 16
    assert matrix.topology_hash == live_hash


# ── nway_policy (J4c — N-way active-set admission) ───────────────────


def _nway_test_matrix():
    """Synthetic matrix: one measured-block triple + one measured-allow triple,
    with the production light/heavy role classification."""
    Nway = contention.Nway
    return contention.ContentionMatrix(
        version=1,
        measured_at="",
        host="",
        topology_hash="t",
        default_floor=0.85,
        n_way={
            ("frontdoor", "ingest_long_context", "vision_escalation"): Nway(
                roles=("frontdoor", "ingest_long_context", "vision_escalation"),
                ratio=0.847,
                verdict="block",
                contains_heavy=True,
            ),
            ("frontdoor", "vision_escalation", "worker_general"): Nway(
                roles=("frontdoor", "vision_escalation", "worker_general"),
                ratio=1.607,
                verdict="allow",
            ),
        },
        light_roles=frozenset(
            {"frontdoor", "vision_escalation", "worker_general", "worker_vision"}
        ),
        heavy_roles=frozenset({"ingest_long_context", "architect_general"}),
    )


def test_nway_policy_measured_block_queues_even_when_pairs_allow():
    """The crux of J4c: a measured-block N-way set queues even though every
    constituent pair is allow ({frontdoor,ingest,vision} = 0.847)."""
    m = _nway_test_matrix()
    roles = ["frontdoor", "ingest_long_context", "vision_escalation"]
    assert (
        contention.nway_policy(roles, contention.TrafficClass.BACKGROUND, matrix=m)
        == contention.PairDecision.QUEUE
    )
    assert (
        contention.nway_policy(roles, contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m)
        == contention.PairDecision.QUEUE
    )


def test_nway_policy_measured_allow():
    m = _nway_test_matrix()
    assert (
        contention.nway_policy(
            ["frontdoor", "vision_escalation", "worker_general"],
            contention.TrafficClass.BACKGROUND,
            matrix=m,
        )
        == contention.PairDecision.ALLOW
    )


def test_nway_policy_all_light_unmeasured_allows_both_classes():
    """Unmeasured but all-light (covers mixed multi-instance via role-set dedup)."""
    m = _nway_test_matrix()
    roles = ["frontdoor", "worker_general", "worker_vision"]  # not in n_way, all light
    assert (
        contention.nway_policy(roles, contention.TrafficClass.BACKGROUND, matrix=m)
        == contention.PairDecision.ALLOW
    )
    assert (
        contention.nway_policy(roles, contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m)
        == contention.PairDecision.ALLOW
    )


def test_nway_policy_heavy_unmeasured_fail_open_fg_closed_bg():
    m = _nway_test_matrix()
    roles = ["ingest_long_context", "worker_vision"]  # not in n_way, contains heavy
    assert (
        contention.nway_policy(roles, contention.TrafficClass.BACKGROUND, matrix=m)
        == contention.PairDecision.QUEUE
    )
    assert (
        contention.nway_policy(roles, contention.TrafficClass.FOREGROUND_INTERACTIVE, matrix=m)
        == contention.PairDecision.ALLOW
    )


def test_nway_policy_single_role_is_allow():
    m = _nway_test_matrix()
    assert contention.nway_policy(["frontdoor"], matrix=m) == contention.PairDecision.ALLOW
    assert contention.nway_policy([], matrix=m) == contention.PairDecision.ALLOW


def test_nway_policy_order_independent():
    m = _nway_test_matrix()
    a = contention.nway_policy(
        ["vision_escalation", "frontdoor", "ingest_long_context"], "background", matrix=m
    )
    b = contention.nway_policy(
        ["ingest_long_context", "vision_escalation", "frontdoor"], "background", matrix=m
    )
    assert a == b == contention.PairDecision.QUEUE


# ── Generator: a pairwise re-bench must NOT truncate runtime policy ───
#
# Regression pin for the defect behind the two `real_matrix` failures above:
# `scripts/server/contention_matrix.py::_emit_yaml` rendered ONLY `pairs` and
# `cmd_bench` wrote it with a whole-file `write_text`, so every pairwise
# re-bench silently deleted the hand-authored `nway_*` / `same_role` / `n_way`
# policy this module's admission gate reads (cd42def3, again at a517793c).
# The gate then degraded to conservative QUEUE with no error anywhere.


def _load_matrix_generator():
    sys.path.insert(0, str(ROOT / "scripts" / "server"))
    return importlib.import_module("contention_matrix")


def test_emitter_carries_hand_authored_policy_across_a_rebench(tmp_path: Path) -> None:
    """A fresh pairwise emit must preserve every non-`pairs` policy section."""
    gen = _load_matrix_generator()

    existing = tmp_path / "contention_matrix.yaml"
    existing.write_text(
        """version: 1
measured_at: "2026-01-01T00:00:00Z"
host: "test"
topology_hash: "OLD"
default_floor: 0.85

pairs:
  - roles: ['frontdoor', 'worker_general']
    ratio: 1.10
    verdict: "allow"

# provenance comment that must travel with its section
nway_light_roles: ["frontdoor", "worker_general"]
nway_heavy_roles: ["ingest_long_context"]
same_role:
  - role: "vision_escalation"
    verdict: "allow"
n_way:
  - roles: ["frontdoor", "ingest_long_context", "worker_general"]
    ratio: 1.535
    verdict: "allow"
"""
    )

    preserved = gen._carry_forward_sections(existing)
    assert [key for key, _ in preserved] == [
        "nway_light_roles",
        "nway_heavy_roles",
        "same_role",
        "n_way",
    ]

    rewritten = gen._emit_yaml(
        [],
        topology_hash="NEW",
        host="test",
        preserve_sections=preserved,
    )

    # The regenerated header/pairs are fresh...
    assert 'topology_hash: "NEW"' in rewritten
    assert '"OLD"' not in rewritten
    # ...and every hand-authored section survived, comments included.
    assert 'nway_light_roles: ["frontdoor", "worker_general"]' in rewritten
    assert 'nway_heavy_roles: ["ingest_long_context"]' in rewritten
    assert 'role: "vision_escalation"' in rewritten
    assert "ratio: 1.535" in rewritten
    assert "# provenance comment that must travel with its section" in rewritten

    # And the rewritten document still parses into a usable policy.
    out = tmp_path / "rewritten.yaml"
    out.write_text(rewritten)
    m = contention.load_contention_matrix(out)
    assert m.light_roles == frozenset({"frontdoor", "worker_general"})
    assert m.heavy_roles == frozenset({"ingest_long_context"})
    assert m.get_same_role("vision_escalation").verdict == "allow"
    assert ("frontdoor", "ingest_long_context", "worker_general") in m.n_way


def test_emitter_freshly_measured_same_role_supersedes_the_preserved_block(
    tmp_path: Path,
) -> None:
    """A `bench-within-role` result must WIN over the carried-forward block."""
    gen = _load_matrix_generator()

    existing = tmp_path / "contention_matrix.yaml"
    existing.write_text(
        "version: 1\ndefault_floor: 0.85\n\n"
        'same_role:\n  - role: "worker_general"\n    verdict: "block"\n'
        'nway_light_roles: ["frontdoor"]\n'
    )
    preserved = gen._carry_forward_sections(existing)

    rewritten = gen._emit_yaml(
        [],
        same_role_verdicts={"worker_general": "allow"},
        preserve_sections=preserved,
    )
    assert rewritten.count("same_role:") == 1
    assert 'verdict: "allow"' in rewritten
    assert 'verdict: "block"' not in rewritten
    # Unrelated policy is still carried through.
    assert 'nway_light_roles: ["frontdoor"]' in rewritten
