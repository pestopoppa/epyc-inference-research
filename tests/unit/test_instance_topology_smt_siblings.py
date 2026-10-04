"""REGION-SIBLING-1 — SMT siblings claim the regions of their physical cores.

`instance_topology.parse_cpu_list` DISCARDS logical CPUs 96-191, the SMT
siblings of 0-95 on the EPYC 9655, so `region-lock run --cpu-list 160-183` could
never conflict with a window held on the cores it physically shares (64-87).

Folding is OPT-IN (`smt_siblings="fold"`). The library default stays "drop" so
every existing caller — the in-process placement model whose GPU host lane on
184-191 is deliberately region-free, and AutoKernel's `loop/claim.py` — keeps
its exact prior meaning. Only the `region-lock run` CLI opts in, via
`--fold-siblings` (default on there; `--no-fold-siblings` = legacy).
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.runtime import instance_topology as it
from src.runtime.instance_topology import (
    CpuTopologyUnavailable,
    build_instance_regions,
    cpu_list_to_regions,
    parse_cpu_list,
    read_sibling_map,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Synthetic EPYC-9655-shaped sibling map, built as a TEST FIXTURE (production
# code reads it from the kernel): logical N and N+96 share one physical core.
EPYC_9655_SIBLINGS = {cpu: cpu % 96 for cpu in range(192)}


def _host_pairs_160_with_64() -> bool:
    try:
        return read_sibling_map().get(160) == 64
    except CpuTopologyUnavailable:
        return False


# ── Required cases ──────────────────────────────────────────────────────


def test_sibling_list_maps_to_regions_of_its_physical_cores() -> None:
    """160-183 are the siblings of 64-87 -> the same regions as 64-87."""
    sib = cpu_list_to_regions("160-183", smt_siblings="fold", sibling_map=EPYC_9655_SIBLINGS)
    assert sib == cpu_list_to_regions("64-87") == frozenset({"q2", "q3"})
    assert parse_cpu_list(
        "160-183", smt_siblings="fold", sibling_map=EPYC_9655_SIBLINGS
    ) == set(range(64, 88))


def test_mixed_list_maps_to_the_union() -> None:
    def fold(cpus: str) -> frozenset[str]:
        return cpu_list_to_regions(cpus, smt_siblings="fold", sibling_map=EPYC_9655_SIBLINGS)

    assert fold("0-23,160-183") == frozenset({"q0", "q2", "q3"})
    # Primary + its own siblings: the siblings add nothing new.
    assert fold("0-23,96-119") == frozenset({"q0"})
    # Siblings of q1 alongside primaries of q0.
    assert fold("0-23,120-143") == frozenset({"q0", "q1"})


@pytest.mark.parametrize(
    "cpu_list, regions",
    [
        ("0-95", {"q0", "q1", "q2", "q3"}),
        ("0-23", {"q0"}),
        ("24-47", {"q1"}),
        ("48-71", {"q2"}),
        ("72-95", {"q3"}),
        ("0-47", {"q0", "q1"}),
        ("23,24", {"q0", "q1"}),
        ("", set()),
    ],
)
def test_primary_only_list_is_unchanged(cpu_list: str, regions: set[str]) -> None:
    """Lists inside 0-95 resolve identically in both modes, without any topology read."""

    def _boom(*_a, **_k):  # a primary-only list must never consult the host map
        raise AssertionError("topology read for a primary-only list")

    original = it._host_sibling_map
    it._host_sibling_map = _boom  # type: ignore[assignment]
    try:
        assert cpu_list_to_regions(cpu_list) == frozenset(regions)
        assert cpu_list_to_regions(cpu_list, smt_siblings="fold") == frozenset(regions)
        assert parse_cpu_list(cpu_list) == parse_cpu_list(cpu_list, smt_siblings="fold")
    finally:
        it._host_sibling_map = original  # type: ignore[assignment]


# ── Library default is UNCHANGED (drop) for every other caller ──────────


def test_library_default_is_legacy_drop_and_never_reads_topology(monkeypatch) -> None:
    """Any caller that does not opt in sees exactly the pre-fix meaning."""

    def _boom():
        raise AssertionError("default mode must not consult the host topology")

    monkeypatch.setattr(it, "_host_sibling_map", _boom)
    assert parse_cpu_list("184-191") == set()
    assert cpu_list_to_regions("184-191") == frozenset()
    assert cpu_list_to_regions("160-183") == frozenset()
    assert cpu_list_to_regions("0-23,160-183") == frozenset({"q0"})
    assert parse_cpu_list("0-23,96-119") == set(range(24))
    assert parse_cpu_list("0-23,96-119", smt_siblings="drop") == set(range(24))


def test_build_instance_regions_keeps_ht_only_instances_region_free() -> None:
    """The placement model is primary-only by explicit choice: the GPU host lane
    (architect_critic on 184-191) must not start taking q3 locks via this fix."""
    cfg = {
        "architect_critic": {"instances": [("184-191", 8083)]},
        "frontdoor": {"instances": [("0-95", 8070), ("0-47,96-143", 8080), ("48-95,144-191", 8180)]},
    }
    regions = build_instance_regions(cfg)
    assert regions[("architect_critic", 0)] == frozenset()
    assert regions[("frontdoor", 0)] == frozenset({"q0", "q1", "q2", "q3"})
    assert regions[("frontdoor", 1)] == frozenset({"q0", "q1"})
    assert regions[("frontdoor", 2)] == frozenset({"q2", "q3"})


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        parse_cpu_list("0-3", smt_siblings="primary")  # type: ignore[arg-type]


# ── Fail closed: never guess a sibling's core ────────────────────────────


def test_unmapped_sibling_refuses_rather_than_dropping() -> None:
    with pytest.raises(CpuTopologyUnavailable):
        parse_cpu_list("200", smt_siblings="fold", sibling_map=EPYC_9655_SIBLINGS)


def test_sibling_folding_outside_region_table_refuses() -> None:
    with pytest.raises(CpuTopologyUnavailable):
        parse_cpu_list("150", smt_siblings="fold", sibling_map={150: 120})


# ── Topology readers (no hardcoded +96) ──────────────────────────────────


def _fake_cpu(root: Path, cpu: int, *, siblings: str | None, core_id: int, pkg: int = 0) -> None:
    topo = root / f"cpu{cpu}" / "topology"
    topo.mkdir(parents=True)
    if siblings is not None:
        (topo / "thread_siblings_list").write_text(siblings + "\n")
    (topo / "core_id").write_text(f"{core_id}\n")
    (topo / "physical_package_id").write_text(f"{pkg}\n")


def test_sysfs_reader_uses_thread_siblings_list(tmp_path: Path) -> None:
    for cpu in (0, 1, 2, 3):
        _fake_cpu(tmp_path, cpu, siblings=f"{cpu % 2},{cpu % 2 + 2}", core_id=7 + cpu % 2)
    (tmp_path / "cpufreq").mkdir()  # non-cpuN entries are ignored
    assert read_sibling_map(tmp_path) == {0: 0, 1: 1, 2: 0, 3: 1}


def test_sysfs_reader_groups_by_package_and_core_id_without_siblings_list(tmp_path: Path) -> None:
    """core_id is a HARDWARE id (cpu64/cpu160 read 88 on the live Zen5 host), so it
    is only a grouping key; the primary is the lowest logical CPU in the group."""
    _fake_cpu(tmp_path, 64, siblings=None, core_id=88)
    _fake_cpu(tmp_path, 160, siblings=None, core_id=88)
    _fake_cpu(tmp_path, 65, siblings=None, core_id=89)
    _fake_cpu(tmp_path, 161, siblings=None, core_id=89)
    _fake_cpu(tmp_path, 300, siblings=None, core_id=88, pkg=1)  # other socket
    assert read_sibling_map(tmp_path) == {64: 64, 160: 64, 65: 65, 161: 65, 300: 300}


def test_lscpu_fallback_when_sysfs_is_unreadable(tmp_path: Path, monkeypatch) -> None:
    out = "# CPU,Core,Socket\n0,0,0\n1,1,0\n2,0,0\n3,1,0\n"

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "lscpu"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(it.subprocess, "run", fake_run)
    assert read_sibling_map(tmp_path / "missing") == {0: 0, 1: 1, 2: 0, 3: 1}


def test_both_sources_unavailable_raises(tmp_path: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("lscpu")

    monkeypatch.setattr(it.subprocess, "run", fake_run)
    with pytest.raises(CpuTopologyUnavailable):
        read_sibling_map(tmp_path / "missing")


@pytest.mark.skipif(not _host_pairs_160_with_64(), reason="host is not the EPYC 9655 SMT layout")
def test_live_host_topology_folds_160_183_onto_64_87() -> None:
    assert parse_cpu_list("160-183", smt_siblings="fold") == set(range(64, 88))
    assert cpu_list_to_regions("160-183", smt_siblings="fold") == frozenset({"q2", "q3"})
    assert cpu_list_to_regions("184-191", smt_siblings="fold") == frozenset({"q3"})
    # ...and the default is still the legacy mapping on the live host.
    assert cpu_list_to_regions("184-191") == frozenset()


# ── region-lock CLI: --fold-siblings (default on) / --no-fold-siblings ───


def _cli(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT", "1")
    from src.runtime import region_lock_cli

    return region_lock_cli


def _parse_run(region_lock_cli, *extra: str) -> argparse.Namespace:
    return region_lock_cli.build_parser().parse_args(["run", *extra, "--", "true"])


def test_cli_fold_siblings_defaults_on(monkeypatch) -> None:
    cli = _cli(monkeypatch)
    assert _parse_run(cli, "--cpu-list", "160-183").fold_siblings is True
    assert _parse_run(cli, "--cpu-list", "160-183", "--fold-siblings").fold_siblings is True
    assert _parse_run(cli, "--cpu-list", "160-183", "--no-fold-siblings").fold_siblings is False


def test_cli_folds_sibling_cpu_list_to_physical_regions(monkeypatch, capsys) -> None:
    cli = _cli(monkeypatch)
    monkeypatch.setattr(it, "_host_sibling_map", lambda: EPYC_9655_SIBLINGS)
    assert cli._resolve_regions(_parse_run(cli, "--cpu-list", "160-183")) == frozenset({"q2", "q3"})
    assert "includes SMT siblings" in capsys.readouterr().err
    assert cli._resolve_regions(_parse_run(cli, "--cpu-list", "0-23,160-183")) == frozenset(
        {"q0", "q2", "q3"}
    )
    assert "instead of the primary-only ['q0']" in capsys.readouterr().err
    # Primary-only lists are unchanged and print no fold notice.
    assert cli._resolve_regions(_parse_run(cli, "--cpu-list", "0-95")) == frozenset(
        {"q0", "q1", "q2", "q3"}
    )
    assert "includes SMT siblings" not in capsys.readouterr().err


def test_cli_no_fold_siblings_restores_legacy_mapping(monkeypatch) -> None:
    cli = _cli(monkeypatch)

    def _boom():
        raise AssertionError("--no-fold-siblings must not consult the host topology")

    monkeypatch.setattr(it, "_host_sibling_map", _boom)
    with pytest.raises(SystemExit, match="maps to no regions"):
        cli._resolve_regions(_parse_run(cli, "--cpu-list", "160-183", "--no-fold-siblings"))
    assert cli._resolve_regions(
        _parse_run(cli, "--cpu-list", "0-23,160-183", "--no-fold-siblings")
    ) == frozenset({"q0"})


def test_cli_refuses_unresolvable_sibling(monkeypatch) -> None:
    cli = _cli(monkeypatch)

    def unavailable():
        raise CpuTopologyUnavailable("no topology")

    monkeypatch.setattr(it, "_host_sibling_map", unavailable)
    with pytest.raises(SystemExit, match="refusing rather than under-claiming"):
        cli._resolve_regions(_parse_run(cli, "--cpu-list", "160-183"))


def _run_cli(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the real CLI with q3 held by a 'peer', in an ISOLATED lock dir
    (ORCHESTRATOR_TMP_DIR=tmp_path) — no production lock is touched."""
    env = os.environ.copy()
    env["ORCHESTRATOR_TMP_DIR"] = str(tmp_path)
    env["ORCHESTRATOR_PATHS_TMP_DIR"] = str(tmp_path)
    marker = tmp_path / "child-ran"
    held = open(tmp_path / "cpu_region.GLOBAL.q3.lock", "a")
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(
            [
                sys.executable, "-m", "src.runtime.region_lock_cli", "run",
                *extra, "--no-preflight", "--timeout-s", "0.5",
                "--tag", "region-sibling-test", "--", "touch", str(marker),
            ],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False, timeout=60,
        )
    finally:
        held.close()
    return result, marker


@pytest.mark.skipif(not _host_pairs_160_with_64(), reason="host is not the EPYC 9655 SMT layout")
def test_cli_sibling_run_blocks_on_window_held_over_its_physical_cores(tmp_path: Path) -> None:
    """The incident: a peer holds q3; a script pinned to 160-183 must NOT run."""
    result, marker = _run_cli(tmp_path, "--cpu-list", "160-183")
    assert result.returncode == 75, result.stderr
    assert "['q2', 'q3']" in result.stderr
    assert not marker.exists()


@pytest.mark.skipif(not _host_pairs_160_with_64(), reason="host is not the EPYC 9655 SMT layout")
def test_cli_no_fold_siblings_reproduces_the_legacy_gap(tmp_path: Path) -> None:
    """Opt-out is exactly the old behaviour: the mixed list claims only q0 and
    runs straight through a window held on q3 (the gap this fix closes)."""
    result, marker = _run_cli(tmp_path, "--cpu-list", "0-23,160-183", "--no-fold-siblings")
    assert result.returncode == 0, result.stderr
    assert "['q0']" in result.stderr
    assert marker.exists()
