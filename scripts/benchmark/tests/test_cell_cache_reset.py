#!/usr/bin/env python3
"""Offline tests for scripts/lib/cell_cache_reset.py (BRF-T3 E5 per-cell reset).

Every primitive that could touch the host (region-lock status, sync,
drop_caches, numa_evict, sysfs) is injected as a fake; ``subprocess.run`` and
``subprocess.Popen`` are additionally guarded so an un-injected call fails the
test instead of reaching a shared host. Runs under pytest or standalone:

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/tests/test_cell_cache_reset.py
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _BENCHMARK_DIR.parent
for _p in (str(_SCRIPTS_DIR / "lib"), str(_BENCHMARK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cell_cache_reset as ccr  # noqa: E402

ME = 4242
REGIONS = ("q0", "q1", "q2", "q3")
# Host shape: 4 NUMA nodes, 24 physical cores each, SMT siblings +96.
CPU_TO_NODE = {cpu: cpu // 24 for cpu in range(96)} | {cpu + 96: cpu // 24 for cpu in range(96)}
INTERLEAVE_FULL = [("0-95", "interleave=all")]
TWO_HALVES = [("0-47,96-143", "none"), ("48-95,144-191", "none")]
HALF0 = [("0-47,96-143", "none")]


def _guard(*args, **kwargs):
    raise AssertionError(f"real subprocess call in an offline test: {args!r}")


@contextlib.contextmanager
def no_subprocess():
    saved = subprocess.run, subprocess.Popen
    subprocess.run = subprocess.Popen = _guard
    try:
        yield
    finally:
        subprocess.run, subprocess.Popen = saved


def lock_status(region_pid=ME, gpu_pid=ME, gpu_mode="exclusive", regions=REGIONS, gpu_held=True):
    return {
        "regions": [
            {
                "region": region,
                "global_held": region in regions,
                "holders": [{"pid": region_pid, "regions": list(REGIONS)}] if region in regions else [],
            }
            for region in REGIONS
        ],
        "gpu_quiet": {
            "held": gpu_held,
            "mode": gpu_mode if gpu_held else None,
            "holders": [{"pid": gpu_pid, "mode": gpu_mode}] if gpu_held else [],
        },
    }


def snap(tag):
    return {"captured_at": tag, "free_kib_by_node": {"N0": 1}, "pages_by_node": {"N0": 2}}


class Recorder:
    def __init__(self, rc=0, raises=None):
        self.calls = []
        self.rc = rc
        self.raises = raises

    def drop(self):
        self.calls.append(("drop",))
        if self.raises:
            raise self.raises
        return [{"argv": ["sync"], "returncode": 0}, {"argv": ccr.DROP_CACHES_ARGV, "returncode": self.rc}]

    def evict(self, nodes, target):
        self.calls.append(("evict", list(nodes), target))
        if self.raises:
            raise self.raises
        return [{"argv": ccr.numa_evict_argv(nodes, target), "returncode": self.rc}]


def reset(manifest, instances, rec, status, model_bytes=None):
    snaps = iter(["before", "after"])
    with no_subprocess():
        return ccr.perform_cell_cache_reset(
            cell_id="c",
            manifest=manifest,
            instances=instances,
            model_bytes=model_bytes,
            status_reader=lambda: status,
            ancestors=lambda: {ME, 1},
            cpu_to_node=lambda: dict(CPU_TO_NODE),
            snapshot=lambda: snap(next(snaps)),
            drop_fn=rec.drop,
            evict_fn=rec.evict,
            clock=lambda: "2026-11-02T00:00:00+00:00",
        )


DG = {"decision_grade_intent": True, "cache_reset": "per_cell"}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def test_resolve_policy_defaults():
    assert ccr.resolve_policy({"decision_grade_intent": True}) == "per_cell"
    assert ccr.resolve_policy({"decision_grade_intent": False}) == "per_group"
    assert ccr.resolve_policy({}) == "per_group"
    assert ccr.resolve_policy({"decision_grade_intent": True, "cache_reset": "none"}) == "none"


def test_violations_refuse_decision_grade_without_per_cell():
    for policy in ("per_group", "none"):
        manifest = {"decision_grade_intent": True, "cache_reset": policy}
        errors = ccr.cache_reset_violations(manifest, require_declared=False)
        assert errors and "warm_cache_control=true" in errors[0], errors
        manifest["warm_cache_control"] = True
        assert ccr.cache_reset_violations(manifest, require_declared=True) == []


def test_violations_declared_requirement_is_schema_only():
    manifest = {"decision_grade_intent": True}
    assert ccr.cache_reset_violations(manifest, require_declared=False) == []
    errors = ccr.cache_reset_violations(manifest, require_declared=True)
    assert len(errors) == 1 and "must declare cache_reset" in errors[0]


def test_violations_contradiction_enum_and_type():
    assert "contradicts" in ccr.cache_reset_violations(
        {"decision_grade_intent": True, "cache_reset": "per_cell", "warm_cache_control": True},
        require_declared=True,
    )[0]
    assert "is not one of" in ccr.cache_reset_violations(
        {"cache_reset": "per_model"}, require_declared=False
    )[0]
    assert "must be a boolean" in ccr.cache_reset_violations(
        {"cache_reset": "per_group", "warm_cache_control": 1}, require_declared=False
    )[0]
    # Non-decision-grade cells may relax freely.
    for policy in ("per_group", "none", "per_cell"):
        assert ccr.cache_reset_violations(
            {"decision_grade_intent": False, "cache_reset": policy}, require_declared=True
        ) == []


# ---------------------------------------------------------------------------
# Whole-host evidence
# ---------------------------------------------------------------------------


def test_whole_host_requires_all_regions_and_gpu_quiet_exclusive_in_our_tree():
    ours = {ME, 1}
    assert ccr.whole_host_evidence(lock_status(), ours) == {"held": True, "reasons": []}
    foreign = ccr.whole_host_evidence(lock_status(region_pid=999, gpu_pid=999), ours)
    assert not foreign["held"] and len(foreign["reasons"]) == 5
    partial = ccr.whole_host_evidence(lock_status(regions=("q0", "q1")), ours)
    assert not partial["held"] and partial["reasons"] == ["q2 not held", "q3 not held"]
    shared = ccr.whole_host_evidence(lock_status(gpu_mode="shared"), ours)
    assert not shared["held"] and "not exclusive" in shared["reasons"][0]
    no_gpu = ccr.whole_host_evidence(lock_status(gpu_held=False), ours)
    assert no_gpu["reasons"] == ["gpu-quiet not held"]
    assert ccr.whole_host_evidence(None, ours)["held"] is False


def test_ancestor_pids_walks_proc_stat_with_spaces_in_comm():
    with tempfile.TemporaryDirectory() as tmp:
        proc = Path(tmp)
        for pid, ppid in ((30, 20), (20, 10), (10, 0)):
            (proc / str(pid)).mkdir()
            (proc / str(pid) / "stat").write_text(f"{pid} (a b) S {ppid} 0 0\n")
        assert ccr.ancestor_pids(30, proc) == {30, 20, 10}


# ---------------------------------------------------------------------------
# Topology + snapshot
# ---------------------------------------------------------------------------


def test_cell_memory_nodes_scopes_by_policy():
    assert ccr.cell_memory_nodes(INTERLEAVE_FULL, CPU_TO_NODE) == [0, 1, 2, 3]
    assert ccr.cell_memory_nodes(HALF0, CPU_TO_NODE) == [0, 1]
    assert ccr.cell_memory_nodes(TWO_HALVES, CPU_TO_NODE) == [0, 1, 2, 3]
    assert ccr.cell_memory_nodes([("72-95,168-191", "none")], CPU_TO_NODE) == [3]


def test_node_memory_snapshot_reads_free_and_file_pages():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for node, (free, files) in enumerate(((1000, 400), (2000, 800))):
            (base / f"node{node}").mkdir()
            (base / f"node{node}" / "meminfo").write_text(
                f"Node {node} MemTotal:  9999 kB\nNode {node} MemFree:  {free} kB\n"
                f"Node {node} FilePages:  {files} kB\n"
            )
            (base / f"node{node}" / "cpulist").write_text(f"{node * 2}-{node * 2 + 1}\n")
        shot = ccr.node_memory_snapshot(base, page_kib=4)
        assert shot["free_kib_by_node"] == {"N0": 1000, "N1": 2000}
        assert shot["pages_by_node"] == {"N0": 100, "N1": 200}
        assert ccr.read_cpu_to_node(base) == {0: 0, 1: 0, 2: 1, 3: 1}


def test_evict_target_never_below_floor_and_covers_model_share():
    assert ccr.evict_target_gib(None, 4) == 40
    assert ccr.evict_target_gib(36 * ccr.GIB, 4) == 40
    assert ccr.evict_target_gib(400 * ccr.GIB, 2) == 198  # capped for numa_evict
    assert ccr.evict_target_gib(200 * ccr.GIB, 4) == 50


# ---------------------------------------------------------------------------
# The reset + witness
# ---------------------------------------------------------------------------


def test_whole_host_uses_drop_caches_and_writes_full_witness():
    rec = Recorder()
    witness = reset(DG, TWO_HALVES, rec, lock_status())
    assert rec.calls == [("drop",)]
    assert witness["ok"] is True and witness["performed"] is True
    assert witness["method"] == "drop_caches"
    assert witness["scope_nodes"] == "host"
    assert witness["cold_cache_guaranteed"] is True
    assert witness["timestamp"] == "2026-11-02T00:00:00+00:00"
    assert witness["before"]["captured_at"] == "before"
    assert witness["after"]["captured_at"] == "after"
    assert witness["before"]["pages_by_node"] and witness["after"]["free_kib_by_node"]
    assert witness["commands"][1]["argv"] == ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"]
    assert witness["schema"] == "cell-cache-reset-witness/1"


def test_without_whole_host_uses_scoped_numa_evict_never_drop_caches():
    rec = Recorder()
    witness = reset(DG, HALF0, rec, lock_status(region_pid=999), model_bytes=200 * ccr.GIB)
    assert rec.calls == [("evict", [0, 1], 100)]
    assert witness["method"] == "numa_evict"
    assert witness["scope_nodes"] == [0, 1]
    assert witness["evict_target_gib"] == 100
    assert witness["cold_cache_guaranteed"] is False
    assert "whole host NOT held" in witness["method_reason"]
    assert witness["ok"] is True and witness["performed"] is True
    argv = witness["commands"][0]["argv"]
    assert argv[1].endswith("scripts/utils/numa_evict.py")
    assert argv[2:] == ["--nodes", "0,1", "--target-gib", "100"]


def test_gpu_quiet_shared_is_not_whole_host():
    rec = Recorder()
    witness = reset(DG, INTERLEAVE_FULL, rec, lock_status(gpu_mode="shared"))
    assert rec.calls == [("evict", [0, 1, 2, 3], 40)]
    assert witness["method"] == "numa_evict"


def test_unreadable_lock_status_falls_back_to_scoped_eviction():
    rec = Recorder()
    witness = reset(DG, HALF0, rec, None)
    assert [c[0] for c in rec.calls] == ["evict"]
    assert witness["whole_host"]["reasons"] == ["region-lock status unavailable"]


def test_per_group_and_none_do_nothing_but_still_record():
    for policy in ("per_group", "none"):
        rec = Recorder()
        manifest = {"decision_grade_intent": False, "cache_reset": policy}
        witness = reset(manifest, HALF0, rec, lock_status())
        assert rec.calls == []
        assert witness["performed"] is False and witness["method"] is None
        assert witness["ok"] is True and witness["policy"] == policy
        assert witness["before"]["captured_at"] == "before"


def test_failed_reset_is_witnessed_not_raised():
    rec = Recorder(rc=1)
    witness = reset(DG, HALF0, rec, lock_status())
    assert witness["ok"] is False and witness["performed"] is False
    assert "drop_caches failed" in witness["error"] and "rc=1" in witness["error"]
    assert witness["after"]["captured_at"] == "after"

    rec = Recorder(raises=RuntimeError("numactl missing"))
    witness = reset(DG, HALF0, rec, lock_status(region_pid=999))
    assert witness["ok"] is False and "numactl missing" in witness["error"]


def test_real_drop_argv_matches_the_existing_reanchor_mechanism():
    # Reuse, not reinvention: the same argv fg4b_a4_cpu_optimized_reanchor uses.
    source = (_BENCHMARK_DIR / "fg4b_a4_cpu_optimized_reanchor.py").read_text()
    assert 'drop_command = ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"]' in source
    assert ccr.DROP_CACHES_ARGV == ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"]
    assert ccr.NUMA_EVICT.is_file()


def _run_all() -> int:
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
