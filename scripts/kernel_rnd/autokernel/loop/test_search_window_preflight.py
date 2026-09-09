"""Original read path versus pure captured reduction; raw fixtures are synthetic."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from ..resource import preflight as pf


def _stat(pid, parent):
    return f"{pid} (fixture) " + " ".join(["S", str(parent)] + ["0"] * 17 + ["123"] + ["0"] * 12)


def _fixture(tmp_path, holder):
    proc = tmp_path / "proc"
    locks = tmp_path / "claims"
    proc.mkdir()
    locks.mkdir()
    for pid, parent in ((10, 2), (2, 1), (11, 10), (20, 1)):
        root = proc / str(pid)
        root.mkdir()
        (root / "stat").write_text(_stat(pid, parent))
        (root / "cgroup").write_text("0::/test-owned" if pid != 20 else "0::/foreign")
        (root / "cmdline").write_bytes(b"/fixture/python\0")
    path = locks / "cpu_region.GLOBAL.region-a.lock"
    path.write_text('{"schema_version":1,"pid":20,"request_tag":"synthetic"}')
    info = path.stat()
    original_key = f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}"
    line = "" if holder is None else f"1: FLOCK ADVISORY WRITE {holder} {original_key} 0 EOF\n"
    (proc / "locks").write_text(line)
    return pf.ClaimSources(locks, proc=pf.ProcSource(proc, 10)), path, line


@pytest.mark.parametrize("holder", [None, 2, 10, 11, 20, -1])
@pytest.mark.parametrize("incomplete", [False, True])
def test_read_and_captured_cpu_paths_agree_without_reopening(tmp_path, monkeypatch,
                                                          holder, incomplete):
    sources, path, raw_locks = _fixture(tmp_path, holder)
    scope = pf.PreflightScope.cpu("fixture", ["region-a"])
    if incomplete:
        (sources.proc.root / "11" / "stat").write_text("malformed synthetic stat")
    expected = pf.claim_witness_preflight(scope, sources, now=lambda: "fixture-time")
    owned = pf.read_own_scope(sources.proc)
    stat_rows = {int(item.name): ((item / "stat").read_text(), None)
                 for item in sources.proc.root.iterdir() if item.is_dir()}
    captured_owned = pf.reduce_owned_scope(self_pid=10, ancestor_stats=stat_rows,
        pid_stats=stat_rows, cgroup="/test-owned")
    assert captured_owned.to_dict() == owned.to_dict()
    info = path.stat()
    key = (f"{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}", info.st_ino)
    holders = pf.parse_proc_locks(raw_locks).get(key, pf.LockHolders())
    claim = pf.parse_region_claim(role="GLOBAL", region="region-a", lock_path=str(path),
                                 holders=holders, raw=path.read_text())
    assert claim.to_dict() == pf.read_region_claims(sources.region_lock_dir, sources.proc)[0].to_dict()
    descriptions = {pid: pf._describe_pid(sources.proc, pid)
                    for pid in holders.holder_pids if not owned.owns(pid)}
    material = pf.CapturedClaimWitness(owned, region_claims=(claim,), region_error=None,
                                     holder_descriptions=descriptions)
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("pure reducer read"))
    monkeypatch.setattr(Path, "stat", lambda *_args, **_kwargs: pytest.fail("pure reducer stat"))
    assert pf.reduce_claim_witness(scope, material, observed_at="fixture-time").to_dict() == expected.to_dict()


def test_missing_namespace_is_unknown_not_an_empty_pass():
    owned = pf.OwnedScope(10, "/fixture", frozenset({10}), {10: "self"})
    result = pf.reduce_claim_witness(pf.PreflightScope.cpu("fixture", ["a"]),
        pf.CapturedClaimWitness(owned), observed_at="fixture-time")
    assert result.verdict == pf.COULD_NOT_CHECK
    assert "not captured" in result.reasons[0]


@pytest.mark.parametrize("gpu_case", ["missing", "unavailable", "broken", "own", "foreign", "empty"])
def test_gpu_diagnostics_and_predicate_are_identical(tmp_path, gpu_case):
    sources, _path, _locks = _fixture(tmp_path, None)
    own = pf.read_own_scope(sources.proc)
    scope = pf.PreflightScope.gpu("fixture", ["gpu-a"])
    claim = pf.GpuClaimWitness("gpu-a", 10 if gpu_case == "own" else 20,
                               "synthetic observed holder", "fixture-original-device-receipt")
    def reader():
        if gpu_case == "unavailable":
            raise pf.PreflightUnavailable("synthetic unavailable")
        if gpu_case == "broken":
            raise ValueError("synthetic broken")
        return () if gpu_case == "empty" else (claim,)
    sources = replace(sources, gpu_claim_reader=None if gpu_case == "missing" else reader)
    expected = pf.claim_witness_preflight(scope, sources, now=lambda: "fixture-time")
    error = ("GPU device claim witness unavailable: synthetic unavailable"
             if gpu_case == "unavailable" else
             "GPU device claim reader raised ValueError: synthetic broken"
             if gpu_case == "broken" else None)
    captured = pf.CapturedClaimWitness(own, gpu_reader_present=gpu_case != "missing",
        gpu_error=error, gpu_claims=() if gpu_case in ("missing", "empty", "broken", "unavailable") else (claim,))
    assert pf.reduce_claim_witness(scope, captured, observed_at="fixture-time").to_dict() == expected.to_dict()


def test_partial_ancestry_never_invents_ownership():
    result = pf.reduce_owned_scope(self_pid=10, ancestor_stats={10: (None, "unreadable")},
        pid_stats={20: ("malformed", None), 11: (None, "denied")}, cgroup=None,
        cgroup_error="unreadable")
    assert result.pids == frozenset({10})
    assert len(result.incomplete) == 4
    with pytest.raises(pf.PreflightUnavailable, match="pid 1"):
        pf.reduce_owned_scope(self_pid=1, ancestor_stats={}, pid_stats={}, cgroup=None)


def test_original_lock_table_parser_retains_inode_waiters_and_ofd():
    rows = pf.parse_proc_locks("1: FLOCK ADVISORY WRITE 20 08:01:123 0 EOF\n"
        "2: -> FLOCK ADVISORY WRITE 30 08:01:123 0 EOF\n"
        "3: OFDLCK ADVISORY WRITE -1 08:02:123 0 EOF\nmalformed\n")
    assert rows[("08:01", 123)].holder_pids == (20,)
    assert rows[("08:01", 123)].waiter_pids == (30,)
    assert rows[("08:02", 123)].unattributed_holders == 1

