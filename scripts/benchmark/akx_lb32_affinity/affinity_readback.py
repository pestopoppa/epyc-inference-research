#!/usr/bin/env python3
"""LB3.2b llama-bench affinity readback that survives OMP_PROC_BIND.

Root cause of the rc=68 refusals (lb32-cv2-20261026): the legacy check read
os.sched_getaffinity(pid) -- the MAIN TID only -- after the benchmark loaded,
and required it to equal the whole grant. With OMP_PROC_BIND=spread and
OMP_PLACES=cores, libgomp pins the master thread to one place at the first
parallel region, so the main TID mask is a single core and the check refused a
correctly placed run.

Fix (process-wide readback): read every task of the process in two matching
snapshots under a pidfd; once OpenMP is live (bench=True) each mask must be
nonempty and a subset of the grant; pre-init launcher tasks (bench=False) must
equal the grant exactly. Point-in-time mask evidence only, never residency or
utilization. Logic promoted verbatim from the cv5 affinity-v3 package
(/mnt/raid0/llm/tmp/autokernel-all-devices-all-dimensions/lb32-cv5-20261104/control_checks.py),
which passed its G-t48 arm on 2026-11-04 (48 tasks, RC 0).
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import select
import sys


def legacy_affinity(pid, cpus):
    """Pre-v3 check, kept only so the fixture can show it refusing."""
    actual = os.sched_getaffinity(pid)
    grant = expand(cpus)
    status = Path(f"/proc/{pid}/status").read_text()
    readback = expand(next(l.split(":", 1)[1].strip() for l in status.splitlines()
                           if l.startswith("Cpus_allowed_list:")))
    assert readback == actual
    assert actual and actual <= grant
    assert actual == grant, f"main TID mask {sorted(actual)} != grant {sorted(grant)}"
    return {"pid": pid, "actual": sorted(actual), "grant": sorted(grant), "RC": 0}


def expand(text):
    out = set()
    for part in text.split(','):
        a, sep, b = part.strip().partition('-')
        lo, hi = int(a), int(b if sep else a)
        if lo < 0 or hi < lo:
            raise ValueError('invalid CPU range')
        out.update(range(lo, hi + 1))
    if not out:
        raise ValueError('empty grant')
    return out


def _starttime(path):
    # comm may contain spaces and parentheses; fields after the final ')' start
    # at field 3 (state), and starttime is field 22.
    fields = path.read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ValueError('process/task is a zombie')
    return int(fields[19])


def _snapshot(task_dir, grant, bench):
    before = {int(p.name) for p in task_dir.iterdir()}
    if not before:
        raise ValueError('process has no readable tasks')
    tasks, identities = {}, {}
    for tid in sorted(before):
        entry = task_dir / str(tid)
        started = _starttime(entry / 'stat')
        allowed = os.sched_getaffinity(tid)
        status = (entry / 'status').read_text()
        readback = expand(next(line.split(':', 1)[1].strip()
                               for line in status.splitlines()
                               if line.startswith('Cpus_allowed_list:')))
        if readback != allowed or _starttime(entry / 'stat') != started:
            raise ValueError(f'TID {tid} changed during readback')
        if not allowed or not allowed <= grant:
            raise ValueError(f'TID {tid} escapes grant: {sorted(allowed)}')
        if not bench and allowed != grant:
            raise ValueError(f'TID {tid} pre-init mask differs from grant')
        tasks[tid], identities[tid] = sorted(allowed), started
    if before != {int(p.name) for p in task_dir.iterdir()}:
        raise ValueError('task membership changed during readback')
    return tasks, identities


def affinity(pid, cpus, bench=False, expected_starttime=None):
    """Read two matching task snapshots while a pidfd pins process identity.

    This is point-in-time evidence of masks, not continuous containment, CPU
    utilization, or coverage of every granted CPU. Fail on observed churn;
    callers must sample during their actual loaded progress event. The caller
    may bind the observation to a previously captured process starttime.
    """
    if pid <= 0:
        raise ValueError('PID must be positive')
    grant = expand(cpus)
    proc = Path(f'/proc/{pid}')
    pidfd = os.pidfd_open(pid)
    try:
        started = _starttime(proc / 'stat')
        if expected_starttime is not None and started != expected_starttime:
            raise ValueError('process identity does not match expected starttime')
        first = _snapshot(proc / 'task', grant, bench)
        second = _snapshot(proc / 'task', grant, bench)
        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        if poller.poll(0) or _starttime(proc / 'stat') != started:
            raise ValueError('process exited or changed during readback')
        if first != second:
            raise ValueError('task membership, identity or masks changed between snapshots')
        tasks, identities = second
        union = set().union(*(set(mask) for mask in tasks.values()))
        return {'pid': pid, 'leader_tid': pid, 'starttime_ticks': started,
                'readback_at': datetime.now(timezone.utc).isoformat(),
                'tasks': tasks, 'task_starttime_ticks': identities,
                'union': sorted(union), 'grant': sorted(grant),
                'scope': 'all_tasks_two_matching_snapshots',
                'proof': 'point_in_time_mask_subset' if bench else 'point_in_time_mask_equals_grant',
                'RC': 0}
    finally:
        os.close(pidfd)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--cpus", required=True)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    try:
        r = legacy_affinity(a.pid, a.cpus) if a.legacy else affinity(a.pid, a.cpus, a.bench)
        print(json.dumps(r))
        return 0
    except (AssertionError, OSError, ValueError, StopIteration) as exc:
        print(json.dumps({"RC": 68, "error": str(exc)}))
        return 68


if __name__ == "__main__":
    sys.exit(main())
