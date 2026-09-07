#!/usr/bin/env python3
"""SMT-correct foreign-load sampler for CPU benchmark arms (INF-70 MEAS-1).

Answers one question: *while this arm was running, how much foreign work was on the
physical cores it was timing?* Two properties make it correct where the obvious
implementation is not, and both were learned by being wrong first.

1. **LIVE %CPU from /proc/<pid>/stat utime+stime deltas, never `ps %CPU`.**
   `ps %CPU` is a cumulative average over the whole process lifetime, so it structurally
   cannot see a burst inside one arm -- and the burst is the discriminator, not the mean.
   Measured 2026-09-07: the slowest of three A/A arms carried the LOWEST median foreign
   load and was undone by a single 3218% peak (32 cores). Unpinned noise (`ps`, `htop`)
   dominates a median and moves nothing.

2. **SMT sibling expansion, read from sysfs and never assumed.**
   A bench region taken as `taskset -c 0-95` covers 96 PHYSICAL cores, but logical CPU N
   and N+96 are the two threads of one physical core (verified: cpu0 -> "0,96",
   cpu88 -> "88,184"). A process pinned to 184-191 is therefore on physical cores 88-95,
   INSIDE the region -- yet a literal `Cpus_allowed_list` vs `0-95` comparison reports it
   disjoint. That exact mislabel called a contended window clean for the whole campaign
   before 2026-09-07, which is why arms measured before then carry labels that are WRONG
   rather than merely missing.

On a 96-physical-core host presenting 192 logical CPUs, every logical CPU is a sibling of
a bench core, so isolation between two CPU-heavy tenants is achievable only by TIME, never
by placement. This sampler exists to measure that, not to fix it.

Use as a library (`sample_once`, `bench_logical_cpus`) or as a CLI logger.

CLI:
    foreign_load.py --label <arm> --server-pid <pid> --out <path.jsonl> [--interval 10]
                    [--bench-cpus 0-95]
"""
import argparse
import json
import os
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")
_SIBLINGS = "/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list"


def parse_cpu_list(spec):
    """Parse a taskset-style list ("0-95", "3,5,88-91") into a set of ints.

    Tolerant by design: this parses /proc data written by other processes, and a
    malformed field must not take down a sampler that a measurement depends on.
    """
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                out |= set(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        except ValueError:
            continue
    return out


def bench_logical_cpus(bench_physical, sibling_path=_SIBLINGS, max_cpu=4096):
    """Every logical CPU whose PHYSICAL core is in `bench_physical`.

    Expands via sysfs `thread_siblings_list`. Returns the input set unchanged if sysfs
    is unreadable -- a sampler that silently narrowed its own scope would report clean
    windows, which is the failure this function exists to prevent.
    """
    bench = set(bench_physical)
    out = set()
    seen_any = False
    for c in range(max_cpu):
        p = sibling_path.format(c)
        if not os.path.exists(p):
            if c > 256:
                break
            continue
        seen_any = True
        try:
            with open(p) as fh:
                sibs = parse_cpu_list(fh.read().strip())
        except OSError:
            continue
        if sibs & bench:
            out |= sibs
    if not seen_any or not out:
        return bench
    return out


def _cpus_allowed(pid, proc="/proc"):
    try:
        with open(f"{proc}/{pid}/status") as fh:
            for line in fh:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _snapshot(proc="/proc"):
    """pid -> (cpu_ticks, comm, last_cpu). Skips anything that vanishes mid-read."""
    out = {}
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            with open(f"{proc}/{entry}/stat") as fh:
                st = fh.read()
            close = st.rindex(")")
            fields = st[close + 2:].split()
            comm = st[st.index("(") + 1:close]
            out[int(entry)] = (int(fields[11]) + int(fields[12]), comm, int(fields[36]))
        except (OSError, ValueError, IndexError):
            continue
    return out


def _own_pids(pid, proc="/proc"):
    out = {pid}
    try:
        with open(f"{proc}/{pid}/task/{pid}/children") as fh:
            for line in fh:
                out |= {int(x) for x in line.split()}
    except (OSError, ValueError):
        pass
    return out


def sample_once(prev, dt, bench_logical, own_pids, min_pct=2.0, proc="/proc"):
    """One sample. Returns (record, snapshot) -- pass the snapshot back as `prev`.

    A process counts as foreign when its Cpus_allowed_list intersects `bench_logical`,
    i.e. when it MAY run on a bench physical core. That is deliberately the permissive
    test: a process free to land on the cores being timed is a risk to the measurement
    whether or not it happened to be there at sample time.
    """
    cur = _snapshot(proc)
    total = 0.0
    own = 0.0
    top = []
    for pid, (ticks, comm, last_cpu) in cur.items():
        if pid not in prev:
            continue
        pct = 100.0 * (ticks - prev[pid][0]) / HZ / dt if dt > 0 else 0.0
        if pct < min_pct:
            continue
        if pid in own_pids:
            own += pct
            continue
        allowed = _cpus_allowed(pid, proc)
        if not (parse_cpu_list(allowed) & bench_logical):
            continue
        total += pct
        top.append(dict(pid=pid, comm=comm, pct=round(pct, 1), last_cpu=last_cpu,
                        allowed=allowed, on_bench_core=last_cpu in bench_logical))
    top.sort(key=lambda r: -r["pct"])
    return dict(foreign_pct=round(total, 1), server_pct=round(own, 1),
                n_foreign=len(top), top=top[:8]), cur


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="arm label, recorded in each row")
    ap.add_argument("--server-pid", type=int, required=True,
                    help="the arm's own server; it and its children are excluded as self")
    ap.add_argument("--out", required=True, help="jsonl output path")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--bench-cpus", default="0-95",
                    help="PHYSICAL cores the region lock covers (default 0-95); "
                         "siblings are expanded from sysfs, never assumed")
    args = ap.parse_args(argv)

    bench_logical = bench_logical_cpus(parse_cpu_list(args.bench_cpus))
    with open(args.out, "a") as fh:
        fh.write(json.dumps(dict(meta="bench_logical_cpus", label=args.label,
                                 n=len(bench_logical), lo=min(bench_logical),
                                 hi=max(bench_logical))) + "\n")

    prev, tprev = _snapshot(), time.time()
    while True:
        time.sleep(args.interval)
        now = time.time()
        rec, prev = sample_once(prev, now - tprev, bench_logical,
                                _own_pids(args.server_pid))
        tprev = now
        rec.update(t=round(now, 1), label=args.label,
                   iso=time.strftime("%H:%M:%S", time.gmtime(now)))
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
