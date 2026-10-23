#!/usr/bin/env python3
"""INF-70 RETEST-1 AMENDMENT-1: host-level in-window sampler.

The per-pid CPU sampler (foreign_load.py) answers "how much foreign CPU was on my cores".
That is the wrong question for a bandwidth-bound decode, and RETEST-1's failed A/A calibration
is the proof: decode fell 7% while prefill stayed flat, every arm passed the CPU screen, NUMA
placement was constant to the digit and AnonHugePages was flat. A tenant that consumes DRAM
bandwidth without consuming cores is invisible to a per-pid CPU sampler and is exactly what that
signature describes.

So this records the two things that sampler cannot:

  busy_cores   from /proc/stat -- ALL work on the host, including processes born and reaped
               between samples, which a per-pid delta sampler can never attribute (it skips any
               pid absent from the previous snapshot).
  pgpgin/out   from /proc/vmstat -- block-layer traffic into and out of the page cache, the
               memory-bandwidth proxy. Units are KB.

loadavg is recorded but is NOT a drop criterion: with OMP_WAIT_POLICY=active the arm's own 48
spinning threads dominate it, so it cannot discriminate foreign load.

  usage: host_load.py --label <arm> --out <path.jsonl> [--interval 10]
"""
import argparse, json, os, time


def procstat():
    for line in open("/proc/stat"):
        if line.startswith("cpu "):
            f = [int(x) for x in line.split()[1:]]
            return sum(f), f[3] + f[4]          # total, idle+iowait
    return 0, 0


def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k, _, v = line.partition(" ")
        if k in ("pgpgin", "pgpgout", "pgmajfault", "pgfault"):
            d[k] = int(v)
    return d


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        p = line.split()
        if p[0].rstrip(":") in ("MemFree", "Cached", "Dirty"):
            d[p[0].rstrip(":")] = int(p[1])
    return d


def node_free():
    out = []
    try:
        for n in range(4):
            with open(f"/sys/devices/system/node/node{n}/meminfo") as fh:
                for line in fh:
                    if "MemFree:" in line:
                        out.append(int(line.split()[3]) // 1024)   # MB
                        break
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()
    ncpu = os.cpu_count() or 192
    t0, (tot0, idle0), vm0 = time.time(), procstat(), vmstat()
    while True:
        time.sleep(a.interval)
        t1, (tot1, idle1), vm1 = time.time(), procstat(), vmstat()
        dt_tot = tot1 - tot0
        busy = (1.0 - (idle1 - idle0) / dt_tot) if dt_tot > 0 else 0.0
        rec = dict(t=round(t1, 1), iso=time.strftime("%H:%M:%S", time.gmtime(t1)),
                   label=a.label, dt=round(t1 - t0, 1),
                   busy_cores=round(busy * ncpu, 2), busy_pct=round(100 * busy, 2),
                   pgpgin_kb=vm1.get("pgpgin", 0) - vm0.get("pgpgin", 0),
                   pgpgout_kb=vm1.get("pgpgout", 0) - vm0.get("pgpgout", 0),
                   pgmajfault=vm1.get("pgmajfault", 0) - vm0.get("pgmajfault", 0),
                   loadavg=open("/proc/loadavg").read().split()[0],
                   node_free_mb=node_free(), **{k.lower(): v for k, v in meminfo().items()})
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        t0, tot0, idle0, vm0 = t1, tot1, idle1, vm1


if __name__ == "__main__":
    main()
