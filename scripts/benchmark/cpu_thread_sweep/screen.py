#!/usr/bin/env python3
"""Pre-registered contention screens for the R23-64 thread sweep.

DERIVED FROM the INF-70 RETEST-1 screen (sha256 52d4f146d195...a187, see PROVENANCE.txt); the thresholds are unchanged:

    foreign CPU, OTHER max (foreign_load.py)   <200 CLEAN   200..400 FLAGGED   >=400 DROPPED
    host busy cores - OWN THREADS - announced  > 6  -> DROPPED
    pgpgin over the launch                     > 8 GB -> DROPPED

TWO deliberate changes, both forced by this sweep and neither a threshold change:

  1. OWN_THREADS is a PARAMETER. The original hard-coded 48 (the champion's -t), which for the t96 arm
     would charge the arm's own 48 extra OMP threads to "foreign load" and DROP every t96 launch.
  2. The ANNOUNCED lane defaults to EMPTY. The original exempted processes pinned inside 184-191 (the
     GPU session's host threads). R23-64 requires the GPU loop DOWN, and 184-191 are SMT siblings of
     cores 88-95, i.e. INSIDE the bench region, so a process there is contention, not common-mode.

A DROPPED launch is excluded BEFORE any statistic is computed (analyze_sweep.py routes every number
through screen_launch()); a launch with no sampler files is NO-SAMPLES and is excluded too, unless the
caller asks otherwise explicitly.
"""
import argparse
import json
import os

CLEAN, FLAG, DROP = 200.0, 400.0, 400.0
EXCESS_DROP = 6.0
PGPGIN_DROP_KB = 8 * 1024 * 1024


def parse_cpu_list(spec):
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


def screen(path, announced=frozenset()):
    """Foreign-CPU screen over a foreign_load.py record. Verdict is written on the OTHER max, never the
    median: the discriminator for a ruined launch is a burst, and a burst is invisible to an average."""
    if not os.path.exists(path):
        return None
    announced = set(announced)
    other, ann, n, heavy = [], [], 0, {}
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("meta") or "foreign_pct" not in r:
            continue
        n += 1
        o = a = 0.0
        for t in r.get("top", []):
            allowed = parse_cpu_list(t.get("allowed", ""))
            pct = float(t.get("pct", 0))
            if announced and allowed and allowed <= announced:
                a += pct
            else:
                o += pct
                if pct >= 50:
                    k = t.get("comm", "?")
                    heavy[k] = max(heavy.get(k, 0.0), pct)
        # `top` is capped at 8 entries; the remainder is unattributed and goes to OTHER (conservative).
        resid = float(r.get("foreign_pct", 0)) - (o + a)
        if resid > 0:
            o += resid
        other.append(o)
        ann.append(a)
    if not n:
        return None
    other.sort()
    ann.sort()
    omax, omed = max(other), other[len(other) // 2]
    verdict = "DROPPED" if omax >= DROP else ("FLAGGED" if omax >= CLEAN else "CLEAN")
    return dict(label=os.path.basename(path).split(".")[0], n_samples=n,
                other_p50=round(omed, 1), other_max=round(omax, 1),
                announced_p50=round(ann[len(ann) // 2], 1), announced_max=round(max(ann), 1),
                heavy=dict(sorted(heavy.items(), key=lambda kv: -kv[1])[:5]), verdict=verdict)


def host_screen(path, own_threads, announced_cores=0.0):
    """Host-level screen over a host_load.py record: busy cores beyond the launch's own threads, and
    page-cache traffic (the memory-bandwidth proxy a per-pid CPU sampler cannot see)."""
    if not os.path.exists(path):
        return None
    busy, pgin, pgout, majf, samples, load = [], 0, 0, 0, 0, []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        samples += 1
        busy.append(float(r.get("busy_cores", 0)))
        pgin += int(r.get("pgpgin_kb", 0))
        pgout += int(r.get("pgpgout_kb", 0))
        majf += int(r.get("pgmajfault", 0))
        try:
            load.append(float(r.get("loadavg", 0)))
        except (TypeError, ValueError):
            pass
    if not samples:
        return None
    busy.sort()
    bmax, bmed = max(busy), busy[len(busy) // 2]
    excess = bmax - float(own_threads) - float(announced_cores)
    v = "DROPPED" if (excess > EXCESS_DROP or pgin > PGPGIN_DROP_KB) else "CLEAN"
    return dict(n=samples, busy_max=round(bmax, 2), busy_p50=round(bmed, 2), own_threads=own_threads,
                excess_cores=round(excess, 2), pgpgin_gb=round(pgin / 1048576, 2),
                pgpgout_gb=round(pgout / 1048576, 2), pgmajfault=majf,
                loadavg_max=round(max(load), 2) if load else None, verdict=v)


def screen_launch(runs_dir, label, own_threads):
    """(verdict, detail) for one launch: DROPPED if either instrument says so, NO-SAMPLES if either
    record is absent/empty, else the worst of the two (FLAGGED > CLEAN)."""
    c = screen(os.path.join(runs_dir, f"{label}.foreign.jsonl"))
    h = host_screen(os.path.join(runs_dir, f"{label}.host.jsonl"), own_threads)
    if c is None or h is None:
        return "NO-SAMPLES", dict(cpu=c, host=h)
    if c["verdict"] == "DROPPED" or h["verdict"] == "DROPPED":
        return "DROPPED", dict(cpu=c, host=h)
    return ("FLAGGED" if c["verdict"] == "FLAGGED" else "CLEAN"), dict(cpu=c, host=h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["one", "host"])
    ap.add_argument("path")
    ap.add_argument("--own-threads", type=float, default=0.0, help="host mode: the launch's -t")
    a = ap.parse_args()
    if a.mode == "one":
        s = screen(a.path)
        print("no samples" if not s else
              f"other p50={s['other_p50']} max={s['other_max']} | n={s['n_samples']} | {s['verdict']} | heavy={s['heavy']}")
    else:
        h = host_screen(a.path, a.own_threads)
        print("no host samples" if not h else
              f"busy_cores p50={h['busy_p50']} max={h['busy_max']} own={h['own_threads']} excess={h['excess_cores']} "
              f"| pgpgin={h['pgpgin_gb']}GB pgpgout={h['pgpgout_gb']}GB majf={h['pgmajfault']} "
              f"| loadavg_max={h['loadavg_max']} | n={h['n']} | {h['verdict']}")


if __name__ == "__main__":
    main()
