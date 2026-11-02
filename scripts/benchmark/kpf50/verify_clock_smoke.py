#!/usr/bin/env python3
"""Require a profiler-observed HIP kernel to fit a bracketed host/HSA clock."""
import argparse
import csv
import json
import re
from pathlib import Path
from hsa_clock import offset_interval


def verify(before, after, log, rows):
    lo, hi = offset_interval([before, after])
    m = re.search(r'KPF50_CLOCK_WITNESS begin_ns=(\d+) end_ns=(\d+)', log)
    if not m: raise ValueError('missing host witness span')
    start, end = (int(x) for x in m.groups())
    if max(x['host_after_ns'] for x in before['samples']) > start or min(x['host_before_ns'] for x in after['samples']) < end:
        raise ValueError('clock samples do not bracket the witness')
    found = [r for r in rows if 'kpf50_clock_witness' in r['KernelName']]
    if len(found) != 1: raise ValueError('expected exactly one witness kernel')
    b, e = int(found[0]['BeginNs']), int(found[0]['EndNs'])
    if not (start <= b + lo <= b + hi and e + lo <= e + hi <= end and b < e):
        raise ValueError('profiler/HSA clock alignment not proved')
    return {'schema':'epyc.kpf50.clock-smoke.v1', 'verified':True,
            'host_start_ns':start, 'host_end_ns':end, 'gpu_start_ns':b, 'gpu_end_ns':e,
            'host_minus_hsa_ns':[lo,hi]}


def main():
    ap=argparse.ArgumentParser()
    for arg in ('before', 'after', 'stdout', 'csv', 'out'): ap.add_argument('--'+arg, required=True)
    a=ap.parse_args()
    try:
        before=json.loads(Path(a.before).read_text()); after=json.loads(Path(a.after).read_text())
        with open(a.csv, newline='') as f: rows=list(csv.DictReader(f))
        result=verify(before, after, Path(a.stdout).read_text(), rows)
        Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e: ap.exit(2, f'KPF50 clock smoke refused: {e}\n')

if __name__ == '__main__': main()
