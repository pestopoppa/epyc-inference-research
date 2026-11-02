#!/usr/bin/env python3
"""Capture a bracketed CLOCK_MONOTONIC/HSA system-clock correspondence.

Run before and after the measured window while holding its exclusive GPU claim.
HSA initialization touches the GPU runtime: do not run during another job.
"""
import argparse
import ctypes
import json
import os
import time
from pathlib import Path


def sample(lib, n):
    get_info = lib.hsa_system_get_info
    get_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
    get_info.restype = ctypes.c_int
    frequency = ctypes.c_uint64()
    if get_info(3, ctypes.byref(frequency)) != 0 or frequency.value == 0:
        raise RuntimeError('HSA timestamp frequency unavailable')
    rows = []
    for _ in range(n):
        ticks = ctypes.c_uint64()
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        status = get_info(2, ctypes.byref(ticks))
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        if status != 0 or after < before:
            raise RuntimeError('HSA timestamp unavailable')
        rows.append({'host_before_ns': before, 'hsa_ticks': ticks.value, 'host_after_ns': after})
    return {'schema': 'epyc.kpf50.hsa-clock.v1', 'frequency_hz': frequency.value, 'samples': rows}


def offset_interval(captures):
    """Bound host monotonic minus HSA nanoseconds using exact integer rounding."""
    if len(captures) < 2:
        raise ValueError('need before and after captures')
    freqs = {c.get('frequency_hz') for c in captures}
    if len(freqs) != 1 or not isinstance(next(iter(freqs)), int) or next(iter(freqs)) <= 0:
        raise ValueError('missing or changed frequency')
    frequency = next(iter(freqs))
    intervals = []
    last_tick = -1
    for capture in captures:
        if capture.get('schema') != 'epyc.kpf50.hsa-clock.v1' or not capture.get('samples'):
            raise ValueError('missing HSA sample')
        for row in capture['samples']:
            b, t, e = row['host_before_ns'], row['hsa_ticks'], row['host_after_ns']
            if not all(type(x) is int for x in (b, t, e)) or b > e or t < last_tick:
                raise ValueError('invalid or unordered sample')
            last_tick = t
            scaled_floor = t * 1000000000 // frequency
            scaled_ceil = (t * 1000000000 + frequency - 1) // frequency
            intervals.append((b - scaled_ceil, e - scaled_floor))
    lo = max(a for a, _ in intervals)
    hi = min(b for _, b in intervals)
    if lo > hi:
        raise ValueError('HSA/host offset not stable within measured brackets')
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--samples', type=int, default=20)
    args = ap.parse_args()
    if args.samples < 2 or args.samples > 10000:
        ap.error('sample count outside [2,10000]')
    if os.environ.get('HSA_OVERRIDE_CPU_AFFINITY_DEBUG') != '0':
        ap.error('require HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0')
    lib = ctypes.CDLL('libhsa-runtime64.so.1')
    if lib.hsa_init() != 0:
        raise RuntimeError('hsa_init failed')
    try:
        out = sample(lib, args.samples)
        out['pid'] = os.getpid()
        Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    finally:
        lib.hsa_shut_down()


if __name__ == '__main__':
    main()
