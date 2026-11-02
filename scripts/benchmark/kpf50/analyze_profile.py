#!/usr/bin/env python3
"""Fail-closed KPF-50 step attribution from calibrated native GPU dispatches.

The trunk marginal is a *counterfactual*, not a timed section of a fused kernel.
A valid output is descriptive only; this tool never emits the proposed gate verdict.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
from hsa_clock import offset_interval
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interval_union(intervals):
    ordered = sorted(intervals)
    total = 0
    last_begin = last_end = None
    for begin, end in ordered:
        require(isinstance(begin, int) and isinstance(end, int) and 0 <= begin < end, 'invalid interval')
        if last_end is None or begin > last_end:
            if last_end is not None:
                total += last_end - last_begin
            last_begin, last_end = begin, end
        else:
            last_end = max(last_end, end)
    return total + (last_end - last_begin if last_end is not None else 0)


def analyze_profile(manifest, dispatches, clock_bounds):
    require(manifest.get('schema') == 'epyc.kpf50.profile.v1', 'unsupported schema')
    require(manifest.get('classification') == 'prospective-diagnostic', 'must remain diagnostic')
    for name in ('model_sha256', 'server_sha256', 'hip_library_sha256', 'prompt_sha256', 'fork_witness_sha256', 'profiler_sha256'):
        require(re.fullmatch(r'[a-f0-9]{64}', str(manifest.get(name, ''))) is not None, f'missing {name}')
    require(manifest.get('gpu_resident') is True and manifest.get('exclusive_gpu_claim') is True, 'missing live GPU residency or exclusive claim')
    require(manifest.get('target') == {'sequences': 4, 'context_tokens': 80000, 'trunk_tokens': 60000, 'drafter': 'DFlash2'}, 'incorrect target shape')
    require(manifest.get('fork_witnesses') == 3, 'three witnessed child forks required')
    require(manifest.get('clock_domain') == 'hsa-ns' and manifest.get('rocprof_hsa_alignment_verified') is True, 'missing validated rocprof/HSA clock alignment')
    require(isinstance(clock_bounds, tuple) and len(clock_bounds) == 2 and all(type(x) is int for x in clock_bounds), 'missing HSA/host calibration')
    clock_lo, clock_hi = clock_bounds
    require(clock_lo <= clock_hi and clock_hi - clock_lo <= manifest.get('max_clock_uncertainty_ns', -1), 'clock uncertainty exceeds predeclared bound')
    patterns = manifest.get('attention_kernel_patterns')
    require(isinstance(patterns, list) and patterns and all(isinstance(p, str) and p for p in patterns), 'missing predeclared attention patterns')
    regexes = [re.compile(p) for p in patterns]
    steps = manifest.get('steps')
    require(isinstance(steps, list) and steps, 'no step boundaries')
    ordered_steps = sorted(steps, key=lambda x: x['start_ns'])
    for i, s in enumerate(ordered_steps):
        require(s.get('phase') in ('decode', 'verify'), 'unknown or prefill step')
        require(s.get('complete') is True and s.get('synchronized') is True, 'step lacks synchronized complete boundary')
        require(isinstance(s.get('start_ns'), int) and isinstance(s.get('end_ns'), int) and s['start_ns'] < s['end_ns'], 'invalid step')
        require(i == 0 or ordered_steps[i-1]['end_ns'] <= s['start_ns'], 'overlapping steps')
    summaries = []
    for step in ordered_steps:
        all_gpu, attn = [], []
        # host = GPU + offset. For every admissible offset, the inner window
        # is certainly inside the host step; the outer window possibly overlaps.
        inner_start, inner_end = step['start_ns'] - clock_lo, step['end_ns'] - clock_hi
        outer_start, outer_end = step['start_ns'] - clock_hi, step['end_ns'] - clock_lo
        require(inner_start < inner_end, 'step shorter than clock uncertainty')
        for d in dispatches:
            begin, end = d['begin_ns'], d['end_ns']
            if begin < outer_end and end > outer_start:
                require(inner_start <= begin < end <= inner_end, 'dispatch crosses or ambiguously overlaps step boundary')
                all_gpu.append((begin, end))
                if any(p.search(d['kernel']) for p in regexes):
                    attn.append((begin, end))
        require(all_gpu and attn, 'step missing dispatches or attention kernels')
        duration = step['end_ns'] - step['start_ns']
        attention = interval_union(attn)
        summaries.append({'id': step['id'], 'phase': step['phase'], 'step_ns': duration,
                          'gpu_busy_ns': interval_union(all_gpu), 'attention_busy_ns': attention,
                          'attention_share': attention / duration})
    # Counterfactual times require real hardware captures with exact matching kernel and input
    # scope. Even valid captures estimate a marginal and are NOT a direct trunk timing.
    pairs = manifest.get('counterfactual_pairs', [])
    estimates = []
    for pair in pairs:
        require(pair.get('complete') is True and pair.get('same_query_mask_layout') is True,
                'counterfactual lacks matched query/mask/layout')
        require(pair.get('same_kernel') is True and pair.get('alternate_order') is True,
                'counterfactual kernel/order mismatch')
        full, private = pair.get('full_ns'), pair.get('private_only_ns')
        require(type(full) is int and type(private) is int and full > 0 and private > 0,
                'invalid counterfactual time')
        estimates.append(full - private)
    return {'schema': 'epyc.kpf50.analysis.v1', 'classification': 'observation',
            'steps': summaries,
            'mean_attention_share': statistics.fmean(s['attention_share'] for s in summaries),
            'trunk_attention_ns': None,
            'trunk_time_reason': 'fused attention does not separately time trunk cells',
            'counterfactual_marginal_ns': estimates,
            'clock_offset_host_minus_hsa_ns': [clock_lo, clock_hi],
            'gate_verdict': 'UNRESOLVED',
            'gate_reason': 'trunk timing is not directly observed and the 20% threshold is proposed'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--dispatch-csv', required=True, help='rocprof CSV with KernelName, BeginNs, EndNs')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    try:
        manifest = json.loads(Path(args.manifest).read_text())
        require(sha256(args.dispatch_csv) == manifest.get('dispatch_sha256'), 'dispatch hash mismatch')
        with open(args.dispatch_csv, newline='') as f:
            rows = [{'kernel': r['KernelName'], 'begin_ns': int(r['BeginNs']), 'end_ns': int(r['EndNs'])} for r in csv.DictReader(f)]
        captures = []
        for key in ('clock_before', 'clock_after'):
            c = manifest[key]
            path = Path(c['path'])
            require(sha256(path) == c['sha256'], f'{key} hash mismatch')
            captures.append(json.loads(path.read_text()))
        steps = manifest['steps']
        require(max(s['host_after_ns'] for s in captures[0]['samples']) <= min(s['start_ns'] for s in steps), 'before clock capture does not precede steps')
        require(min(s['host_before_ns'] for s in captures[1]['samples']) >= max(s['end_ns'] for s in steps), 'after clock capture does not follow steps')
        result = analyze_profile(manifest, rows, offset_interval(captures))
        result['manifest_sha256'] = sha256(args.manifest)
        result['dispatch_sha256'] = sha256(args.dispatch_csv)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    except (ValueError, KeyError, OSError, json.JSONDecodeError, re.error) as exc:
        ap.exit(2, f'KPF-50 refused: {exc}\n')


if __name__ == '__main__':
    main()
