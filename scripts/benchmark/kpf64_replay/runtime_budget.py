#!/usr/bin/env python3
"""Estimate KPF-64 F/FB successor runtime from original service records.

This is a capacity-planning aid, not a guarantee or an inference/measurement verdict.
It reads and hashes private records but emits only aggregate numbers and digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import family_successor


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path.name}:{lineno}: invalid JSON') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{path.name}:{lineno}: row is not an object')
        rows.append(row)
    return rows


def select(rows: list[dict], expected: set[tuple[str, int]], *, complete: bool) -> list[dict]:
    chosen = {}
    for row in rows:
        key = (row.get('session'), row.get('k'))
        if key not in expected:
            continue
        if key in chosen or 'error' in row:
            raise ValueError('duplicate or failed eligible historical request')
        for field in ('prompt_n', 'total_tok', 'predicted_n', 'prompt_ms', 'svc_ms', 't_start', 't_end'):
            value = row.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid or missing {field}')
        if row['prompt_n'] <= 0 or row['total_tok'] < row['prompt_n'] or row['predicted_n'] <= 0:
            raise ValueError('invalid token counts')
        if row['prompt_ms'] <= 0 or row['svc_ms'] < row['prompt_ms'] or row['t_end'] < row['t_start']:
            raise ValueError('invalid timing')
        chosen[key] = row
    if not chosen or (complete and set(chosen) != expected):
        raise ValueError('missing eligible historical requests')
    return list(chosen.values())


def estimate(cohort: Path, report_hash: str, manifest_hash: str, f_path: Path, fb_path: Path,
             ctx_per_slot: int, predict_cap: int, gap_cap: float, startup: float,
             factor: float, planning_factor: float) -> dict:
    report = family_successor.verify(cohort, report_hash, manifest_hash)
    expected = {tuple(x) for x in report['expected']}
    if not expected or len(expected) != report['retained_requests']:
        raise ValueError('invalid successor request count')
    if ctx_per_slot <= 0 or predict_cap <= 0 or gap_cap < 0 or startup < 0 or factor < 1 or planning_factor < 1:
        raise ValueError('invalid sizing assumptions')
    f = select(records(f_path), expected, complete=True)
    fb = select(records(fb_path), expected, complete=False)
    prompt = sum(x['prompt_n'] for x in fb)
    prompt_s = sum(x['prompt_ms'] for x in fb) / 1000
    predicted = sum(x['predicted_n'] for x in fb)
    decode_s = sum(x['svc_ms'] - x['prompt_ms'] for x in fb) / 1000
    if not prompt or prompt_s <= 0 or not predicted or decode_s <= 0:
        raise ValueError('historical throughput unavailable')
    historical_full_tokens = sum(x['total_tok'] for x in f)
    historical_predicted = sum(x['predicted_n'] for x in f)
    pps, dps = prompt / prompt_s, predicted / decode_s
    # F is the same set of requests but a different cache configuration and burst
    # scheduling; future generated replies and prompts can differ. This is only a
    # conditional extrapolation based on observed per-request service counters.
    weighted_s = historical_full_tokens / pps + historical_predicted / dps
    worst_prompt_s_per_token = max(x['prompt_ms'] / 1000 / x['prompt_n'] for x in fb)
    worst_decode_s_per_token = max((x['svc_ms'] - x['prompt_ms']) / 1000 / x['predicted_n'] for x in fb)
    # Serialize all requests and assume every request reaches both declared caps.
    # Even this is not a hard runtime bound: historical maxima are not future minima.
    stress_s = len(expected) * (ctx_per_slot * worst_prompt_s_per_token +
                                predict_cap * worst_decode_s_per_token + gap_cap) + startup
    stress_window_s = math.ceil(stress_s * factor / 3600) * 3600
    planning_s = math.ceil((planning_factor * weighted_s + len(expected) * gap_cap + startup) / 86400) * 86400
    return {
        'schema': 'epyc.kpf64.runtime-budget.v1',
        'kind': 'historical_capacity_estimate_not_a_runtime_bound',
        'cohort_manifest_sha256': manifest_hash, 'cohort_report_sha256': report_hash,
        'f_records_sha256': sha(f_path), 'fb_records_sha256': sha(fb_path),
        'cohort_requests': len(expected), 'f_matched': len(f), 'fb_matched': len(fb),
        'previously_exposed': True, 'future_prompt_and_output_sizes_may_differ': True,
        'f_full_prompt_tokens': historical_full_tokens, 'f_generated_tokens': historical_predicted,
        'fb_observed_prompt_tokens': prompt, 'fb_observed_prompt_seconds': prompt_s,
        'fb_observed_generated_tokens': predicted, 'fb_observed_decode_seconds': decode_s,
        'fb_observed_span_seconds': max(x['t_end'] for x in fb) - min(x['t_start'] for x in fb),
        'fb_observed_summed_service_seconds': sum(x['svc_ms'] for x in fb) / 1000,
        'fb_weighted_prefill_tokens_per_s': pps, 'fb_weighted_decode_tokens_per_s': dps,
        'conditional_serial_weighted_seconds': weighted_s,
        'observed_max_prefill_s_per_token': worst_prompt_s_per_token,
        'observed_max_decode_s_per_token': worst_decode_s_per_token,
        'assumed_max_prompt_tokens_per_request': ctx_per_slot,
        'assumed_max_generated_tokens_per_request': predict_cap,
        'assumed_gap_cap_seconds_per_request': gap_cap,
        'assumed_load_and_cleanup_seconds': startup,
        'sequential_observed_maxima_stress_seconds': stress_s,
        'stress_margin_factor': factor, 'stress_window_seconds': stress_window_s,
        'planning_margin_factor': planning_factor, 'proposed_contiguous_window_seconds': planning_s,
        'confidence': 'none: observed maxima and demand are not proven bounds; scheduling changes can alter service rates',
        'on_cutoff': 'preserve incomplete exposure; do not claim completion or replay exposed requests silently',
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--report-sha256', required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--f-records', type=Path, required=True)
    p.add_argument('--fb-records', type=Path, required=True)
    p.add_argument('--ctx-per-slot', type=int, default=65536)
    p.add_argument('--predict-cap', type=int, default=128)
    p.add_argument('--gap-cap-s', type=float, default=60)
    p.add_argument('--startup-cleanup-s', type=float, default=1800)
    p.add_argument('--stress-margin', type=float, default=1.1)
    p.add_argument('--planning-margin', type=float, default=4.0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    try:
        result = estimate(a.cohort, a.report_sha256, a.manifest_sha256, a.f_records, a.fb_records,
                          a.ctx_per_slot, a.predict_cap, a.gap_cap_s, a.startup_cleanup_s, a.stress_margin, a.planning_margin)
        with a.out.open('x') as out:
            json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
            out.write('\n')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, f'KPF-64 runtime estimate refused: {exc}\n')
    print(f"FB: {result['cohort_requests']} requests; {result['proposed_contiguous_window_seconds'] / 3600:.0f} h planning estimate; {result['stress_window_seconds'] / 3600:.0f} h stress (neither is a bound)")


if __name__ == '__main__':
    main()
