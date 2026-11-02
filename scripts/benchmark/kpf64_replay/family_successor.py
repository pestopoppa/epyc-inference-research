#!/usr/bin/env python3
"""Build the structurally eligible KPF-64 family successor without redrawing sessions.

Outputs contain private transcript text and must remain in private scratch. The report
contains hashes, counts and exclusions only. See FAMILY_SUCCESSOR.md before use.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import adapter
from cohort_audit import audit_cohort


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, dest: Path, expected_sha: str) -> dict:
    manifest_path = source / 'manifest.json'
    source_sha = sha(manifest_path)
    if source_sha != expected_sha:
        raise ValueError('source manifest digest mismatch')
    if dest.exists():
        raise ValueError('output already exists')
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get('schema') != 'kpf64.sessions.v2':
        raise ValueError('unsupported manifest schema')
    audit = audit_cohort(source, children='bursts')
    entries = source_manifest['sessions']
    source_files = {r['id']: sha(source / f"{r['id']}.json") for r in entries}
    by_id = {row['id']: row for row in entries}
    # An invalid or unknown timed child is excluded as a whole; a parent defect
    # refuses the entire transformation instead of inventing lineage/timing.
    issues = audit['issues'] + audit['unknown_timing']
    reasons: dict[str, set[str]] = {}
    for issue in issues:
        sid = issue['session']
        if not by_id[sid].get('parent'):
            raise ValueError(f'parent timing/lineage defect: {sid}')
        reasons.setdefault(sid, set()).add(issue['status'])
    retained = [dict(row) for row in entries if row['id'] not in reasons]
    retained_ids = {row['id'] for row in retained}
    sessions = {}
    unknown_gaps = []
    for row in retained:
        sid = row['id']
        path = source / f'{sid}.json'
        obj = json.loads(path.read_text())
        # First-turn gap is naturally absent; every later gap must be a real,
        # finite, nonnegative value when present. Unknown is reported, not made up.
        for i, turn in enumerate(obj['turns']):
            gap = turn.get('gap_s')
            if gap is None:
                unknown_gaps.append({'session': sid, 'turn': i, 'first_turn': i == 0})
            elif type(gap) not in (int, float) or not math.isfinite(gap) or gap < 0:
                raise ValueError(f'invalid gap: {sid}:{i}')
        if not row.get('parent'):
            obj['children'] = [child for child in obj.get('children', []) if child['id'] in retained_ids]
            row['children'] = len(obj['children'])
        sessions[sid] = obj
    # Cross-session bound must be recomputed: removing a family changes which
    # earlier prefixes are available. Never carry stale sample-wide aggregates.
    head = adapter.HeadIndex()
    for row in retained:
        sid = row['id']
        turns = sessions[sid]['turns']
        prof = adapter.ideal_profile(turns)
        cross = union = 0
        for (_, prompt, _), (plen, within) in zip(adapter.iter_prompts(turns), prof):
            c = min(head.match_and_add(prompt, sid), plen)
            cross += c
            union += max(within, c)
        pc = sum(p for p, _ in prof)
        row.update(adapter.session_stats(turns), ideal_cross_only_chars=cross,
                   ideal_union_chars=union, ideal_share_cross_head=union / pc if pc else 0.0,
                   provider=adapter.provider_share(turns))
    top = [r for r in retained if not r.get('parent')]
    spans = [{k: r.get(k) for k in ('id', 'parent', 'start', 'end', 'requests')} for r in retained]
    links = Counter(r['link'] for r in retained if isinstance(r.get('link'), str))
    result = dict(source_manifest)
    result.update(sessions=retained, totals=adapter.totals(top),
                  totals_incl_children=adapter.totals(retained), fanout=adapter.fanout_stats(spans),
                  child_links=dict(links),
                  successor={'policy': 'exclude-invalid-child-whole-v1', 'source_manifest_sha256': source_sha,
                             'excluded': [{'id': sid, 'reasons': sorted(reason),
                                           'requests': by_id[sid]['requests'],
                                           'sha256': sha(source / f'{sid}.json')}
                                          for sid, reason in sorted(reasons.items())],
                             'first_turn_gap': 'unknown, no pre-first-request wait',
                             'later_missing_gap': 'unknown, driver applies no wait; retain as unknown'})
    # Original files remain untouched; new files are private and creation is exclusive.
    dest.mkdir(mode=0o700, parents=True)
    try:
        for row in retained:
            sid = row['id']
            target = dest / f'{sid}.json'
            source_bytes = (source / f'{sid}.json').read_bytes()
            # Only parent child lists may change; preserve all retained turns byte-for-byte
            # semantically and the actual original file hash in the source catalog.
            original = json.loads(source_bytes)
            if sessions[sid] == original:
                target.write_bytes(source_bytes)
            else:
                target.write_text(json.dumps(sessions[sid], separators=(',', ':')))
        (dest / 'manifest.json').write_text(json.dumps(result, indent=1) + '\n')
        derived = audit_cohort(dest, children='bursts')
        if not derived['valid'] or derived['unknown_timing']:
            raise ValueError('derived cohort failed integrity audit')
        if sha(manifest_path) != source_sha or any(sha(source / f'{sid}.json') != digest for sid, digest in source_files.items()):
            raise ValueError('source drift during materialization')
        report = {'source_manifest_sha256': source_sha, 'manifest_sha256': sha(dest / 'manifest.json'),
                  'source_files': source_files,
                  'derived_files': {row['id']: sha(dest / f"{row['id']}.json") for row in retained},
                  'excluded': result['successor']['excluded'], 'retained_sessions': len(retained),
                  'retained_requests': len(derived['expected']),
                  'retained_child_requests': sum(r['requests'] for r in retained if r.get('parent')),
                  'expected': derived['expected'], 'unknown_gaps': unknown_gaps,
                  'eligible_population_changed': bool(reasons),
                  'source_issue_counts': dict(Counter(x['status'] for x in issues))}
        (dest / 'SUCCESSOR_REPORT.json').write_text(json.dumps(report, indent=1) + '\n')
        return report
    except Exception:
        # Never present a partially written directory as a valid cohort.
        (dest / "INVALID").write_text("materialization failed; preserve for diagnosis\n")
        raise


def verify(dest: Path, report_sha: str, manifest_sha: str) -> dict:
    """Verify a separately predeclared successor before model startup."""
    report_path = dest / 'SUCCESSOR_REPORT.json'
    if sha(report_path) != report_sha:
        raise ValueError('successor report digest mismatch')
    report = json.loads(report_path.read_text())
    if sha(dest / 'manifest.json') != manifest_sha or report['manifest_sha256'] != manifest_sha:
        raise ValueError('successor manifest digest mismatch')
    manifest = json.loads((dest / 'manifest.json').read_text())
    ids = {r['id'] for r in manifest['sessions']}
    if ids != set(report['derived_files']):
        raise ValueError('successor session set mismatch')
    for sid, digest in report['derived_files'].items():
        if sha(dest / f'{sid}.json') != digest:
            raise ValueError(f'successor session digest mismatch: {sid}')
    audit = audit_cohort(dest, children='bursts')
    if not audit['valid'] or audit['unknown_timing'] or [list(k) for k in audit['expected']] != report['expected']:
        raise ValueError('successor integrity audit failed')
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path)
    p.add_argument('--dest', required=True, type=Path)
    p.add_argument('--source-sha256')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--report-sha256')
    p.add_argument('--manifest-sha256')
    args = p.parse_args()
    if args.verify:
        if not args.report_sha256 or not args.manifest_sha256:
            p.error('--verify requires --report-sha256 and --manifest-sha256')
        report = verify(args.dest, args.report_sha256, args.manifest_sha256)
    else:
        if not args.source or not args.source_sha256:
            p.error('materialization requires --source and --source-sha256')
        report = build(args.source, args.dest, args.source_sha256)
    print(json.dumps({k: report[k] for k in ('manifest_sha256', 'retained_sessions', 'retained_requests',
                                               'retained_child_requests', 'source_issue_counts')}))


if __name__ == '__main__':
    main()
