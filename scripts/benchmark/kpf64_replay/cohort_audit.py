#!/usr/bin/env python3
"""Read-only KPF-64 cohort and replay coverage audit; never emits transcript text."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_cohort(directory, children='bursts', only=None):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['sessions']
    ids = [e['id'] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate manifest session id')
    by_id = {e['id']: e for e in entries}
    selected = set(only) if only else {e['id'] for e in entries if not e.get('parent')}
    if not selected <= {e['id'] for e in entries if not e.get('parent')}:
        raise ValueError('selected id is not a manifest parent')
    problems, sessions, expected = [], [], set()
    for parent_id in sorted(selected):
        parent_path = directory / f'{parent_id}.json'
        parent = json.loads(parent_path.read_text())
        if parent.get('id') != parent_id:
            raise ValueError(f'session id mismatch: {parent_id}')
        replayable = parent.get('replayable', True)
        if replayable:
            expected.update((parent_id, k) for k in range(len(parent['turns'])))
        children_by_id = {}
        if children == 'bursts':
            for child in parent.get('children', []):
                cid = child['id']
                if cid in children_by_id:
                    raise ValueError(f'duplicate child declaration: {cid}')
                children_by_id[cid] = child
            listed = {e['id'] for e in entries if e.get('parent') == parent_id}
            if listed != set(children_by_id):
                raise ValueError(f'manifest and parent child sets differ: {parent_id}')
        all_items = [(parent_id, parent, parent_path)]
        for cid, decl in children_by_id.items():
            child_path = directory / f'{cid}.json'
            child = json.loads(child_path.read_text())
            if child.get('id') != cid or child.get('parent') != parent_id:
                raise ValueError(f'child identity mismatch: {cid}')
            k = decl.get('parent_turn')
            if child.get('parent_turn') != k or by_id[cid].get('parent_turn') != k:
                raise ValueError(f'child parent_turn mismatch: {cid}')
            issue = None
            if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                issue = 'parent_turn_unknown_or_invalid'
            elif replayable and k >= len(parent['turns']):
                issue = 'parent_turn_outside_retained_parent'
            if issue:
                problems.append({'session': cid, 'parent': parent_id, 'parent_turn': k,
                                 'status': issue, 'requests': len(child['turns']),
                                 'source_hash': by_id[cid].get('source_hash'), 'file_sha256': _sha(child_path)})
            expected.update((cid, n) for n in range(len(child['turns'])))
            all_items.append((cid, child, child_path))
        for sid, sess, path in all_items:
            n = len(sess['turns'])
            if 'requests' in by_id[sid] and by_id[sid]['requests'] != n:
                raise ValueError(f'manifest request count mismatch: {sid}')
            sessions.append({'id': sid, 'requests': n, 'sha256': _sha(path),
                             'source_hash': by_id[sid].get('source_hash')})
            for k, turn in enumerate(sess['turns']):
                start, end = turn.get('ts_start'), turn.get('ts_end')
                if isinstance(start, (int, float)) and isinstance(end, (int, float)) and start > end:
                    problems.append({'session': sid, 'turn': k, 'status': 'reversed_timestamp'})
    return {'schema': 'kpf64.cohort-audit.v1', 'manifest_sha256': _sha(manifest_path),
            'sessions': sessions, 'expected': sorted(expected), 'issues': problems,
            'valid': not problems}


def audit_coverage(expected, path):
    expected = {tuple(x) for x in expected}
    counts, failed, malformed = Counter(), set(), 0
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                key = (rec['session'], rec['k'])
                if not isinstance(key[0], str) or not isinstance(key[1], int):
                    raise ValueError('invalid key')
                counts[key] += 1
                if 'error' in rec:
                    failed.add(key)
            except (ValueError, KeyError, TypeError):
                malformed += 1
    observed = set(counts)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(k for k, n in counts.items() if n != 1)
    return {'expected': len(expected), 'observed_unique': len(observed), 'missing': missing,
            'unexpected': unexpected, 'duplicates': duplicates, 'failed': sorted(failed),
            'malformed': malformed,
            'complete': not (missing or unexpected or duplicates or failed or malformed)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions', required=True)
    p.add_argument('--children', choices=('off', 'bursts'), default='bursts')
    p.add_argument('--requests')
    p.add_argument('--only', help='comma-separated parent IDs')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    report = audit_cohort(a.sessions, a.children, a.only.split(',') if a.only else None)
    if a.requests:
        report['coverage'] = audit_coverage(report['expected'], a.requests)
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['valid'] and (not a.requests or report['coverage']['complete']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
