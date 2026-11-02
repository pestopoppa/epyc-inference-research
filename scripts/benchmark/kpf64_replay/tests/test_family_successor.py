"""Structural successor filtering without changing retained turns."""
import hashlib
import json
from pathlib import Path

import pytest

import family_successor
from cohort_audit import audit_cohort


def turn(name, start=1, end=2):
    return {'pre': name, 'ref': 'response', 'reset': False, 'gap_s': None,
            'ts_start': start, 'ts_end': end, 'spawns': [], 'usage': None}


def fixture(root):
    source = root / 'src'
    source.mkdir()
    sessions = {
        'p': {'id': 'p', 'store': 'claude', 'replayable': True,
              'turns': [turn('p')], 'children': [
                  {'id': 'p.c0', 'parent_turn': 0}, {'id': 'p.c1', 'parent_turn': 7},
                  {'id': 'p.c2', 'parent_turn': 0}]},
        'p.c0': {'id': 'p.c0', 'store': 'claude', 'parent': 'p', 'parent_turn': 0,
                 'turns': [turn('kept')]},
        'p.c1': {'id': 'p.c1', 'store': 'claude', 'parent': 'p', 'parent_turn': 7,
                 'turns': [turn('out-of-range')]},
        'p.c2': {'id': 'p.c2', 'store': 'claude', 'parent': 'p', 'parent_turn': 0,
                 'turns': [turn('bad-timestamp', 2, 1)]},
    }
    entries = []
    for sid, obj in sessions.items():
        (source / f'{sid}.json').write_text(json.dumps(obj))
        entries.append({'id': sid, 'store': 'claude', 'parent': obj.get('parent'),
                        'parent_turn': obj.get('parent_turn'), 'requests': len(obj['turns']),
                        'source_hash': 'test', 'start': 1, 'end': 2,
                        'link': 'explicit' if obj.get('parent') else None})
    (source / 'manifest.json').write_text(json.dumps({'schema': 'kpf64.sessions.v2',
                                                       'sessions': entries, 'params': {}}))
    return source, hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest()


def test_excludes_whole_invalid_children_and_keeps_turns(tmp_path):
    src, digest = fixture(tmp_path)
    dest = tmp_path / 'dest'
    result = family_successor.build(src, dest, digest)
    assert result['retained_requests'] == 2
    assert result['retained_child_requests'] == 1
    assert {x['id'] for x in result['excluded']} == {'p.c1', 'p.c2'}
    assert (dest / 'p.c0.json').read_bytes() == (src / 'p.c0.json').read_bytes()
    assert json.loads((dest / 'p.json').read_text())['turns'] == json.loads((src / 'p.json').read_text())['turns']
    assert [x['id'] for x in json.loads((dest / 'p.json').read_text())['children']] == ['p.c0']
    assert audit_cohort(dest)['valid']
    with pytest.raises(ValueError, match='already exists'):
        family_successor.build(src, dest, digest)


def test_reject_wrong_source_pin_and_invalid_parent(tmp_path):
    src, digest = fixture(tmp_path)
    with pytest.raises(ValueError, match='digest mismatch'):
        family_successor.build(src, tmp_path / 'dest', '0' * 64)
    obj = json.loads((src / 'p.json').read_text())
    obj['turns'][0]['ts_end'] = 0
    (src / 'p.json').write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='parent timing'):
        family_successor.build(src, tmp_path / 'dest', digest)


def test_fb_is_f_matched_except_cache_ram():
    path = Path(family_successor.__file__).with_name('arms.json')
    arms = json.loads(path.read_text())['arms']
    f, fb = dict(arms['F']), dict(arms['FB'])
    assert f.pop('cache_ram_mib') == 32768
    assert fb.pop('cache_ram_mib') == 0
    f.pop('label')
    fb.pop('label')
    assert f == fb


def test_verify_requires_bound_report_and_all_session_bytes(tmp_path):
    src, source_digest = fixture(tmp_path)
    dest = tmp_path / 'dest'
    report = family_successor.build(src, dest, source_digest)
    report_digest = hashlib.sha256((dest / 'SUCCESSOR_REPORT.json').read_bytes()).hexdigest()
    assert family_successor.verify(dest, report_digest, report['manifest_sha256'])['retained_requests'] == 2
    with pytest.raises(ValueError, match='report digest'):
        family_successor.verify(dest, '0' * 64, report['manifest_sha256'])
    (dest / 'p.c0.json').write_text('{}')
    with pytest.raises(ValueError, match='session digest'):
        family_successor.verify(dest, report_digest, report['manifest_sha256'])
