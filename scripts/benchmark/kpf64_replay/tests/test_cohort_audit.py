"""Cohort integrity and exactly-once coverage, without loading model data."""
import json

import cohort_audit


def family(tmp_path, parent_replayable=True, child_turn=0):
    def write(name, obj):
        (tmp_path / name).write_text(json.dumps(obj))
    turns = [{'pre': 'prompt', 'ref': 'answer', 'ts_start': 1, 'ts_end': 2}] if parent_replayable else []
    write('p.json', {'id': 'p', 'turns': turns, 'replayable': parent_replayable,
                     'children': [{'id': 'c', 'parent_turn': child_turn}]})
    write('c.json', {'id': 'c', 'parent': 'p', 'parent_turn': child_turn,
                     'turns': [{'pre': 'child', 'ref': 'response'}]})
    write('manifest.json', {'sessions': [{'id': 'p', 'requests': len(turns)},
          {'id': 'c', 'parent': 'p', 'parent_turn': child_turn, 'requests': 1}]})
    return tmp_path


def test_reject_truncated_parent_but_allow_virtual_anchor(tmp_path):
    report = cohort_audit.audit_cohort(family(tmp_path, child_turn=94))
    assert not report['valid']
    assert report['issues'][0]['status'] == 'parent_turn_outside_retained_parent'
    assert report['issues'][0]['requests'] == 1
    assert 'prompt' not in json.dumps(report)
    anchor = tmp_path / 'anchor'
    anchor.mkdir()
    assert cohort_audit.audit_cohort(family(anchor, parent_replayable=False, child_turn=94))['valid']


def test_reversed_timing_and_disagreeing_lineage_refuse(tmp_path):
    family(tmp_path)
    parent = json.loads((tmp_path / 'p.json').read_text())
    parent['turns'][0]['ts_end'] = 0
    (tmp_path / 'p.json').write_text(json.dumps(parent))
    report = cohort_audit.audit_cohort(tmp_path)
    assert any(x['status'] == 'reversed_timestamp' for x in report['issues'])
    child = json.loads((tmp_path / 'c.json').read_text())
    child['parent_turn'] = 1
    (tmp_path / 'c.json').write_text(json.dumps(child))
    import pytest
    with pytest.raises(ValueError, match='parent_turn mismatch'):
        cohort_audit.audit_cohort(tmp_path)


def test_coverage_requires_every_attempt_exactly_once_and_success(tmp_path):
    expected = [['p', 0], ['c', 0]]
    path = tmp_path / 'requests.jsonl'
    path.write_text(json.dumps({'session': 'p', 'k': 0}) + '\n')
    report = cohort_audit.audit_coverage(expected, path)
    assert not report['complete'] and report['missing'] == [('c', 0)]
    with path.open('a') as out:
        out.write(json.dumps({'session': 'c', 'k': 0, 'error': 'interrupted'}) + '\n')
        out.write(json.dumps({'session': 'p', 'k': 0}) + '\n')
    report = cohort_audit.audit_coverage(expected, path)
    assert not report['complete'] and report['duplicates'] == [('p', 0)]
    assert report['failed'] == [('c', 0)]
    path.write_text('\n'.join(json.dumps({'session': sid, 'k': k}) for sid, k in expected) + '\n')
    assert cohort_audit.audit_coverage(expected, path)['complete']


def test_driver_rejects_invalid_family_before_opening_stub(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    family(sessions, child_turn=94)
    out = tmp_path / 'out'
    env = dict(os.environ, WINDOW_CONFIRMED='1')
    driver = Path(cohort_audit.__file__).with_name('replay_driver.py')
    result = subprocess.run([sys.executable, str(driver), '--sessions', str(sessions),
                             '--out', str(out), '--dry-run', '--children', 'bursts'],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert 'invalid cohort' in result.stderr
    assert json.loads((out / 'cohort_audit.json').read_text())['valid'] is False
    assert not (out / 'stub_server.log').exists()
    assert not (out / 'requests.jsonl').exists()


def test_interrupted_existing_rows_cannot_pass_as_complete(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    family(sessions)
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'requests.jsonl').write_text(json.dumps({'session': 'p', 'k': 0}) + '\n')
    env = dict(os.environ, WINDOW_CONFIRMED='1')
    driver = Path(cohort_audit.__file__).with_name('replay_driver.py')
    result = subprocess.run([sys.executable, str(driver), '--sessions', str(sessions),
                             '--out', str(out), '--dry-run', '--children', 'bursts'],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    coverage = json.loads((out / 'coverage_audit.json').read_text())
    assert not coverage['complete'] and coverage['missing'] == [['c', 0]]
