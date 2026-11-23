import sqlite3

import pytest

from . import community_intake as intake


def test_review_bundle_joins_bandwidth_hotspot_and_null(tmp_path):
    db = tmp_path / 'experiments.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE experiments (attempt_id, epoch_sha256, target_surface, target_symbol, status, mechanism_id, recorded_at)')
        conn.execute('INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)',
                     ('a1', 'epoch', 'gpu', 'mul_mat', 'measured_null', 'akm-x', '2026'))
    claim = {'title': 'Port X', 'mechanism_id': 'akm-x', 'target_symbol': 'mul_mat',
             'target_surface': 'gpu', 'statement': 'change access', 'falsifier': 'no gain', 'url': 'https://example.org/pr',
             'tier': 'upstream', 'achieved_gb_s': 400, 'ceiling_gb_s': 800,
             'bandwidth_basis': 'source GPU and measured ceiling'}
    result = intake.prepare([claim], [{'signature': 'gpu_mul_mat', 'share_of_device_time': 0.4}], db)
    assert '50.0%' in result and 'gpu_mul_mat (0.4' in result
    assert 'a1 / epoch / gpu / mul_mat' in result
    assert 'operator review' in result and 'not merge authorization' in result
    assert '## akm-x (proposed)' in result and '**Falsifier:** no gain' in result
    assert 'gfx90a port hypotheses' in result
    claim['achieved_gb_s'] = 900
    assert 'inspect cache residency' in intake.prepare([claim], [], db)
    claim.pop('ceiling_gb_s')
    with pytest.raises(intake.IntakeRefused):
        intake.prepare([claim], [], db)


def test_missing_database_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        intake.prepare([], [], tmp_path / 'missing.db')


def test_community_and_paper_unknown_bandwidth_remains_unknown(tmp_path):
    db = tmp_path / 'experiments.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE experiments (attempt_id, epoch_sha256, target_surface, target_symbol, status, mechanism_id, recorded_at)')
    claims = [dict(title=tier, mechanism_id=tier, target_symbol='kernel',
                   target_surface='gpu', statement='candidate', falsifier='no gain',
                   url=f'https://example.org/{tier}', tier=tier)
              for tier in ('community', 'paper')]
    result = intake.prepare(claims, [], db)
    assert result.count('unknown (source did not report') == 2
    assert result.count('Source is a design prior, not a local measurement.') == 2
    assert 'no matching signature; applicability unproven' in result


def test_cli_requires_review_outside_live_inbox_and_never_overwrites(tmp_path):
    store = tmp_path / 'store'
    store.mkdir()
    db = store / 'experiments.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE experiments (attempt_id, epoch_sha256, target_surface, target_symbol, status, mechanism_id, recorded_at)')
    claims, hotspots = tmp_path / 'claims.json', tmp_path / 'hotspots.json'
    claims.write_text('{"claims": []}')
    hotspots.write_text('[]')
    inbox = tmp_path / 'inbox'
    inbox.mkdir()
    common = ['--claims', str(claims), '--hotspots', str(hotspots), '--experiments-db', str(db)]
    with pytest.raises(SystemExit):
        intake.main([*common, '--out', str(inbox / 'seed.md')])
    assert not list(inbox.iterdir())
    with pytest.raises(SystemExit):
        intake.main([*common, '--out', str(store / 'review.md')])
    output = tmp_path / 'review.md'
    assert intake.main([*common, '--out', str(output)]) == 0
    with pytest.raises(FileExistsError):
        intake.main([*common, '--out', str(output)])


def test_upstream_enumeration_pins_tip_and_provenance(monkeypatch, tmp_path):
    from types import SimpleNamespace
    calls = []
    tip = 'a' * 40
    def git(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=(tip + '\n' if 'rev-parse' in argv
                                       else 'b' * 40 + '\tCUDA fix\n'))
    monkeypatch.setattr(intake.subprocess, 'run', git)
    actual_tip, candidates = intake.upstream_candidates(tmp_path, 'upstream/master')
    assert actual_tip == tip and '0db32c06..' + tip in calls[1]
    assert candidates == ['- https://github.com/ggml-org/llama.cpp/commit/' + 'b' * 40 + ' — CUDA fix']
