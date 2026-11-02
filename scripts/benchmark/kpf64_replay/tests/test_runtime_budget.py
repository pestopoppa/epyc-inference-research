import json
import math

import pytest

import runtime_budget as budget


def row(session, k, *, prompt=100, total=200, predicted=10, pms=2000, sms=7000):
    return dict(session=session, k=k, prompt_n=prompt, total_tok=total, predicted_n=predicted,
                prompt_ms=pms, svc_ms=sms, t_start=1, t_end=8)


def test_conservative_stress_uses_caps_not_old_cohort_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(budget.family_successor, 'verify', lambda *_: {
        'expected': [['a', 0], ['b', 0]], 'retained_requests': 2})
    f, fb = tmp_path / 'f', tmp_path / 'fb'
    f.write_text('\n'.join(json.dumps(x) for x in [row('a', 0), row('b', 0)]) + '\n')
    fb.write_text(json.dumps(row('a', 0)) + '\n')
    out = budget.estimate(tmp_path, 'a' * 64, 'b' * 64, f, fb, 1000, 20, 5, 10, 1.1, 4.0)
    assert (out['cohort_requests'], out['f_matched'], out['fb_matched']) == (2, 2, 1)
    assert out['conditional_serial_weighted_seconds'] == 18
    assert out['sequential_observed_maxima_stress_seconds'] == 2 * (1000 * 0.02 + 20 * 0.5 + 5) + 10
    assert out['stress_window_seconds'] == 3600
    assert out['proposed_contiguous_window_seconds'] == 86400
    assert 'not a runtime bound' in out['kind'].replace('_', ' ')


def test_rejects_missing_f_duplicate_failed_and_bad_timings():
    expected = {('a', 0), ('b', 0)}
    with pytest.raises(ValueError, match='missing'):
        budget.select([row('a', 0)], expected, complete=True)
    with pytest.raises(ValueError, match='duplicate'):
        budget.select([row('a', 0), row('a', 0)], expected, complete=False)
    with pytest.raises(ValueError, match='failed'):
        budget.select([{'session': 'a', 'k': 0, 'error': 'timeout'}], expected, complete=False)
    bad = row('a', 0); bad['svc_ms'] = -1
    with pytest.raises(ValueError, match='svc_ms'):
        budget.select([bad], expected, complete=False)
    bad = row('a', 0); bad['prompt_ms'] = math.nan
    with pytest.raises(ValueError, match='prompt_ms'):
        budget.select([bad], expected, complete=False)
