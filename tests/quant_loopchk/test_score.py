import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('loopscore', ROOT / 'scripts/quant_loopchk/score.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def inputs(tmp_path, text='normal completion', finish='stop'):
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'id': str(i)} for i in range(5)]))
    records = [{'id': str(i), 'text': text, 'cap': 1024, 'finish_reason': finish, 'wrong': False} for i in range(5)]
    captures = tmp_path / 'captures.json'
    captures.write_text(json.dumps({'reference': records, 'candidate': records}))
    return prompts, captures


def test_synthetic_loop_and_normal():
    assert m.looped('alpha beta gamma ' * 4)
    assert not m.looped('alpha beta gamma delta epsilon')


def test_paired_threshold_zero_and_futility(tmp_path):
    p, c = inputs(tmp_path)
    data = json.loads(c.read_text())
    data['candidate'][0]['text'] = 'alpha beta gamma ' * 4
    c.write_text(json.dumps(data))
    result = m.score(p, c, 1024)
    assert result['arms']['candidate']['labels'][0]['looped']
    assert result['flags']['looped'] and result['verdict'] == 'flag'
    data['reference'][0]['finish_reason'] = 'length'
    data['reference'][1]['finish_reason'] = 'length'
    c.write_text(json.dumps(data))
    assert m.score(p, c, 1024)['verdict'] == 'redeclare-cap'


def test_rejects_missing_and_unreviewed(tmp_path):
    p, c = inputs(tmp_path)
    data = json.loads(c.read_text())
    data['candidate'][0]['wrong'] = None
    c.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='human review'):
        m.score(p, c, 1024)
    data['candidate'].pop()
    c.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='coverage'):
        m.score(p, c, 1024)


def test_checked_in_dry_fixture():
    result = m.score(ROOT / 'data/quant-loopchk/hard-prompts.json', ROOT / 'data/quant-loopchk/synthetic-captures.json', 8192)
    labels = result['arms']['candidate']['labels']
    assert labels[0]['looped'] and labels[2]['cut_off']
    assert result['prompt_sha256'] == '90040cc66140a81319ebe4b99c4c54f12cfa349b5be9a3dd88768101f46a19cf'
