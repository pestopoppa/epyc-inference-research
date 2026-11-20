"""Unit check for act_dump's validator on a synthetic dump (no model, no GPU)."""
import json, os, struct, sys

sys.path.insert(0, os.path.dirname(__file__))
from check_dump import check  # noqa: E402


def _w(d, key, typ, ne, payload, tensors):
    open(os.path.join(d, key + '.bin'), 'wb').write(payload)
    tensors.append({'key': key, 'file': key + '.bin', 'type': typ, 'ne': ne, 'bytes': len(payload),
                    'stats': {'nonfinite': 0} if typ in ('f32', 'f16') else None})


def _mk(d, act_gate=None, fa=True):
    t = []
    act = struct.pack('<64f', *range(64))  # ne [32, 2]
    _w(d, 'act_ffn_up', 'f32', [32, 2, 1, 1], act, t)
    _w(d, 'act_ffn_gate', 'f32', [32, 2, 1, 1], act_gate or act, t)
    _w(d, 'w_ffn_up', 'q8_0', [32, 4, 1, 1], b'\0' * 34 * 4, t)
    if fa:
        _w(d, 'fa_q', 'f32', [8, 2, 4, 1], b'\0' * 4 * 64, t)
        _w(d, 'fa_k', 'f16', [8, 4, 2, 1], b'\0' * 2 * 64, t)
        _w(d, 'fa_v', 'f16', [8, 4, 2, 1], b'\0' * 2 * 64, t)
    json.dump({'layer': 3, 'n_tokens': 2, 'tensors': t}, open(os.path.join(d, 'manifest.json'), 'w'))


def test_valid(tmp_path):
    _mk(str(tmp_path))
    r = check(str(tmp_path))
    assert r['ok'], r['errors']


def test_mismatched_ffn_inputs(tmp_path):
    _mk(str(tmp_path), act_gate=struct.pack('<64f', *([1.0] * 64)))
    assert 'act_ffn_up and act_ffn_gate inputs differ' in check(str(tmp_path))['errors']


def test_missing_fa(tmp_path):
    _mk(str(tmp_path), fa=False)
    assert not check(str(tmp_path))['ok']
    assert check(str(tmp_path), require_fa=False)['ok']


def test_truncated_bin(tmp_path):
    _mk(str(tmp_path))
    open(os.path.join(str(tmp_path), 'w_ffn_up.bin'), 'wb').write(b'\0' * 10)
    assert any(e.startswith('w_ffn_up: size') for e in check(str(tmp_path))['errors'])


def _mk_head(d, hidden_rows=2):
    t = []
    _w(d, 'head_hidden', 'f32', [8, hidden_rows, 1, 1], b'\0' * 4 * 8 * hidden_rows, t)
    _w(d, 'head_w.f32', 'f32', [8, 10, 1, 1], b'\0' * 4 * 80, t)
    _w(d, 'head_topk_ids', 'i32', [4, 2, 1, 1], b'\0' * 4 * 8, t)
    _w(d, 'head_topk_val', 'f32', [4, 2, 1, 1], b'\0' * 4 * 8, t)
    json.dump({'head_mode': True, 'vocab': 10, 'n_tokens': 2, 'tensors': t}, open(os.path.join(d, 'manifest.json'), 'w'))


def test_head_mode_valid(tmp_path):
    _mk_head(str(tmp_path))
    r = check(str(tmp_path))
    assert r['ok'] and r['head_mode'], r['errors']


def test_head_mode_row_mismatch(tmp_path):
    _mk_head(str(tmp_path), hidden_rows=3)
    assert not check(str(tmp_path))['ok']
