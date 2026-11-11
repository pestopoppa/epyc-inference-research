import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mmap_prefetch_witness as w


def arm(populate, line=False, tokens=((1, 2),)):
    top = None if populate is None else {'len': 10, 'populate': 1 if populate else 0, 'target': 'm.gguf', 'ok': 1}
    return {'top': top, 'line': line, 'tokens': [list(t) for t in tokens]}


GOOD = {'unset': arm(True), 'zero': arm(True), 'one-noprefetch': arm(False, line=True)}


def test_pass():
    assert w.classify(GOOD)[0] == 'PASS'


def test_control_not_separating_is_futility_not_verdict():
    # r3 failure mode: the observable is identical in every arm
    bad = {k: arm(True, line=(k == 'one-noprefetch')) for k, _ in w.ARMS}
    assert w.classify(bad)[0] == 'FUTILITY_CONTROL'


def test_missing_gguf_mapping_is_futility():
    bad = dict(GOOD, zero=arm(None))
    assert w.classify(bad)[0] == 'FUTILITY_CONTROL'


def test_wrong_pattern_is_fail():
    bad = dict(GOOD, unset=arm(False))
    assert w.classify(bad)[0] == 'FAIL'


def test_token_divergence_is_fail():
    bad = dict(GOOD, zero=arm(True, tokens=((9, 9),)))
    assert w.classify(bad)[0] == 'FAIL'


def test_parse_picks_largest_ok_gguf():
    rows = [{'ok': 1, 'target': 'a.gguf', 'len': 5, 'populate': 0}, {'ok': 1, 'target': 'a.gguf', 'len': 50, 'populate': 1},
            {'ok': 0, 'target': 'a.gguf', 'len': 500, 'populate': 0}, {'ok': 1, 'target': 'libx.so', 'len': 900, 'populate': 0}]
    top = w.parse_shim_log('\n'.join(json.dumps(r) for r in rows) + '\nnot json\n')
    assert top['len'] == 50 and w.parse_shim_log('') is None
