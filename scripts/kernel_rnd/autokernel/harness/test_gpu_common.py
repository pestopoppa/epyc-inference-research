"""Unit tests for the canonical gpu_common residency proof against a fake sub-second process (no GPU, no KFD, no sampler)."""
import os
import shutil
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_common as G  # noqa: E402

EMPTY_PROOF = {'kfd_pids_ours': {}, 'maps': {}, 'vram_delta_b': 212631552, 'samples': 2, 'kfd_foreign': []}


@pytest.fixture
def row(tmp_path, monkeypatch):
    cc = shutil.which('cc') or shutil.which('gcc')
    if not cc:
        pytest.skip('no C compiler for the fake libggml-hip.so')
    bdir = tmp_path / 'build'
    (bdir / 'bin').mkdir(parents=True)
    c = tmp_path / 'fake.c'
    c.write_text('int fake_hip_init(void) { return 0; }\n')
    subprocess.run([cc, '-shared', '-fPIC', '-o', str(bdir / 'bin' / 'libggml-hip.so'), str(c)], check=True)
    G.init('test-row', str(tmp_path), 'test-task')
    G.G['DEADLINE'] = 2 ** 40

    def fake_run_sampled(argv, log, env=None, timeout_at=None, stopfile=None, cwd=None, errlog=None):
        with open(log, 'wb') as f, open(errlog, 'wb') as fe:
            rc = subprocess.run(argv, stdout=f, stderr=fe, env=env, cwd=cwd).returncode
        return rc, None, dict(EMPTY_PROOF)  # the sampler saw nothing: a sub-second process between samples

    monkeypatch.setattr(G, 'gpu_proof', lambda: types.SimpleNamespace(run_sampled=fake_run_sampled))
    return str(bdir)


def fake_tbo(lib, witness=True, status='OK'):
    body = f"import ctypes; ctypes.CDLL({lib!r})"
    if witness:
        body += "; import sys; print('ggml_cuda_init: found 1 ROCm devices', file=sys.stderr); print('\\x1b[1;32mBackend ROCm0: " + status + "\\x1b[0m')"
    return [sys.executable, '-I', '-c', body]


def test_subsecond_process_passes_on_witness(row):
    rc, err, pr, txt, etxt = G.run_sampled(fake_tbo(row + '/bin/libggml-hip.so'), 'ctl-fake', row)
    r = G.residency_from_proof(pr, row)
    assert rc == 0 and pr['_elapsed_s'] < G.SHORT_S
    assert r['gate'] == 'witness' and r['ok'] and not r['sampler_ok']
    assert r['witness_rocm_init'] and r['witness_backend_status'] == 'OK' and r['ld_debug_hip_from_row_build']
    assert 'libggml-hip' not in etxt  # LD_DEBUG went to its own file, not into the receipts stream


def test_lib_from_another_build_fails(row, tmp_path):
    other = tmp_path / 'other'
    other.mkdir()
    shutil.copy(row + '/bin/libggml-hip.so', other / 'libggml-hip.so')
    _, _, pr, _, _ = G.run_sampled(fake_tbo(str(other / 'libggml-hip.so')), 'ctl-other', row)
    r = G.residency_from_proof(pr, row)
    assert not r['ld_debug_hip_from_row_build'] and not r['ok']


def test_missing_witness_lines_fail(row):
    _, _, pr, _, _ = G.run_sampled(fake_tbo(row + '/bin/libggml-hip.so', witness=False), 'ctl-silent', row)
    r = G.residency_from_proof(pr, row)
    assert r['ld_debug_hip_from_row_build'] and not r['ok']


def test_long_process_still_needs_sampler(row):
    _, _, pr, _, _ = G.run_sampled(fake_tbo(row + '/bin/libggml-hip.so'), 'cand-fake', row)
    pr['_elapsed_s'] = 5.0
    r = G.residency_from_proof(pr, row)
    assert r['gate'] == 'sampler+witness' and r['witness_ok'] and not r['ok']


def test_legacy_proof_without_witness_data_unchanged(row):
    r = G.residency_from_proof(dict(EMPTY_PROOF), row)
    assert r['gate'] == 'sampler' and not r['ok'] and 'witness_ok' not in r


def test_expected_failure_control_still_witnessed(row):
    """A positive control is meant to fail (rc != 0, 'Backend ROCm0: FAIL'); it still ran on ROCm0 (r3 refused it on 'OK')."""
    _, _, pr, _, _ = G.run_sampled(fake_tbo(row + '/bin/libggml-hip.so', status='FAIL'), 'ctl-positive', row)
    r = G.residency_from_proof(pr, row)
    assert r['witness_backend_status'] == 'FAIL' and r['ok']
