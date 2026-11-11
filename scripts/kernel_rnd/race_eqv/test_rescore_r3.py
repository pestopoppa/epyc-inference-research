#!/usr/bin/env python3
"""Re-score r3 (race-eqv-gpu-r3-20261109, BASELINE|Q|0) through THIS row's job.run_tbo + gate.
r3 filed INVALID_BASELINE_FAILS with n_fail_lines=0; under the 2026-11-10 gate the same log must parse
312 reconciled failures, be a USABLE baseline, and score PASS against itself.  No GPU, writes only to a temp dir."""
import gzip, hashlib, os, shutil, sys, tempfile
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
R3LOG = '/mnt/raid0/llm/tmp/race-eqv-gpu-r3-20261109/out/logs/current.log'
R3_SHA = '67e4760557e509a3dcb5aa4ba7e237f28df0029a4b52c042e3d64679880b0801'
import job
from gpu_signals import newfail as nf


def main():
    if not os.path.exists(R3LOG):
        print('SKIP: r3 log absent (tmp cleaned)'); return
    assert hashlib.sha256(open(R3LOG, 'rb').read()).hexdigest() == R3_SHA
    td = tempfile.TemporaryDirectory(prefix='r4-rescore-'); tmp = td.name; job.LOGS = tmp; job.RAW = tmp
    def fake_run(a, log, **kw):
        shutil.copyfile(R3LOG, log); return 1, None, {'kfd_pids_ours': {1: 863903744}, 'maps': {}}
    job.gp.run_sampled = fake_run
    try:
        r = job.run_tbo('/nonexistent', 'Q', 1000, 'rescore-r3', 10)
        p = r['parse']
        assert (p['passed'], p['total'], p['n_fail'], p['consistent']) == (2295, 2607, 312, True), p
        assert r['n_fail_lines'] == 312 and not r['strict_pass']
        assert nf.baseline_usable(p) == (True, 'ok')                       # r3: INVALID_BASELINE_FAILS -> r4: USABLE
        assert nf.compare(p, p)['verdict'] == 'PASS'                        # baseline numeric error cancels
        assert nf.determinism(p, p)['deterministic']
        mut = dict(p, passed=2294, n_fail=313, fail_counts=dict(p['fail_counts'], **{'MUL_MAT(synthetic)': 1}))
        assert nf.compare(p, mut)['verdict'] == 'NEW_FAIL'
        assert 'FILLER_ID' in job.tbo_env('/x', {})
    finally:
        td.cleanup()
    print('rescore-r3 OK: 2295/2607, 312 fail keys reconciled, baseline USABLE, self-compare PASS, synthetic new-fail caught')


if __name__ == '__main__':
    main()


def test_rescore_r3():
    main()
