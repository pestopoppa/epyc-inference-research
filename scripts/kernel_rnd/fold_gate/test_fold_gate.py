"""Unit tests for the fold gate.  Fixtures are compiled at test time (gcc for host ELF, ROCm clang for gfx90a device
code); nothing binary is committed.  Run: python3 -m pytest scripts/kernel_rnd/fold_gate -q"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fold_gate
import hipdev
import libcmp

GCC = shutil.which('gcc')
CLANG = '/opt/rocm/llvm/bin/clang'
needs_gcc = pytest.mark.skipif(not GCC or not shutil.which('readelf') or not shutil.which('objdump'), reason='binutils/gcc missing')
needs_hip = pytest.mark.skipif(not os.path.exists(CLANG) or not os.path.exists('/opt/rocm/llvm/bin/llvm-objdump'), reason='ROCm clang missing')

BASE_SRC = r'''
volatile long sink;
const char tag[] = "BUILDINFO";
long work(long x) { return x * 3 + 1; }
long other(long x) { return x ^ 0x55; }
'''
UI_SRC = r'''
volatile long ui_sz0, ui_sz1;
const unsigned char ui_blob[] = {%BLOB%};
static void ui_init(void) __asm__("_GLOBAL__sub_I_ui.cpp");
static void __attribute__((constructor)) ui_init(void) { ui_sz0 = %S0%; ui_sz1 = %S1%; %EXTRA% }
long work(long x) { return x * 3 + 1; }
'''


def build(tmp, name, src, extra=()):
    c = tmp / f'{name}.c'
    c.write_text(src)
    so = tmp / f'{name}.so'
    subprocess.run([GCC, '-O1', '-shared', '-fPIC', *extra, str(c), '-o', str(so)], check=True)
    return str(so)


def ui_src(s0, s1, blob, extra=''):
    return UI_SRC.replace('%S0%', str(s0)).replace('%S1%', str(s1)).replace('%BLOB%', blob).replace('%EXTRA%', extra)


@needs_gcc
def test_same(tmp_path):
    a = build(tmp_path, 'a', BASE_SRC)
    assert libcmp.classify(a, a)[0] == 'SAME'


@needs_gcc
def test_buildid_and_buildinfo_string_are_explained(tmp_path):
    a = build(tmp_path, 'a', BASE_SRC.replace('BUILDINFO', 'commit 49550eda5'), ['-Wl,--build-id=0x11223344'])
    b = build(tmp_path, 'b', BASE_SRC.replace('BUILDINFO', 'commit 141e4bcf5'), ['-Wl,--build-id=0xaabbccdd'])
    v, det = libcmp.classify(a, b)
    assert v == 'EXPLAINED', det


@needs_gcc
def test_real_code_change_is_code_diff(tmp_path):
    a = build(tmp_path, 'a', BASE_SRC)
    b = build(tmp_path, 'b', BASE_SRC.replace('x * 3 + 1', 'x * 5 + 1'))
    v, det = libcmp.classify(a, b)
    assert v == 'CODE-DIFF' and any('work' in l for l in det)


@needs_gcc
def test_unexplained_rodata_string(tmp_path):
    a = build(tmp_path, 'a', BASE_SRC.replace('BUILDINFO', 'hello there'))
    b = build(tmp_path, 'b', BASE_SRC.replace('BUILDINFO', 'goodbye now'))
    assert libcmp.classify(a, b)[0] == 'UNEXPLAINED'


@needs_gcc
def test_ui_only_diff_needs_the_source_proof(tmp_path):
    a = build(tmp_path, 'a', ui_src(0x247113, 0x3a, '1,2,3,4'))
    b = build(tmp_path, 'b', ui_src(0x247114, 0x3c, '1,2,3,9'))
    assert libcmp.classify(a, b)[0] == 'CODE-DIFF'
    v, det = libcmp.classify(a, b, ui_source_identical=True)
    assert v == 'EXPLAINED-UI', det


@needs_gcc
def test_ui_class_refuses_other_function_change(tmp_path):
    a = build(tmp_path, 'a', ui_src(0x10, 0x20, '1,2'))
    b = build(tmp_path, 'b', ui_src(0x11, 0x20, '1,2').replace('x * 3 + 1', 'x * 7 + 1'))
    assert libcmp.classify(a, b, ui_source_identical=True)[0] == 'CODE-DIFF'


@needs_gcc
def test_ui_class_refuses_non_immediate_change_in_initialiser(tmp_path):
    a = build(tmp_path, 'a', ui_src(0x10, 0x20, '1,2', 'ui_sz0 += 1;'))
    b = build(tmp_path, 'b', ui_src(0x10, 0x20, '1,2', 'ui_sz0 += ui_sz1;'))
    assert libcmp.classify(a, b, ui_source_identical=True)[0] == 'CODE-DIFF'


def git(repo, *a):
    subprocess.run(['git', '-C', str(repo), *a], check=True, capture_output=True)


def test_ui_source_identical_is_proved_by_git(tmp_path):
    r = tmp_path / 'r'
    (r / 'tools/ui').mkdir(parents=True)
    (r / 'tools/server').mkdir(parents=True)
    (r / 'src').mkdir()
    git(r.parent, 'init', '-q', str(r))
    for k, v in (('user.email', 't@t'), ('user.name', 't')):
        git(r, 'config', k, v)
    (r / 'tools/ui/a.ts').write_text('1')
    (r / 'src/x.c').write_text('1')
    git(r, 'add', '-A'); git(r, 'commit', '-qm', 'base')
    (r / 'src/x.c').write_text('2')
    git(r, 'commit', '-qam', 'src only')
    (r / 'tools/ui/a.ts').write_text('2')
    git(r, 'commit', '-qam', 'ui change')
    rev = lambda n: subprocess.run(['git', '-C', str(r), 'rev-parse', f'HEAD~{n}'], capture_output=True, text=True).stdout.strip()
    ok, _ = fold_gate.ui_source_identical(str(r), rev(2), rev(1), ['tools/ui', 'tools/server'])
    assert ok
    ok, msg = fold_gate.ui_source_identical(str(r), rev(2), rev(0), ['tools/ui', 'tools/server'])
    assert not ok and 'changed' in msg
    assert not fold_gate.ui_source_identical(None, None, None, [])[0]


@needs_gcc
def test_driver_end_to_end(tmp_path):
    ba, nb = tmp_path / 'base' / 'bin', tmp_path / 'new' / 'bin'
    ba.mkdir(parents=True); nb.mkdir(parents=True)
    for d, n, id_ in ((ba, 10411, '0x01'), (nb, 10412, '0x02')):
        sub = d.parent
        so = build(sub, 'libllama-common.so.0.0.%d' % n if False else 'tmp1', BASE_SRC.replace('BUILDINFO', str(n)), [f'-Wl,--build-id={id_}'])
        shutil.move(so, d / f'libllama.so.0.0.{n}')
        so = build(sub, 'tmp2', ui_src(10 + n, 5, '1,2,%d' % (n % 200)), [f'-Wl,--build-id={id_}'])
        shutil.move(so, d / 'libllama-server-impl.so')
        so = build(sub, 'tmp3', BASE_SRC.replace('x * 3', 'x * %d' % (n % 7 + 2)))
        shutil.move(so, d / 'libllama-bench-impl.so')
        os.symlink(f'libllama.so.0.0.{n}', d / 'libllama.so.0')
    lines = []
    a = fold_gate.argparse.Namespace(base_image=str(ba), new_image=str(nb), src_repo=None, base_commit=None, cand_commit=None,
                                     expected_changed=['libllama-bench-impl.so'], ui_path=['tools/ui'])
    assert not fold_gate.run(a, lines.append)      # no source proof -> ui lib is CODE-DIFF
    assert any('CODE-DIFF' in l and 'server-impl' in l for l in lines)
    a2 = fold_gate.argparse.Namespace(**{**vars(a)})
    orig = fold_gate.ui_source_identical
    fold_gate.ui_source_identical = lambda *x: (True, 'stub')
    try:
        lines2 = []
        assert fold_gate.run(a2, lines2.append), lines2
        assert any('EXPLAINED-UI' in l for l in lines2) and any('EXPECTED' in l for l in lines2)
        a3 = fold_gate.argparse.Namespace(**{**vars(a), 'expected_changed': []})
        assert not fold_gate.run(a3, [].append)    # bench-impl change not declared -> FAIL
    finally:
        fold_gate.ui_source_identical = orig


# ------------------------------------------------------------------ device code
HIP = '''#include <hip/hip_runtime.h>
__device__ __attribute__((noinline)) void ndc(const char** o, long* n) { o[0] = __FILE__; n[0] = __builtin_strlen(__FILE__); }
__global__ void k(const char** o, long* n) { ndc(o, n); n[1] = n[2] * %d; }
'''


def hip_build(tmp, root, mult=3):
    d = tmp / root / 'ggml' / 'src'
    d.mkdir(parents=True)
    f = d / 'k.hip'
    f.write_text(HIP % mult)
    out = tmp / f'{root}.co'
    env = {**os.environ, 'TMPDIR': str(tmp)}
    subprocess.run([CLANG, '-x', 'hip', '--offload-arch=gfx90a', '--cuda-device-only', '-nogpulib', '-O2', '-c', str(f), '-o', str(out)],
                   check=True, capture_output=True, env=env)
    return out.read_bytes()


@needs_hip
def test_hip_path_length_only_difference_is_classified_not_hidden(tmp_path):
    a, b = hip_build(tmp_path, 'aa'), hip_build(tmp_path, 'bbbbbbbbbbbbbbbb')
    ok, line, bad = hipdev.compare_fatbins(a, b)
    assert ok and '1 differ only in 1 path-length-dependent' in line, (line, bad)


@needs_hip
def test_hip_equal_length_roots_are_exact(tmp_path):
    a, b = hip_build(tmp_path, 'aaaa'), hip_build(tmp_path, 'bbbb')
    ok, line, _ = hipdev.compare_fatbins(a, b)
    assert ok and 'mode=exact' in line and '1 identical' in line, line


@needs_hip
def test_hip_real_device_code_change_fails(tmp_path):
    a, b = hip_build(tmp_path, 'aa', 3), hip_build(tmp_path, 'bbbbbbbbbbbbbbbb', 5)
    ok, line, bad = hipdev.compare_fatbins(a, b)
    assert not ok and bad, line
