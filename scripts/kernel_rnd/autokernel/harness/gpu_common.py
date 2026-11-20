"""Canonical shared helpers for autokernel GPU measurement rows.

New rows IMPORT this file (sys.path.insert(0, '/mnt/raid0/llm/epyc-inference-research/scripts/kernel_rnd/autokernel/harness')) and pin
its sha256 in the row's SOURCE.sha256 instead of vendoring a copy. The five pre-2026-11-20 row dirs carry byte-identical vendored copies
(md5 f7e8f3e8...) and are left as they are. 2026-11-20 change: run_sampled runs children under LD_DEBUG=libs and records wall time, and
residency_from_proof gates sub-second processes (< SHORT_S) on the in-process witness + LD_DEBUG library proof, with the VRAM sampler
kept as supplementary evidence (origin: ak-akx-all-29-4-guardband-gpu-20261119-r2, FUTILITY_RESIDENCY from a 2-sample miss).

Every function reads the module globals set by init(). Writes go only under <row>/out (or out-smoke). Children carry FILLER_ID and are
plain Popen children of the row job (never a new session); rc 75 only on STOP file / FILLER_DEADLINE; missing input is a verdict, not rc 75.
"""
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
import urllib.request

G = {}
KFD = '/sys/class/kfd/kfd/proc'
VERIFY = '/mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh'
UIREF = '/mnt/raid0/llm/kernels/builds/cpu-20261103-181483cfc-lbt/tools/ui/dist'
C10_UI = '/mnt/raid0/llm/kernels/builds/gpu-20261109-7f1bd044b-c10fold/tools/ui/dist'
IN = '/mnt/raid0/llm/kernels/inputs/llguidance-v1.0.1-d795912'
C14_STORE = '/mnt/raid0/llm/kernels/builds/gpu-20261118-eee3da838-c14'
C14_SHA = 'eee3da838e3c8306a856f2c72c4b04048fded698'
OLD = '/mnt/raid0/llm/tmp/c11-fold-gpu-20261114'  # path baked into the pinned configure argv template
REPO = '/mnt/raid0/llm/llama.cpp-experimental'  # read-only `git archive` of commit objects
MODEL27 = '/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf'
MODEL27_SIZE = 29047086048
DRAFTER27 = '/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf'
MEMBIND = ['numactl', '--membind=3', '--']
stopped = False


def _sig(*_):
    global stopped
    stopped = True


def init(row, d, task, port=None):
    G.update(ROW=row, D=d, TASK=task, PORT=port)
    G['SMOKE'] = os.environ.get('FILLER_DRYRUN') == '1' or os.environ.get('FILLER_SMOKE') == '1'
    G['BASE'] = d + '/out'
    G['OUT'] = d + ('/out-smoke' if G['SMOKE'] else '/out')
    G['LOGS'] = G['OUT'] + '/logs'
    G['RES'] = G['OUT'] + '/results.jsonl'
    G['RESULT'] = G['OUT'] + '/RESULT.json'
    G['STOPF'] = os.environ.get('FILLER_STOPFILE', d + '/STOP')
    G['DEADLINE'] = int(os.environ.get('FILLER_DEADLINE', str(int(time.time()) + 8 * 3600)))
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    os.makedirs(G['LOGS'], exist_ok=True)
    sys.path.insert(0, d)


def now(): return time.strftime('%FT%TZ', time.gmtime())
def stop(margin=300): return stopped or os.path.exists(G['STOPF']) or time.time() > G['DEADLINE'] - margin
def smoke(): return G['SMOKE']


def cpus():
    """The granted cpus. WS58_REGIONS is honoured in smoke and live (its cpu tokens only)."""
    ws = ','.join(t for t in os.environ.get('WS58_REGIONS', '').split(',') if t and t[0].isdigit())
    return ws or os.environ.get('FILLER_CPUS') or '88-95,184-191'


def ncpus():
    n = 0
    for t in cpus().split(','):
        if t:
            a, _, b = t.partition('-')
            n += int(b or a) - int(a) + 1
    return n or 16


def srv_cpus():
    """The 8-thread GPU host server/bench runs on the GPU host threads 184-191 only (literal -t 8) when the grant holds them."""
    c = cpus()
    return '184-191' if (not smoke() and '184-191' in c) else c


def live_cpu_check():
    """Core checks run in the LIVE run only. Returns a list of problems."""
    if smoke():
        return []
    have = set()
    for t in cpus().split(','):
        if t:
            a, _, b = t.partition('-')
            have.update(range(int(a), int(b or a) + 1))
    miss = [c for c in range(184, 192) if c not in have]
    return [f'live grant {cpus()} lacks GPU host threads 184-191 ({miss})'] if miss else []


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jsave(p, o):
    with open(p + '.tmp', 'w') as f:
        json.dump(o, f, indent=1, sort_keys=True)
    os.replace(p + '.tmp', p)


def append(u):
    with open(G['RES'], 'a') as f:
        f.write(json.dumps({**u, 'ts': now()}, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def done():
    r = {}
    try:
        for ln in open(G['RES']):
            try:
                u = json.loads(ln)
            except ValueError:
                continue
            if u.get('unit_id') and u.get('final'):
                r[u['unit_id']] = u
    except OSError:
        pass
    return r


def end(verdict, code=0, **kw):
    """Write RESULT.json (complete) and exit. FUTILITY_* / KEEP / DROP / terminal verdicts are rc 0; only STOPPED is rc 75."""
    jsave(G['RESULT'], {'row': G['ROW'], 'task': G['TASK'], 'at': now(), 'complete': verdict != 'STOPPED', 'verdict': verdict,
                        'smoke': smoke(), **kw})
    print('VERDICT', verdict, flush=True)
    sys.exit(code)


def pidfile(pid):
    pf = os.environ.get('FILLER_PIDFILE')
    if pf:
        with open(pf, 'a') as f:
            f.write('%d\n' % pid)


def rd(p, d=0):
    try:
        return int(open(p).read().strip())
    except (OSError, ValueError):
        return d


# ------------------------------------------------------------------ environments
def gpu_env(bdir, extra=None):
    e = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/home/node'),
         'LD_LIBRARY_PATH': bdir + '/bin:/opt/rocm/lib', 'HIP_VISIBLE_DEVICES': '0', 'ROCM_PATH': '/opt/rocm',
         'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0', 'FILLER_ID': os.environ.get('FILLER_ID', G.get('ROW', ''))}
    e.update(extra or {})
    return e


def build_env():
    e = dict(os.environ)
    e.update(PATH=IN + '/rust-1.99.1/bin:/usr/bin:/bin', CARGO_HOME=G['BASE'] + '/cargo-home-empty', CARGO_NET_OFFLINE='true',
             CARGO_BUILD_JOBS=str(ncpus()))
    os.makedirs(G['BASE'] + '/cargo-home-empty', exist_ok=True)
    return e


# ------------------------------------------------------------------ source + in-row build
def tree_of(sha):
    return subprocess.run(['git', '-C', REPO, 'rev-parse', sha + '^{tree}'], capture_output=True, text=True).stdout.strip()


def extract(sha, tree, src, marker=None):
    """git archive of a commit object into src (never a checkout). marker = (relpath, text, must_be_present). Idempotent."""
    stamp = src + '.stamp.json'
    if os.path.exists(stamp) and os.path.isdir(src):
        try:
            if json.load(open(stamp)).get('tree') == tree:
                return True
        except ValueError:
            pass
    if os.path.exists(src):
        shutil.rmtree(src)
    os.makedirs(src)
    p = subprocess.Popen(['git', '-C', REPO, 'archive', '--format=tar', sha], stdout=subprocess.PIPE)
    x = subprocess.run(['tar', '-x', '-C', src], stdin=p.stdout)
    p.stdout.close()
    if p.wait() != 0 or x.returncode != 0:
        return False
    if marker:
        f, txt, present = marker
        if (txt in open(src + '/' + f).read()) != present:
            return False
    jsave(stamp, {'sha': sha, 'tree': tree, 'at': now()})
    return True


def configure_argv(template, src, bdir):
    v = [x.replace(OLD + '/build-hip', bdir).replace(OLD + '/src', src) for x in json.load(open(template))]
    v[v.index('-S') + 1] = src
    v[v.index('-B') + 1] = bdir
    v = [x for x in v if not x.startswith('-DLLAMA_USE_PREBUILT_UI')]
    return v + ['-DLLAMA_USE_PREBUILT_UI:BOOL=OFF']


def build(label, sha, tree, src, bdir, targets, template, marker=None, sha_hip_must_differ_from=None):
    """In-row HIP build (configure + cmake --build of `targets`). Returns a proof dict; None on STOP/deadline."""
    if os.path.exists(bdir + '/BUILD.json'):
        try:
            b = json.load(open(bdir + '/BUILD.json'))
            if b.get('ok') and b.get('source_tree') == tree and all(os.path.exists(f'{bdir}/bin/{t}') for t in targets):
                return {**b, 'stage': 'reused'}
        except ValueError:
            pass
    if not extract(sha, tree, src, marker):
        return {'ok': False, 'stage': 'extract', 'label': label}
    os.makedirs(bdir + '/tools/ui', exist_ok=True)
    if not os.path.isdir(bdir + '/tools/ui/dist'):
        shutil.copytree(C10_UI, bdir + '/tools/ui/dist')
    if not os.path.exists(src + '/tools/ui/dist/index.html'):
        shutil.copytree(UIREF, src + '/tools/ui/dist', dirs_exist_ok=True)
    argv = configure_argv(template, src, bdir)
    os.makedirs(G['BASE'] + '/logs', exist_ok=True)
    with open(f"{G['BASE']}/logs/configure-{label}.log", 'wb') as f:
        crc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, cwd=G['D'], env=build_env()).returncode
    cc = open(bdir + '/CMakeCache.txt', errors='replace').read() if os.path.exists(bdir + '/CMakeCache.txt') else ''
    if crc != 0 or 'LLAMA_USE_PREBUILT_UI:BOOL=OFF' not in cc:
        return {'ok': False, 'stage': 'configure', 'rc': crc, 'label': label}
    t0 = time.time()
    with open(f"{G['BASE']}/logs/build-{label}.log", 'ab') as f:
        p = subprocess.Popen(['taskset', '-c', cpus(), 'nice', '-n', '10', '/usr/bin/cmake', '--build', bdir, '--parallel',
                              str(ncpus()), '--target'] + list(targets), stdout=f, stderr=subprocess.STDOUT, env=build_env(), cwd=G['D'])
        pidfile(p.pid)
        while p.poll() is None:
            if stop(0):
                p.terminate()
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                return None
            time.sleep(5)
    if p.returncode != 0:
        return {'ok': False, 'stage': 'build', 'rc': p.returncode, 'label': label}
    link = {}
    for t in targets:
        with open(f"{G['BASE']}/linkage-{label}-{t}.txt", 'wb') as f:
            link[t] = subprocess.run(['/bin/bash', VERIFY, f'{bdir}/bin/{t}', bdir + '/bin'], env=gpu_env(bdir), stdout=f,
                                     stderr=subprocess.STDOUT).returncode
    hip = bdir + '/bin/libggml-hip.so'
    proof = {'cmake_home': f'CMAKE_HOME_DIRECTORY:INTERNAL={src}\n' in cc, 'hip_lib': os.path.exists(hip),
             'linkage_rc0': all(v == 0 for v in link.values()), 'targets_exist': all(os.path.exists(f'{bdir}/bin/{t}') for t in targets)}
    shas = {n: sha256(f'{bdir}/bin/{n}') for n in ['libggml-hip.so', 'libggml-cpu.so', 'libllama.so'] + list(targets)
            if os.path.exists(f'{bdir}/bin/{n}')}
    ok = all(proof.values())
    if sha_hip_must_differ_from and shas.get('libggml-hip.so') == sha_hip_must_differ_from:
        proof['hip_sha_differs'] = False
        ok = False
    r = {'schema': 'epyc.build-identity.v1', 'ok': ok, 'stage': 'done', 'label': label, 'source_sha': sha, 'source_tree': tree,
         'source': src, 'output': bdir, 'configure_argv': argv, 'targets': list(targets), 'content_proof': proof, 'sha256': shas,
         'linkage_rc': link, 'build_s': round(time.time() - t0, 1), 'built_at': now(), 'built_by_row': G['ROW']}
    jsave(bdir + '/BUILD.json', r)
    return r


def store_bound(store=C14_STORE, sha=C14_SHA):
    try:
        return json.load(open(store + '/BUILD-OK.json')).get('source_sha') == sha
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------ GPU residency / sampled one-shot commands
def gpu_proof():
    """The row's own gpuproof.py when the row vendors one (sys.path[0] = row dir), else gpuproof_row.py beside this file."""
    try:
        import gpuproof
    except ImportError:
        sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
        import gpuproof_row as gpuproof
    return gpuproof


SHORT_S = 2.0  # below this wall time the VRAM sampler is supplementary evidence, not a gate
WITNESS_INIT = 'ggml_cuda_init: found 1 ROCm devices'
# the backend RAN on ROCm0: OK or FAIL both prove it (an expected-failure control prints FAIL; requiring OK refused it, r3)
WITNESS_BACKEND = re.compile(r'Backend ROCm0: (OK|FAIL)\b')
_ANSI = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')


def ld_debug_libs(ld_prefix, bdir, lib='libggml-hip.so'):
    """Sampling-free library-residency proof: the LD_DEBUG=libs records (LD_DEBUG_OUTPUT=<ld_prefix>, one file per pid) that show `lib`
    from the row build initialised or mapped by the process. Probe lines ('trying file=') never count. Returns (lines, pids)."""
    import glob
    real = os.path.realpath(bdir) + '/'
    lines, pids = [], set()
    for f in sorted(glob.glob(ld_prefix + '.*')):
        for ln in open(f, errors='replace'):
            if 'trying file=' in ln or not ('calling init:' in ln or 'generating link map' in ln):
                continue
            m = re.search(r'(\S*' + re.escape(lib) + r'\S*)', ln)
            if m and os.path.realpath(m.group(1).split('[')[0].rstrip(';')).startswith(real):
                lines.append(ln.strip()[:200])
                pids.add(f.rsplit('.', 1)[1])
    return lines, sorted(pids)


def witness_residency(txt, etxt, ld_prefix, bdir):
    """In-process witness (ROCm init line + 'Backend ROCm0: OK|FAIL', ANSI stripped) plus the LD_DEBUG library proof."""
    clean = _ANSI.sub('', (txt or '') + '\n' + (etxt or ''))
    w_init = WITNESS_INIT in clean
    m = WITNESS_BACKEND.search(clean)
    lines, pids = ld_debug_libs(ld_prefix, bdir) if ld_prefix else ([], [])
    return {'witness_rocm_init': w_init, 'witness_backend_ran': bool(m), 'witness_backend_status': m.group(1) if m else None,
            'ld_debug_hip_from_row_build': bool(lines), 'ld_debug_pids': pids, 'ld_debug_lines': lines[:4],
            'witness_ok': w_init and bool(m) and bool(lines)}


def residency_from_proof(pr, bdir):
    """Sampler proof (own-pid KFD VRAM > 0 + libggml-hip mapped from the row build). When run_sampled recorded the run's wall time,
    text and LD_DEBUG prefix, the witness proof is added: under SHORT_S it alone gates (the sampler can miss a sub-second process and
    is kept as supplementary evidence); at >= SHORT_S both must hold. A proof without witness data gates on the sampler as before."""
    v = max(pr['kfd_pids_ours'].values(), default=0)
    libs = [ln for m in pr['maps'].values() for ln in m['libs']]
    hip = any('libggml-hip' in ln for ln in libs)
    own = any('libggml-hip' in ln and ln.startswith(os.path.realpath(bdir) + '/') for ln in libs)
    s_ok = v > 0 and hip and own
    r = {'kfd_vram_max_b': v, 'hip_mapped': hip, 'hip_from_arm_build': own, 'vram_delta_b': pr.get('vram_delta_b'),
         'samples': pr.get('samples'), 'kfd_foreign': pr.get('kfd_foreign'), 'sampler_ok': s_ok, 'ok': s_ok, 'gate': 'sampler'}
    if '_elapsed_s' in pr:
        w = witness_residency(pr.get('_text', ''), pr.get('_errtext', ''), pr.get('_ld_prefix'), bdir)
        short = pr['_elapsed_s'] < SHORT_S
        r.update(w, elapsed_s=round(pr['_elapsed_s'], 3), gate='witness' if short else 'sampler+witness',
                 ok=w['witness_ok'] if short else (s_ok and w['witness_ok']))
    return r


def run_sampled(argv, uid, bdir, env_extra=None, cap_s=3600, cwd=None):
    """Run argv (own session sampler: own-pid KFD VRAM + libggml-hip mapping) under LD_DEBUG=libs (per-pid file, never stderr), timed.
    Returns (rc, err, proof, text, errtext) or None on stop; proof carries _elapsed_s/_ld_prefix/_text/_errtext for residency_from_proof."""
    gp = gpu_proof()
    log = f"{G['LOGS']}/{uid}.log"
    ldp = f"{G['LOGS']}/{uid}.lddebug.{int(time.time())}"
    env = gpu_env(bdir, {'LD_DEBUG': 'libs', 'LD_DEBUG_OUTPUT': ldp, **(env_extra or {})})
    t0 = time.time()
    rc, err, pr = gp.run_sampled(argv, log, env=env, timeout_at=min(G['DEADLINE'] - 240, time.time() + cap_s),
                                 stopfile=G['STOPF'], cwd=cwd or G['OUT'], errlog=log + '.err')
    el = time.time() - t0
    if err == 'stop' or stop(0):
        return None
    txt = open(log, errors='replace').read()
    etxt = open(log + '.err', errors='replace').read() if os.path.exists(log + '.err') else ''
    pr = dict(pr, _elapsed_s=el, _ld_prefix=ldp, _text=txt, _errtext=etxt)
    return rc, err, pr, txt, etxt


# ------------------------------------------------------------------ servers
def http(port, path, payload=None, timeout=3600):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='GET' if payload is None else 'POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or 'null')


def own_port_ok(port, pid):
    return subprocess.run(['/usr/bin/python3', '-I', G['D'] + '/own_port_proof.py', 'owner', str(port), str(pid)]).returncode == 0


def port_busy(port):
    return subprocess.run(['/usr/bin/python3', '-I', G['D'] + '/own_port_proof.py', 'listening', str(port)]).returncode == 0


def meminfo():
    o = {}
    try:
        for ln in open('/proc/meminfo'):
            k, _, v = ln.partition(':')
            if k in ('MemFree', 'MemAvailable', 'Cached', 'Mapped', 'Buffers', 'Dirty', 'AnonHugePages', 'Shmem'):
                o[k + '_kb'] = int(v.split()[0])
    except OSError:
        pass
    return o


def page_cache_resident(path):
    """Fraction of a file's pages resident in the page cache (libc mmap + mincore; no data is read)."""
    import ctypes
    try:
        size = os.path.getsize(path)
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED
        finally:
            os.close(fd)
        if addr in (None, ctypes.c_void_p(-1).value):
            return None
        pg = os.sysconf('SC_PAGE_SIZE')
        npg = (size + pg - 1) // pg
        vec = (ctypes.c_ubyte * npg)()
        rc = libc.mincore(addr, size, vec)
        libc.munmap(addr, size)
        return None if rc != 0 else sum(1 for b in bytes(vec) if b & 1) / npg
    except Exception:  # noqa: BLE001
        return None


class Srv:
    """One fresh llama-server per launch (a launch == a unit == a process)."""

    def __init__(self, uid, bdir, port, args, env_extra=None, cpus_=None, verbosity=None):
        self.uid, self.bdir, self.port, self.p = uid, bdir, port, None
        self.argv = MEMBIND + ['taskset', '-c', cpus_ or srv_cpus(), bdir + '/bin/llama-server', '--host', '127.0.0.1', '--port',
                               str(port), '--no-webui'] + list(args) + (['-lv', str(verbosity)] if verbosity is not None else [])
        self.env = gpu_env(bdir, env_extra)

    def start(self, timeout=1800):
        os.makedirs(G['OUT'] + '/servers', exist_ok=True)
        jsave(f"{G['OUT']}/servers/{self.uid}.argv.json", {'argv': self.argv, 'env': self.env})
        self.log = f"{G['OUT']}/servers/{self.uid}.server.log"
        self.lf = open(self.log, 'wb')
        self.t_start = time.time()
        self.p = subprocess.Popen(self.argv, stdout=self.lf, stderr=subprocess.STDOUT, env=self.env, cwd=G['D'])
        pidfile(self.p.pid)
        ready = False
        while time.time() - self.t_start < timeout and self.p.poll() is None and not stop(0):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2).read()
                ready = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        self.load_s = round(time.time() - self.t_start, 1)
        if not ready:
            return 'stopped' if stop(0) else 'not-ready'
        # numactl/taskset exec: the Popen pid is the server; the own-port proof still binds the listener to it
        if not own_port_ok(self.port, self.p.pid):
            return 'not-own-pid'
        return 'ok'

    def residency(self, min_gib=1.0):
        pid = self.p.pid
        d = f'{KFD}/{pid}'
        try:
            vram = sum(rd(os.path.join(d, f)) for f in os.listdir(d) if f.startswith('vram_'))
        except OSError:
            vram = 0
        try:
            m = sorted({os.path.realpath(ln.split()[-1]) for ln in open(f'/proc/{pid}/maps') if 'libggml' in ln or 'libllama' in ln})
        except OSError:
            m = []
        pre = os.path.realpath(self.bdir) + '/bin/'
        hip = any(os.path.basename(x).startswith('libggml-hip') and x.startswith(pre) for x in m)
        own = bool(m) and all(x.startswith(pre) for x in m)
        return {'kfd_pid': pid, 'own_vram_b': vram, 'hip_mapped_from_build': hip, 'own_libs': own, 'libs': m,
                'resident': vram >= int(min_gib * (1 << 30)) and hip and own}

    def gen(self, msgs, npred, timeout=3600):
        t0 = time.time()
        b = http(self.port, '/v1/chat/completions',
                 {'messages': msgs, 'max_tokens': npred, 'temperature': 0, 'top_k': 1, 'seed': 42, 'stream': False,
                  'cache_prompt': False, 'ignore_eos': True, 'chat_template_kwargs': {'enable_thinking': False}}, timeout)
        tm = b.get('timings') or {}
        c = (b.get('choices') or [{}])[0].get('message', {}).get('content') or ''
        return {'predicted_n': tm.get('predicted_n'), 'predicted_ms': tm.get('predicted_ms'), 'prompt_n': tm.get('prompt_n'),
                'prompt_ms': tm.get('prompt_ms'), 'draft_n': tm.get('draft_n'), 'draft_n_accepted': tm.get('draft_n_accepted'),
                'wall_s': round(time.time() - t0, 2), 'content_sha256': hashlib.sha256(c.encode()).hexdigest()}

    def reap(self):
        p = self.p
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(90)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        try:
            self.lf.close()
        except Exception:  # noqa: BLE001
            pass
        return p is None or p.poll() is not None


# ------------------------------------------------------------------ contention label (host busy on the claimed cpus minus own ticks)
def cpu_list(spec):
    out = []
    for t in spec.split(','):
        if t:
            a, _, b = t.partition('-')
            out += list(range(int(a), int(b or a) + 1))
    return out


def cpu_ticks(cpu_set):
    """(busy, total) jiffies summed over cpu_set from /proc/stat."""
    busy = tot = 0
    want = {f'cpu{c}' for c in cpu_set}
    try:
        for ln in open('/proc/stat'):
            f = ln.split()
            if f and f[0] in want:
                v = [int(x) for x in f[1:9]]
                tot += sum(v)
                busy += sum(v) - v[3] - v[4]
    except OSError:
        pass
    return busy, tot


def proc_ticks(pid):
    """utime+stime over the pid and all descendants."""
    seen, stack, total = set(), [pid], 0
    kids = {}
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:
                f = open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()
                kids.setdefault(int(f[1]), []).append(int(d))
            except (OSError, IndexError, ValueError):
                pass
    while stack:
        q = stack.pop()
        if q in seen:
            continue
        seen.add(q)
        stack += kids.get(q, [])
        try:
            f = open(f'/proc/{q}/stat').read().rsplit(')', 1)[1].split()
            total += int(f[11]) + int(f[12])
        except (OSError, IndexError, ValueError):
            pass
    return total


def foreign_kfd(own_pid):
    """KFD pids that are not ours and hold >= 64 MiB VRAM (monitors with 0 VRAM do not count)."""
    out = {}
    try:
        for p in os.listdir(KFD):
            if p.isdigit() and int(p) != own_pid:
                d = os.path.join(KFD, p)
                v = sum(rd(os.path.join(d, f)) for f in os.listdir(d) if f.startswith('vram_'))
                if v >= (64 << 20):
                    out[p] = v
    except OSError:
        pass
    return out


# ------------------------------------------------------------------ statistics (no scipy)
def t_crit(df, conf=0.95):
    """Two-sided Student t critical value (Cornish-Fisher around the normal; table below df 6)."""
    small = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    if df in small:
        return small[df]
    z = st.NormalDist().inv_cdf(0.5 + conf / 2)
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def mean_ci(xs, conf=0.95):
    n = len(xs)
    if n < 2:
        return {'n': n, 'mean': xs[0] if xs else None, 'sd': None, 'lo': None, 'hi': None}
    m, s = st.mean(xs), st.stdev(xs)
    h = t_crit(n - 1, conf) * s / math.sqrt(n)
    return {'n': n, 'mean': m, 'sd': s, 'lo': m - h, 'hi': m + h}


def sd_ci(xs, conf=0.95):
    """Chi-square interval for a standard deviation (Wilson-Hilferty quantiles)."""
    n = len(xs)
    if n < 3:
        return None
    s = st.stdev(xs)
    df = n - 1
    nd = st.NormalDist()

    def chi(p):
        z = nd.inv_cdf(p)
        return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3
    return {'sd': s, 'lo': s * math.sqrt(df / chi(0.5 + conf / 2)), 'hi': s * math.sqrt(df / chi(0.5 - conf / 2))}


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def fnum(rx, txt):
    m = re.findall(rx, txt)
    return float(m[-1]) if m else None


def chunk_rows(txt):
    rows = []
    for ln in txt.splitlines():
        t = [x for x in ln.replace('%', ' ').split() if x != '±']
        if len(t) == 11 and t[0].isdigit():
            try:
                v = [float(x) for x in t[1:]]
            except ValueError:
                continue
            rows.append({'chunk': int(t[0]), 'ppl_q': v[0], 'ln_ratio': v[2], 'kld': v[4], 'dp_rms_pct': v[6], 'same_top_p_pct': v[8]})
    return rows


# ------------------------------------------------------------------ prompts and production recipes
P4K_CHARS = 13200  # ~3.3 chars per token -> ~4k prompt tokens; actual prompt_n is recorded and bounded by the rows


def corpus():
    return open(G['D'] + '/corpus.txt', encoding='utf-8', errors='replace').read()


def prompt4k(k=0):
    off = 7000 + k * 190000
    user = f'Source excerpt:\n{corpus()[off:off + P4K_CHARS]}\n\nWalk through what this code does, function by function, with as much detail as you can.'
    return [{'role': 'system', 'content': 'You are a code-reading assistant. Answer in detail.'}, {'role': 'user', 'content': user}]


SHORT_MSGS = [{'role': 'user', 'content': 'Say hello in one paragraph.'}]
VL30 = '/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf'
VL30_SIZE = 18556687200
VL30_MMPROJ = '/mnt/raid0/llm/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf'
VL30_MMPROJ_SIZE = 1083499584
DRAFTER27_SIZE = 2056414752


def recipe_args(name, ctx=16384):
    """Production GPU workload recipes (W1 = 27B Q8_0 + DFlash2 drafter, np 1; W2 = Qwen3-VL-30B-A3B Q4_K_M + mmproj F16, 1 slot)."""
    if name == 'W1':
        return ['-m', MODEL27, '-md', DRAFTER27, '--spec-type', 'draft-dflash', '--spec-draft-n-max', '8', '-ngl', '99', '-ngld', '99',
                '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16', '-c', str(ctx), '-np', '1', '-b', '2048', '-ub', '2048', '-t', '8',
                '--cache-ram', '0', '--jinja', '--reasoning', 'off', '--device', 'ROCm0', '--device-draft', 'ROCm0']
    if name == 'W2':
        return ['-m', VL30, '--mmproj', VL30_MMPROJ, '-ngl', '99', '-fa', 'on', '-ctk', 'f16', '-ctv', 'f16', '-c', str(ctx), '-np', '1',
                '-b', '2048', '-ub', '2048', '-t', '8', '--cache-ram', '0', '--jinja', '--device', 'ROCm0']
    raise ValueError(name)


def recipe_inputs(name):
    if name == 'W1':
        return [(MODEL27, MODEL27_SIZE), (DRAFTER27, DRAFTER27_SIZE)]
    return [(VL30, VL30_SIZE), (VL30_MMPROJ, VL30_MMPROJ_SIZE)]


def sizes_missing(pairs):
    miss = []
    for p, sz in pairs:
        if not os.path.exists(p):
            miss.append(p)
        elif os.path.getsize(p) != sz:
            miss.append(f'{p} size != {sz}')
    return miss


def decode_window(srv, msgs, npred):
    """One generation with the contention label: host busy on the claimed GPU host cpus minus the server's own ticks, plus foreign KFD."""
    cs = cpu_list('184-191') if not smoke() else cpu_list(srv_cpus())
    b0, t0 = cpu_ticks(cs)
    p0 = proc_ticks(srv.p.pid)
    r = srv.gen(msgs, npred)
    b1, t1 = cpu_ticks(cs)
    p1 = proc_ticks(srv.p.pid)
    tot = max(1, t1 - t0)
    r['foreign_busy_pct'] = round(max(0.0, ((b1 - b0) - (p1 - p0))) / tot * 100.0, 2)
    r['host_busy_pct'] = round((b1 - b0) / tot * 100.0, 2)
    r['foreign_kfd'] = foreign_kfd(srv.p.pid)
    return r
