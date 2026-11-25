#!/usr/bin/python3
"""ak-27b-served-ub-arrival-sweep-gpu-20261125-r2 (autokernel; autokernel-akx-program.md:SERVED-27B-UB-ARRIVAL-SWEEP).
RULE: SCREENING (operator 2026-11-25, not a gate): per metric, the arrival rate at which B2048 stops beating B512
KNOBS (witness required): none (the arm delta is argv; witness = the -b/-ub values in each launch's own argv)
Scaffolded by mk_ak_row.py v2, filled by hand from ak-27b-served-ub-sweep-gpu-20261124 (same served argv, binary, corpus).

PRE-REGISTERED (before any run):
QUESTION  Under open-loop Poisson arrivals on the served 27B architect_critic recipe (np 4, -c 393216, kv-unified, DFlash2
          n-max 7, MI210), at what arrival rate does -b/-ub 2048 stop beating the served -b/-ub 512? Origin: the ub-sweep
          (np4 bursts) found B2048 +25% per-slot decode, lower long-prompt TTFT, but slower short prefill and 3.5x ITL p95
          during a concurrent prefill; the operator's p95-based break-even estimate (~1 prompt per slot per ~15 s) is to
          be replaced by the measured crossover.
ARMS      B512 (served argv, -b 512 -ub 512) and B2048 (-b 2048 -ub 2048). Both + `-lv 4`. Env = the ub-sweep's served env
          (LD_LIBRARY_PATH = binary's own bin + /opt/rocm/lib, HIP_VISIBLE_DEVICES 0, taskset to the grant's 184-191).
          No V5I knobs: V5I (GGML_HIP_Q8_MF) is not in any served env and its strings are absent from the served v10 lib.
WORKLOAD  per unit = (arm, IAT, repeat), ONE fresh server: greedy probe, then N_ARR = N_WARM + N_MEAS open-loop arrivals
          with exponential inter-arrival times of mean IAT seconds (seeded per (IAT, repeat), the SAME schedule for both
          arms: paired). Prompt mix fixed by arrival index: i % 4 == 3 -> long (~16k tok, ub-sweep W2L prompt), else
          short (~3.6k tok, ub-sweep W1S prompt); 25% long. max_tokens MAX_TOK, ignore_eos, served sampling T 1.0
          top_p 0.95 top_k 20, per-request seed; SSE stream; cache_prompt false. The first N_WARM arrivals are warm-up.
RATES     IAT_S = 40, 25, 15, 8 s per server (rho ~ 0.35, 0.55, 0.9, 1.7 of prefill capacity at ~500 tok/s aggregate
          prefill). 60 s and 4 s were cut for the 3 h budget (operator: cut rates, not repeats): 4 s is deep overload.
SATURATE  open loop: if SAT_OUTSTANDING requests are in flight at a scheduled arrival, no further arrivals are sent and the
          cell is flagged saturated (its metrics cover what completed). Drain cap DRAIN_CAP_S after the last arrival.
METRICS   per measured request: tok/s = predicted_n / (t_end - t_first); TTFT = t_first - t_send; total latency
          = t_end - t_send; ITL = inter-SSE-event gaps (an event can carry several tokens under speculative decoding, same
          in both arms). Per cell (arm, IAT), median over repeats of per-repeat: tok_s_mean, tok_s_p95, tok_s_p05, itl_p95,
          ttft_p50/p95, latency_p50/p95, aggregate_tok_s = measured output tokens / (last end - first send).
DECISION  per metric and IAT, B2048 'beats' iff its cell value is better than B512's (direction per metric). Crossover per
          metric = the first rate (low -> high) at which B2048 stops beating B512: status BEATS_ALL / NEVER_BEATS /
          CROSSOVER (bracket [last beating IAT, first not-beating IAT] + log-rate interpolation of ratio 1) / NON_MONOTONE.
          repeats_agree per rate (both repeats on the same side). Headline: crossover of tok_s_mean and of latency_p50/p95.
          Correctness: every B2048 greedy probe sha == B512's, else verdict CORRECTNESS_FAIL. Row verdict SCREENED.
STOP      on-answer: all 16 units valid. on-futility: B512 cannot launch (FUTILITY_BASELINE); > MAX_INVALID invalid units
          (FUTILITY_INVALID_UNITS). STOP / FILLER_DEADLINE: rc 75, resumable per (arm, IAT, repeat) unit (results.jsonl).
SMOKE     FILLER_SMOKE=1 / FILLER_DRYRUN=1: 30 s startup probe of both arms at -c 393216, then ONE reduced unit on B512
          (IAT 8, 4 arrivals, 32 tokens) -> TERMINAL-smoke.json + smoke_measure.json.
rc: 0 verdict written | 75 stop/deadline | 3 smoke not ok | --selftest: 0 pass, 1 fail (no server)
"""
# GENERATED-BY: mk_ak_row.py v2 harness_sha256=e6724337984071091808c6540f7ac448eb6b42692551cf7c9468f698b9c3e757
import atexit
import hashlib
import json
import math
import os
import random
import signal
import statistics as st
import subprocess
import sys
import threading
import time

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import ak_row_harness as H  # noqa: E402

ROW_ID = 'ak-27b-served-ub-arrival-sweep-gpu-20261125-r2'
KNOBS = []
N_SERVERS = 1
# fresh server starts per arm (only when spec fresh_server_per_visit): one per (IAT, repeat) unit
FRESH_SERVER_STARTS = 8
OUT = os.path.join(D, "out")
OUT_SMOKE = os.path.join(D, "out-smoke")
SMOKE = os.environ.get("FILLER_SMOKE") == "1" or os.environ.get("FILLER_DRYRUN") == "1"
STOPF = os.path.join(D, "STOP")
_SERVERS = []
# The production server argv, DERIVED from the orchestrator registry (spec.served_ref), + the declared `-lv 4` delta.
SERVER_ARGV = ['/mnt/raid0/llm/kernels/builds/gpu-20260921-ffc1bac82/bin/llama-server', '-m', '/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf', '--host', '127.0.0.1', '--port', '{PORT}', '-np', '4', '-c', '393216', '-t', '8', '-b', '512', '-ub', '512', '--flash-attn', 'on', '--jinja', '-ctk', 'q8_0', '-ctv', 'q8_0', '--kv-unified', '--no-mmap', '-ngl', 'all', '--cache-ram', '65536', '--ctx-checkpoints', '32', '--chat-template-file', '/mnt/raid0/llm/models/chat-templates/sha256/1443ea9ab4bb0a2663b1002a5d4baad970e9f0b2f42ca4ec12b4683838284551.jinja', '--no-cache-idle-slots', '-md', '/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf', '-ngld', '99', '--spec-type', 'draft-dflash', '--spec-draft-n-max', '7', '--device', 'ROCm0', '--slot-save-path', '{SLOTDIR}', '--device-draft', 'ROCm0', '-lv', '4']

BASE, ALT = 'B512', 'B2048'
ARMS = {'B512': {}, 'B2048': {'-b': '2048', '-ub': '2048'}}
ARM_ORDER = [BASE, ALT]
IAT_S = [40, 25, 15, 8]          # mean inter-arrival per server, low rate -> high rate
REPEATS = 2
N_WARM, N_MEAS = 3, 12
MAX_TOK = 256
SAT_OUTSTANDING = 12
DRAIN_CAP_S = 900
REQ_TIMEOUT_S = 1500
MAX_INVALID = 4
PROBE_HOLD_S = 30
SERVED_SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}
SHORT_CHARS, LONG_CHARS = 12500, 53000   # ub-sweep W1S (~3.6k tok) / W2L (~16k tok) prompt windows
PROBE_NPRED = 64
SMOKE_IAT, SMOKE_NARR, SMOKE_TOK = 8, 4, 32
OOM_MARKERS = ('out of memory', 'hipErrorOutOfMemory', 'failed to allocate', 'unable to allocate', 'ErrorOutOfMemory',
               'failed to create context', 'failed to load model')
KFD = '/sys/class/kfd/kfd/proc'
# metric -> +1 higher is better, -1 lower is better
METRICS = {'tok_s_mean': 1, 'tok_s_p95': 1, 'tok_s_p05': 1, 'itl_p95_s': -1, 'ttft_p50_s': -1, 'ttft_p95_s': -1,
           'latency_p50_s': -1, 'latency_p95_s': -1, 'aggregate_tok_s': 1}


def server_argv(port, arm=BASE):
    slots = os.path.join(OUT, "kv_slots")
    a = [str(port) if x == "{PORT}" else slots if x == "{SLOTDIR}" else x for x in SERVER_ARGV]
    for flag, val in ARMS[arm].items():
        a[a.index(flag) + 1] = val
    return a


def cpus():
    """The granted cpus (ONE env source: WS58_CPUS, then FILLER_CPUS)."""
    return H.grant_cpus()


def cpu_str():
    return H.cpu_str(cpus())


def ports():
    env = [int(x) for x in os.environ.get("FILLER_PORTS", "").split(",") if x.strip().isdigit()]
    return env[:N_SERVERS] if len(env) >= N_SERVERS else H.alloc_ports(ROW_ID, N_SERVERS)


def _kill_servers(*_a):
    """Kill only the servers this job started (captured Popen handles) and their children, TERM then KILL, by pid."""
    while _SERVERS:
        p = _SERVERS.pop()
        H.kill_tree(p)


def _on_term(signum, _frame):
    _kill_servers()
    sys.exit(128 + signum)


atexit.register(_kill_servers)
signal.signal(signal.SIGTERM, _on_term)


def server_env():
    b = os.path.dirname(SERVER_ARGV[0])
    e = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': os.environ.get('HOME', '/home/node'),
         'LD_LIBRARY_PATH': b + ':/opt/rocm/lib', 'HIP_VISIBLE_DEVICES': '0', 'ROCM_PATH': '/opt/rocm',
         'HSA_OVERRIDE_CPU_AFFINITY_DEBUG': '0'}
    for k in ('FILLER_ID', 'FILLER_DRYRUN', 'WS58_CPUS', 'FILLER_CPUS'):
        if k in os.environ:
            e[k] = os.environ[k]
    return e


def host_cpus():
    """Host threads for the server: the grant's cpus in 184-191 (the -t 8 GPU host lane), else the whole grant."""
    try:
        g = sorted(H.parse_cpus(cpus()))
    except Exception:  # noqa: BLE001
        return ''
    hi = [c for c in g if 184 <= c <= 191]
    return H.compress_cpus(hi or g)


def start_server(argv, log_path, env=None):
    """Start a server process owned (and killed at exit) by this job."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "ab")
    hc = host_cpus()
    full = (['/usr/bin/taskset', '-c', hc] if hc else []) + list(argv)
    p = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, env=env or server_env())   # same session/pgrp as the job: the keeper attributes GPU/KFD use to the grant
    _SERVERS.append(p)
    return p


def request(port, payload, timeout=1800):
    """POST one completion. cache_prompt is forced False: no slot/prompt-cache carry-over."""
    import urllib.request
    body = dict(payload, cache_prompt=False)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream_request(port, payload, timeout=REQ_TIMEOUT_S):
    """Streaming /completion (SSE). Returns {t_send, t_first, t_end, event_times, final} (monotonic seconds)."""
    import urllib.request
    body = dict(payload, cache_prompt=False, stream=True)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_send = time.monotonic()
    times, final = [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            ln = raw.decode('utf-8', 'replace').strip()
            if not ln.startswith('data:'):
                continue
            try:
                ev = json.loads(ln[5:].strip())
            except ValueError:
                continue
            if ev.get('content'):
                times.append(time.monotonic())
            if ev.get('stop'):
                final = ev
                break
    return {'t_send': t_send, 't_first': times[0] if times else None, 't_end': times[-1] if times else None,
            'event_times': times, 'final': final}


# ------------------------------------------------------------------------------------------------------------ workload
_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = open(os.path.join(D, 'corpus.txt'), encoding='utf-8', errors='replace').read()
    return _CORPUS


def prompt(chars, k):
    """The ub-sweep prompt windows: short = prompt(12500, 0..3) (W1S), long = prompt(53000, 1..3) (W2L)."""
    off = 7000 + k * 190000
    off = off % max(1, len(corpus()) - chars)
    return ('Source excerpt:\n' + corpus()[off:off + chars] +
            '\n\nWalk through what this code does, function by function, with as much detail as you can.\n')


def is_long(i):
    return i % 4 == 3


def arrival_prompt(i):
    return prompt(LONG_CHARS, 1 + (i // 4) % 3) if is_long(i) else prompt(SHORT_CHARS, i % 4)


def schedule_times(iat, rep, n):
    """Poisson arrival offsets (s) from unit start; seeded per (IAT, repeat), identical for both arms. First at 0."""
    rng = random.Random(100003 * int(iat) + 17 * rep + 1)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(1.0 / iat)
    return out


PROBE_PROMPT = 'List the first twelve prime numbers, then explain in two sentences why 1 is not prime.\n'


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(len(v) - 1, int(k) + 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def garbage(text):
    t = text or ''
    if len(t.strip()) < 20:
        return 'too short'
    pr = sum(1 for c in t if c.isprintable() or c in '\n\t') / len(t)
    if pr < 0.95:
        return f'printable ratio {pr:.2f}'
    w = t.split()
    if len(w) >= 8 and len(set(w)) / len(w) < 0.2:
        return f'distinct-word ratio {len(set(w)) / len(w):.2f}'
    if not any(ch.isdigit() for ch in t):
        return 'probe answer has no digits (primes expected)'
    return None


def probe(port):
    r = request(port, {'prompt': PROBE_PROMPT, 'n_predict': PROBE_NPRED, 'temperature': 0.0, 'top_k': 1, 'seed': 1})
    c = r.get('content') or ''
    return {'sha256': hashlib.sha256(c.encode()).hexdigest(), 'garbage': garbage(c), 'text': c[:400]}


def run_arrivals(port, iat, rep, n_arr, max_tok, stopf=STOPF):
    """Open-loop: send arrival i at its scheduled offset in its own thread. Returns (per-request records, flags)."""
    offs = schedule_times(iat, rep, n_arr)
    recs = [None] * n_arr
    lock = threading.Lock()
    outstanding = [0]

    def one(i, t_sched):
        try:
            r = stream_request(port, {'prompt': arrival_prompt(i), 'n_predict': max_tok, 'ignore_eos': True,
                                      'seed': 5000 + i, **SERVED_SAMPLING})
            f = r['final'] or {}
            tm = f.get('timings') or {}
            ev = r['event_times']
            recs[i] = {'i': i, 'long': is_long(i), 't_sched': t_sched, 't_send': r['t_send'], 't_first': r['t_first'],
                       't_end': r['t_end'], 'prompt_n': tm.get('prompt_n') or f.get('tokens_evaluated'),
                       'predicted_n': tm.get('predicted_n') or f.get('tokens_predicted'),
                       'gaps': [b - a for a, b in zip(ev, ev[1:])], 'n_events': len(ev),
                       'draft_n': tm.get('draft_n'), 'draft_n_accepted': tm.get('draft_n_accepted')}
        except Exception as e:  # noqa: BLE001
            recs[i] = {'i': i, 'long': is_long(i), 'error': f'{e!r}'[:300]}
        finally:
            with lock:
                outstanding[0] -= 1
    t0 = time.monotonic()
    th, saturated, stopped, sent = [], False, False, 0
    for i, off in enumerate(offs):
        while time.monotonic() - t0 < off:
            time.sleep(min(0.05, max(0.0, off - (time.monotonic() - t0))))
        if H.should_stop(stopf):
            stopped = True
            break
        with lock:
            if outstanding[0] >= SAT_OUTSTANDING:
                saturated = True
            else:
                outstanding[0] += 1
        if saturated:
            break
        t = threading.Thread(target=one, args=(i, t0 + off), daemon=True)
        t.start()
        th.append(t)
        sent += 1
    t_last = time.monotonic()
    for t in th:
        t.join(timeout=max(1.0, DRAIN_CAP_S - (time.monotonic() - t_last)))
    drain_capped = any(t.is_alive() for t in th)
    return recs, {'saturated': saturated, 'stopped': stopped, 'sent': sent, 'drain_capped': drain_capped,
                  'arrival_span_s': t_last - t0, 'offsets': offs}


def unit_metrics(recs, n_warm, max_tok):
    """Per-unit summary over the measured (index >= n_warm) completed requests."""
    m = [r for r in recs if r and r['i'] >= n_warm and not r.get('error') and r.get('t_first')]
    tok_s = [r['predicted_n'] / (r['t_end'] - r['t_first']) for r in m
             if r.get('predicted_n') and r['t_end'] and r['t_end'] > r['t_first']]
    ttft = [r['t_first'] - r['t_send'] for r in m]
    lat = [r['t_end'] - r['t_send'] for r in m]
    gaps = [g for r in m for g in r['gaps']]
    tot = sum(int(r['predicted_n'] or 0) for r in m)
    span = (max(r['t_end'] for r in m) - min(r['t_send'] for r in m)) if m else 0
    return {'n_measured': len(m), 'n_short': sum(1 for r in m if not r['long']), 'n_long': sum(1 for r in m if r['long']),
            'tok_s_mean': st.mean(tok_s) if tok_s else None, 'tok_s_p95': pct(tok_s, .95), 'tok_s_p05': pct(tok_s, .05),
            'itl_p95_s': pct(gaps, .95), 'itl_p50_s': pct(gaps, .5), 'ttft_p50_s': pct(ttft, .5), 'ttft_p95_s': pct(ttft, .95),
            'latency_p50_s': pct(lat, .5), 'latency_p95_s': pct(lat, .95),
            'aggregate_tok_s': tot / span if span > 0 else None, 'measured_tokens': tot,
            'short_ttft_p50_s': pct([r['t_first'] - r['t_send'] for r in m if not r['long']], .5),
            'long_ttft_p50_s': pct([r['t_first'] - r['t_send'] for r in m if r['long']], .5),
            'bad_predicted_n': [r['i'] for r in m if r.get('predicted_n') != max_tok]}


def residency(pid):
    d = f'{KFD}/{pid}'
    vram = 0
    try:
        for f in os.listdir(d):
            if f.startswith('vram_'):
                try:
                    vram += int(open(os.path.join(d, f)).read().strip() or 0)
                except (OSError, ValueError):
                    pass
    except OSError:
        pass
    try:
        libs = sorted({os.path.realpath(ln.split()[-1]) for ln in open(f'/proc/{pid}/maps') if 'libggml-hip' in ln})
    except OSError:
        libs = []
    return {'own_vram_b': vram, 'hip_libs': libs, 'resident': vram > (8 << 30) and bool(libs)}


def log_oom(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 400000))
            txt = f.read().decode('utf-8', 'replace')
    except OSError:
        return None
    for ln in txt.splitlines():
        if any(m.lower() in ln.lower() for m in OOM_MARKERS):
            return ln.strip()[:300]
    return None


def launch(arm, uid, port, out_dir, hold_s=0, ready_timeout=900):
    """Start ONE fresh server for `arm`; wait /health; optional hold (startup probe). Returns (proc, info)."""
    import urllib.request
    log = os.path.join(out_dir, 'servers', f'{uid}.log')
    argv = server_argv(port, arm)
    os.makedirs(os.path.join(OUT, 'kv_slots'), exist_ok=True)
    t0 = time.time()
    p = start_server(argv, log)
    ready = False
    while time.time() - t0 < ready_timeout and p.poll() is None:
        if H.should_stop(STOPF):
            break
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).read()
            ready = True
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    load_s = time.time() - t0
    t_h = time.time()
    while ready and time.time() - t_h < hold_s and p.poll() is None:
        time.sleep(1)
    alive = p.poll() is None
    info = {'uid': uid, 'arm': arm, 'argv': argv, 'log': log, 'load_s': load_s, 'ready': ready, 'alive': alive,
            'oom_line': log_oom(log), 'witness_b_ub': [argv[argv.index('-b') + 1], argv[argv.index('-ub') + 1]]}
    info['residency'] = residency(p.pid) if alive else None   # taskset execs the server: Popen pid IS the server pid
    return p, info


def stop_one(p):
    if p in _SERVERS:
        _SERVERS.remove(p)
    H.kill_tree(p)


def measure_unit(port, iat, rep, small=False):
    """ONE unit on an already-ready server: greedy probe, then the open-loop arrival run."""
    pr = probe(port)
    n_arr, tok = (SMOKE_NARR, SMOKE_TOK) if small else (N_WARM + N_MEAS, MAX_TOK)
    recs, flags = run_arrivals(port, iat, rep, n_arr, tok)
    n_warm = 1 if small else N_WARM
    return {'probe': pr, 'requests': recs, 'flags': flags, 'metrics': unit_metrics(recs, n_warm, tok)}


def valid(rec, small=False):
    bad = []
    fl, m = rec.get('flags') or {}, rec.get('metrics') or {}
    if fl.get('stopped'):
        bad.append('stopped mid-unit')
    errs = [r['i'] for r in rec.get('requests') or [] if r and r.get('error')]
    if errs and not fl.get('saturated'):
        bad.append(f'request errors {errs}')
    if not fl.get('saturated') and not fl.get('drain_capped') and m.get('bad_predicted_n'):
        bad.append(f"predicted_n != max_tokens for {m['bad_predicted_n']}")
    if not m.get('n_measured'):
        bad.append('no measured request completed')
    return bad


# ------------------------------------------------------------------------------------------------------------ decision
def schedule():
    """(repeat, IAT, arm) units: per repeat, rates low -> high, arm order alternating per rate and per repeat."""
    out = []
    for rep in range(REPEATS):
        for k, iat in enumerate(IAT_S):
            arms = ARM_ORDER if (k + rep) % 2 == 0 else ARM_ORDER[::-1]
            out += [(rep, iat, a) for a in arms]
    return out


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def crossover(rows, sign):
    """rows: [(iat, b512, b2048, agree)] ordered low rate -> high rate. First rate at which B2048 stops beating B512."""
    beats = [(None if (a is None or b is None) else (b - a) * sign > 0) for _, a, b, _ in rows]
    known = [x for x in beats if x is not None]
    out = {'beats': [{'iat_s': r[0], 'b512': r[1], 'b2048': r[2], 'ratio': (r[2] / r[1]) if r[1] and r[2] else None,
                      'b2048_beats': x, 'repeats_agree': r[3]} for r, x in zip(rows, beats)]}
    if not known:
        out['status'] = 'NO_DATA'
        return out
    if all(known):
        out['status'] = 'BEATS_ALL'
        return out
    if not beats[0]:
        out['status'] = 'NEVER_BEATS' if not any(known) else 'NON_MONOTONE'
        out['stops_at_iat_s'] = rows[0][0]
        return out
    j = next(k for k, x in enumerate(beats) if x is False)
    out['status'] = 'NON_MONOTONE' if any(x for x in beats[j + 1:] if x is not None) else 'CROSSOVER'
    out['stops_at_iat_s'] = rows[j][0]
    out['bracket_iat_s'] = [rows[j - 1][0], rows[j][0]]
    try:   # interpolate ln(ratio) = 0 linearly in ln(rate)
        (i0, a0, b0, _), (i1, a1, b1, _) = rows[j - 1], rows[j]
        y0, y1 = math.log(b0 / a0), math.log(b1 / a1)
        x0, x1 = math.log(1.0 / i0), math.log(1.0 / i1)
        x = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
        out['interp_iat_s'] = 1.0 / math.exp(x)
    except (ZeroDivisionError, ValueError, TypeError):
        out['interp_iat_s'] = None
    return out


def decide(units):
    """units: valid unit records {arm, iat, rep, rec}. Returns (verdict, table)."""
    cells, per_rep = {}, {}
    for arm in ARM_ORDER:
        for iat in IAT_S:
            us = [u for u in units if u['arm'] == arm and u['iat'] == iat]
            c = {'n_repeats': len(us), 'saturated_any': any(u['rec']['flags'].get('saturated') for u in us),
                 'drain_capped_any': any(u['rec']['flags'].get('drain_capped') for u in us),
                 'n_measured': sum(u['rec']['metrics']['n_measured'] for u in us)}
            for m in list(METRICS) + ['itl_p50_s', 'short_ttft_p50_s', 'long_ttft_p50_s']:
                c[m] = med([u['rec']['metrics'].get(m) for u in us])
            cells[f'{arm}@{iat}'] = c
            per_rep[(arm, iat)] = {u['rep']: u['rec']['metrics'] for u in us}
    if not any(cells[f'{BASE}@{i}']['n_repeats'] for i in IAT_S):
        return 'FUTILITY_BASELINE', {'cells': cells}
    xo = {}
    for m, sign in METRICS.items():
        rows = []
        for iat in IAT_S:
            a, b = cells[f'{BASE}@{iat}'][m], cells[f'{ALT}@{iat}'][m]
            sides = []
            for rep in range(REPEATS):
                ra, rb = per_rep[(BASE, iat)].get(rep, {}).get(m), per_rep[(ALT, iat)].get(rep, {}).get(m)
                if ra is not None and rb is not None:
                    sides.append((rb - ra) * sign > 0)
            rows.append((iat, a, b, (len(set(sides)) == 1) if len(sides) >= 2 else None))
        xo[m] = crossover(rows, sign)
    b_sha = {u['rec']['probe']['sha256'] for u in units if u['arm'] == BASE}
    alt_sha = {u['rec']['probe']['sha256'] for u in units if u['arm'] == ALT}
    garb = [u['rec']['probe']['garbage'] for u in units if u['rec']['probe']['garbage']]
    corr = {'b512_shas': sorted(b_sha), 'b2048_shas': sorted(alt_sha),
            'identical': bool(alt_sha) and alt_sha <= b_sha and len(b_sha) == 1, 'garbage': garb[:3]}
    table = {'cells': cells, 'crossover': xo, 'correctness': corr,
             'headline': {'per_user_decode_tok_s_mean': xo['tok_s_mean'], 'total_latency_p50': xo['latency_p50_s'],
                          'total_latency_p95': xo['latency_p95_s']}}
    verdict = 'SCREENED' if corr['identical'] and not garb else 'CORRECTNESS_FAIL'
    return verdict, table


def witness_ok(witness):
    """Every knob in KNOBS must be witnessed (present and truthy) in RESULT["witness"]."""
    return all(witness.get(k) for k in KNOBS)


def finish(units, witness, extra=None):
    verdict, table = decide(units)
    if KNOBS and not witness_ok(witness):
        verdict = "FUTILITY_WITNESS"
    slim = [{k: v for k, v in u.items() if k != 'rec'} | {'metrics': u['rec']['metrics'], 'flags': u['rec']['flags'],
             'probe': u['rec']['probe']} for u in units]
    res = {"complete": True, "verdict": verdict, **table, "witness": witness, "units": slim, "row": ROW_ID,
           "utc": H.now(), **(extra or {})}
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "RESULT.json.tmp")
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, os.path.join(OUT, "RESULT.json"))
    return res


def end(verdict, rc=0, **kw):
    res = {"complete": verdict != 'STOPPED', "verdict": verdict, "row": ROW_ID, "utc": H.now(), **kw}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "RESULT.json.tmp"), "w") as f:
        json.dump(res, f, indent=1)
    os.replace(os.path.join(OUT, "RESULT.json.tmp"), os.path.join(OUT, "RESULT.json"))
    _kill_servers()
    sys.exit(rc)


def start_production_server(port):
    """The smoke and main() start the SAME server: server_argv(port) (the production argv), log under out/."""
    if not SERVER_ARGV:
        raise RuntimeError("SERVER_ARGV is empty")
    return start_server(server_argv(port), os.path.join(OUT, f"server-{port}.log"))


def smoke():
    """Feasibility: the 30 s startup probe of BOTH arms at the served -c; then ONE reduced unit on B512."""
    try:
        port = ports()[0]
        feas = {}
        for arm in ARM_ORDER:
            p, info = launch(arm, f'smoke-probe-{arm}', port, OUT_SMOKE, hold_s=PROBE_HOLD_S)
            stop_one(p)
            feas[arm] = {k: info[k] for k in ('ready', 'alive', 'oom_line', 'load_s', 'residency', 'witness_b_ub')}
            H.append_unit({'smoke': True, 'kind': 'startup_probe', 'arm': arm, **feas[arm]}, OUT_SMOKE)
        if not (feas[BASE]['ready'] and feas[BASE]['alive']):
            return H.end_smoke("FAIL", 3, f"baseline B512 startup probe failed: {feas[BASE]}", OUT)
        p, info = launch(BASE, 'smoke-unit', port, OUT_SMOKE)
        u0 = time.time()
        r = measure_unit(port, SMOKE_IAT, 0, small=True)
        unit_s = time.time() - u0
        stop_one(p)
        bad = valid(r, small=True)
        if not (info['residency'] or {}).get('resident'):
            bad.append('not resident (own KFD VRAM <= 8 GiB or no libggml-hip mapped)')
        H.append_unit(dict(r, unit=0, smoke=True, arm=BASE, invalid_why=bad), OUT_SMOKE)
        H.write_smoke_measure(OUT_SMOKE, unit_s, 1, info['load_s'], unit_name='reduced unit (probe + 4 arrivals x 32 tok)',
                              extra={'feasibility': {a: {'ready': f['ready'], 'alive': f['alive'], 'oom_line': f['oom_line']}
                                                     for a, f in feas.items()}, 'units': len(schedule())})
        if bad:
            return H.end_smoke("FAIL", 3, f"smoke unit invalid: {bad}", OUT)
        drop = [a for a, f in feas.items() if not (f['ready'] and f['alive']) or f['oom_line']]
        if drop:
            return H.end_smoke("FAIL", 3, f"startup probe failed for {drop}: {[feas[a] for a in drop]}", OUT)
        return H.end_smoke("PASS", 0, f"1 reduced unit in {unit_s:.1f} s; both arms start at -c 393216", OUT)
    except Exception as e:  # noqa: BLE001 - the smoke must always say why it failed
        return H.end_smoke("FAIL", 3, f"{e.__class__.__name__}: {e}", OUT)
    finally:
        _kill_servers()


def done_units():
    p = os.path.join(OUT, H.RESULTS_JSONL)
    out = []
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def main():
    H.require_grant()   # no FILLER_ID/FILLER_DRYRUN -> exit 64: never run a row (or its smoke) by hand
    if SMOKE:
        return smoke()
    try:
        port = ports()[0]
        prev = done_units()
        n_invalid = sum(1 for u in prev if u.get('kind') == 'unit' and not u.get('valid'))
        have = {(u['rep'], u['iat'], u['arm']) for u in prev if u.get('kind') == 'unit' and u.get('valid')}
        probed = {u['arm'] for u in prev if u.get('kind') == 'unit'}
        for rep, iat, arm in schedule():
            if (rep, iat, arm) in have:
                continue
            if H.should_stop(STOPF):
                end('STOPPED', 75)
            first = arm not in probed
            p, info = launch(arm, f'r{rep}-i{iat}-{arm}-{int(time.time())}', port, OUT, hold_s=PROBE_HOLD_S if first else 0)
            probed.add(arm)
            if not (info['ready'] and info['alive']) or (first and info['oom_line']):
                stop_one(p)
                if H.should_stop(STOPF):
                    end('STOPPED', 75)
                if arm == BASE and first:
                    end('FUTILITY_BASELINE', info=info)
                rec, bad = {}, [f"server not ready/alive: {info['oom_line']}"]
            else:
                try:
                    rec = measure_unit(port, iat, rep)
                    bad = valid(rec)
                except Exception as e:  # noqa: BLE001
                    rec, bad = {}, [f'error: {e!r}'[:300]]
                info['residency_post'] = residency(p.pid) if p.poll() is None else None
                info['oom_line_post'] = log_oom(info['log'])
                stop_one(p)
                if not (info['residency'] or {}).get('resident'):
                    bad.append('not resident (own KFD VRAM <= 8 GiB or no libggml-hip mapped)')
            if (rec.get('flags') or {}).get('stopped'):
                end('STOPPED', 75)
            ok = not bad
            H.append_unit({'kind': 'unit', 'unit_id': info['uid'], 'rep': rep, 'iat': iat, 'arm': arm, 'valid': ok,
                           'invalid_why': bad, 'rec': rec, 'info': info}, OUT)
            if ok:
                have.add((rep, iat, arm))
            else:
                n_invalid += 1
                if n_invalid > MAX_INVALID:
                    end('FUTILITY_INVALID_UNITS', n_invalid=n_invalid)
        units = [u for u in done_units() if u.get('kind') == 'unit' and u.get('valid')]
        finish(units, {'b_ub_per_unit': {u['unit_id']: u['info']['witness_b_ub'] for u in units}},
               extra={'n_invalid': n_invalid, 'server_binary': SERVER_ARGV[0], 'iat_s': IAT_S, 'repeats': REPEATS,
                      'n_warm': N_WARM, 'n_meas': N_MEAS, 'max_tokens': MAX_TOK})
        return 0
    finally:
        _kill_servers()


def selftest():
    """No compute: schedule, argv deltas, arrivals, metrics, crossover, decide(); then smoke() against stubs."""
    import tempfile
    global KNOBS, OUT, OUT_SMOKE, launch, measure_unit, stop_one
    fails = []

    def chk(c, m):
        if not c:
            fails.append(m)
    sc = schedule()
    chk(len(sc) == REPEATS * len(IAT_S) * 2 and len(set(sc)) == len(sc), 'schedule size/unique')
    chk(sc[0][2] != sc[2][2] and sc[0][2] != sc[2 * len(IAT_S)][2], 'arm order alternates per rate and per repeat')
    for arm, dl in ARMS.items():
        a = server_argv(1234, arm)
        chk(a[a.index('-b') + 1] == dl.get('-b', '512') and a[a.index('-ub') + 1] == dl.get('-ub', '512'), f'argv delta {arm}')
        chk(a[a.index('-c') + 1] == '393216' and '1234' in a and a[a.index('-lv') + 1] == '4', f'served -c, port, -lv {arm}')
        chk(len(a) == len(SERVER_ARGV), f'no flags added {arm}')
    s1, s2 = schedule_times(15, 0, 200), schedule_times(15, 0, 200)
    chk(s1 == s2 and s1[0] == 0 and all(b > a for a, b in zip(s1, s1[1:])), 'arrivals deterministic, increasing')
    chk(10 < s1[-1] / 199 < 20, f'mean IAT ~15 ({s1[-1] / 199:.1f})')
    chk(schedule_times(15, 1, 5) != schedule_times(15, 0, 5), 'repeats differ')
    chk(sum(is_long(i) for i in range(N_WARM + N_MEAS)) == 3, 'fixed 25% long mix')
    chk(len(arrival_prompt(3)) > 4 * len(arrival_prompt(0)), 'long vs short prompt')
    chk(garbage('2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37. One is not prime because it has one divisor.') is None, 'good probe')
    chk(garbage('the the the the the the the the the the') is not None, 'garbage probe')
    recs = [{'i': i, 'long': is_long(i), 't_send': 10.0 * i, 't_first': 10.0 * i + 2, 't_end': 10.0 * i + 12,
             'predicted_n': 100, 'gaps': [0.1, 0.2]} for i in range(6)]
    um = unit_metrics(recs, 2, 100)
    chk(um['n_measured'] == 4 and abs(um['tok_s_mean'] - 10.0) < 1e-9 and abs(um['latency_p50_s'] - 12) < 1e-9
        and abs(um['ttft_p50_s'] - 2) < 1e-9 and not um['bad_predicted_n'], f'unit_metrics {um}')
    x = crossover([(40, 10, 12, True), (25, 10, 11, True), (15, 10, 9, True), (8, 10, 8, True)], 1)
    chk(x['status'] == 'CROSSOVER' and x['stops_at_iat_s'] == 15 and x['bracket_iat_s'] == [25, 15]
        and 15 < x['interp_iat_s'] < 25, f'crossover {x}')
    chk(crossover([(40, 5, 4, None), (25, 5, 4, None)], -1)['status'] == 'BEATS_ALL', 'lower-better beats all')
    chk(crossover([(40, 5, 6, None), (25, 5, 6, None)], -1)['status'] == 'NEVER_BEATS', 'never beats')
    chk(crossover([(40, 1, 2, None), (25, 1, 0.5, None), (15, 1, 2, None)], 1)['status'] == 'NON_MONOTONE', 'non-monotone')

    def unit(arm, iat, rep, tok, lat, sha='s'):
        m = {k: None for k in METRICS}
        m.update(tok_s_mean=tok, tok_s_p95=tok, tok_s_p05=tok, itl_p95_s=1.0, ttft_p50_s=lat / 2, ttft_p95_s=lat,
                 latency_p50_s=lat, latency_p95_s=lat * 1.5, aggregate_tok_s=tok * 3, n_measured=N_MEAS)
        return {'arm': arm, 'iat': iat, 'rep': rep, 'unit_id': f'{arm}{iat}{rep}', 'info': {'witness_b_ub': ['512', '512']},
                'rec': {'probe': {'sha256': sha, 'garbage': None}, 'flags': {'saturated': iat == 8}, 'metrics': m}}
    us = []
    for rep in range(REPEATS):
        for iat in IAT_S:
            us.append(unit(BASE, iat, rep, 20.0, 30.0))
            us.append(unit(ALT, iat, rep, 25.0 if iat >= 25 else 18.0, 25.0 if iat >= 15 else 40.0))
    v, t = decide(us)
    chk(v == 'SCREENED', f'screened {v}')
    chk(t['headline']['per_user_decode_tok_s_mean']['status'] == 'CROSSOVER'
        and t['headline']['per_user_decode_tok_s_mean']['stops_at_iat_s'] == 15, 'tok/s crossover at 15')
    chk(t['headline']['total_latency_p50']['stops_at_iat_s'] == 8, 'latency crossover at 8')
    chk(t['cells'][f'{BASE}@8']['saturated_any'], 'saturation flag')
    chk(decide(us + [unit(ALT, 40, 0, 1, 1, sha='other')])[0] == 'CORRECTNESS_FAIL', 'sha mismatch')
    chk(decide([u for u in us if u['arm'] != BASE])[0] == 'FUTILITY_BASELINE', 'no baseline')
    saved = (KNOBS, OUT, OUT_SMOKE, launch, measure_unit, stop_one)
    try:
        KNOBS = ["GGML_TEST_KNOB"]
        chk(not witness_ok({}) and witness_ok({"GGML_TEST_KNOB": "on in log"}), 'witness rule')
        KNOBS = []
        with tempfile.TemporaryDirectory() as td:
            chk(H.end_smoke("FAIL", 3, "selftest", td, stream=open(os.devnull, "w")) == 3, 'end_smoke rc')
        with tempfile.TemporaryDirectory() as td:
            OUT, OUT_SMOKE = os.path.join(td, "out"), os.path.join(td, "out-smoke")
            res = finish(us, {})
            chk(res['verdict'] == 'SCREENED' and os.path.exists(os.path.join(OUT, 'RESULT.json')), 'finish writes RESULT')
            launch = lambda arm, uid, port, od, hold_s=0, ready_timeout=900: (None, {  # noqa: E731 - stub: no process
                'ready': True, 'alive': True, 'oom_line': None, 'load_s': 1.0,
                'residency': {'resident': True}, 'witness_b_ub': ['512', '512']})
            stop_one = lambda p: None  # noqa: E731
            measure_unit = lambda port, iat, rep, small=False: {  # noqa: E731
                'probe': {'sha256': 's', 'garbage': None}, 'requests': [], 'flags': {},
                'metrics': {'n_measured': 3, 'bad_predicted_n': []}}
            chk(smoke() == 0, 'smoke() against the stub did not PASS')
            term = json.load(open(os.path.join(OUT, H.TERMINAL_SMOKE)))
            chk(term["ok"] is True, f'smoke terminal {term}')
            chk(os.path.exists(os.path.join(OUT_SMOKE, H.SMOKE_MEASURE)), 'smoke_measure written')
            chk(os.path.exists(os.path.join(OUT_SMOKE, H.RESULTS_JSONL)), 'smoke results written')
    finally:
        KNOBS, OUT, OUT_SMOKE, launch, measure_unit, stop_one = saved
    print('SELFTEST', 'PASS' if not fails else 'FAIL', fails)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv[1:] else main())
