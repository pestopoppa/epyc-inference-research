#!/usr/bin/env python3
"""[ak-asm-t1-harness-20261105: copy of rock-aiter-tests-20261105/op_runner.py + a hang watchdog: once a gfx90a code object
has been loaded (`LoadKernel ... /gfx90a/`) and no JIT build is in progress, `idle_s` seconds without any output kills the test and
reports hang=true; JIT builds keep the plain `timeout`.]
In-container runner for rock-aiter-tests-20261105 (one op run per container, one fresh test process per run).
  python3 /pk/op_runner.py '<json run spec>'          cwd=/aiter
Steps: (1) optional script patch (copy under op_tests/, exact-once substitutions, unified diff logged); (2) `--help` of the
script, logged between ==HELP== markers; (3) adapt the planned args to the help text (flag spellings/aliases, choices,
tuple form), recording every adaptation; (4) run the test in a fresh subprocess with a hard timeout, streaming its output;
(5) print `RUNNER_RESULT {json}`. Special spec kind 'moe_sort_probe' runs /pk/moe_sort_probe.py instead."""
import difflib, json, os, re, subprocess, sys, time

TAB_OK = False


def emit(tag, d): print(f'{tag} ' + json.dumps(d), flush=True)


def ensure_tabulate():
    global TAB_OK
    r = subprocess.run([sys.executable, '-c', 'import tabulate;print(tabulate.__version__)'], capture_output=True, text=True)
    if r.returncode == 0:
        TAB_OK = True; return {'tabulate': r.stdout.strip(), 'source': 'preinstalled/pydeps'}
    p = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--target', '/pydeps', 'tabulate'], capture_output=True, text=True)
    r = subprocess.run([sys.executable, '-c', 'import tabulate;print(tabulate.__version__)'], capture_output=True, text=True)
    TAB_OK = (r.returncode == 0)
    return {'tabulate': r.stdout.strip() if TAB_OK else None, 'source': 'pip --target /pydeps (fallback)', 'pip_tail': (p.stdout + p.stderr)[-300:]}


def apply_patch(spec):
    pt = spec.get('patch')
    if not pt: return spec['script'], None
    src = open(spec['script']).read(); lines = src.split('\n'); out = list(lines); notes = []
    for op in pt['ops']:
        idx = [i for i, l in enumerate(out) if re.search(op['regex'], l)]
        if len(idx) != 1: raise RuntimeError(f"patch regex {op['regex']!r} matched {len(idx)} lines (need exactly 1)")
        i = idx[0]
        if op['do'] == 'comment': out[i] = '# [patched out] ' + out[i]
        elif op['do'] == 'insert_before': out[i:i] = [op['text']]
        else: raise RuntimeError('bad patch op')
        notes.append({'do': op['do'], 'line': i + 1})
    dst = pt['dst']
    open(dst, 'w').write('\n'.join(out))
    diff = ''.join(difflib.unified_diff([l + '\n' for l in lines], [l + '\n' for l in out], spec['script'], dst, n=1))
    print('==PATCH-DIFF==\n' + diff + '==END-PATCH-DIFF==', flush=True)
    return dst, {'dst': dst, 'ops': notes, 'diff': diff[:3000]}


def flag_present(help_text, flag):
    return re.search(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])', help_text) is not None


def flag_block(help_text, flag):
    """help text from the flag's line to the next option line."""
    ls = help_text.split('\n'); out = []; on = False
    for l in ls:
        if re.match(r'\s+-', l) and flag_present(l, flag): on = True; out.append(l); continue
        if on and re.match(r'\s+-', l): break
        if on: out.append(l)
    return '\n'.join(out)


def adapt(groups, help_text):
    """groups: [{'flags': ['-d','--dtype'], 'values': [...], 'tuple': 'comma'|None}] -> (argv, notes)."""
    argv = []; notes = []
    for g in groups:
        fl = next((f for f in g['flags'] if flag_present(help_text, f)), None)
        if fl is None:
            notes.append({'dropped': g['flags'], 'reason': 'flag not in --help'}); continue
        if fl != g['flags'][0]: notes.append({'flag_alias': [g['flags'][0], fl]})
        vals = list(g.get('values', []))
        blk = flag_block(help_text, fl)
        m = re.search(r'\{([^{}]+)\}', blk)
        if m and vals and not g.get('tuple'):
            ch = {c.strip() for c in m.group(1).replace(',', ' ').split()}
            keep = [v for v in vals if str(v) in ch or str(v).lower() in {c.lower() for c in ch}]
            if not keep:   # choices are not plain words (e.g. torch dtypes): do not guess, keep the planned values
                notes.append({'flag': fl, 'choices_unmatched_kept_planned': sorted(ch)[:12]})
            else:
                if keep != vals: notes.append({'flag': fl, 'values_dropped_not_in_choices': [v for v in vals if v not in keep], 'choices': sorted(ch)})
                vals = keep
        if g.get('tuple') == 'comma' and re.search(re.escape(fl) + r'\s+\d+\s+\d+\s+\d+(?!\s*,)', blk) and not re.search(re.escape(fl) + r'\s+\d+,\d+', blk):
            # help shows space-separated triples: flatten "a,b,c" -> a b c
            vals = [x for v in vals for x in str(v).split(',')]; notes.append({'flag': fl, 'tuple_form': 'space (from help example)'})
        argv.append(fl); argv += [str(v) for v in vals]
    return argv, notes


def main():
    spec = json.loads(sys.argv[1]); t0 = time.time()
    res = {'run': spec['run'], 'op': spec['op'], 'kind': spec.get('kind', 'test')}
    res['tabulate'] = ensure_tabulate()
    print('TABULATE', json.dumps(res['tabulate']), flush=True)
    if spec.get('kind') == 'probe':
        cmd = [sys.executable, spec['script']]
        r = subprocess.run(cmd, cwd='/aiter', timeout=spec.get('timeout', 600)); res.update(rc=r.returncode, wall_s=round(time.time() - t0, 1))
        emit('RUNNER_RESULT', res); return 0
    try:
        script, pinfo = apply_patch(spec); res['patch'] = pinfo
        h = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True, timeout=900)
        help_text = h.stdout + h.stderr
        print('==HELP== ' + script + f' rc={h.returncode}\n' + help_text + '\n==ENDHELP==', flush=True)
        res['help_rc'] = h.returncode
        argv, notes = adapt(spec['groups'], help_text)
        res['planned'] = spec['groups']; res['argv'] = argv; res['adaptations'] = notes
        cmd = [sys.executable, script] + argv + list(spec.get('extra_argv', []))
        print('==RUN== ' + ' '.join(cmd), flush=True)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        t_end = time.time() + spec['timeout']; jit = []; timed_out = False; hang = False
        idle_s = spec.get('idle_s', 600); st = {'last': time.time(), 'loaded90a': False, 'building': 0}
        import threading
        def killer():
            nonlocal timed_out, hang
            while p.poll() is None:
                if time.time() > t_end: timed_out = True; p.kill(); return
                if st['loaded90a'] and st['building'] <= 0 and time.time() - st['last'] > idle_s:
                    hang = True; print(f'RUNNER_HANG no output for {idle_s} s after a gfx90a LoadKernel', flush=True); p.kill(); return
                time.sleep(2)
        threading.Thread(target=killer, daemon=True).start()
        for l in p.stdout:
            sys.stdout.write(l); sys.stdout.flush(); st['last'] = time.time()
            if 'start build [' in l: jit.append(l.strip()[:200]); st['building'] += 1
            if 'finish build [' in l: st['building'] -= 1
            if 'LoadKernel' in l and '/gfx90a/' in l: st['loaded90a'] = True
        rc = p.wait()
        res.update(rc=rc, timed_out=timed_out, hang=hang, jit_during_test=jit[:10], wall_s=round(time.time() - t0, 1))
    except BaseException as e:  # noqa: BLE001
        res.update(rc=None, runner_error=f'{type(e).__name__}: {e}'[:400], wall_s=round(time.time() - t0, 1))
    emit('RUNNER_RESULT', res); return 0


if __name__ == '__main__':
    sys.exit(main())
