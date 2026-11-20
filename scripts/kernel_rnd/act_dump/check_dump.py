#!/usr/bin/env python3
"""Validate an act_dump output dir (stdlib only). Exit 0 ok, 1 on any defect; prints one JSON summary line.

Checks: manifest present; every listed .bin exists with bytes == nelements * type_size / block_size; f32/f16 stats have
no non-finite values; activations feed their weights (act_<p>.ne0 == w_<p>.ne0, act ne1 == n_tokens); ffn_up and ffn_gate
see the identical input; FA q/k/v share head dim and k/v share kv length (>= n_tokens) when an FA node was dumped.
"""
import json, os, sys

TS = {'i32': (4, 1), 'f32': (4, 1), 'f16': (2, 1), 'bf16': (2, 1), 'q8_0': (34, 32), 'q4_0': (18, 32), 'q6_K': (210, 256), 'q4_K': (144, 256)}


def check(d, require_fa=True):
    errs = []
    try:
        m = json.load(open(os.path.join(d, 'manifest.json')))
    except (OSError, ValueError) as e:
        return {'ok': False, 'errors': [f'manifest: {e}']}
    by = {t['key']: t for t in m.get('tensors', [])}
    n_tok = m.get('n_tokens')
    for k, t in by.items():
        p = os.path.join(d, t['file'])
        ne = t['ne']; n = ne[0] * ne[1] * ne[2] * ne[3]
        if t['type'] not in TS:
            errs.append(f'{k}: unknown type {t["type"]}'); continue
        sz, bs = TS[t['type']]
        want = n // bs * sz
        if not os.path.exists(p) or os.path.getsize(p) != want or t['bytes'] != want:
            errs.append(f'{k}: size {os.path.getsize(p) if os.path.exists(p) else None} != {want}')
        s = t.get('stats')
        if s and s.get('nonfinite'):
            errs.append(f'{k}: {s["nonfinite"]} non-finite values')
    if m.get('head_mode'):  # --head dump: hidden [d,N], dequantized weight [d,V], top-K ids/vals [K,N]
        h, w, i, v = (by.get(x) for x in ('head_hidden', 'head_w.f32', 'head_topk_ids', 'head_topk_val'))
        if not (h and w and i and v):
            errs.append('head mode: missing head_hidden / head_w.f32 / head_topk_ids / head_topk_val')
        else:
            if h['ne'][1] != n_tok or i['ne'][1] != n_tok or v['ne'] != i['ne']:
                errs.append('head mode: row count != n_tokens or ids/vals shape differ')
            if h['ne'][0] != w['ne'][0]:
                errs.append('head mode: hidden width != head weight ne0')
            if m.get('vocab') and w['ne'][1] != m['vocab']:
                errs.append(f'head mode: head rows {w["ne"][1]} != vocab {m["vocab"]}')
        return {'ok': not errs, 'errors': errs, 'n_tensors': len(by), 'n_tokens': n_tok, 'head_mode': True,
                'shapes': {k: by[k]['ne'] for k in sorted(by)}}
    acts = [k for k in by if k.startswith('act_')]
    if not acts:
        errs.append('no act_* tensors')
    for k in acts:
        a = by[k]; w = by.get('w_' + k[4:])
        if a['ne'][1] != n_tok:
            errs.append(f'{k}: ne1 {a["ne"][1]} != n_tokens {n_tok}')
        if w and w['ne'][0] != a['ne'][0]:
            errs.append(f'{k}: ne0 {a["ne"][0]} != weight ne0 {w["ne"][0]}')
    if 'act_ffn_up' in by and 'act_ffn_gate' in by:
        if open(os.path.join(d, by['act_ffn_up']['file']), 'rb').read() != open(os.path.join(d, by['act_ffn_gate']['file']), 'rb').read():
            errs.append('act_ffn_up and act_ffn_gate inputs differ')
    fa = [by.get(x) for x in ('fa_q', 'fa_k', 'fa_v')]
    if all(fa):
        q, k, v = fa
        if q['ne'][0] != k['ne'][0]:
            errs.append('fa: q/k head dim differ')
        if k['ne'][1] != v['ne'][1] or k['ne'][1] < n_tok or q['ne'][1] != n_tok:
            errs.append('fa: kv length / q length mismatch')
        if q['ne'][2] % k['ne'][2]:
            errs.append('fa: q heads not a multiple of kv heads')
    elif require_fa:
        errs.append('no FA q/k/v dumped')
    return {'ok': not errs, 'errors': errs, 'n_tensors': len(by), 'n_tokens': n_tok, 'layer': m.get('layer'),
            'shapes': {k: by[k]['ne'] for k in sorted(by)}}


if __name__ == '__main__':
    r = check(sys.argv[1], require_fa='--no-fa' not in sys.argv)
    print(json.dumps(r, sort_keys=True))
    sys.exit(0 if r['ok'] else 1)
