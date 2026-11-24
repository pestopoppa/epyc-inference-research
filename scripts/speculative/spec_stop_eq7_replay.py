#!/usr/bin/env python3
"""SPEC-STOP-EQ7-1: offline replay of the cost-aware draft-stop rule (intake-2876 Eq 7)
with the marginal expert-union verify cost (intake-2896) on the CAFE-1-SERVED-T A3B MTP
CPU captures, against fixed p_min {0, 0.5, 0.75}. No model, no server.

Step 1  C(K) = a + b*K + c*E_union(K) by least squares on the CAFE-2 r2 native verify
        timings (24 threads, widths 1..9 rows = K+1), E_union = mean per-layer routed-expert
        union from the same row's census. Futility if R^2 < 0.9.
Calib   served-context scale s and per-drafted-token MTP cost d from the CAFE-1-SERVED-T
        time-phase requests: predicted_ms = s * R * C(Kbar) + d * draft_n  (NNLS, 2 params).
Step 2  On the A0 (D=4 full-chain) dist-phase captures, each round gives draft probs p1..p4
        and the accepted length a. A policy choosing K yields min(a,K)+1 tokens for
        d*K + s*C(K) ms. Fixed p_min: K = leading run of p_i >= p_min. Eq 7: K* = argmin_K
        (d*K + s*C(K)) / (1 + sum_{i<=K} prod_{j<=i} p_j). Tasks split 50/50 (seeded hash);
        on the held-out half, task-level bootstrap of the tok/s ratio Eq7 / fixed.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

KMAX = 4

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def fit_cost(cafe2_dir):
    res = json.load(open(os.path.join(cafe2_dir, 'RESULT.json')))
    eu = {int(k): v for k, v in res['stats']['mean_layer_union'].items()}
    cells = []
    for l in open(os.path.join(cafe2_dir, 'results.jsonl')):
        r = json.loads(l)
        if r.get('mode') != 'full' or r.get('threads') != 24 or 'rows' not in r:
            continue
        if r['unit_id'].startswith('timing-i0'):  # warm-up invocation
            continue
        cells.append((r['rows'], r['ns'] / 1e6))
    w = np.array([c[0] for c in cells]); y = np.array([c[1] for c in cells])
    K = w - 1
    X = np.c_[np.ones_like(K, dtype=float), K, [eu[int(x)] for x in w]]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2_cells = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ws = sorted(set(w.tolist()))
    med = np.array([np.median(y[w == x]) for x in ws])
    Xm = np.c_[np.ones(len(ws)), np.array(ws) - 1, [eu[x] for x in ws]]
    coef_m, *_ = np.linalg.lstsq(Xm, med, rcond=None)
    pm = Xm @ coef_m
    r2_med = 1 - ((med - pm) ** 2).sum() / ((med - med.mean()) ** 2).sum()
    return dict(a=coef_m[0], b=coef_m[1], c=coef_m[2], r2_medians=r2_med,
                cells_coef=coef.tolist(), r2_cells=r2_cells, n_cells=len(cells),
                widths=ws, median_ms=med.tolist(), e_union={str(k): eu[k] for k in ws})

def C(fit, K):
    eu = fit['_eu']
    return fit['a'] + fit['b'] * K + fit['c'] * eu(K)

def calibrate(fit, cafe1_out):
    rows, ys = [], []
    for l in open(os.path.join(cafe1_out, 'results.jsonl')):
        r = json.loads(l)
        if r.get('phase') != 'time' or r.get('status', 'ok') not in ('ok', None, 'pass'):
            pass
        if r.get('phase') != 'time':
            continue
        for q in r['requests']:
            R = q['predicted_n'] - q['draft_n_accepted']
            if R <= 0:
                continue
            kbar = q['draft_n'] / R
            rows.append([R * C(fit, kbar), q['draft_n']]); ys.append(q['predicted_ms'])
    X = np.array(rows); y = np.array(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = np.maximum(coef, 0)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return dict(s=float(coef[0]), d_ms=float(coef[1]), r2=float(r2), n_requests=len(y))

def load_rounds(cap_dir, T):
    tasks = {}
    files = sorted(glob.glob(os.path.join(cap_dir, f'cap-dist-b*-T{T}-A0.jsonl')))
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r['k'] != 'round' or r['n_draft'] != KMAX:
                continue
            p = []
            for i, t in enumerate(r['chain']):
                top = {e[0]: e[2] for e in r['mtp_top'][i]}
                p.append(top.get(t, 0.0))
            tasks.setdefault(f'{os.path.basename(f)}:{r["task"]}', []).append((p, len(r['acc']) - 1))
    return files, tasks

def policy_K(p, rule, fit, cal):
    if rule == 'eq7':
        best, bk, cum, e = None, 0, 1.0, 1.0
        for K in range(0, KMAX + 1):
            if K > 0:
                cum *= p[K - 1]; e += cum
            v = (cal['d_ms'] * K + cal['s'] * C(fit, K)) / e
            if best is None or v < best:
                best, bk = v, K
        return bk
    K = 0
    while K < KMAX and p[K] >= rule:
        K += 1
    return K

def task_stats(rounds, rule, fit, cal):
    tok = ms = 0.0
    for p, a in rounds:
        K = policy_K(p, rule, fit, cal)
        tok += min(a, K) + 1
        ms += cal['d_ms'] * K + cal['s'] * C(fit, K)
    return tok, ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cafe1-out', default='/mnt/raid0/llm/tmp/ak-cafe1-served-t-cpu-20261123-r2/out')
    ap.add_argument('--cafe2-out', default='/mnt/raid0/llm/tmp/inf50-runs-20261107/cafe2-verify-cost-native-r2/out')
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20261124)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fit = fit_cost(a.cafe2_out)
    eu_k = np.array([fit['e_union'][str(w)] for w in fit['widths']]); ks = np.array(fit['widths']) - 1
    fit['_eu'] = lambda K: float(np.interp(K, ks, eu_k))
    cal = calibrate(fit, a.cafe1_out)
    out = dict(schema='spec-stop-eq7-1.v1',
               step1=dict(model='C(K)=a+b*K+c*E_union(K), ms, 24 threads, CAFE-2 r2',
                          a_ms=fit['a'], b_ms=fit['b'], c_ms_per_expert=fit['c'],
                          r2_medians=fit['r2_medians'], r2_cells=fit['r2_cells'], n_cells=fit['n_cells'],
                          cells_coef=fit['cells_coef'], median_ms=fit['median_ms'], e_union=fit['e_union'],
                          futility=bool(fit['r2_medians'] < 0.9)),
               calibration=cal, temps={}, inputs={})
    for p in [os.path.join(a.cafe2_out, 'RESULT.json'), os.path.join(a.cafe2_out, 'results.jsonl'),
              os.path.join(a.cafe1_out, 'results.jsonl')]:
        out['inputs'][p] = sha(p)
    rng = np.random.default_rng(a.seed)
    rules = ['eq7', 0.0, 0.5, 0.75]
    for T in ['0.3', '1.0']:
        files, tasks = load_rounds(os.path.join(a.cafe1_out, 'capture'), T)
        for f in files:
            out['inputs'][f] = sha(f)
        names = sorted(tasks)
        held = [n for n in names if int(hashlib.sha256(f'{a.seed}:{n}'.encode()).hexdigest(), 16) % 2 == 1]
        st = {r: np.array([task_stats(tasks[n], r, fit, cal) for n in held]) for r in rules}
        kdist = np.bincount([policy_K(p, 'eq7', fit, cal) for n in held for p, _ in tasks[n]], minlength=KMAX + 1)
        tps = {str(r): float(st[r][:, 0].sum() / st[r][:, 1].sum() * 1000) for r in rules}
        idx = rng.integers(0, len(held), size=(a.boot, len(held)))
        cmp = {}
        for r in rules[1:]:
            e7 = st['eq7'][idx].sum(1); fx = st[r][idx].sum(1)
            ratio = (e7[:, 0] / e7[:, 1]) / (fx[:, 0] / fx[:, 1])
            point = tps['eq7'] / tps[str(r)]
            cmp[f'p_min={r}'] = dict(ratio=point, gain_pct=(point - 1) * 100,
                                     ci95=[float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))])
        out['temps'][T] = dict(n_tasks=len(names), n_heldout=len(held),
                               n_rounds_heldout=int(sum(len(tasks[n]) for n in held)),
                               modelled_tok_s=tps, eq7_K_hist=kdist.tolist(), vs_fixed=cmp,
                               fixed_model_check_P05_over_A0=tps['0.5'] / tps['0.0'],
                               fixed_model_check_P075_over_A0=tps['0.75'] / tps['0.0'])
    # verdict per box: KEEP if gain>=3% & CI lower>1.0 at both T (vs every fixed p_min);
    # CLOSE if gain<1% & CI upper<1.02 at both T (vs best fixed); else INCONCLUSIVE.
    best = {T: max(out['temps'][T]['vs_fixed'].values(), key=lambda v: -v['ratio']) for T in out['temps']}
    keep = all(v['gain_pct'] >= 3 and v['ci95'][0] > 1.0 for T in out['temps'] for v in out['temps'][T]['vs_fixed'].values())
    close = all(best[T]['gain_pct'] < 1 and best[T]['ci95'][1] < 1.02 for T in best)
    if out['step1']['futility']:
        verdict = 'FUTILITY (step-1 R^2 < 0.9)'
    else:
        verdict = 'KEEP candidate' if keep else ('CLOSE' if close else 'INCONCLUSIVE')
    out['verdict'] = verdict
    out['verdict_basis'] = 'vs the strongest fixed p_min per T (min ratio)'
    out['limits'] = ['C(K) shape from one frozen 7-token prompt (CAFE-2 r2), scaled to served context by s fitted on CAFE-1 time requests',
                     'acceptance replay assumes the A0 greedy MTP chain prefix is the chain any shorter K would draft (true for greedy chain drafting)',
                     'modelled, not measured, tok/s; per-round draft cost d is a fitted constant']
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'RESULT.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('step1', 'calibration', 'verdict')}, indent=1, default=float))
    for T, v in out['temps'].items():
        print(T, json.dumps(v, default=float))

if __name__ == '__main__':
    main()
