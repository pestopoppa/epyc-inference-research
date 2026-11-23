#!/usr/bin/env python3
"""Offline re-analysis of CAFE-1-SERVED-T r2 DIST captures (no inference).
1) recompute per-arm token counts / rank hist, 2) selection-matched A0 (emulate p-min truncation on A0 rounds using
the MTP drafter confidence that is logged), 3) request-level block bootstrap / permutation chi2, 4) KS-PIT recheck."""
import glob, json, math, random, sys, collections
sys.path.insert(0, '/mnt/raid0/llm/tmp/ak-cafe1-served-t-cpu-20261123-r2')
import importlib.util
spec = importlib.util.spec_from_file_location('job', '/mnt/raid0/llm/tmp/ak-cafe1-served-t-cpu-20261123-r2/job.py')
CAP = '/mnt/raid0/llm/tmp/ak-cafe1-served-t-cpu-20261123-r2/out/capture'
OUT = '/mnt/raid0/llm/tmp/pmin-dist-investigation-20261123/results.json'


def chi2_sf(x, k):
    # regularized upper gamma via series/continued fraction
    a, x2 = k / 2.0, x / 2.0
    if x2 <= 0: return 1.0
    if x2 < a + 1:
        s = t = 1.0 / a; n = a
        for _ in range(1000):
            n += 1; t *= x2 / n; s += t
            if abs(t) < abs(s) * 1e-15: break
        return 1 - s * math.exp(-x2 + a * math.log(x2) - math.lgamma(a))
    b = x2 + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 1000):
        an = -i * (i - a); b += 2; d = an * d + b; d = 1 / d if d else 1e300; c = b + an / c; h *= d * c
        if abs(d * c - 1) < 1e-15: break
    return math.exp(-x2 + a * math.log(x2) - math.lgamma(a)) * h


def homog(a, b, minpool=10):
    # same as job.chi2_homog: merge tail bins to >= minpool pooled
    bins = []; ca = cb = 0
    for x, y in zip(a, b):
        ca += x; cb += y
        if ca + cb >= minpool:
            bins.append((ca, cb)); ca = cb = 0
    if ca + cb and bins:
        x, y = bins[-1]; bins[-1] = (x + ca, y + cb)
    na, nb = sum(x for x, _ in bins), sum(y for _, y in bins)
    if len(bins) < 2 or not na or not nb: return 0.0, 0, 1.0
    chi = 0.0
    for x, y in bins:
        t = x + y; ea = t * na / (na + nb); eb = t * nb / (na + nb)
        chi += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    return chi, len(bins) - 1, chi2_sf(chi, len(bins) - 1)


def ks_p(u):
    u = sorted(u); n = len(u)
    d = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(u))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return max(0.0, min(1.0, p)), d


def load(T, arm, thr=None):
    """Return list of requests; each request = list of (rank, U-ish, conf_ok) per accepted token.
    thr: if given, emulate p-min truncation: keep a row only if the drafter confidence of the draft at that row >= thr
    (row starts a draft) or the previous row's draft had confidence >= thr (bonus row after an accepted confident chain)."""
    reqs = collections.defaultdict(list)
    rng = random.Random(12345)
    for f in sorted(glob.glob(f'{CAP}/cap-dist-b*-T{T}-{arm}.jsonl')):
        for ln in open(f):
            r = json.loads(ln)
            if r.get('k') == 'task': continue
            key = (f, r['task'])
            conf = [m[0][2] if m else 0.0 for m in r.get('mtp_top') or []]
            for i, x in enumerate(r['acc']):
                cands = sorted(((int(t), float(p)) for t, p in r['rows'][i][1]), key=lambda z: -z[1])
                d = dict(cands)
                if x not in d: continue
                if thr is not None:
                    # row i exists in a p-min round iff the round exists (conf[0] >= thr) and the chain was not truncated
                    # before it (conf[0..i-1] >= thr); row n_draft is the bonus row.
                    if not conf or conf[0] < thr or any(c < thr for c in conf[:i]): continue
                px = d[x]; s = sum(p for _, p in cands if p < px)
                rk = min(20, [t for t, _ in cands].index(x))
                reqs[key].append((rk, s + rng.random() * px))
    return list(reqs.values())


def hist(reqs):
    h = [0] * 21
    for q in reqs:
        for rk, _ in q: h[rk] += 1
    return h


def perm_p(ra, rb, n=2000, seed=1):
    """Request-level permutation test of the chi2 statistic: shuffle whole requests between arms (respects within-sequence
    dependence; exchangeable under H0 'same law' given matched prompts/seeds design)."""
    obs = homog(hist(ra), hist(rb))[0]
    pool = ra + rb; na = len(ra); rng = random.Random(seed); ge = 0
    for _ in range(n):
        rng.shuffle(pool)
        if homog(hist(pool[:na]), hist(pool[na:]))[0] >= obs: ge += 1
    return obs, (ge + 1) / (n + 1)


def first_k_per_req(reqs, k):
    return [q[:k] for q in reqs]


def thin(reqs, step):
    return [q[::step] for q in reqs]


res = {}
for T in ('0.3', '1.0'):
    A0 = load(T, 'A0')
    res[T] = {}
    for arm, thr in (('P05', 0.5), ('P075', 0.75)):
        B = load(T, arm)
        A0m = load(T, 'A0', thr)
        r = {'tokens_arm': sum(map(len, B)), 'tokens_A0': sum(map(len, A0)), 'tokens_A0_matched': sum(map(len, A0m)),
             'rank0_frac_A0': hist(A0)[0] / max(1, sum(hist(A0))), 'rank0_frac_A0_matched': hist(A0m)[0] / max(1, sum(hist(A0m))),
             'rank0_frac_arm': hist(B)[0] / max(1, sum(hist(B)))}
        r['naive_chi2'] = homog(hist(A0), hist(B))
        r['naive_perm_p'] = perm_p(A0, B)[1]
        r['thin8_chi2'] = homog(hist(thin(A0, 8)), hist(thin(B, 8)))
        r['matched_chi2'] = homog(hist(A0m), hist(B))
        r['matched_perm_p'] = perm_p(A0m, B)[1]
        r['ks_arm'] = ks_p([u for q in B for _, u in q])
        r['ks_A0_matched'] = ks_p([u for q in A0m for _, u in q])
        res[T][arm] = r
        print(T, arm, json.dumps(r))
json.dump(res, open(OUT, 'w'), indent=1)
