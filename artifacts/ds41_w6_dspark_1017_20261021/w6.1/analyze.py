import json, glob, os, random, sys
# usage: analyze.py <runs_dir> <out_json>
runs, outp = sys.argv[1], sys.argv[2]
random.seed(20261021)
def load(label):
    f = os.path.join(runs, label, "rows.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))
pairs = {}
for d in sorted(glob.glob(os.path.join(runs, "p[0-9]*A"))):
    p = os.path.basename(d)[1:-1]
    a, b = load(f"p{p}A"), load(f"p{p}B")
    if not a or not b:
        continue
    def stoch(rows, n):
        r = [x for x in rows if x["mode"] == "stoch" and x["n_max"] == n]
        return {x["prompt"]: (x["draft_n_accepted"], x["draft_n"]) for x in r}
    pairs[p] = {n: (stoch(a, n), stoch(b, n)) for n in (2, 3) if len(stoch(a, n)) == 16 and len(stoch(b, n)) == 16}
res = {"n_pairs": len(pairs), "pairs": sorted(pairs), "blocks": {}}
for n in (2, 3):
    # unit = (pair, prompt) cell; per-cell A/B accepted & drafted
    cells = []
    pb = {p: v for p, v in pairs.items() if n in v}
    for p, v in pb.items():
        A, B = v[n]
        for i in range(16):
            cells.append((p, i, A[i], B[i]))
    def alpha(cs, arm):
        acc = sum(c[2 + arm][0] for c in cs); dr = sum(c[2 + arm][1] for c in cs)
        return acc / dr if dr else float("nan"), acc, dr
    aA, accA, drA = alpha(cells, 0); aB, accB, drB = alpha(cells, 1)
    # per-pair alpha (pooled over prompts) diffs
    perpair = {}
    for p in pb:
        cs = [c for c in cells if c[0] == p]
        perpair[p] = (alpha(cs, 0)[0], alpha(cs, 1)[0])
    # paired bootstrap: resample prompts (cluster) with replacement; within each draw, use all pairs of that prompt
    prompts = list(range(16))
    byp = {i: [c for c in cells if c[1] == i] for i in prompts}
    def diff_prompts(ps):
        cs = [c for i in ps for c in byp[i]]
        return alpha(cs, 0)[0] - alpha(cs, 1)[0]
    boots_prompt = []
    for _ in range(10000):
        boots_prompt.append(diff_prompts([random.choice(prompts) for _ in prompts]))
    boots_prompt.sort()
    # two-way: resample pairs and prompts
    plist = list(pb)
    bypair = {p: [c for c in cells if c[0] == p] for p in plist}
    boots_two = []
    for _ in range(10000):
        ps = [random.choice(plist) for _ in plist]
        qs = [random.choice(prompts) for _ in prompts]
        cs = [c for p in ps for c in bypair[p] if c[1] in set(qs) for _ in range(qs.count(c[1]))]
        boots_two.append(alpha(cs, 0)[0] - alpha(cs, 1)[0])
    boots_two.sort()
    q = lambda b, x: b[int(x * (len(b) - 1))]
    # wins/losses/ties on per-cell alpha
    w = l = t = 0
    for c in cells:
        x = c[2][0] / max(1, c[2][1]); y = c[3][0] / max(1, c[3][1])
        if abs(x - y) < 1e-12: t += 1
        elif x > y: w += 1
        else: l += 1
    res["blocks"][n] = {"n_pairs": len(pb),
        "alpha_A_1017": aA, "A_acc": accA, "A_draft": drA, "alpha_B_sept": aB, "B_acc": accB, "B_draft": drB,
        "diff_A_minus_B": aA - aB,
        "ci95_prompt_cluster_bootstrap": [q(boots_prompt, 0.025), q(boots_prompt, 0.975)],
        "ci95_pair_x_prompt_bootstrap": [q(boots_two, 0.025), q(boots_two, 0.975)],
        "cells_A_win_loss_tie": [w, l, t],
        "per_pair_alpha_A_B": perpair,
    }
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res, indent=1))
