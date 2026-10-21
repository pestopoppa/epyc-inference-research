#!/usr/bin/env python3
"""Post-run analysis for the DS41 W5a / W5b / W6.2 slot (2026-10-22T22:00Z -> 2026-10-23T00:30Z).

Run by the main session after /mnt/raid0/llm/tmp/compute-flags/DS41_W62_ENDED exists:

  /mnt/raid0/llm/epyc-inference-research/.venv/bin/python -I \
      /mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/analyze_post.py --agent <roster-id> [--commit] [--push]

What it does:
  1. creates a research worktree from origin/main (/mnt/raid0/llm/worktrees/research-ds41-w62-post-<ts>);
  2. copies the slot outputs in: W5a kernel NMSE rows, W5b probe rows, W6.2 run rows + client logs, timeline;
  3. W6.2: pools the complete pairs (old p1-p2 from w6.1/runs + new p3.. from w6.2/runs), computes per-block
     alpha, A-B, a paired t CI over per-pair diffs and the W6.1 pair x prompt / prompt-cluster bootstraps;
     greedy text identity of p3A vs p3B (same config);
  4. writes artifacts/ds41_w6_dspark_1017_20261021/w6.2/{analysis.json,SUMMARY.md};
  5. --commit: `git add` of exactly those paths, `git commit -F <msgfile>` alone; --push: serialized_push.py.
Without --commit it only writes into the worktree and prints the summary.
"""
import argparse, datetime as dt, glob, json, math, os, random, shutil, subprocess, sys

W = "/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021"
RESEARCH = "/mnt/raid0/llm/epyc-inference-research"
TL = "/mnt/raid0/llm/tmp/compute-flags/DS41_W62_TIMELINE.txt"
ENDF = "/mnt/raid0/llm/tmp/compute-flags/DS41_W62_ENDED"
A6 = "artifacts/ds41_w6_dspark_1017_20261021"
D5 = "data/ds41-1017-w5-20261021"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def sh(*a, cwd=None, check=True):
    r = subprocess.run(list(a), cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {' '.join(a)}\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def load_rows(d):
    f = os.path.join(d, "rows.json")
    return json.load(open(f)) if os.path.exists(f) else None


def stoch(rows, n):
    return {x["prompt"]: (x["draft_n_accepted"], x["draft_n"]) for x in rows if x["mode"] == "stoch" and x["n_max"] == n}


def w62(old_runs, new_runs):
    random.seed(20261021)
    pairs, config = {}, {}
    for root, cfg in ((old_runs, "p1-p2: A and B concurrent, -t 24 each on 0-23/24-47 (W6.1)"),
                      (new_runs, "p3+: sequential, one server, -t 48 on 0-47 (W6.2)")):
        for d in sorted(glob.glob(os.path.join(root, "p[0-9]*A"))):
            p = os.path.basename(d)[1:-1]
            a, b = load_rows(d), load_rows(os.path.join(root, f"p{p}B"))
            if not a or not b:
                continue
            blocks = {n: (stoch(a, n), stoch(b, n)) for n in (2, 3) if len(stoch(a, n)) == 16 and len(stoch(b, n)) == 16}
            if blocks:
                pairs[p] = blocks
                config[p] = cfg
    res = {"pairs": sorted(pairs, key=int), "pair_config": config, "blocks": {}}
    for n in (2, 3):
        pb = {p: v[n] for p, v in pairs.items() if n in v}
        cells = [(p, i, A[i], B[i]) for p, (A, B) in pb.items() for i in range(16)]
        if not cells:
            res["blocks"][str(n)] = {"n_pairs": 0}
            continue

        def alpha(cs, arm):
            acc = sum(c[2 + arm][0] for c in cs); dr = sum(c[2 + arm][1] for c in cs)
            return (acc / dr if dr else float("nan")), acc, dr
        aA, accA, drA = alpha(cells, 0); aB, accB, drB = alpha(cells, 1)
        perpair = {p: (alpha([c for c in cells if c[0] == p], 0)[0], alpha([c for c in cells if c[0] == p], 1)[0]) for p in pb}
        diffs = [a - b for a, b in perpair.values()]
        k = len(diffs)
        mean = sum(diffs) / k
        if k >= 2:
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (k - 1))
            half = T975.get(k - 1, 1.96) * sd / math.sqrt(k)
            tci = [mean - half, mean + half]
        else:
            sd, tci = None, None
        prompts = list(range(16))
        byp = {i: [c for c in cells if c[1] == i] for i in prompts}
        boots_prompt = []
        for _ in range(10000):
            qs = [random.choice(prompts) for _ in prompts]
            cs = [c for i in qs for c in byp[i]]
            boots_prompt.append(alpha(cs, 0)[0] - alpha(cs, 1)[0])
        boots_prompt.sort()
        plist = list(pb)
        bypair = {p: [c for c in cells if c[0] == p] for p in plist}
        boots_two = []
        for _ in range(10000):
            ps = [random.choice(plist) for _ in plist]
            qs = [random.choice(prompts) for _ in prompts]
            cnt = {q: qs.count(q) for q in set(qs)}
            cs = [c for p in ps for c in bypair[p] if c[1] in cnt for _ in range(cnt[c[1]])]
            boots_two.append(alpha(cs, 0)[0] - alpha(cs, 1)[0])
        boots_two.sort()
        q = lambda b, x: b[int(x * (len(b) - 1))]
        w = l = t = 0
        for c in cells:
            x = c[2][0] / max(1, c[2][1]); y = c[3][0] / max(1, c[3][1])
            if abs(x - y) < 1e-12: t += 1
            elif x > y: w += 1
            else: l += 1
        res["blocks"][str(n)] = {
            "n_pairs": k, "pairs": sorted(pb, key=int),
            "alpha_A_1017": aA, "A_acc": accA, "A_draft": drA, "alpha_B_sept": aB, "B_acc": accB, "B_draft": drB,
            "diff_A_minus_B_pooled": aA - aB,
            "per_pair_alpha_A_B": perpair, "per_pair_diff": dict(zip(perpair, diffs)),
            "mean_per_pair_diff": mean, "sd_per_pair_diff": sd, "ci95_paired_t_over_pairs": tci,
            "ci95_prompt_cluster_bootstrap": [q(boots_prompt, 0.025), q(boots_prompt, 0.975)],
            "ci95_pair_x_prompt_bootstrap": [q(boots_two, 0.025), q(boots_two, 0.975)],
            "cells_A_win_loss_tie": [w, l, t]}
    return res


def greedy(new_runs, pair="3"):
    a, b = load_rows(os.path.join(new_runs, f"p{pair}A")), load_rows(os.path.join(new_runs, f"p{pair}B"))
    if not a or not b:
        return {"pair": pair, "status": "missing"}
    ga = {x["prompt"]: x for x in a if x["mode"] == "greedy"}
    gb = {x["prompt"]: x for x in b if x["mode"] == "greedy"}
    both = sorted(set(ga) & set(gb))
    rows = []
    for i in both:
        ta, tb = ga[i]["text"], gb[i]["text"]
        fd = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if fd is None and len(ta) != len(tb):
            fd = min(len(ta), len(tb))
        rows.append({"prompt": i, "identical": ga[i]["text_sha"] == gb[i]["text_sha"], "first_diff_char": fd,
                     "alpha_A": ga[i]["draft_n_accepted"] / max(1, ga[i]["draft_n"]),
                     "alpha_B": gb[i]["draft_n_accepted"] / max(1, gb[i]["draft_n"])})
    return {"pair": pair, "config": "same server config both arms (-t 48, 0-47, mmap)", "n_A": len(ga), "n_B": len(gb),
            "n_compared": len(both), "n_identical": sum(r["identical"] for r in rows), "complete": len(both) == 16,
            "rows": rows}


def w5b(rows_f):
    if not os.path.exists(rows_f):
        return {"status": "missing"}
    rows = [json.loads(l) for l in open(rows_f)]
    out = []
    for r in rows:
        lp = r.get("first_eos_logprob")
        out.append({"id": r["id"], "kind": r["kind"], "prompt_tokens": r["prompt_tokens"], "first_is_eos": r["first_is_eos"],
                    "p_eos_first": (math.exp(lp) if lp is not None and lp <= 0 else lp),
                    "top3": (r.get("first_top10") or [])[:3], "text_head": r["text"][:100]})
    return {"status": "ok", "n": len(out), "rows": out}


def w5a_kernel(f):
    if not os.path.exists(f):
        return {"status": "missing"}
    rows = [json.loads(l) for l in open(f)]
    return {"status": "ok", "n": len(rows), "all_repacked": all(r["repacked"] for r in rows),
            "all_pass": all(r["pass"] for r in rows),
            "max_nmse_repack_vs_f32ref": max(r["nmse_repack_vs_f32ref"] for r in rows),
            "max_nmse_plain_vs_f32ref": max(r["nmse_plain_vs_f32ref"] for r in rows),
            "max_nmse_repack_vs_plain": max(r["nmse_repack_vs_plain"] for r in rows), "rows": rows}


BIN = "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin"
TGT = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf"
DRAFT = {"A": "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf",
         "B": "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf"}
ENV = {"LD_LIBRARY_PATH": BIN, "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "OMP_WAIT_POLICY": "active",
       "OMP_DYNAMIC": "false", "GGML_IQK": "1"}
REGION = {"role": "build", "cpu_list": "0-47", "quarter": "q0+q1", "tag": "ds41-w62-slot"}


def capture_rows(wt, an):
    """Write side (VB-DS41-1017-WIRE): one producer row per gate number, via the research producer
    ds41_1017_capture.py imported FROM THE WORKTREE (so scored artifacts resolve inside it). Only numbers
    measured inside the slot's build-role claim are captured; nothing is re-derived from prose."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ds41_1017_capture", os.path.join(wt, "scripts/benchmark/ds41_1017_capture.py"))
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    d5 = os.path.join(wt, D5); d62 = os.path.join(wt, A6, "w6.2")
    out = []
    k = an["w5a_kernel"]
    if k["status"] == "ok" and k["all_repacked"]:
        argv = ["taskset", "-c", "0-47", f"{W}/w5a/kernel/mxfp4_repack_nmse", "48", "w5a_kernel.jsonl", "<4 cases, see run_w5a_kernel.sh>"]
        for metric, key in (("mxfp4_cpu_repack_mul_mat_id_nmse_vs_f32ref_max", "max_nmse_repack_vs_f32ref"),
                            ("mxfp4_cpu_repack_vs_generic_mul_mat_id_nmse_max", "max_nmse_repack_vs_plain")):
            out.append(C.capture(d5, gate="nmse", arm="1017", metric=metric, value=k[key], unit="nmse",
                                 metric_direction="lower_better", reps=k["n"], reps_basis="4 tensors x 9 widths, 8 real experts, n_used 6",
                                 scored=os.path.join(d5, "w5a", "w5a_kernel.jsonl"), recipe={"argv": argv, "env": ENV},
                                 region=REGION, binary=f"{BIN}/llama-server",
                                 claim="CPU_REPACK mxfp4_8x8_q8_0 MUL_MAT_ID vs f32-dequant reference, cap 5e-4 (DS41-W5a)",
                                 extra={"all_pass": k["all_pass"], "cap": 5e-4, "harness": "data/ds41-1017-w5-20261021/w5a/mxfp4_repack_nmse.cpp"}))
    wf = os.path.join(d5, "w5a", "weight_fidelity_slot.json")
    if os.path.exists(wf):
        rows = json.load(open(wf))["rows"]
        for gtype in ("MXFP4", "Q8_0", "F16"):
            rs = [r for r in rows if r["gguf_type"] == gtype]
            if rs:
                out.append(C.capture(d5, gate="nmse", arm="1017", metric=f"weight_nmse_vs_hf_source_max_{gtype.lower()}",
                                     value=max(r["nmse"] for r in rs), unit="nmse", metric_direction="lower_better",
                                     reps=len(rs), reps_basis="tensors compared", scored=wf,
                                     recipe={"argv": ["python", "-I", "weight_fidelity.py", "index.json", "gguf_tensors.json"], "env": {"OMP_NUM_THREADS": "1"}},
                                     region=REGION, binary=f"{BIN}/llama-server",
                                     claim=f"GGUF {gtype} dequant vs HF 1017 source tensors (rev 1bae0e10), max over checked tensors",
                                     extra={"tensors": [r["tensor"] for r in rs]}))
    b = an["w5b"]
    if b["status"] == "ok":
        raw = [r for r in b["rows"] if r["kind"] == "raw"]
        out.append(C.capture(d5, gate="coherence", arm="1017", metric="w5b_raw_probe_first_token_eos_count",
                             value=sum(r["first_is_eos"] for r in raw), unit="count", metric_direction="lower_better",
                             reps=len(raw), reps_basis="raw-completion probes", scored=os.path.join(d5, "w5b", "run", "w5b.rows.jsonl"),
                             recipe={"argv": json.load(open(os.path.join(d5, "w5b", "run", "w5b.recipe.json")))["argv"], "env": ENV},
                             region=REGION, binary=f"{BIN}/llama-server",
                             claim="DS41-W5b first-token EOS probe set (greedy, n_probs=10), target-only 1017 server",
                             extra={"per_probe": {r["id"]: [r["first_is_eos"], r["p_eos_first"]] for r in b["rows"]}}))
    scored62 = os.path.join(d62, "analysis.json")
    dig = {}
    if any(blk.get("n_pairs") for blk in an["w62"]["blocks"].values()):
        for arm, path in DRAFT.items():
            dig[arm] = {"sha256": C.sha256_file(path), "path": path, "bytes": os.path.getsize(path)}
    for n, blk in an["w62"]["blocks"].items():
        if not blk.get("n_pairs"):
            continue
        for arm, key, label in (("A", "alpha_A_1017", "1017_drafter"), ("B", "alpha_B_sept", "sept_drafter")):
            argv = ["taskset", "-c", "0-47", f"{BIN}/llama-server", "--no-webui", "-t", "48", "-tb", "48", "-np", "1", "-c", "4096",
                    "-m", TGT, "-fa", "on", "-md", DRAFT[arm], "--spec-type", "draft-dspark", "--spec-draft-n-max", "3"]
            out.append(C.capture(d62, gate="alpha", arm="1017", metric=f"dspark_alpha_block{n}_{label}_temp0.7",
                                 value=blk[key], unit="fraction", metric_direction="higher_better", reps=blk["n_pairs"],
                                 reps_basis=f"complete pairs {blk['pairs']} x 16 prompts (pooled accepted/drafted)",
                                 scored=scored62, recipe={"argv": argv, "env": ENV}, region=REGION, binary=f"{BIN}/llama-server",
                                 drafter=dig[arm],
                                 claim=f"DSpark alpha at block {n}, 1017 target, temperature 0.7 top_p 1.0 top_k 0 (DS41-W6.2)",
                                 extra={"pair_config": an["w62"]["pair_config"], "diff_A_minus_B_pooled": blk["diff_A_minus_B_pooled"],
                                        "ci95_paired_t_over_pairs": blk["ci95_paired_t_over_pairs"],
                                        "ci95_pair_x_prompt_bootstrap": blk["ci95_pair_x_prompt_bootstrap"],
                                        "other_drafter_sha256": dig["B" if arm == "A" else "A"]["sha256"]}))
    return [r["measurement_id"] for r in out]


def fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, float) else str(x))


def summary_md(an):
    L = ["# DS41 W5a / W5b / W6.2 slot results", "",
         f"Slot timeline: `TIMELINE.txt`; end flag: `{an['ended']}`.", ""]
    L += ["## W5a: NMSE spot checks", "",
          "Weight fidelity vs the HF 1017 source (`w5a/weight_fidelity.json`, measured 2026-10-21 without a server):",
          "", "| tensor | GGUF type | source | NMSE |", "|---|---|---|---|"]
    for r in an["w5a_weight_fidelity"]:
        L.append(f"| {r['tensor']}{' e' + str(r['expert']) if 'expert' in r else ''} | {r['gguf_type']} | {r['src_dtype']} | {r['nmse']:.3e} |")
    k = an["w5a_kernel"]
    L += ["", "Served-path kernel (CPU_REPACK `mxfp4_8x8_q8_0` GEMV inside MUL_MAT_ID) vs f32-dequant reference, cap 5e-4:", ""]
    if k["status"] == "ok":
        L += [f"- cases: {k['n']}, all repacked: {k['all_repacked']}, all pass: {k['all_pass']}",
              f"- max NMSE repack vs f32 ref: {k['max_nmse_repack_vs_f32ref']:.3e}; plain vs f32 ref: {k['max_nmse_plain_vs_f32ref']:.3e}; "
              f"repack vs plain: {k['max_nmse_repack_vs_plain']:.3e}"]
    else:
        L.append(f"- not available ({k['status']})")
    L += ["", "## W5b: first-token EOS probes", ""]
    b = an["w5b"]
    if b["status"] == "ok":
        L += ["| probe | kind | prompt tok | EOS first | P(EOS) | top-3 | text |", "|---|---|---|---|---|---|---|"]
        for r in b["rows"]:
            top = ", ".join(f"{t[1]!r}:{t[2]}" for t in r["top3"])
            pe = "n/a" if r["p_eos_first"] is None else f"{r['p_eos_first']:.3f}"
            L.append(f"| {r['id']} | {r['kind']} | {r['prompt_tokens']} | {r['first_is_eos']} | {pe} | {top} | {r['text_head']!r} |")
    else:
        L.append(f"not available ({b['status']})")
    L += ["", "## W6.2: 1017 vs Sept DSpark drafter alpha on the 1017 target", ""]
    for n, blk in an["w62"]["blocks"].items():
        if not blk.get("n_pairs"):
            L.append(f"- block {n}: no complete pairs"); continue
        L.append(f"- block {n}: pairs {blk['pairs']} (n={blk['n_pairs']}); alpha A(1017) {blk['alpha_A_1017']:.4f} vs "
                 f"B(Sept) {blk['alpha_B_sept']:.4f}; pooled A-B {blk['diff_A_minus_B_pooled'] * 100:+.2f} pp; "
                 f"per-pair mean {blk['mean_per_pair_diff'] * 100:+.2f} pp, paired-t CI95 "
                 f"{'n/a' if blk['ci95_paired_t_over_pairs'] is None else '[%+.2f, %+.2f] pp' % tuple(x * 100 for x in blk['ci95_paired_t_over_pairs'])}; "
                 f"pair x prompt bootstrap [{blk['ci95_pair_x_prompt_bootstrap'][0] * 100:+.2f}, {blk['ci95_pair_x_prompt_bootstrap'][1] * 100:+.2f}] pp; "
                 f"cells A win/loss/tie {blk['cells_A_win_loss_tie']}")
    L += ["", "Pair configs: " + "; ".join(f"p{p}: {c}" for p, c in sorted(an['w62']['pair_config'].items(), key=lambda x: int(x[0])))]
    g = an["greedy"]
    L += ["", f"Greedy text identity (p{g['pair']}A vs p{g['pair']}B, same config): "
          + (f"{g['n_identical']}/{g['n_compared']} identical (A {g['n_A']}, B {g['n_B']} prompts)" if g.get("n_compared") is not None else g.get("status", "?")), ""]
    L += ["Claim class: OBSERVATION / indicative (build-role claim on q0+q1, not a quiet-host timing window). "
          "Sampling for the stochastic blocks: temperature 0.7, top_p 1.0, top_k 0 (the W6 arm, not the 1017 card's 1.0/1.0).", ""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--worktree", default=None, help="reuse an existing worktree instead of creating one")
    ap.add_argument("--skip-capture", action="store_true", help="do not write belief_measurements rows")
    a = ap.parse_args()
    ended = open(ENDF).read().strip() if os.path.exists(ENDF) else "ABSENT (slot not finished?)"
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    wt = a.worktree or f"/mnt/raid0/llm/worktrees/research-ds41-w62-post-{ts}"
    if not a.worktree:
        sh("git", "-C", RESEARCH, "fetch", "-q", "origin", "main")
        sh("git", "-C", RESEARCH, "worktree", "add", "-q", "-b", f"ds41-w62-post-{ts}", wt, "origin/main")
    o62 = os.path.join(wt, A6, "w6.2")
    o5a = os.path.join(wt, D5, "w5a")
    o5b = os.path.join(wt, D5, "w5b", "run")
    for d in (o62, o5a, o5b):
        os.makedirs(d, exist_ok=True)
    added = []
    # copy new W6.2 runs (rows, client log, arm meta); server logs stay in scratch
    for d in sorted(glob.glob(f"{W}/w62/runs/p*")):
        dst = os.path.join(o62, "runs", os.path.basename(d))
        os.makedirs(dst, exist_ok=True)
        for f in ("rows.json", "client.log", "arm.json"):
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), dst)
    for src, dst in ((TL, os.path.join(o62, "TIMELINE.txt")), (f"{W}/w62/arms.log", os.path.join(o62, "arms.log")),
                     (f"{W}/w5a/kernel/w5a_kernel.jsonl", os.path.join(o5a, "w5a_kernel.jsonl")),
                     (f"{W}/w5a/kernel/provenance.txt", os.path.join(o5a, "kernel_provenance.txt")),
                     (f"{W}/w5a/kernel/run.log", os.path.join(o5a, "kernel_run.log")),
                     (f"{W}/w5a/kernel/weight_fidelity_slot.json", os.path.join(o5a, "weight_fidelity_slot.json")),
                     (f"{W}/w5b/run/w5b.rows.jsonl", os.path.join(o5b, "w5b.rows.jsonl")),
                     (f"{W}/w5b/run/w5b.recipe.json", os.path.join(o5b, "w5b.recipe.json")),
                     (f"{W}/w5b/run/w5b.server.log", os.path.join(o5b, "w5b.server.log"))):
        if os.path.exists(src):
            shutil.copy2(src, dst)
    an = {"schema": "ds41.w5ab_w62.post.v1", "generated": ts, "ended": ended,
          "w5a_weight_fidelity": json.load(open(f"{W}/w5a/weight_fidelity.json"))["rows"],
          "w5a_kernel": w5a_kernel(os.path.join(o5a, "w5a_kernel.jsonl")),
          "w5b": w5b(os.path.join(o5b, "w5b.rows.jsonl")),
          "w62": w62(os.path.join(wt, A6, "w6.1", "runs"), os.path.join(o62, "runs")),
          "greedy": greedy(os.path.join(o62, "runs"))}
    json.dump(an, open(os.path.join(o62, "analysis.json"), "w"), indent=1)
    md = summary_md(an)
    open(os.path.join(o62, "SUMMARY.md"), "w").write(md)
    print(md)
    if not a.skip_capture:
        ids = capture_rows(wt, an)
        print(f"belief rows captured: {len(ids)} (ingest: python3 /workspace/scripts/vidya/cli.py ingest ds41-1017-validation "
              f"--path <research>/{D5} and --path <research>/{A6}/w6.2, after the push)")
    if not a.commit:
        print(f"worktree: {wt} (not committed; rerun with --worktree {wt} --commit [--push])")
        return
    paths = [os.path.join(A6, "w6.2"), os.path.join(D5, "w5a"), os.path.join(D5, "w5b")]
    if os.path.exists(os.path.join(wt, D5, "belief_measurements.jsonl")):
        paths.append(os.path.join(D5, "belief_measurements.jsonl"))
    sh("git", "add", "--", *paths, cwd=wt)
    b2 = an["w62"]["blocks"].get("2", {})
    msg = f"{W}/commit-msg-{ts}.txt"
    open(msg, "w").write(
        "ds41 W5a/W5b/W6.2: slot results (kernel NMSE, EOS probes, drafter alpha pairs)\n\n"
        f"W6.2 block2 n_pairs={b2.get('n_pairs')} pooled A-B={fmt(b2.get('diff_A_minus_B_pooled'))}; "
        f"greedy identical {an['greedy'].get('n_identical')}/{an['greedy'].get('n_compared')}; "
        f"W5a kernel all_pass={an['w5a_kernel'].get('all_pass')}; W5b rows={an['w5b'].get('n')}.\n"
        f"End flag: {ended}\n\nCo-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>\n")
    sh("git", "commit", "-F", msg, cwd=wt)
    sha = sh("git", "rev-parse", "HEAD", cwd=wt)
    print("committed", sha)
    if a.push:
        r = subprocess.run([sys.executable, "/workspace/scripts/coordination/serialized_push.py", "--agent", a.agent,
                            "--repo", wt, "--fetch", "--push"], capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-2000:])
        if r.returncode != 0:
            raise SystemExit(f"push failed rc={r.returncode}; commit {sha} is on {wt}")


if __name__ == "__main__":
    main()
