#!/usr/bin/env python3
"""Collect the WG-LFM-1 scout run into one machine-readable record."""
import glob
import hashlib
import json
import os
import re
import subprocess

D = "/workspace/tmp/wg-lfm-1"


def parse_md(path):
    rows = []
    for line in open(path):
        if not line.startswith("| ") or line.startswith("| model") or "---" in line:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 9:
            continue
        m = re.match(r"([\d.]+) ± ([\d.]+)", c[8])
        rows.append(
            {
                "model": c[0],
                "size": c[1],
                "params": c[2],
                "threads": int(c[4]),
                "fa": c[5],
                "mmap": c[6],
                "test": c[7],
                "t_s": float(m.group(1)) if m else None,
                "stddev": float(m.group(2)) if m else None,
            }
        )
    return rows


def parse_time(path):
    if not os.path.exists(path):
        return {}
    t = open(path).read()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    wall = re.search(r"Elapsed \(wall clock\) time [^:]*: ([\d:.]+)", t)
    return {
        "peak_rss_kb": int(rss.group(1)) if rss else None,
        "wall": wall.group(1) if wall else None,
        "iqk_active_lines": len(re.findall(r"\[iqk\] ACTIVE", t)),
    }


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


arms = []
for md in sorted(glob.glob(f"{D}/bench_*.md")):
    tag = os.path.basename(md)[len("bench_") : -3]
    arms.append(
        {
            "arm": tag,
            "rows": parse_md(md),
            **parse_time(md[:-3] + ".time"),
            "md_file": md,
        }
    )

correctness = {}
for f in sorted(glob.glob(f"{D}/correct2_*.txt")):
    tag = os.path.basename(f)[len("correct2_") : -4]
    txt = open(f).read()
    per_q = []
    for b in txt.split("### Q")[1:]:
        a = re.search(r"\[End thinking\](.*?)\[ Prompt:", b, re.S)
        m = re.search(r"\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]", b)
        per_q.append(
            {
                "q": int(b[0]),
                "answer": " ".join(a.group(1).split()) if a else None,
                "truncated": a is None,
                "cli_pp_t_s": float(m.group(1)) if m else None,
                "cli_tg_t_s": float(m.group(2)) if m else None,
            }
        )
    correctness[tag] = per_q

models = {}
for p in [
    "/mnt/raid0/llm/models/LFM2.5-2.6B-Q4_K_M.gguf",
    "/mnt/raid0/llm/models/LFM2.5-2.6B-Q8_0.gguf",
]:
    models[os.path.basename(p)] = {
        "path": p,
        "bytes": os.path.getsize(p),
        "sha256": sha(p),
    }

rec = {
    "task": "WG-LFM-1",
    "grade": "SCOUT — NOT decision-grade",
    "not_decision_grade_because": [
        "host uptime 14.1 d exceeds the 7 d constitutional limit (host-health gate warns)",
        "CPU scope is ONE region (q0 = cores 0-23, NUMA node 0), not the full-machine "
        "canonical 0-95 / -t 96 recipe; absolute numbers are not comparable to headline tables",
        "incumbent arm is base decode only — llama-bench cannot exercise the production "
        "gemma4 MTP self-speculative path (draft_max=2, ~95% acceptance)",
        "no tool-schema-compliance or repair-rate measurement (needs a llama-server arm)",
    ],
    "kernel": {
        "tree": "/mnt/raid0/llm/llama.cpp",
        "branch": "production-consolidated-v9",
        "commit": "0db32c06e3e550065b78311a6031ef3dd2c4f27c",
        "build": "b10125",
    },
    "hf_revision": "b421ad1d549afeda6a0fb2ad3a697cb5a7879adc",
    "cpu_scope": {"region": "q0", "cpus": "0-23", "numa": "membind=0", "threads": 24},
    "env": {
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "OMP_WAIT_POLICY": "active",
        "OMP_DYNAMIC": "false",
        "GGML_IQK": "1",
        "GGML_IQK_Q8_0": "1 (Q8_0 arms only)",
    },
    "models": models,
    "speed_arms": arms,
    "correctness": correctness,
}
out = f"{D}/wg-lfm-1-scout-record.json"
json.dump(rec, open(out, "w"), indent=2)
print(out)
print(json.dumps({a["arm"]: [(r["test"], r["t_s"]) for r in a["rows"]] for a in arms}, indent=2))
