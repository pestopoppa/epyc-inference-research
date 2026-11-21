#!/usr/bin/env python3
"""Extract judge fixtures from retained TOC-CMP-3 live (r3) compaction transcripts.

Per trial with a `summary` wire call: summary handover = that response content; spec actions = the REPL code
the model wrote in the post-boundary generation turns (the continuation after the fold).
Fixture kinds:
  accept            own summary + own post-boundary actions
  reject_mismatch   summary of trial A + post-boundary actions of trial B (different task context)
  reject_redo       own summary + the PRE-boundary turn-1 action replayed (re-does work listed as done)
The redo reject is WEAK: the replayed action is a read-only probe, which the V2-Fast prompt itself says cannot
contradict completed changes, so a faithful judge may legitimately accept it. Firm labels: accept, reject_mismatch.
Labels are by construction from real transcripts; no model output is used as a label.
"""
import glob, json, os, sys

ROOT = sys.argv[1]
OUT = sys.argv[2]


def content(path):
    raw = open(path).read()
    try:
        return json.loads(raw)["choices"][0]["message"]["content"]
    except Exception:
        out = []
        for line in raw.splitlines():
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    out.append(json.loads(line[5:])["choices"][0]["delta"].get("content") or "")
                except Exception:
                    pass
        return "".join(out)


trials = []
for t in sorted(glob.glob(f"{ROOT}/trial-*")):
    g = json.load(open(t + "/GRAPH.json"))
    w = g["wire"]
    summ = [x for x in w if x["call_kind"] == "summary"]
    if not summ:
        continue
    resp = lambda x: content(os.path.join(t, os.path.dirname(x["response_file"]), "response.body"))
    pre = [resp(x) for x in w if x["call_kind"] == "generation" and not x["history_boundaries"]]
    post = [resp(x) for x in w if x["call_kind"] == "generation" and x["history_boundaries"]]
    trials.append({"id": os.path.basename(t), "summary": resp(summ[0]), "pre": pre, "post": post})

cases = []
for i, a in enumerate(trials):
    act = lambda texts: [{"command": c.strip()} for c in texts if c.strip()]
    cases.append({"id": f"{a['id']}-accept", "label": "accept", "strength": "firm", "summary": a["summary"], "actions": act(a["post"])})
    b = trials[(i + 1) % len(trials)]
    cases.append({"id": f"{a['id']}-mismatch-{b['id']}", "label": "reject", "strength": "firm", "summary": a["summary"], "actions": act(b["post"])})
    cases.append({"id": f"{a['id']}-redo", "label": "reject", "strength": "weak", "summary": a["summary"], "actions": act(a["pre"][:1])})
json.dump({"source": ROOT, "cases": cases}, open(OUT, "w"), indent=1)
print(len(trials), "trials", len(cases), "cases")
