#!/usr/bin/env python3
"""AGG-1.2 deterministic pilot-sample selection (zero model calls).
Reads dnd/test.jsonl (singledoc_* only), picks the median-length context window (ties: smaller context_window_id),
then 5 singledoc_rolls + 5 singledoc_spells questions: sort that window's questions by id, take indices 0,2,4,6,8 of each type
(spaced, no outcome information is read: `answer` is read only to build the gold file, after selection).
Usage: select_sample.py DATASET_DIR OUT.json   (python3 -I)"""
import json, sys, hashlib, os
ds, out = sys.argv[1], sys.argv[2]
wins = {}   # cwid -> (len, {type: [(id, question, answer)]})
with open(os.path.join(ds, "dnd", "test.jsonl"), "rb") as f:
    for line in f:
        d = json.loads(line)
        t = d["question_type"]
        if not t.startswith("singledoc_"):
            continue
        w = wins.setdefault(d["context_window_id"], [len(d["context_window_text"]), {}])
        w[1].setdefault(t, []).append((d["id"], d["question"], d["answer"]))
order = sorted(wins, key=lambda k: (wins[k][0], k))
cw = order[(len(order) - 1) // 2]
sel = []
for t in ("singledoc_rolls", "singledoc_spells"):
    qs = sorted(q for q in wins[cw][1][t] if q[2].strip().isdigit())   # integer-gold questions only (list-valued golds excluded; format filter, not an outcome filter)
    assert len(qs) >= 9, (t, len(qs))
    sel += [{"id": q[0], "type": t, "question": q[1], "gold": q[2]} for q in qs[0:9:2]]
res = {"dataset_dir": ds, "context_window_id": cw, "context_chars": wins[cw][0], "n": len(sel),
       "n_windows_singledoc": len(order), "questions": sel}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "questions"}), hashlib.sha256(open(out, "rb").read()).hexdigest())
