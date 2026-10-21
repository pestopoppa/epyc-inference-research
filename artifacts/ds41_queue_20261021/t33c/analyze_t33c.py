"""DS41-T3.3c offline analysis of a t33 result JSON (NO server, NO inference): the BOS decision and where/why each case diverges.

usage: analyze_t33c.py <t33-label.json> [<more.json> ...] [--out <result.json>]

For every case it strips the oracle's leading BOS text (the template does not emit it; the server adds it with add_special=true),
then locates the first residual text divergence between encoding.py's output and the served /apply-template prompt, reports how
far from the END of the prompt it is, and classifies the shape of the residual:
  double_think        served has `<think><think>` where the oracle has one  (template emits a second <think> before the final reasoning)
  reasoning_leak      served inserts `<think>..</think>` after `</think>` in chat mode (reasoning_content re-rendered; oracle drops it)
  task_marker_lost    oracle has `<｜action｜>` (etc.) task token, served rendered the content instead (the `task` field did not reach the template)
  other               anything else (printed verbatim)
BOS decision: pass_add_special_true over the evaluated cases vs pass_add_special_false.
"""
import json, sys, re

BOS = "<｜begin▁of▁sentence｜>"
args = sys.argv[1:]
out = None
if "--out" in args:
    i = args.index("--out"); out = args[i + 1]; del args[i:i + 2]


def classify(o, s, i):
    ctx_o, ctx_s = o[max(0, i - 12):i + 40], s[max(0, i - 12):i + 40]
    w = lambda x: x[max(0, i - 12):i + 24]     # the divergence index can sit inside a token marker: look both sides
    if "<think><think>" in w(s) and "<think><think>" not in w(o):
        return "double_think", ctx_o, ctx_s
    if re.search(r"<｜(action|query|authority|domain|title|read_url)｜>", w(o)):
        return "task_marker_lost", ctx_o, ctx_s
    if "</think><think>" in w(s) and "</think><think>" not in w(o):
        return "reasoning_leak", ctx_o, ctx_s
    return "other", ctx_o, ctx_s


result = {"files": args, "cases": []}
for f in args:
    d = json.load(open(f))
    ev = [r for r in d["rows"] if "served_prompt" in r]
    t = sum(1 for r in ev if r.get("pass_add_special_true")); fa = sum(1 for r in ev if r.get("pass_add_special_false"))
    result["bos_decision"] = {"file": f, "evaluated": len(ev), "identical_add_special_true": t, "identical_add_special_false": fa,
                              "decision": "add_special=true (server adds BOS; template must NOT emit it)" if t > fa else "undecided",
                              "arguments_probe": d.get("arguments_probe")}
    for r in ev:
        o, s = r["oracle_text"], r["served_prompt"]
        o2 = o[len(BOS):] if o.startswith(BOS) and not s.startswith(BOS) else o
        i = next((k for k, (a, b) in enumerate(zip(o2, s)) if a != b), None if len(o2) == len(s) else min(len(o2), len(s)))
        row = {"case": r["case"], "oracle_chars": len(o2), "served_chars": len(s), "text_identical_after_bos_strip": i is None,
               "token_identical_add_special_true": r.get("pass_add_special_true"), "first_residual_div_char": i}
        if i is not None:
            kind, co, cs = classify(o2, s, i)
            row.update({"chars_from_end_of_oracle": len(o2) - i, "class": kind, "oracle_ctx": co, "served_ctx": cs})
        result["cases"].append(row)
bad = [c for c in result["cases"] if not c["text_identical_after_bos_strip"]]
result["summary"] = {"cases": len(result["cases"]), "residual_divergent": [c["case"] for c in bad],
                     "by_class": {k: [c["case"] for c in bad if c["class"] == k] for k in sorted({c["class"] for c in bad})},
                     "all_divergences_in_final_assistant_turn": all(c["chars_from_end_of_oracle"] < 400 for c in bad) if bad else None}
txt = json.dumps(result, indent=1, ensure_ascii=False)
if out:
    open(out, "w").write(txt)
print(txt)
