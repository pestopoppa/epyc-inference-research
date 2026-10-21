"""DS41-T3.3 / T3.3c (v3: robust arguments probe + reasoning_content/task variants for the 4 residual divergences; v2 otherwise): oracle(encoding.py) render -> server /tokenize  vs  server /apply-template -> /tokenize.
usage: t33.py PORT ENCODING_PY_DIR OUTDIR LABEL   (host-side only, no inference)"""
import sys, json, copy, urllib.request, importlib.util, hashlib, os
port, encdir, outdir, label = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)
sp = importlib.util.spec_from_file_location("encoding", os.path.join(encdir, "encoding.py")); enc = importlib.util.module_from_spec(sp); sp.loader.exec_module(enc)
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try: return json.load(urllib.request.urlopen(r, timeout=120))
    except Exception as e: return {"error": str(e)}
def tok(text, add_special=False): return post("/tokenize", {"content": text, "add_special": add_special, "parse_special": True}).get("tokens")
tests = "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding/tests"
cases = []
for i in range(1, 6):
    c = enc.load_cases(f"{tests}/test_input_{i}.json")[0]
    c["name"] = f"golden{i}"; c["golden_text"] = open(f"{tests}/test_output_{i}.txt").read(); c["mode"] = c.get("thinking_mode") or "chat"; cases.append(c)
U = lambda t: {"role": "user", "content": t}; A = lambda t, **k: dict({"role": "assistant", "content": t}, **k)
tool = [{"type": "function", "function": {"name": "get_w", "description": "w", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
tc = {"id": "c1", "type": "function", "function": {"name": "get_w", "arguments": json.dumps({"city": "Paris"})}}
extra = {
 "tool_call_roundtrip": ([{"role": "system", "content": "sys", "tools": tool}, U("weather?"), A("", tool_calls=[tc]), {"role": "tool", "tool_call_id": "c1", "content": "sunny"}], "chat", None),
 "tool_result_only_user_turn": ([{"role": "system", "content": "sys", "tools": tool}, U("weather?"), A("", tool_calls=[tc]), {"role": "tool", "tool_call_id": "c1", "content": "sunny"}, A("It is sunny."), U("thanks")], "chat", None),
 "thinking": ([U("2+2?")], "thinking", None),
 "thinking_effort_high": ([U("2+2?")], "thinking", "high"),
 "thinking_effort_max": ([U("2+2?")], "thinking", "max"),
 "effort_int_75": ([U("2+2?")], "thinking", 75),
 "mid_system": ([U("hi"), A("hello"), {"role": "system", "content": "be terse"}, U("again")], "chat", None),
 "reasoning_retention_prefix": ([U("q1"), A("a1", reasoning_content="r1"), U("q2")], "thinking", None),
 "reasoning_retention_nonprefix": ([U("q1"), A("a1", reasoning_content="r1"), U("q2"), A("a2", reasoning_content="r2"), U("q3")], "thinking", None),
}
for n, (m, mode, eff) in extra.items(): cases.append({"name": n, "messages": m, "context": None, "mode": mode, "reasoning_effort": eff})
def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return None if len(a) == len(b) else min(len(a), len(b))
def probe_arguments():
    """What the server hands the template for tool_call.function.arguments: string or object?  v3: the tool_call sits in a full
    round trip (assistant tool_calls -> tool result -> user), because v2's probe put it in the LAST message and both forms
    rendered neither the named parameter nor the blob (UNDETERMINED). The raw prompt is kept so a null is inspectable."""
    base = [{"role": "system", "content": "s"}, {"role": "user", "content": "w?"}]
    res = {}
    for form in ("string", "object"):
        a = json.dumps({"city": "Paris"}) if form == "string" else {"city": "Paris"}
        msgs = base + [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_w", "arguments": a}}]},
                       {"role": "tool", "tool_call_id": "c1", "content": "sunny"}, {"role": "user", "content": "thanks"}]
        r = post("/apply-template", {"messages": msgs, "tools": tool, "chat_template_kwargs": {"enable_thinking": False}})
        p = r.get("prompt", "")
        res[form] = {"error": str(r)[:200]} if "prompt" not in r else {"maps_to_named_parameter": 'parameter name="city"' in p, "fell_back_to_arguments_blob": 'parameter name="arguments"' in p,
                                                                      "has_dsml_invoke": "invoke name=\"get_w\"" in p, "prompt_tail": p[-420:]}
    if res["string"].get("maps_to_named_parameter"): res["server_delivers"] = "object (string input is parsed to an object before the template)"; res["send"] = "string"
    elif res["object"].get("maps_to_named_parameter"): res["server_delivers"] = "as sent (string stays string, object stays object)"; res["send"] = "object"
    else: res["server_delivers"] = "UNDETERMINED"; res["send"] = "object"
    return res
ARGPROBE = probe_arguments()
def fix_args(msgs, form):
    msgs = copy.deepcopy(msgs)
    for m in msgs:
        for t in m.get("tool_calls") or []:
            a = t["function"]["arguments"]
            if form == "object" and isinstance(a, str): t["function"]["arguments"] = json.loads(a)
            if form == "string" and not isinstance(a, str): t["function"]["arguments"] = json.dumps(a)
    return msgs
def has_image(msgs): return any(isinstance(m.get("content"), list) and any(isinstance(b, dict) and b.get("type") in ("image", "image_url") for b in m["content"]) for m in msgs)
def pass_of(div): return div is None
rows = []
for c in cases:
    r = {"case": c["name"], "arguments_probe_send": ARGPROBE["send"]}
    if has_image(c["messages"]):
        r["skip"] = "SKIP(no mmproj)"; r["pass"] = None; rows.append(r); continue   # stays in the denominator as skipped
    try: oracle = enc.encode_messages(copy.deepcopy(c["messages"]), thinking_mode=c["mode"], context=c.get("context"), reasoning_effort=c.get("reasoning_effort"))
    except Exception as e: r["oracle_error"] = repr(e); rows.append(r); continue
    if "golden_text" in c: r["oracle_matches_golden_text"] = (oracle == c["golden_text"])
    o_ids = tok(oracle); r["oracle_n"] = len(o_ids or [])
    msgs = [{k: v for k, v in m.items() if k != "tools"} for m in c["messages"]]
    tools = next((m["tools"] for m in c["messages"] if m.get("tools")), None)
    msgs = fix_args(msgs, ARGPROBE["send"])
    body = {"messages": msgs}
    if tools: body["tools"] = tools
    # T3.3c: ALWAYS send enable_thinking explicitly (the server default is thinking ON when the key is absent)
    body["chat_template_kwargs"] = {"enable_thinking": c["mode"] == "thinking", **({"reasoning_effort": c["reasoning_effort"]} if c.get("reasoning_effort") is not None else {})}
    at = post("/apply-template", body)
    if "prompt" not in at: r["served_error"] = str(at)[:300]; rows.append(r); continue
    served = at["prompt"]; r["served_prompt"] = served; r["oracle_text"] = oracle; r["request_chat_template_kwargs"] = body["chat_template_kwargs"]
    r["served_starts_with_bos_text"] = served.startswith("<｜begin▁of▁sentence｜>"); r["oracle_starts_with_bos_text"] = oracle.startswith("<｜begin▁of▁sentence｜>")
    # as served: the chat path tokenizes the rendered text with the model's own BOS rule; try both
    s_ids = tok(served, add_special=False); s_ids_sp = tok(served, add_special=True)
    r.update(served_n=len(s_ids or []), served_n_addspecial=len(s_ids_sp or []))
    r["text_first_div_char"] = first_div(oracle, served)
    r["id_first_div"] = first_div(o_ids, s_ids); r["id_first_div_addspecial"] = first_div(o_ids, s_ids_sp)
    if r["id_first_div"] is not None:
        i = r["id_first_div"]; r["oracle_ctx"] = oracle[max(0, (r["text_first_div_char"] or 0) - 30):(r["text_first_div_char"] or 0) + 40]; r["served_ctx"] = served[max(0, (r["text_first_div_char"] or 0) - 30):(r["text_first_div_char"] or 0) + 40]
    # BOS is NOT decided here: both views are reported. add_special=false = tokenize the rendered text as-is; add_special=true = the
    # server's own chat-path rule (model adds BOS). pass_* are per view; "pass" (headline) stays the add_special=false view for continuity.
    r["pass_add_special_false"] = r["id_first_div"] is None; r["pass_add_special_true"] = r["id_first_div_addspecial"] is None
    r["pass"] = r["pass_add_special_false"]
    rows.append(r)
# v3 variants: do the residual divergences come from the FINAL assistant turn's reasoning_content / task fields?
def last_assistant(msgs):
    for k in range(len(msgs) - 1, -1, -1):
        if msgs[k].get("role") == "assistant": return k
    return None
variants = []
for c in cases:
    if c["name"] not in ("golden1", "golden2", "golden3", "golden4") or has_image(c["messages"]): continue
    k = last_assistant(c["messages"])
    if k is None: continue
    v = {"case": c["name"], "final_assistant_keys": sorted(c["messages"][k].keys())}
    for vn, drop in (("drop_reasoning_content", ("reasoning_content",)), ("drop_task", ("task",))):
        if not any(d in c["messages"][k] for d in drop): v[vn] = "n/a (field absent)"; continue
        m2 = copy.deepcopy(c["messages"]); [m2[k].pop(d, None) for d in drop]
        try: o2 = enc.encode_messages(copy.deepcopy(m2), thinking_mode=c["mode"], context=c.get("context"), reasoning_effort=c.get("reasoning_effort"))
        except Exception as e: v[vn] = {"oracle_error": repr(e)}; continue
        msgs2 = fix_args([{kk: vv for kk, vv in m.items() if kk != "tools"} for m in m2], ARGPROBE["send"])
        tools2 = next((m["tools"] for m in m2 if m.get("tools")), None)
        b2 = {"messages": msgs2, "chat_template_kwargs": {"enable_thinking": c["mode"] == "thinking"}}
        if tools2: b2["tools"] = tools2
        at2 = post("/apply-template", b2); s2 = at2.get("prompt")
        B = "<｜begin▁of▁sentence｜>"; o2s = o2[len(B):] if o2.startswith(B) else o2
        v[vn] = {"text_identical_after_bos_strip": (s2 == o2s) if s2 is not None else None, "first_div_char": first_div(o2s, s2 or "")}
    variants.append(v)
out = {"label": label, "arguments_probe": ARGPROBE, "variants": variants, "encoding_py_sha256": hashlib.sha256(open(os.path.join(encdir, "encoding.py"), "rb").read()).hexdigest(), "rows": rows}
json.dump(out, open(os.path.join(outdir, f"t33-{label}.json"), "w"), indent=1, ensure_ascii=False)
ev = [r for r in rows if "skip" not in r and "oracle_error" not in r and "served_error" not in r]
sk = [r["case"] for r in rows if "skip" in r]
print(f"T33c {label}: denominator={len(rows)} evaluated={len(ev)} skipped={sk}; add_special=false {sum(1 for r in ev if r['pass_add_special_false'])}/{len(ev)} identical; add_special=true {sum(1 for r in ev if r['pass_add_special_true'])}/{len(ev)} identical; "
      f"divergent(false)={[r['case'] for r in ev if not r['pass_add_special_false']]}; divergent(true)={[r['case'] for r in ev if not r['pass_add_special_true']]}; errors={[r['case'] for r in rows if 'oracle_error' in r or 'served_error' in r]}; args_probe={ARGPROBE['send']}/{ARGPROBE['server_delivers']}")
