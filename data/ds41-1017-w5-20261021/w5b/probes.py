#!/usr/bin/env python3
"""DS41-W5b probe: why does an article + "Question: ...?\\nAnswer:" raw completion emit EOS as the first token?

Launches ONE target-only 1017 llama-server (own PID; no drafter, so n_probs is populated), runs a fixed
probe set greedily with n_probs=10 (top first-token logprobs, P(EOS)), then terminates the server and
verifies it is dead.

Static facts already established (2026-10-21, no compute): llama.cpp tokenization == HF tokenizer.json
on all 10 W5 prompts (one BOS, no double BOS); the pre-tokenizer regex is the HF one; the failing family
all end in the merged token "?\\u010a" + "Answer" + ":", the passing q40/p220 end ".\\u010a"/".\\u010a\\u010a" + "Answer" + ":".

usage: probes.py <cpulist> <nthreads> <port> <outdir>
"""
import hashlib, json, os, signal, subprocess, sys, time, urllib.request
import jinja2

cpus, nt, port, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
W = "/mnt/raid0/llm/tmp/ds41-w5ab-w62-20261021/w5b"
B = "/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin"
M = "/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf"
TPL = "/mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja"
EOS = 1
os.makedirs(out, exist_ok=True)

# ---------------------------------------------------------------- probe prompts (deterministic)
L = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(f"{W}/prompts_len.jsonl")}
HEAD = "Read the following encyclopedia article and answer the question at the end.\n\n"
Q = "\n\nQuestion: In one or two sentences, who is the subject of this article and what is his profession?\nAnswer:"
base = L["len1200"]
assert base.startswith(HEAD) and base.endswith(Q), "len1200 layout changed"
art = base[len(HEAD):-len(Q)]
art_sent = art[:art.rfind(" . ") + 2]                      # cut at the last sentence end, not mid-word
dewiki = art.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").replace(" , ", ", ") \
            .replace(" . ", ". ").replace(" 's", "'s").replace(" ( ", " (").replace(" ) ", ") ")
para1 = art[:art.find("\n", art.find("Robert Boulter is"))]
probes = [
    ("P00_len1200_control", base),
    ("P01_short_qa_qmark", "Question: What is the capital of France?\nAnswer:"),
    ("P02_short_qa_qmark_2nl", "Question: What is the capital of France?\n\nAnswer:"),
    ("P03_para1_same_suffix", HEAD + para1 + Q),
    ("P04_q_ends_period", base[:-len(Q)] + "\n\nQuestion: In one or two sentences, name the subject of this article and his profession.\nAnswer:"),
    ("P05_qmark_2nl_answer", base[:-len("\nAnswer:")] + "\n\nAnswer:"),
    ("P06_answer_trailing_space", base + " "),
    ("P07_no_head_instruction", art + Q),
    ("P08_sentence_boundary_cut", HEAD + art_sent + Q),
    ("P09_dewikitext", HEAD + dewiki + Q),
    ("P10_question_first_and_last", "Question: In one or two sentences, who is the subject of the following article and what is his profession?\n\n" + art + Q),
    ("P11_answer_prefix_The", base + " The"),
]
# length ladder on the same article: ~110 / 160 / 210 / 260 tokens of article
for nchar in (450, 700, 950, 1150):
    probes.append((f"P2{nchar // 100:02d}_ladder_{nchar}c", HEAD + art[:nchar] + Q))
# EOS banned: what the model would say instead (does it know the answer?)
banned = [("P30_len1200_eos_banned", base), ("P31_len9000_eos_banned", L["len9000"])]
# chat path: the same content rendered with the epyc-v1 template (1017 encoding.py), non-thinking
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
tpl = env.from_string(open(TPL).read())
BOS_S = "<｜begin▁of▁sentence｜>"
def chat(text):
    s = tpl.render(messages=[{"role": "user", "content": text}], enable_thinking=False, add_generation_prompt=True)
    assert s.startswith(BOS_S)
    return s[len(BOS_S):]                                 # the server adds the BOS itself
chats = [("P40_chat_len1200_noanswer", chat(base[:-len("\nAnswer:")])),
         ("P41_chat_len1200_with_answer", chat(base))]

if os.environ.get("W5B_DRY") == "1":
    for pid, text in probes + banned + chats:
        print(f"DRY {pid} chars={len(text)} tail={text[-70:]!r}")
    raise SystemExit(0)

# ---------------------------------------------------------------- server
env_s = dict(os.environ, OMP_PROC_BIND="spread", OMP_PLACES="cores", OMP_WAIT_POLICY="active",
             OMP_DYNAMIC="false", GGML_IQK="1", LD_LIBRARY_PATH=B)
argv = ["taskset", "-c", cpus, "numactl", "--interleave=all", f"{B}/llama-server", "-m", M, "-np", "1",
        "-c", "8192", "-t", nt, "-tb", nt, "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16",
        "--device", "none", "-ngl", "0", "-fa", "on", "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
json.dump({"argv": argv, "probes": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in probes + banned + chats}},
          open(f"{out}/w5b.recipe.json", "w"), indent=1)
def _term(*_):
    raise SystemExit(143)          # so the finally block below always tears the server down


signal.signal(signal.SIGTERM, _term)
log = open(f"{out}/w5b.server.log", "w")
p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env_s)
open(f"{out}/server.pid", "w").write(str(p.pid))


def call(path, body=None, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def top_first(r):
    cp = r.get("completion_probabilities") or []
    if not cp:
        return None, None
    tops = cp[0].get("top_logprobs") or cp[0].get("top_probs") or []
    peos = None
    for t in tops:
        if t.get("id") == EOS:
            peos = t.get("logprob", t.get("prob"))
    return [(t.get("id"), t.get("token"), round(t.get("logprob", t.get("prob", 0.0)), 4)) for t in tops], peos


rows = []
try:
    t0 = time.time()
    while True:
        if p.poll() is not None:
            raise SystemExit(f"server exited rc={p.returncode}")
        try:
            if call("/health", timeout=5).get("status") == "ok":
                break
        except Exception:
            pass
        if time.time() - t0 > 1800:
            raise SystemExit("load timeout")
        time.sleep(5)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    for kind, items in (("raw", probes), ("eos_banned", banned), ("chat_rendered", chats)):
        for pid, text in items:
            body = {"prompt": text, "n_predict": 48, "temperature": 0, "top_k": 1, "seed": 42, "n_probs": 10,
                    "return_tokens": True, "cache_prompt": False}
            if kind == "eos_banned":
                body["logit_bias"] = [[EOS, False]]
            r = call("/completion", body)
            tops, peos = top_first(r)
            toks = r.get("tokens") or []
            row = {"id": pid, "kind": kind, "prompt_tokens": r.get("tokens_evaluated"), "first_token": toks[0] if toks else None,
                   "first_is_eos": bool(toks) and toks[0] == EOS, "n_gen": len(toks), "stop_type": r.get("stop_type"),
                   "text": r.get("content", ""), "first_top10": tops, "first_eos_logprob": peos,
                   "prompt_tail": text[-120:]}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("id", "prompt_tokens", "first_is_eos", "first_eos_logprob")}), repr(row["text"][:100]), flush=True)
            with open(f"{out}/w5b.rows.jsonl", "w") as fh:
                for x in rows:
                    fh.write(json.dumps(x) + "\n")
finally:
    p.terminate()
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(30)
    alive = os.path.exists(f"/proc/{p.pid}") and p.poll() is None
    print("server pid", p.pid, "rc", p.returncode, "ALIVE" if alive else "dead", flush=True)
