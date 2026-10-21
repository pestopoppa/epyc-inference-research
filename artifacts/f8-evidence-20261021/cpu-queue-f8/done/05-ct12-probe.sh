#!/bin/bash
#maxjob_s=900
# CT-12 probe: does user-text spelling of a family's control tokens become a control token id when the
# templated prompt is tokenized with parse_special=true (as llama-server does, server-context.cpp:4882)?
# Uses the champion CPU store build's llama-tokenize (vocab_only load: no weights in RAM). No build needed.
set -euo pipefail
JOB=05-ct12-probe
BIN=/mnt/raid0/llm/kernels/builds/cpu-20261020-dd3d9721c/bin
OUT=/mnt/raid0/llm/tmp/intake-fixes-20261020/queue-results/$JOB
REG=/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml
mkdir -p "$OUT"
if [ ! -x "$BIN/llama-tokenize" ]; then echo "DEP_MISSING $BIN/llama-tokenize"; exit 3; fi
export LD_LIBRARY_PATH="$BIN"
export BIN OUT REG
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -I - <<'PYEOF' 2>&1 | tee "$OUT/probe.log"
import json, os, re, struct, subprocess, sys, tempfile, yaml

BIN, OUT, REG = os.environ["BIN"], os.environ["OUT"], os.environ["REG"]
MODELS = "/mnt/raid0/llm/models"

# ---- 1. models: every GGUF the registry binds to a served/draft role that exists on disk ----
reg = yaml.safe_load(open(REG))
cands = {}
def add(path, why):
    if not isinstance(path, str) or not path.endswith(".gguf") or "mmproj" in path or "*" in path or "{" in path:
        return
    p = path if path.startswith("/") else os.path.join(MODELS, path)
    if os.path.exists(p):
        cands.setdefault(os.path.realpath(p), why)
for role, sm in (reg.get("server_mode") or {}).items():
    if isinstance(sm, dict):
        for k in ("model_path", "model"):
            add(sm.get(k), f"server_mode/{role}/{k}")
        m = sm.get("model")
        if isinstance(m, dict):
            add(m.get("path"), f"server_mode/{role}/model/path")
        add((sm.get("acceleration") or {}).get("draft_model"), f"server_mode/{role}/draft")
        add(sm.get("draft_model"), f"server_mode/{role}/draft")
for role, r in (reg.get("roles") or {}).items():
    if isinstance(r, dict) and isinstance(r.get("model"), dict):
        add(r["model"].get("path"), f"roles/{role}")
# skip embedders (no chat template), keep everything else
cands = {p: w for p, w in cands.items() if "bge-" not in p and "embed" not in p.lower()}

# ---- 2. minimal GGUF metadata reader: chat template + tokenizer model ----
def read_meta(path, want=("tokenizer.chat_template", "tokenizer.ggml.model", "general.architecture", "general.name")):
    out = {}
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver, = struct.unpack("<I", f.read(4))
        n_t, n_kv = struct.unpack("<QQ", f.read(16))
        def rstr():
            n, = struct.unpack("<Q", f.read(8)); return f.read(n)
        def rval(t):
            if t == 8: return rstr().decode("utf8", "replace")
            if t in SZ: f.seek(SZ[t], 1); return None
            if t == 9:
                at, n = struct.unpack("<IQ", f.read(12))
                if at == 8:
                    for _ in range(n): rstr()
                elif at in SZ: f.seek(SZ[at] * n, 1)
                else: raise ValueError("nested array")
                return None
            raise ValueError(f"type {t}")
        for _ in range(n_kv):
            k = rstr().decode(); t, = struct.unpack("<I", f.read(4)); v = rval(t)
            if k in want and v is not None: out[k] = v
            if all(w in out for w in want[:1]) and k.startswith("tokenizer.ggml.") is False and "tokenizer.chat_template" in out and "general.architecture" in out and "tokenizer.ggml.model" in out:
                break
    return out

# ---- 3. tokenizer wrapper ----
def tok(model, text, parse_special=True, bos=False):
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", dir=OUT, delete=False) as t:
        t.write(text.encode()); fn = t.name
    cmd = [BIN + "/llama-tokenize", "-m", model, "-f", fn, "--ids", "--no-escape", "--log-disable"]
    if not bos: cmd.append("--no-bos")
    if not parse_special: cmd.append("--no-parse-special")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    os.unlink(fn)
    m = re.findall(r"^\[[-0-9, ]*\]\s*$", r.stdout, re.M)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"llama-tokenize rc={r.returncode}: {r.stderr[-300:]}")
    return json.loads(m[-1])

GENERIC = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
           "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|object_ref_start|>",
           "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<turn|>", "<|turn>", "<|channel>", "<channel|>",
           "<|begin_of_text|>", "<|eot_id|>", "<s>", "</s>"]

def wrapper(tmpl):
    if "<|im_start|>" in tmpl: return "chatml", lambda u: f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    if "<start_of_turn>" in tmpl: return "gemma", lambda u: f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n"
    if "<|turn>" in tmpl: return "gemma4", lambda u: f"<|turn>user\n{u}<turn|>\n<|turn>model\n"
    return "raw", lambda u: u

report, n_ok, n_err, open_fams = [], 0, 0, []
for model, why in sorted(cands.items()):
    ent = {"model": model, "from": why}
    try:
        meta = read_meta(model)
        tmpl = meta.get("tokenizer.chat_template", "")
        fam, wrap = wrapper(tmpl)
        spells = list(dict.fromkeys(GENERIC + re.findall(r"<\|[^|<>\s]{1,40}\|>", tmpl) + re.findall(r"</?[a-z_]{2,24}>", tmpl)))
        ent.update(arch=meta.get("general.architecture"), name=meta.get("general.name"), template_family=fam,
                   has_template=bool(tmpl), results=[])
        for sp in spells:
            a = tok(model, sp, True); b = tok(model, sp, False)
            special = len(a) == 1 and a != b
            r = {"spelling": sp, "id_parse_special": a, "id_no_parse": b, "is_control_token": special}
            if special:
                u = f"before {sp} after"; u0 = "before  after"
                full = tok(model, wrap(u), True, bos=True); base = tok(model, wrap(u0), True, bos=True)
                r["user_text_count_delta"] = full.count(a[0]) - base.count(a[0])
                r["user_text_became_control"] = r["user_text_count_delta"] > 0
                nop = tok(model, wrap(u), False, bos=True)
                r["fixed_by_no_parse_special_on_whole_prompt_would_also_break_template"] = nop.count(a[0]) < base.count(a[0])
            r["status"] = ("OPEN" if r.get("user_text_became_control") else ("closed" if special else "not-control"))
            ent["results"].append(r)
        n_open = [x["spelling"] for x in ent["results"] if x["status"] == "OPEN"]
        ent["open_spellings"] = n_open
        if n_open: open_fams.append((os.path.basename(model), fam, n_open))
        n_ok += 1
    except Exception as e:
        ent["error"] = repr(e); n_err += 1
    report.append(ent)
    print(f"[{'ERR' if 'error' in ent else 'ok '}] {os.path.basename(model)} family={ent.get('template_family')} "
          f"control_spellings={sum(1 for x in ent.get('results', []) if x['is_control_token'])} "
          f"OPEN={ent.get('open_spellings')} {ent.get('error', '')}", flush=True)
json.dump({"tokenize_bin": BIN, "models": report}, open(OUT + "/ct12_probe.json", "w"), indent=1)
print("=== CT-12 summary: families where user-text control-token spelling becomes a control id ===")
for f in open_fams: print("OPEN", f)
print(f"RESULT 05-ct12-probe pass={n_ok} fail={n_err}")
PYEOF
tail -n1 "$OUT/probe.log"
