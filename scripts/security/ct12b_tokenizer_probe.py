#!/usr/bin/env python3
"""CT-12.1b: tokenizer-level control-token probe for the NON-Qwen lineup.

Extends CT-12.1 (05-ct12-probe.sh, Qwen-only).  For every GGUF in MODELS it
  1. reads the vocab (tokenizer.ggml.tokens / token_type) and lists every
     CONTROL (3) / USER_DEFINED (4) spelling, collapsing numbered placeholder
     runs (`<unused12>`, `<|place_holder_mm_span_0021|>`) to first+last;
  2. tokenizes each spelling with `llama-tokenize` parse_special ON and OFF
     (no weights are loaded: llama-tokenize opens the GGUF vocab-only);
  3. records whether the spelling is a control id (`1 token` under parse_special,
     multi-token under --no-parse-special) and whether the CURRENT orchestrator
     neutraliser regex (`has_control_tokens`) already matches it.

Usage: ct12b_tokenizer_probe.py --bin <dir-with-llama-tokenize> --orch <orchestrator-root> --out <json>
Writes only to --out.  No server, no inference.
"""
from __future__ import annotations

import argparse, json, os, re, struct, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

M = "/mnt/raid0/llm/models/"
# family -> [(label, path)]; every non-Qwen family with GGUFs on disk and a registry/lineup tie
MODELS = {
    "deepseek41": [
        ("DS41-1017 full (antirez MXFP4 shard 1)", M + "antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf"),
        ("DS41-1017 DSpark drafter (deepseek-ai)", M + "deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf"),
        ("DS41-1017 DSpark Q8_0 (unsloth)", M + "unsloth/DeepSeek-V4.1-Flash-1017-GGUF/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"),
    ],
    "gemma4": [
        ("gemma-4-31B-it Q4_K_M", M + "gemma-4-31B-it-Q4_K_M.gguf"),
        ("gemma-4-31B-it assistant Q8_0", M + "gemma-4-31B-it-assistant-Q8_0.gguf"),
    ],
    "llama3": [("Llama-3.2-1B-Instruct f16", M + "Llama-3.2-1B-Instruct-f16.gguf")],
    "nemotron": [("Nemotron-Labs-Diffusion-14B Q8_0", M + "Nemotron-Labs-Diffusion-14B-Q8_0-GGUF/nemotron-diffusion-14b-Q8_0.gguf")],
    "embedder": [
        ("bge-m3 Q8_0", M + "bge-m3-Q8_0.gguf"),
        ("bge-large-en-v1.5 f16", M + "bge-large-en-v1.5-f16.gguf"),
        ("granite-embedding-97m Q8_0", M + "granite-embedding-97m-multilingual-r2-Q8_0.gguf"),
        ("multilingual-e5-base Q8_0", M + "multilingual-e5-base-Q8_0.gguf"),
    ],
    "speech-tts": [("qwen-talker-0.6b-base Q8_0", M + "Qwen3-TTS-qwentts/qwen-talker-0.6b-base-Q8_0.gguf")],
}
# non-special "added tokens" (HF special:false) that llama.cpp tokenizes as plain TEXT but that the chat
# template emits verbatim: a user spelling is byte-for-byte what the template emits (role/DSML injection).
DS41_TOKENIZER_JSON = M + "deepseek-ai/DeepSeek-V4.1-Flash/tokenizer.json"
INJ_EXTRA = ["[INST]", "[/INST]", "[SEP]", "[CLS]", "[MASK]", "<SPECIAL_18>", "<｜DSML｜", "</｜DSML｜", "<dsml:", "</dsml:", "<|EOT|>"]
GENERIC = ["<s>", "</s>", "<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>",
           "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|begin_of_text|>", "<|eot_id|>", "<|im_start|>", "<|im_end|>"]
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FM = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    f.read(4)
    _, nkv = struct.unpack("<QQ", f.read(16))
    def rs():
        n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf8", "replace")
    def rv(t, keep):
        if t == 8: return rs()
        if t in SZ: return struct.unpack(FM[t], f.read(SZ[t]))[0]
        at, n = struct.unpack("<IQ", f.read(12))
        if at == 8:
            if keep: return [rs() for _ in range(n)]
            for _ in range(n): rs()
            return None
        if keep: return list(struct.unpack("<%d%s" % (n, FM[at][1]), f.read(SZ[at] * n)))
        f.seek(SZ[at] * n, 1); return None
    out = {}
    for _ in range(nkv):
        k = rs(); t, = struct.unpack("<I", f.read(4))
        keep = k in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type")
        v = rv(t, keep)
        if keep or k in ("tokenizer.chat_template", "tokenizer.ggml.model", "tokenizer.ggml.pre", "general.architecture", "general.name"):
            out[k] = v
    return out


def collapse(tokens, types):
    """control/user-defined spellings; numbered runs collapsed to first+last."""
    groups = {}
    for i, (t, y) in enumerate(zip(tokens, types)):
        if y in (3, 4) and not re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t):
            groups.setdefault(re.sub(r"\d+", "#", t), []).append((i, t, y))
    rows = []
    for key, items in groups.items():
        sel = items if len(items) <= 2 else [items[0], items[-1]]
        for i, t, y in sel:
            rows.append({"id": i, "spelling": t, "token_type": y, "group_size": len(items)})
    return rows


class Tok:
    def __init__(self, bin_dir, tmp):
        self.bin, self.tmp = bin_dir, tmp
    def __call__(self, model, text, parse_special):
        with tempfile.NamedTemporaryFile("wb", suffix=".txt", dir=self.tmp, delete=False) as t:
            t.write(text.encode()); fn = t.name
        cmd = [self.bin + "/llama-tokenize", "-m", model, "-f", fn, "--ids", "--no-escape", "--log-disable", "--no-bos"]
        if not parse_special: cmd.append("--no-parse-special")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                               env={**os.environ, "LD_LIBRARY_PATH": self.bin})
        finally:
            os.unlink(fn)
        m = re.findall(r"^\[[-0-9, ]*\]\s*$", r.stdout, re.M)
        if r.returncode != 0 or not m:
            raise RuntimeError(f"llama-tokenize rc={r.returncode}: {r.stderr[-200:]}")
        return json.loads(m[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--orch", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.orch)
    from src.security.control_tokens import has_control_tokens, neutralise_control_tokens
    tmp = os.path.dirname(os.path.abspath(a.out)); os.makedirs(tmp, exist_ok=True)
    tok = Tok(a.bin, tmp)
    report = {"tokenize_bin": a.bin, "orchestrator_root": a.orch, "families": {}}
    for fam, models in MODELS.items():
        for label, path in models:
            ent = {"label": label, "path": path}
            report["families"].setdefault(fam, []).append(ent)
            if not os.path.exists(path):
                ent["error"] = "not on disk"; print("MISSING", label); continue
            try:
                meta = read_gguf(path)
                ent.update(arch=meta.get("general.architecture"), name=meta.get("general.name"),
                           tokenizer_model=meta.get("tokenizer.ggml.model"), tokenizer_pre=meta.get("tokenizer.ggml.pre"),
                           has_chat_template=bool(meta.get("tokenizer.chat_template")))
                toks, types = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"]
                rows = collapse(toks, types)
                seen = {r["spelling"] for r in rows}
                for g in GENERIC:
                    if g not in seen:
                        rows.append({"id": None, "spelling": g, "token_type": None, "group_size": 1, "generic_probe": True})
                if fam == "deepseek41" and os.path.exists(DS41_TOKENIZER_JSON):
                    for a_ in json.load(open(DS41_TOKENIZER_JSON))["added_tokens"]:
                        c_ = a_["content"]
                        if not a_["special"] and "place_holder" not in c_ and c_ not in seen:
                            rows.append({"id": a_["id"], "spelling": c_, "token_type": "hf-added-nonspecial", "group_size": 1}); seen.add(c_)
                for g in INJ_EXTRA:
                    if g not in seen:
                        rows.append({"id": None, "spelling": g, "token_type": None, "group_size": 1, "generic_probe": True}); seen.add(g)
                tmpl = meta.get("tokenizer.chat_template") or ""
                ent["control_token_count"] = sum(1 for y in types if y in (3, 4))
                def probe(r):
                    sp = r["spelling"]
                    on, off = tok(path, sp, True), tok(path, sp, False)
                    ctl = len(on) == 1 and on != off
                    r.update(ids_parse_special=on, ids_no_parse=off, is_control_token=ctl,
                             regex_matches_current=has_control_tokens(sp),
                             non_ascii=not sp.isascii(),
                             in_chat_template=sp in tmpl)
                    r["status"] = ("CLOSED-BY-REGEX" if r["regex_matches_current"] else "OPEN") if ctl else "not-control"
                    if not ctl and r["token_type"] == "hf-added-nonspecial":
                        r["status"] = "TEXT-MARKER-" + ("CLOSED-BY-REGEX" if r["regex_matches_current"] else "OPEN")
                    if ctl:
                        wrapped = f"before {sp} after"
                        w = tok(path, wrapped, True)
                        r["user_text_became_control"] = on[0] in w
                        r["neutralised_text_stays_literal"] = on[0] not in tok(path, neutralise_control_tokens(wrapped), True)
                    return r
                try:
                    tok(path, "x", True)
                    ent["tokenizer_loadable"] = True
                except RuntimeError as e0:
                    # drafter / TTS-talker GGUFs cannot be opened standalone by llama-tokenize: static fallback
                    ent["tokenizer_loadable"] = False; ent["tokenizer_error"] = str(e0)
                if ent["tokenizer_loadable"]:
                    with ThreadPoolExecutor(4) as ex:
                        ent["spellings"] = list(ex.map(probe, rows))
                else:
                    for r in rows:
                        r.update(is_control_token=r["token_type"] in (3, 4), regex_matches_current=has_control_tokens(r["spelling"]),
                                 non_ascii=not r["spelling"].isascii(), in_chat_template=r["spelling"] in tmpl, static_only=True)
                        r["status"] = ("CLOSED-BY-REGEX" if r["regex_matches_current"] else "OPEN") if r["is_control_token"] else "not-control"
                    ent["spellings"] = [r for r in rows if r["token_type"] is not None]
                ctl = [r for r in ent["spellings"] if r["is_control_token"]]
                ent["text_markers"] = [r["spelling"] for r in ent["spellings"] if r["status"].startswith("TEXT-MARKER")]
                ent["open_text_markers"] = [r["spelling"] for r in ent["spellings"] if r["status"] == "TEXT-MARKER-OPEN"]
                ent["open"] = [r["spelling"] for r in ctl if r["status"] == "OPEN"]
                ent["closed"] = [r["spelling"] for r in ctl if r["status"] != "OPEN"]
                print(f"[ok ] {fam:11s} {label}: loadable={ent['tokenizer_loadable']} control spellings tested={len(ctl)} OPEN={len(ent['open'])} closed={len(ent['closed'])} text-marker-OPEN={len(ent['open_text_markers'])}", flush=True)
            except Exception as e:  # noqa: BLE001
                ent["error"] = repr(e); print("[ERR]", label, repr(e), flush=True)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
