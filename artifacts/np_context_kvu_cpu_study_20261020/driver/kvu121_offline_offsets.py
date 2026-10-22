"""KVU-12.1a fix (3), offline: derive the REAL rendered token offsets of the recorded KVU-12.1 run (2026-10-22, out dir
/mnt/raid0/llm/tmp/kvu121-arms/out) without a server, so the fixed simulator can be re-judged against the logged checkpoints.

How: the recorded driver (`kvu_ckpt_arms.py` before KVU-12.1a) built every conversation from a corpus whose sha256 it
logged, with a cursor that is a pure function of (model, L, wave). This script rebuilds that corpus from a checkout (the sha must match the
logged one or it refuses), replays the recorded driver's fit bisection and probe sequence with an OFFLINE renderer (the production template
`epyc-qwen3x-v1-terse.jinja` through jinja2, enable_thinking=false as `--reasoning off`) and an OFFLINE tokenizer (the Qwen3.6 tokenizer.json),
and writes, per request: the prompt length, the user-message starts (server rule: positions of the user delimiter tokens in the prompt), every
message start, and the longest common prefix with the previous prompt.

Self-check (the reason the output can be trusted): every request's offline prompt length must equal the length the live server logged
(`n_prompt_total` in requests.jsonl). The script refuses to write a fixture for a cell where any request disagrees.

Needs `tokenizers` and `jinja2` (the research .venv has both; the driver and its tests do not). Starts no server, loads no model.
Usage: .venv/bin/python kvu121_offline_offsets.py --requests REQS --corpus-root /workspace --out fixtures/kvu121_w0_offsets.json [--cells 8k:0,32k:0]
"""
import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kvu_ckpt_arms as K            # noqa: E402  (constants, conv_messages, edit_message, probe_plan, edit_index_for_back, user_starts_from_ids)

TEMPLATE = "/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja"
TOKENIZER = "/mnt/raid0/llm/models/dflash/Qwen3.6-27B-DFlash/tokenizer.json"    # Qwen3.6 family vocabulary (verified by the self-check)
RECORDED_CORPUS_SHA = "bf5d7c7a65994b51b1a3c3a4b9a4a92f02735f47f02fa9e5f9e7a8c8c70fe74c"   # launch.log of the recorded run, files=674
RECORDED_RATIO = 4.0                 # the recorded driver's fixed chars/token
RECORDED_VERSION = "kvu_ckpt_arms.py before KVU-12.1a (byte-identical on research origin/main 91dc86dca and the f8/kvu12-arms checkout the run used)"


class Corpus:
    """Byte-for-byte the recorded driver's corpus (kvu_cpu_matrix.Corpus), built from a working tree."""
    def __init__(self, roots, cap=24_000_000):
        files = []
        for r in roots:
            files += sorted(Path(r).rglob("*.md"))
        files = [f for f in sorted(set(files)) if f.stat().st_size > 2048]
        blob, n = [], 0
        for f in files:
            t = f.read_text(errors="ignore"); blob.append(t); n += len(t)
            if n >= cap:
                break
        self.text = "\n\n".join(blob)
        self.sha = hashlib.sha256(self.text.encode()).hexdigest()
        self.cursor = 0

    def take(self, nchars):
        if self.cursor + nchars > len(self.text):
            self.cursor = 0
        s = self.text[self.cursor:self.cursor + nchars]
        self.cursor += nchars
        return s


class Renderer:
    def __init__(self, template=TEMPLATE, tokenizer=TOKENIZER):
        import jinja2
        from tokenizers import Tokenizer
        env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)
        env.globals["raise_exception"] = raise_exception
        self.tmpl = env.from_string(Path(template).read_text())
        self.tok = Tokenizer.from_file(tokenizer)
        self.user_delim = self.ids(K.USER_DELIM)
        self.msg_delim = self.ids(K.MSG_DELIM)

    def prompt(self, messages):
        return self.tmpl.render(messages=messages, add_generation_prompt=True, enable_thinking=False, bos_token="", eos_token="<|im_end|>")

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def render_ids(self, messages):
        return self.ids(self.prompt(messages))


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def fit_v1(measure, make, target_tok, lo, hi, tol=0.01, max_iter=24):
    """The RECORDED (pre KVU-12.1a) fit_conversation: closest probe, never refuses. Kept here only to replay the recorded run."""
    best = None
    for _ in range(max_iter):
        k = (lo + hi) // 2
        msgs = make(k); n = measure(msgs)
        if best is None or abs(n - target_tok) < abs(best[1] - target_tok):
            best = (msgs, n, k)
        if abs(n - target_tok) <= tol * target_tok or lo >= hi - 1:
            break
        if n < target_tok: lo = k
        else: hi = k
    return best


def replay_cell(R, corpus, model, unit, wave, succeeded):
    """The recorded run_ckpt_cell's request sequence (texts, the ESTIMATED starts it used for edit indices, extend texts) rendered offline.
    `succeeded(probe)` says whether the live request succeeded: the recorded driver advanced the conversation only then."""
    L = K.LS_ARM[unit]
    corpus.cursor = (int(hashlib.sha1(f"{model}|{unit}|{wave}".encode()).hexdigest()[:8], 16) * 7919) % max(1, len(corpus.text) // 2)
    ratio = RECORDED_RATIO
    pair = K.TURN_USER_TOK + K.TURN_ASST_TOK + 16
    n_pairs = max(1, (L - K.PROBE_NPREDICT - 64 - 64) // pair)
    c0 = corpus.cursor

    def make(k):
        texts = []
        corpus.cursor = c0
        for i in range(2 * n_pairs + 1):
            ntok = K.TURN_USER_TOK if i % 2 == 0 else K.TURN_ASST_TOK
            if i == 2 * n_pairs: ntok = max(16, k // 10)
            texts.append(corpus.take(int(ntok * ratio)))
        return K.conv_messages(texts)

    measure = lambda msgs: len(R.render_ids(msgs))
    msgs, n_total, _k = fit_v1(measure, make, L - K.PROBE_NPREDICT - 64, lo=160, hi=(pair + K.TURN_USER_TOK) * 10, tol=0.01)
    texts_tok = [len(R.ids(m["content"])) for m in msgs]
    starts = [0]
    over = max(0, (n_total - sum(texts_tok)) // max(1, len(msgs)))
    for t in texts_tok[:-1]:
        starts.append(starts[-1] + t + over)
    seq, prev = [], None

    def emit(name, messages):
        nonlocal prev
        ids = R.render_ids(messages)
        seq.append(dict(probe=name, n_prompt_total=len(ids), user_starts=K.user_starts_from_ids(ids, R.user_delim),
                        msg_starts=K.user_starts_from_ids(ids, R.msg_delim), lcp_prev=(lcp(prev, ids) if prev is not None else 0)))
        if succeeded(name):
            prev = ids                       # a refused request (HTTP 400) never reached the slot: the cache still holds the previous prompt
        return len(ids)

    emit("cold_build", msgs)
    cur, cur_starts, n_cur = msgs, starts, n_total
    for name, back in K.probe_plan(L, n_cur):
        if name == "extend_1":
            tx = [corpus.take(int(K.TURN_ASST_TOK * ratio)), corpus.take(int(K.TURN_USER_TOK * ratio))]
            nxt = cur[:] + [{"role": "assistant", "content": tx[0]}, {"role": "user", "content": tx[1]}]
        else:
            idx = K.edit_index_for_back(cur_starts, n_cur, back)
            nxt = K.edit_message(cur, idx, f"[edit w{wave} {name}] ")
        n_new = emit(name, nxt)
        if succeeded(name):
            if name == "extend_1":
                cur_starts = cur_starts + [n_cur, n_cur + K.TURN_ASST_TOK + over]
            cur, n_cur = nxt, n_new
    return seq


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--requests", default="/mnt/raid0/llm/tmp/kvu121-arms/out/requests.jsonl")
    p.add_argument("--corpus-root", default="/workspace")
    p.add_argument("--cells", default="8k:0,32k:0", help="unit:wave list (the recorded wave-0 cells by default)")
    p.add_argument("--model", default="q36_35b")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    corpus = Corpus([f"{a.corpus_root}/handoffs", f"{a.corpus_root}/docs"])
    if corpus.sha != RECORDED_CORPUS_SHA:
        sys.exit(f"REFUSED: corpus sha256 {corpus.sha} != recorded {RECORDED_CORPUS_SHA}; check out the tree the run read")
    R = Renderer()
    logged = [json.loads(x) for x in Path(a.requests).read_text().splitlines() if x.strip()]
    out = dict(schema="epyc.kvu121.offline_offsets.v1", derived_by="kvu121_offline_offsets.py", replays=RECORDED_VERSION, corpus_sha256=corpus.sha,
               template=TEMPLATE, tokenizer=TOKENIZER, user_delim=K.USER_DELIM, user_delim_ids=R.user_delim, cells={})
    bad = []
    for spec in a.cells.split(","):
        unit, wave = spec.split(":"); wave = int(wave)
        # every arm of a wave sends the same text; the dense arm's rows say which requests succeeded
        by_probe = {r["probe"]: r for r in logged if r.get("unit") == unit and r.get("wave") == wave and r.get("policy_arm") == "dense"}
        ok = lambda n: bool(by_probe.get(n) and by_probe[n].get("error") is None and by_probe[n].get("status") == 200)
        rows = replay_cell(R, corpus, a.model, unit, wave, ok)
        for r in rows:
            want = (by_probe.get(r["probe"]) or {}).get("n_prompt_total")
            r["logged_n_prompt_total"] = want
            if ok(r["probe"]) and want != r["n_prompt_total"]:
                bad.append(f"{unit} w{wave} {r['probe']}: offline {r['n_prompt_total']} != logged {want}")
        out["cells"][f"{unit}:w{wave}"] = rows
        print(f"{unit} w{wave}: " + ", ".join(f"{r['probe']} {r['n_prompt_total']}/{r['logged_n_prompt_total']} users={len(r['user_starts'])}" for r in rows))
    if bad:
        sys.exit("REFUSED (self-check): " + "; ".join(bad))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
