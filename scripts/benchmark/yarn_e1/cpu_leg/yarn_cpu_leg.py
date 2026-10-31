#!/usr/bin/env python3
"""INF-59 CPU leg: YaRN f2/f4 to 524k / 1M on the CPU-served hybrid Qwen3.6-35B-A3B, BEFORE the GPU E1.

Coordinator directive 2026-10-04 (operator): "first test it on the CPU models where RAM is abundant."
The model has the 27B's architecture family (qwen35moe: 3 GDN : 1 gated full-attention, 64-of-256-dim
interleaved-MRoPE rotary, theta 1e7, native 262,144) with 10 attention layers and 2 KV heads, so a 1M q8_0
KV cache is 10.6 GiB of host RAM. It measures, the way an agent session builds context (turn by turn,
prefix reuse, never a cold 1M prefill):
  * per-turn APPEND cost vs depth (8k-token turns: prompt_ms at each occupancy);
  * DECODE-at-depth (128 forced tokens, ignore_eos) at every probe depth;
  * needle retrieval at varied relative distances, and a local-coherence probe (recurrent state past 262k);
  * the short-text regression suite (84 items, paired against the native arm), the vendor's YaRN warning;
  * ONE cold prefill point (compaction case: a compacted context is re-prefilled from scratch).

Arms (one dedicated np 1 instance each, scratch port 18070, no drafter):
  CN   native, -c 262144                     -> build to ~248k
  CY2  YaRN f2, -c 524288  (orig 262144)     -> build to ~512k
  CY4  YaRN f4, -c 1048576 (orig 262144)     -> build to ~1.03M, + cold-prefill point at 128k
  FY2  OPTIONAL Qwen3.8-Flash-Next f2: GGML_FUSED_DECODE_OFF=1 is MANDATORY (qwen4exp-fused.cpp hardcodes
       freq_scale=1/ext_factor=0 in fused decode: YaRN prefill + non-YaRN decode, silently) until the
       DCA/fused-fix patch lands; f16 KV as in production.

CPU governance: AutoKernel's campaign is CPU-only and holds q0-q3 (cpus 0-95, SMT siblings are the same
cores). Every server launch is wrapped in
    region-lock run --cpu-list 0-95 --role bench --tag yarn -- ...
and a step starts only if cpu_window_gate says the window is open with enough time left for it. At a
window boundary the runner SAVES the slot (/slots/0?action=save, right after a turn, so the saved tokens are
an exact prefix of the next request), stops ITS OWN server (captured PIDs only) and resumes later with
/slots/0?action=restore. A 1M build spans several windows by design.

Modes: --build-inputs | --dry-run | --arm CN|CY2|CY4|FY2 --i-own-the-window [--max-hours H] | --gate
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from yarn_proof import check_proof, PROOF_LOG_FLAGS
import yarn_needle as YN  # noqa: E402  (shared I/O, chat, prompts, coherence_gate loader)
import cpu_window_gate as WG  # noqa: E402

INPUTS = HERE / "inputs"
RESULTS = Path(os.environ.get("YARN_CPU_RESULTS", HERE / "results"))
SLOTS = RESULTS / "slots"
PORT = int(os.environ.get("YARN_CPU_PORT", "18070"))
CPUBIN = "/mnt/raid0/llm/kernels/builds/cpu-20260921-ffc1bac82/bin"
STORE = "/mnt/raid0/llm/kernels/production/cpu"
REGION_LOCK = "/workspace/repos/epyc-orchestrator/scripts/region-lock"
TEMPLATE = "/mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja"
M35 = "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf"
MFN = "/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
TURN_TOKENS = 8192
NEEDLE_EVERY = 25_000
CORPUS_TOKENS = 1_045_000
SEED = "inf59-yarn-cpu-leg-20261004"
COLD_DEPTH = 128_000


def yarn_flags(arch: str, ctx: int, factor: int) -> list[str]:
    ov = (f"{arch}.context_length=int:{ctx},{arch}.rope.scaling.type=str:yarn,{arch}.rope.scaling.factor=float:{factor},"
          f"{arch}.rope.scaling.original_context_length=int:262144")
    return ["--rope-scaling", "yarn", "--rope-scale", str(factor), "--yarn-orig-ctx", "262144", "--override-kv", ov]


ARMS = {
    "CN": {"model": M35, "arch": "qwen35moe", "ctx": 262_144, "kv": ["-ctk", "q8_0", "-ctv", "q8_0"], "flags": [],
           "probes": [32_000, 64_000, 128_000, 192_000, 248_000], "env": {}},
    "CY2": {"model": M35, "arch": "qwen35moe", "ctx": 524_288, "kv": ["-ctk", "q8_0", "-ctv", "q8_0"],
            "flags": yarn_flags("qwen35moe", 524_288, 2), "probes": [32_000, 128_000, 248_000, 320_000, 400_000, 510_000], "env": {}},
    "CY4": {"model": M35, "arch": "qwen35moe", "ctx": 1_048_576, "kv": ["-ctk", "q8_0", "-ctv", "q8_0"],
            "flags": yarn_flags("qwen35moe", 1_048_576, 4),
            "probes": [32_000, 128_000, 248_000, 400_000, 510_000, 640_000, 768_000, 896_000, 1_030_000], "env": {},
            "cold": COLD_DEPTH},
    "FY2": {"model": MFN, "arch": "qwen4exp", "ctx": 524_288, "kv": ["-ctk", "f16", "-ctv", "f16"],
            "flags": yarn_flags("qwen4exp", 524_288, 2), "probes": [32_000, 128_000, 248_000, 400_000, 510_000],
            "env": {"GGML_FUSED_DECODE_OFF": "1"}},
}


def proof_patterns(arm: str) -> list[str]:
    a = ARMS[arm]
    ctx = a["ctx"]
    pats = [rf"print_info: n_ctx_train\s*=\s*{ctx}\b", rf"llama_context: n_ctx_seq\s*=\s*{ctx}\b",
            rf"n_slots = 1, n_ctx_slot = {ctx}\b"]
    if a["flags"]:
        f = 1 / int(a["flags"][3])
        pats += [r"print_info: rope scaling\s*=\s*yarn", rf"print_info: freq_scale_train\s*=\s*{f:g}\s*$",
                 r"print_info: n_ctx_orig_yarn\s*=\s*262144\b", rf"llama_context: freq_scale\s*=\s*{f:g}\s*$"]
    else:
        pats += [r"llama_context: freq_scale\s*=\s*1\s*$"]
    if a["arch"] == "qwen35moe":
        kv_mib = ctx * 10_880 / 2**20
        pats.append(rf"llama_kv_cache: size =\s*{kv_mib:.2f} MiB \(\s*{ctx} cells,\s*10 layers")
    return pats


FORBID = YN.FORBID


# ------------------------------------------------------------------------------ inputs
NEEDLE_THINGS = ["survey archive", "water gauge", "seed collection", "tide table", "parish register", "rail depot",
                 "pumping station", "field notebook", "boundary stone", "signal box", "grain store", "ferry ledger"]
SYLL = ["bra", "cal", "dor", "fen", "gal", "hol", "kes", "lin", "mor", "nar", "ost", "pel", "quin", "ros", "sal",
        "tor", "ulm", "var", "wen", "yar"]


def build_inputs(args) -> int:
    tok = YN.hf_tokenizer()
    if tok is None:
        print("needs `tokenizers`: run with the orchestrator venv python", file=sys.stderr)
        return 2
    import random
    rng = random.Random(SEED)
    INPUTS.mkdir(parents=True, exist_ok=True)
    docs = []
    for line in YN.GOVREPORT.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            d = json.loads(line)
            body = re.sub(r"\n{3,}", "\n\n", d["input"][d["document_start_index"]:d["document_end_index"]].strip())
            if len(body) > 2000:
                docs.append((hashlib.sha256(f"{SEED}:{d['id']}".encode()).hexdigest(), d["id"], body))
    docs.sort()
    parts, total, ids = [], 0, []
    for _, did, body in docs:
        txt = f"Report {len(parts) + 1}\n\n{body}\n\n"
        parts.append(txt)
        ids.append(did)
        total += len(tok.encode(txt, add_special_tokens=False).ids)
        if total >= CORPUS_TOKENS + 20_000:
            break
    corpus = "".join(parts)
    enc = tok.encode(corpus, add_special_tokens=False)
    offs = enc.offsets
    # needles every 25k tokens, at paragraph boundaries; distinct fictional places, unique codes
    places, needles = set(), []
    for k, t in enumerate(range(NEEDLE_EVERY // 2, CORPUS_TOKENS, NEEDLE_EVERY)):
        while True:
            pl = "".join(rng.choice(SYLL) for _ in range(3)).capitalize()
            if pl not in places and pl not in corpus:
                break
        places.add(pl)
        while True:
            code = f"{rng.choice('BCDFGHJKLMNPRSTVWXZ')}{rng.choice('BCDFGHJKLMNPRSTVWXZ')}-{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
            if code not in corpus:
                break
        thing = NEEDLE_THINGS[k % len(NEEDLE_THINGS)]
        c = corpus.find("\n\n", offs[min(t, len(offs) - 1)][0]) + 2
        needles.append({"needle": f"C{k:02d}", "corpus_token": t, "char": c, "expected": [code],
                        "statement": f"The {thing} at {pl} is catalogued under reference code {code}.",
                        "question": f"Under what reference code is the {thing} at {pl} catalogued?"})
    # splice needles in, then cut 8k-token turns at paragraph boundaries
    pieces, last = [], 0
    for n in sorted(needles, key=lambda x: x["char"]):
        pieces += [corpus[last:n["char"]], n["statement"] + "\n\n"]
        last = n["char"]
    pieces.append(corpus[last:])
    text = "".join(pieces)
    enc2 = tok.encode(text, add_special_tokens=False)
    starts = [o[0] for o in enc2.offsets]
    turns, start, cum, tpos = [], 0, 0, 0
    while tpos < min(len(enc2.ids), CORPUS_TOKENS):
        tend = min(tpos + TURN_TOKENS, len(enc2.ids) - 1)
        cend = text.find("\n\n", starts[tend])
        cend = len(text) if (cend < 0 or tend >= len(enc2.ids) - 1) else cend + 2
        seg = text[start:cend]
        n = len(tok.encode(seg, add_special_tokens=False).ids)
        cum += n
        turns.append({"turn": len(turns) + 1, "text": seg, "tokens_hf": n, "cum_tokens_hf": cum,
                      "needles": [x["needle"] for x in needles if x["statement"] in seg]})
        start = cend
        tpos = bisect.bisect_left(starts, cend)
        if cend >= len(text):
            break
    for n in needles:
        n["turn"] = next(t["turn"] for t in turns if n["needle"] in t["needles"])
        n["approx_pos"] = next(t["cum_tokens_hf"] - t["tokens_hf"] for t in turns if t["turn"] == n["turn"]) + \
            len(tok.encode(next(t["text"] for t in turns if t["turn"] == n["turn"]).split(n["statement"])[0],
                           add_special_tokens=False).ids)
    YN.write_jsonl(INPUTS / "turns.jsonl", turns)
    YN.write_jsonl(INPUTS / "needles.jsonl", needles)
    man = {"built_at": YN.now_iso(), "seed": SEED, "corpus_documents": ids, "turn_tokens": TURN_TOKENS,
           "short_suite": str(YN.INPUTS / "short_items.jsonl"),
           "short_suite_sha256": YN.sha256_file(YN.INPUTS / "short_items.jsonl"),
           "files": {p.name: YN.sha256_file(p) for p in sorted(INPUTS.glob("*.jsonl"))}}
    YN.write_json(INPUTS / "manifest.json", man)
    print(f"{len(turns)} turns, {turns[-1]['cum_tokens_hf']:,} tokens (HF); {len(needles)} needles")
    return 0


def load_inputs() -> dict:
    man = json.loads((INPUTS / "manifest.json").read_text())
    bad = [n for n, h in man["files"].items() if YN.sha256_file(INPUTS / n) != h]
    return {"manifest": man, "bad": bad, "turns": YN.read_jsonl(INPUTS / "turns.jsonl"),
            "needles": YN.read_jsonl(INPUTS / "needles.jsonl"),
            "short_items": YN.read_jsonl(YN.INPUTS / "short_items.jsonl"),
            "short_truth": YN.read_jsonl(YN.INPUTS / "short_truth.jsonl")}


def messages(turns: list[dict], k: int, tail: str | None = None) -> list[dict]:
    """Turns 1..k as user messages separated by the fixed ack. tail=None -> ends on user turn k (a TURN
    request); tail=str -> ack after turn k, then the probe as the last user message."""
    msgs = []
    for t in turns[:k]:
        if msgs:
            msgs.append({"role": "assistant", "content": YN.ACK})
        msgs.append({"role": "user", "content": YN.DOC_HEAD.format(k=t["turn"]) + t["text"]})
    if tail is not None:
        msgs += [{"role": "assistant", "content": YN.ACK}, {"role": "user", "content": tail}]
    return msgs


# ------------------------------------------------------------------------------ cost model (ETA)
class Cost:
    """Per-turn append and per-token decode seconds vs occupancy n. Seeded from fa_depth results (C0) if
    present, refit online from this arm's own turns and decode probes."""

    def __init__(self, arm: str):
        self.turn_obs: list[tuple[int, float]] = []
        self.dec_obs: list[tuple[int, float]] = []
        # conservative priors until C0 / the arm's own turns refit them (RUNBOOK section 9 bracket):
        self.a_turn, self.b_turn = 60.0, 0.9    # s per 8k turn at depth 0; + s per 1k occupied cells
        self.a_dec, self.b_dec = 0.030, 1.0e-6  # s/token; + s/token per occupied cell (10 attn layers)
        fd = HERE / "results" / "fa_depth.txt"
        shape = "q35moe"
        if fd.exists():
            for line in fd.read_text().splitlines():
                m = re.search(rf"shape={shape} type=q8_0 nb=(\d+) n_kv=(\d+) threads=96 .*: ([\d.]+) us/op", line)
                if m and int(m.group(2)) >= 262144:
                    nb, kv, us = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    if nb == 1:
                        self.b_dec = 10 * us * 1e-6 / kv            # 10 attention layers
                    elif nb == 2048:
                        self.b_turn = 10 * us * 1e-6 / kv * (TURN_TOKENS / 2048) * 1000

    def fit(self):
        for obs, attr in ((self.turn_obs, ("a_turn", "b_turn", 1000.0)), (self.dec_obs, ("a_dec", "b_dec", 1.0))):
            if len(obs) >= 3:
                xs = [o[0] / attr[2] for o in obs[-12:]]
                ys = [o[1] for o in obs[-12:]]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                vx = sum((x - mx) ** 2 for x in xs)
                if vx > 0:
                    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
                    setattr(self, attr[1], max(b, 0.0))
                    setattr(self, attr[0], max(my - b * mx, 0.0))

    def turn_s(self, n: int) -> float:
        return self.a_turn + self.b_turn * n / 1000

    def dec_s(self, n: int) -> float:
        return self.a_dec + self.b_dec * n

    def probe_s(self, n: int) -> float:
        return 7 * 10.0 + (5 * YN.NEEDLE_MAX_TOKENS + 128 + YN.LOCAL_MAX_TOKENS) * self.dec_s(n)


# ------------------------------------------------------------------------------ server lifecycle
class Server:
    def __init__(self, arm: str, out: Path):
        self.arm, self.out = arm, out
        self.log = RESULTS / f"{arm}.server.log"
        self.wrapper: subprocess.Popen | None = None
        self.pid: int | None = None

    def argv(self) -> list[str]:
        a = ARMS[self.arm]
        return ([REGION_LOCK, "run", "--cpu-list", "0-95", "--role", "bench", "--tag", "yarn", "--timeout-s", "600", "--",
                 "numactl", "--interleave=all", "--", "taskset", "-c", "0-95", f"{CPUBIN}/llama-server",
                 "-m", a["model"], "--host", "127.0.0.1", "--port", str(PORT), "-np", "1", "-c", str(a["ctx"]),
                 "-t", "96", "-ub", "2048", "--flash-attn", "on", "--jinja", *a["kv"], "--no-mmap", "--cache-ram", "0",
                 "--chat-template-file", TEMPLATE, "--reasoning", "off", "--device", "none",
                 "--slot-save-path", str(SLOTS) + "/", *PROOF_LOG_FLAGS, *a["flags"]])  # -lv 4: proof lines

    external = False   # selftest only: talk to an already-running (mock) server, launch/kill nothing

    def start(self) -> None:
        if self.external:
            self.pid = YN.port_pid(PORT) or -1
            return
        if os.path.realpath(STORE) != CPUBIN:
            raise SystemExit(f"REFUSING: {STORE} -> {os.path.realpath(STORE)}, expected {CPUBIN}")
        if YN.port_pid(PORT):
            raise SystemExit(f"REFUSING: :{PORT} already in use")
        SLOTS.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ("LLAMA_SERVER_SLOTS_DEBUG", "GGML_CUDA_GDN_STATE_BF16")}
        env.update({"LD_LIBRARY_PATH": CPUBIN, "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores",
                    "OMP_WAIT_POLICY": "active", "OMP_DYNAMIC": "false", "GGML_IQK": "1", **ARMS[self.arm]["env"]})
        with open(self.log, "a") as fh:
            fh.write(f"=== launch {YN.now_iso()} arm {self.arm}\n{' '.join(self.argv())}\n")
        self.wrapper = subprocess.Popen(self.argv(), stdout=open(self.log, "a"), stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 1800:              # region-lock wait (<= 600 s) + 37 GB --no-mmap load
            if self.wrapper.poll() is not None:
                raise RuntimeError(f"server wrapper exited rc={self.wrapper.returncode} (region-lock timeout or load failure); see {self.log}")
            st, _ = YN.http_json(f"http://127.0.0.1:{PORT}/health", timeout=5)
            if st == 200:
                self.pid = YN.port_pid(PORT)
                return
            time.sleep(5)
        raise RuntimeError("server not healthy after 30 min")

    def proof(self) -> dict:
        if self.external:
            return {"ok": True, "external": True}
        log = self.log.read_text(errors="replace")
        res = check_proof(log, proof_patterns(self.arm), FORBID)
        return res

    def stop(self) -> None:
        """Kill ONLY our captured PIDs: the llama-server on our port (verified by cmdline), then the wrapper."""
        if self.external:
            self.pid = None
            return
        for pid in [p for p in (self.pid,) if p]:
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode()
            except OSError:
                continue
            if f"--port\0{PORT}" not in cmd:
                print(f"REFUSING to kill {pid}: not our :{PORT} server", flush=True)
                continue
            os.kill(pid, signal.SIGTERM)
            for _ in range(60):
                if not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(1)
            if Path(f"/proc/{pid}").exists():
                os.kill(pid, signal.SIGKILL)
                time.sleep(2)
            print(f"server PID {pid} {'STILL ALIVE' if Path(f'/proc/{pid}').exists() else 'dead'}", flush=True)
        if self.wrapper and self.wrapper.poll() is None:
            try:
                self.wrapper.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.wrapper.terminate()
                self.wrapper.wait(timeout=30)
        self.pid, self.wrapper = None, None


# ------------------------------------------------------------------------------ run
def run(args) -> int:
    if not args.i_own_the_window:
        print("RUN needs --i-own-the-window", file=sys.stderr)
        return 2
    inp = load_inputs()
    if inp["bad"]:
        print(f"REFUSING: inputs changed: {inp['bad']}", file=sys.stderr)
        return 2
    arm, a = args.arm, ARMS[args.arm]
    out = RESULTS / arm
    out.mkdir(parents=True, exist_ok=True)
    sink = YN.Sink(out / "calls.jsonl")
    stf = out / "state.json"
    st = json.loads(stf.read_text()) if stf.exists() else {"arm": arm, "turn": 0, "depth": 0, "saved_turn": 0,
                                                            "probes_done": [], "short_done": [], "cold_done": False}
    turns, needles = inp["turns"], inp["needles"]
    limit = a["probes"][-1]
    cost = Cost(arm)
    for r in YN.read_jsonl(out / "turns.jsonl") if (out / "turns.jsonl").exists() else []:
        if r.get("prompt_ms") and r.get("depth_before") and r.get("prompt_n", 0) > 1000:
            cost.turn_obs.append((r["depth_before"], r["prompt_ms"] / 1000))
    cost.fit()
    base = f"http://127.0.0.1:{PORT}"
    srv = Server(arm, out)
    if args.selftest_external:
        srv.external = True
        WG.check = lambda *a, **k: (True, "selftest: window gate bypassed", {})  # noqa: E731
    deadline = time.time() + args.max_hours * 3600
    tmap = {t["id"]: t for t in inp["short_truth"]}
    CG = YN.load_cg(Path(args.cg_path))

    def save() -> None:
        stt, body = YN.http_json(f"{base}/slots/0?action=save", {"filename": f"{arm}.bin"}, timeout=3600)
        sink.add({"kind": "slot_save", "http": stt, "body": body, "turn": st["turn"], "depth": st["depth"]})
        if stt == 200:
            st["saved_turn"] = st["turn"]
            YN.write_json(stf, st)

    def need_next() -> tuple[str, float]:
        if len(st["short_done"]) < len(inp["short_items"]):
            return "short", 15 * 60.0
        nxt = [p for p in a["probes"] if p not in st["probes_done"] and st["depth"] >= p]
        if nxt:
            return "probe", cost.probe_s(st["depth"])
        if a.get("cold") and not st["cold_done"] and st["depth"] >= a["cold"]:
            inc = [r for r in (YN.read_jsonl(out / "turns.jsonl") if (out / "turns.jsonl").exists() else [])
                   if r["depth_after"] <= a["cold"] + TURN_TOKENS]
            return "cold", 1.1 * sum((r.get("prompt_ms") or 0) for r in inc) / 1000 + 120
        if st["depth"] >= limit or st["turn"] >= len(turns):
            return "done", 0.0
        return "turn", cost.turn_s(st["depth"])

    last_step = None
    while True:
        step, eta = need_next()
        if step == "done":
            break
        if time.time() > deadline:
            print("max-hours reached", flush=True)
            if srv.pid and last_step == "turn" and st["saved_turn"] != st["turn"]:
                save()
            break
        boot = 0.0 if srv.pid else 6 * 60.0
        if args.block_until:
            # operator-granted dedicated CPU block (AK CPU campaign paused): the window file is not consulted,
            # region-lock still is (the server runs under it)
            left = (dt.datetime.fromisoformat(args.block_until.replace("Z", "+00:00")) - dt.datetime.now(dt.timezone.utc)).total_seconds()
            ok = left > eta + boot + args.margin_min * 60
            why, info = f"dedicated block: {left / 60:.0f} min left, step needs {(eta + boot) / 60:.0f}+{args.margin_min:.0f}", {}
        else:
            ok, why, info = WG.check((eta + boot) / 60, args.margin_min, args.budget_cpus)
        if not ok:
            if srv.pid:
                print(f"window will not fit {step} ({eta / 60:.1f} min): {why}; stopping our server (slot saved at turn {st['saved_turn']})", flush=True)
                if st["saved_turn"] != st["turn"]:
                    print("WARN: last turn not saved (probes ran after the save); resume restores the earlier save", flush=True)
                srv.stop()
            block_over = bool(args.block_until) and dt.datetime.now(dt.timezone.utc) >= \
                dt.datetime.fromisoformat(args.block_until.replace("Z", "+00:00")) - dt.timedelta(minutes=args.margin_min)
            if not args.wait or block_over or args.block_until:
                # in a dedicated block a step that does not fit now will not fit later either: stop cleanly
                print(f"not starting {step}: {why}", flush=True)
                break
            time.sleep(60)
            continue
        if not srv.pid:
            try:
                srv.start()
            except RuntimeError as e:
                print(f"server start failed: {e}", flush=True)
                srv.stop()
                if not args.wait:
                    return 7
                time.sleep(120)
                continue
            pr = srv.proof()
            YN.write_json(out / f"preflight_{int(time.time())}.json", {**pr, "pid": srv.pid, "argv": srv.argv()})
            if not pr["ok"] and not args.force_preflight:
                srv.stop()
                print(f"REFUSING: proof lines missing / forbidden lines: {pr}", file=sys.stderr)
                return 3
            if st["saved_turn"]:
                stt, body = YN.http_json(f"{base}/slots/0?action=restore", {"filename": f"{arm}.bin"}, timeout=3600)
                sink.add({"kind": "slot_restore", "http": stt, "body": body, "saved_turn": st["saved_turn"]})
                if stt != 200:
                    srv.stop()
                    print(f"slot restore failed: {body}", file=sys.stderr)
                    return 4
                st["turn"] = st["saved_turn"]
                st["depth"] = (body or {}).get("n_restored", st["depth"])
                YN.write_json(stf, st)
        if step == "short":
            last_step = "short"
            for it in inp["short_items"]:
                if it["id"] in st["short_done"]:
                    continue
                r = YN.chat(base, [{"role": "user", "content": it["prompt"]}], it["max_tokens"], cache_prompt=False, timeout=3600)
                row = YN.answer_row(base, it["id"], r, {"set": it["set"], "arm": arm, "prompt": it["prompt"]})
                sink.add({"kind": "short", **row})
                with open(out / "short.jsonl", "a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                st["short_done"].append(it["id"])
                YN.write_json(stf, st)
            st["saved_turn"] = 0          # the slot now holds a short prompt; the build restarts from turn 0
            st["turn"], st["depth"] = 0, 0
            YN.write_json(stf, st)
        elif step == "turn":
            last_step = "turn"
            k = st["turn"] + 1
            depth_before = st["depth"]
            r = YN.chat(base, messages(turns, k), 1, cache_prompt=True, timeout=6 * 3600)
            pn, cn = r.get("prompt_n") or 0, r.get("cache_n") or 0
            rec = {"turn": k, "depth_before": depth_before, "depth_after": pn + cn, "prompt_n": pn, "cache_n": cn,
                   "prompt_ms": r.get("prompt_ms"), "turn_tokens_hf": turns[k - 1]["tokens_hf"], "http": r["http"],
                   "error": r["error"], "at": YN.now_iso()}
            sink.add({"kind": "turn", **rec})
            with open(out / "turns.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if not r["http_ok"]:
                print(f"turn {k} failed: {r['error']}", file=sys.stderr)
                srv.stop()
                return 5
            if k > 1 and pn > turns[k - 1]["tokens_hf"] * 1.1 + 256:
                print(f"ABORT: prefix reuse failed at turn {k}: prompt_n {pn} for a {turns[k - 1]['tokens_hf']}-token turn", file=sys.stderr)
                srv.stop()
                return 6
            st["turn"], st["depth"] = k, pn + cn
            YN.write_json(stf, st)
            if pn > 1000:
                cost.turn_obs.append((depth_before, (r.get("prompt_ms") or 0) / 1000))
                cost.fit()
            print(json.dumps({"arm": arm, "turn": k, "depth": st["depth"], "prompt_ms": r.get("prompt_ms"),
                              "eta_next_s": round(cost.turn_s(st["depth"]))}), flush=True)
            due = any(p not in st["probes_done"] and st["depth"] >= p for p in a["probes"])
            if args.block_until:
                left = (dt.datetime.fromisoformat(args.block_until.replace("Z", "+00:00")) - dt.datetime.now(dt.timezone.utc)).total_seconds()
                nxt_fits = left > cost.turn_s(st["depth"]) + args.margin_min * 60
            else:
                nxt_fits, _, _ = WG.check(cost.turn_s(st["depth"]) / 60, args.margin_min, args.budget_cpus)
            if due or not nxt_fits or k % args.save_every == 0 or (a.get("cold") and not st["cold_done"] and st["depth"] >= a["cold"]):
                save()
            if k == 2 and not st.get("restore_selftest"):
                # prove save -> restore leaves an EXACT prefix (no GDN rollback needed) while a failure costs
                # seconds: turn 3's prompt_n check (prefix reuse) then validates the restored state
                save()
                stt, body = YN.http_json(f"{base}/slots/0?action=restore", {"filename": f"{arm}.bin"}, timeout=600)
                sink.add({"kind": "restore_selftest", "http": stt, "body": body, "depth": st["depth"]})
                if stt != 200 or (body or {}).get("n_restored") not in (st["depth"], None):
                    print(f"ABORT: restore self-test: {stt} {body} (depth {st['depth']})", file=sys.stderr)
                    srv.stop()
                    return 8
                st["restore_selftest"] = True
                YN.write_json(stf, st)
        elif step == "probe":
            last_step = "probe"
            p = next(p for p in a["probes"] if p not in st["probes_done"] and st["depth"] >= p)
            D, k = st["depth"], st["turn"]
            avail = [n for n in needles if n["turn"] <= k and n["approx_pos"] < D - 2000]
            chosen = []
            for f in (0.1, 0.3, 0.5, 0.7, 0.9):
                c = min((n for n in avail if n not in chosen), key=lambda n: abs(n["approx_pos"] - f * D), default=None)
                if c:
                    chosen.append(c)
            rows = []
            for n in chosen:
                r = YN.chat(base, messages(turns, k, YN.Q_HEAD + n["question"]), YN.NEEDLE_MAX_TOKENS, True, 6 * 3600)
                row = YN.answer_row(base, n["needle"], r, {"arm": arm, "probe_depth": p, "depth": D,
                                                           "needle_pos": n["approx_pos"], "distance": D - n["approx_pos"],
                                                           "expected": n["expected"]})
                row["grade"] = CG.grade(r["text"], {"id": n["needle"], "expected": n["expected"], "grader": "needle",
                                                    "case_sensitive": False})["status"]
                sink.add({"kind": "needle", **row})
                rows.append(row)
            body = {"messages": messages(turns, k, "Continue writing the last report above, in the same style."),
                    "max_tokens": 128, "temperature": 0.0, "top_k": 1, "ignore_eos": True, "cache_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
            stt, resp = YN.http_json(f"{base}/v1/chat/completions", body, timeout=6 * 3600)
            t = (resp or {}).get("timings") or {} if isinstance(resp, dict) else {}
            dec = {"kind": "decode_probe", "arm": arm, "probe_depth": p, "depth": D, "http": stt,
                   "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                   "decode_tps": t.get("predicted_per_second"), "prompt_n": t.get("prompt_n")}
            sink.add(dec)
            if t.get("predicted_n") and t.get("predicted_ms"):
                cost.dec_obs.append((D, t["predicted_ms"] / 1000 / t["predicted_n"]))
                cost.fit()
            r = YN.chat(base, messages(turns, k, YN.LOCAL_PROBE), YN.LOCAL_MAX_TOKENS, True, 6 * 3600)
            lrow = YN.answer_row(base, f"L_{p}", r, {"arm": arm, "probe_depth": p, "depth": D})
            cls = CG.classify(r["text"], lrow["token_ids"], r["finish"])
            lrow["degeneracy"] = {"cls": cls["cls"], "reasons": cls.get("reasons")}
            sink.add({"kind": "local", **lrow})
            YN.write_json(out / f"probe_{p}.json", {"needles": rows, "decode": dec, "local": lrow})
            st["probes_done"].append(p)
            YN.write_json(stf, st)
            print(json.dumps({"arm": arm, "probe": p, "depth": D, "correct": sum(x["grade"] == "correct" for x in rows),
                              "decode_tps": dec["decode_tps"], "local": cls["cls"]}), flush=True)
        elif step == "cold":
            last_step = "cold"
            k = next(i for i, _ in enumerate(turns, 1) if i <= st["turn"] and
                     YN.read_jsonl(out / "turns.jsonl")[i - 1]["depth_after"] >= a["cold"])
            r = YN.chat(base, messages(turns, k), 1, cache_prompt=False, timeout=12 * 3600)
            inc = sum((x.get("prompt_ms") or 0) for x in YN.read_jsonl(out / "turns.jsonl")[:k]) / 1000
            rec = {"kind": "cold_prefill", "arm": arm, "turns": k, "prompt_n": r.get("prompt_n"),
                   "cold_s": round((r.get("prompt_ms") or 0) / 1000, 1), "incremental_sum_s": round(inc, 1)}
            sink.add(rec)
            YN.write_json(out / "cold_prefill.json", rec)
            st["cold_done"] = True
            # the slot now holds turns 1..k re-prefilled; the build resumes from the last SAVE
            if k != st["turn"]:
                stt, body = YN.http_json(f"{base}/slots/0?action=restore", {"filename": f"{arm}.bin"}, timeout=3600)
                sink.add({"kind": "slot_restore_after_cold", "http": stt, "body": body})
                st["turn"] = st["saved_turn"]
                st["depth"] = (body or {}).get("n_restored", st["depth"])
            YN.write_json(stf, st)
    if srv.pid:
        srv.stop()
    YN.write_json(out / "report.json", {"arm": arm, "state": st, "finished_at": YN.now_iso(),
                                        "cost_fit": vars(cost) | {"turn_obs": len(cost.turn_obs), "dec_obs": len(cost.dec_obs)}})
    return 0


# ------------------------------------------------------------------------------ dry run / gate
def dry_run(args) -> int:
    probs = []
    for p in (M35, YN.GOVREPORT, YN.INPUTS / "short_items.jsonl", REGION_LOCK, INPUTS / "manifest.json"):
        if not Path(p).exists():
            probs.append(f"missing {p}")
    if probs:
        print("\n".join("PROBLEM " + p for p in probs))
        return 1
    inp = load_inputs()
    if inp["bad"]:
        probs.append(f"inputs changed: {inp['bad']}")
    text = "".join(t["text"] for t in inp["turns"])
    for n in inp["needles"]:
        if text.count(n["expected"][0]) != 1 or text.count(n["statement"]) != 1:
            probs.append(f"{n['needle']}: value/statement not unique")
    if os.path.realpath(STORE) != CPUBIN:
        probs.append(f"{STORE} -> {os.path.realpath(STORE)} != {CPUBIN}")
    for arm, a in ARMS.items():
        lim = a["probes"][-1]
        need = next((t for t in inp["turns"] if t["cum_tokens_hf"] >= lim), None)
        if need is None:
            probs.append(f"{arm}: corpus shorter than {lim}")
        elif need["cum_tokens_hf"] + 40 * need["turn"] + 512 > a["ctx"]:
            probs.append(f"{arm}: turn {need['turn']} reaches {need['cum_tokens_hf']} (+template) > ctx {a['ctx']}")
        print(f"{arm}: {' '.join(Server(arm, RESULTS).argv())}")
        print(f"   proof: {proof_patterns(arm)}")
    ok, why, info = WG.check(30)
    print(f"window now: {why} {info}")
    print(f"turns {len(inp['turns'])}, last cum {inp['turns'][-1]['cum_tokens_hf']:,}; needles {len(inp['needles'])}")
    for p in probs:
        print("PROBLEM", p)
    print("DRY-RUN", "OK" if not probs else "FAILED")
    return 0 if not probs else 1


def gate(args) -> int:
    CG = YN.load_cg(Path(args.cg_path))
    rep: dict = {"schema": "inf59.yarn_cpu_leg.v1", "at": YN.now_iso(), "arms": {}}
    md = ["# INF-59 CPU leg (Qwen3.6-35B-A3B, CPU) — results", ""]
    for arm in ARMS:
        d = RESULTS / arm
        if not d.exists():
            continue
        ar: dict = {}
        tr = YN.read_jsonl(d / "turns.jsonl") if (d / "turns.jsonl").exists() else []
        ar["turn_cost"] = [(r["depth_before"], round((r.get("prompt_ms") or 0) / 1000, 1)) for r in tr if r.get("prompt_n", 0) > 1000][::8]
        pr = sorted(d.glob("probe_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        ar["probes"] = []
        for p in pr:
            j = json.loads(p.read_text())
            ar["probes"].append({"depth": j["decode"]["depth"], "decode_tps": j["decode"]["decode_tps"],
                                 "needles": [(n["id"], round(n["distance"] / 1000), n["grade"]) for n in j["needles"]],
                                 "correct": sum(n["grade"] == "correct" for n in j["needles"]),
                                 "local": j["local"]["degeneracy"]["cls"]})
        if (d / "cold_prefill.json").exists():
            ar["cold"] = json.loads((d / "cold_prefill.json").read_text())
        rep["arms"][arm] = ar
        md += [f"## {arm}", "", "| depth | decode tok/s | needles (id, distance k, status) | local |", "|---|---|---|---|"]
        md += [f"| {x['depth']} | {x['decode_tps']} | {x['needles']} | {x['local']} |" for x in ar["probes"]]
        md += ["", f"turn append cost (depth, s per 8k turn), every 8th: {ar['turn_cost']}"]
        if "cold" in ar:
            md.append(f"cold prefill {ar['cold']['prompt_n']} tok: {ar['cold']['cold_s']} s vs incremental sum {ar['cold']['incremental_sum_s']} s")
        md.append("")
    # paired short suite vs CN, and pre-registered thresholds (same as GPU E1 C1/C2/C3)
    base = RESULTS / "CN" / "short.jsonl"
    for arm in ("CY2", "CY4"):
        c = RESULTS / arm / "short.jsonl"
        if base.exists() and c.exists():
            r = CG.evaluate(YN.read_jsonl(base), YN.read_jsonl(c), YN.read_jsonl(YN.INPUTS / "short_truth.jsonl"))
            rep[f"C1_short_{arm}_vs_CN"] = {k: r["aggregate"][k] for k in ("gate", "regressions", "identical", "both_bad", "needs_review", "n")}
            md.append(f"- C1 short {arm} vs CN: {rep[f'C1_short_{arm}_vs_CN']}")
    for arm in ("CY2", "CY4"):
        ar = rep["arms"].get(arm, {})
        beyond = [x for x in ar.get("probes", []) if x["depth"] > 262_144]
        if beyond:
            rep[f"C3_{arm}"] = {"pass": all(x["correct"] >= 4 and x["local"] == "OK" for x in beyond),
                                "per_depth": {x["depth"]: x["correct"] for x in beyond}}
            md.append(f"- C3 {arm} beyond native (>= 4/5 and local OK at every depth): {rep[f'C3_{arm}']}")
    YN.write_json(RESULTS / "verdict.json", rep)
    (RESULTS / "VERDICT.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--build-inputs", action="store_true")
    m.add_argument("--dry-run", action="store_true")
    m.add_argument("--arm", choices=sorted(ARMS))
    m.add_argument("--gate", action="store_true")
    ap.add_argument("--i-own-the-window", action="store_true")
    ap.add_argument("--wait", action="store_true", help="keep waiting for the next open window instead of exiting")
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--margin-min", type=float, default=5.0)
    ap.add_argument("--budget-cpus", help="a NUMA budget that must not overlap AutoKernel (none exists while it holds 0-95)")
    ap.add_argument("--block-until", help="ISO time: end of an operator-granted dedicated CPU block (AK paused); replaces the window gate")
    ap.add_argument("--save-every", type=int, default=8, help="also save the slot every N turns (cache-loss insurance)")
    ap.add_argument("--force-preflight", action="store_true")
    ap.add_argument("--selftest-external", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cg-path", default=str(YN.CG_DEFAULT))
    a = ap.parse_args()
    if a.build_inputs:
        return build_inputs(a)
    if a.dry_run:
        return dry_run(a)
    if a.gate:
        return gate(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
